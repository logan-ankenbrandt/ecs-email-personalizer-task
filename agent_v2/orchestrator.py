"""Opus orchestrator: one session per recipient.

run_for_recipient is the V2 entry point. The pipeline calls it from
PersonalizerPipeline._personalize_one_recipient when ORCHESTRATOR_V2=1.

The orchestrator runs a while-loop via call_with_tools_loop. The model
decides when the session is done by producing a text-only response (or
by exhausting budget). The orchestrator never writes copy, never reads
draft text unless it asks, and never bypasses the sub-agent dispatchers.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config import (
    JUDGE_MODEL_FINAL,
    QUALITY_HARD_FLOOR,
    QUALITY_HARD_FLOOR_NO_RESEARCH,
    QUALITY_THRESHOLD,
    QUALITY_THRESHOLD_NO_RESEARCH,
)
from utils.cost import CostAccumulator, usd_for_tokens
from utils.merge_fields import build_merge_dict, resolve_merge_fields
from utils.mongo import upsert_personalized_email
from utils.research import build_recipient_summary

from agent_v2.budget import Budget
from agent_v2.loop import ToolResult, call_with_tools_loop
from agent_v2.memory import RecipientMemory
from agent_v2.schemas import ORCHESTRATOR_TOOLS
from agent_v2.tools.dispatchers import (
    handle_dispatch_critic,
    handle_dispatch_researcher,
    handle_dispatch_writer,
    handle_get_recipient_brief,
)
from agent_v2.tools.drafts_store import read_draft_fields
from agent_v2.tools.gaps import list_recipient_gaps

logger = logging.getLogger(__name__)


# ============================================================
# System prompt loading (cached at module level)
# ============================================================

_PROMPT_DIR = Path(__file__).parent / "system_prompts"
_prompt_cache: Dict[str, str] = {}


def _load_prompt(name: str) -> str:
    """Load a system prompt by name. Cached after first load.

    Falls back to empty string if the file is missing so unit tests that
    stub the directory don't crash. Production ECS image will have all
    four prompts present.
    """
    if name not in _prompt_cache:
        path = _PROMPT_DIR / f"{name}.md"
        try:
            _prompt_cache[name] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("System prompt file missing: %s", path)
            _prompt_cache[name] = ""
    return _prompt_cache[name]


# ============================================================
# Result type
# ============================================================

@dataclass
class OrchestratorResult:
    any_step_succeeded: bool
    steps_submitted: List[int] = field(default_factory=list)
    steps_skipped: Dict[int, str] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    turns_used: int = 0
    budget_exhausted: Optional[str] = None  # "max_turns" / "max_usd" / "max_sec" / None


# ============================================================
# Main entry point
# ============================================================

def run_for_recipient(
    recipient: Dict[str, Any],
    sequence_doc: Dict[str, Any],
    template_emails: List[Dict[str, Any]],
    target_steps: Optional[Set[int]],
    is_rewrite: bool,
    feedback: Optional[str],
    previous_versions_fn: Callable[[str, int], Optional[Dict[str, Any]]],
    sequence_id: str,
    personalization_run_id: str,
    cost_tracker: CostAccumulator,
    quality_example: str,
    writer_system_prompt_extra: str = "",
) -> OrchestratorResult:
    """Run one orchestrator session for one recipient.

    Args:
        recipient: Mongo recipient doc.
        sequence_doc: Mongo email_sequences doc.
        template_emails: list of sequence_emails docs (ordered by step).
        target_steps: restrict to these steps if set (targeted rewrite);
            None = all template steps.
        is_rewrite: True when the outer pipeline is in rewrite mode.
        feedback: optional user feedback for rewrite.
        previous_versions_fn: callable(rid, step) -> prior personalized doc
            snapshot or None. Only called in rewrite mode.
        sequence_id: string _id.
        personalization_run_id: run identifier for this batch.
        cost_tracker: pipeline-level CostAccumulator (thread-safe).
        quality_example: cached 0.92 reference email (may be empty).
        writer_system_prompt_extra: optional suffix appended to the writer
            system prompt. Unused today but kept for future extensibility.

    Returns OrchestratorResult.
    """
    rid = str(recipient["_id"])

    memory = RecipientMemory(recipient_id=rid, sequence_id=str(sequence_id))
    budget = Budget()

    # Load all four system prompts once. They're cached after first call.
    orchestrator_prompt = _load_prompt("orchestrator")
    researcher_prompt = _load_prompt("researcher")
    writer_prompt = _load_prompt("writer") + (writer_system_prompt_extra or "")
    critic_prompt = _load_prompt("critic")

    recipient_summary = build_recipient_summary(recipient)
    merge_dict = build_merge_dict(recipient)
    available_merge_keys = sorted(merge_dict.keys())

    # Determine template step universe + target filter.
    all_template_steps: Set[int] = {int(t.get("step", 0)) for t in template_emails}
    if target_steps:
        active_steps = {s for s in target_steps if s in all_template_steps}
    else:
        active_steps = all_template_steps

    # Build a quick step -> template dict for upsert lookups.
    template_by_step: Dict[int, Dict[str, Any]] = {
        int(t.get("step", 0)): t for t in template_emails
    }

    # Quality thresholds depend on whether we have research capability.
    # We don't yet know if the researcher will succeed; start with the
    # has-research tier, then downgrade after gap detection if needed.
    has_website = bool(
        recipient.get("company_website")
        or recipient.get("website")
        or (recipient.get("custom_fields") or {}).get("company_website")
    )
    effective_threshold = QUALITY_THRESHOLD if has_website else QUALITY_THRESHOLD_NO_RESEARCH
    effective_floor = QUALITY_HARD_FLOOR if has_website else QUALITY_HARD_FLOOR_NO_RESEARCH

    # ------------------------------------------------------------
    # submit_step / skip_step: orchestrator-directed terminal tools
    # ------------------------------------------------------------

    def _submit_step(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        step = int(tool_input.get("step", 0))
        draft_id = tool_input.get("draft_id")
        quality_score = float(tool_input.get("quality_score", 0.80))

        if step not in template_by_step:
            return {"ok": False, "reason": f"unknown step {step}"}
        if not draft_id or draft_id not in memory.drafts:
            return {"ok": False, "reason": f"unknown draft_id {draft_id!r}"}
        draft = memory.drafts[draft_id]

        # Hard floor gate: refuse to upsert below the floor.
        if quality_score < effective_floor:
            logger.warning(
                "orchestrator submit_step BELOW_FLOOR recipient=%s step=%d "
                "score=%.2f floor=%.2f",
                rid, step, quality_score, effective_floor,
            )
            return {
                "ok": False,
                "reason": f"score {quality_score:.2f} below floor {effective_floor:.2f}",
            }

        # Regression guard for rewrite mode.
        prev = previous_versions_fn(rid, step) if is_rewrite else None
        prev_score = prev.get("quality_score") if prev else None
        if (
            is_rewrite
            and prev is not None
            and isinstance(prev_score, (int, float))
            and (prev_score - quality_score) > 0.05
        ):
            logger.warning(
                "orchestrator submit_step REGRESSION_BLOCKED recipient=%s step=%d "
                "prev=%.2f new=%.2f delta=%.2f",
                rid, step, prev_score, quality_score, prev_score - quality_score,
            )
            return {
                "ok": False,
                "reason": (
                    f"regression: new score {quality_score:.2f} < prior "
                    f"{prev_score:.2f} by more than 0.05"
                ),
            }

        # Merge-field resolve, then upsert.
        resolved_subject = resolve_merge_fields(draft.get("subject", ""), merge_dict)
        resolved_content = resolve_merge_fields(draft.get("content", ""), merge_dict)

        # Re-run slop validation on the final resolved copy for audit.
        from utils.slop_validation import validate_email
        final_violations = validate_email(resolved_subject, resolved_content, step)
        slop_warnings = [v.to_dict() for v in final_violations]

        template = template_by_step[step]
        try:
            ok = upsert_personalized_email(
                email_sequence_id=sequence_id,
                recipient_id=rid,
                step=step,
                subject=resolved_subject,
                content=resolved_content,
                personalization_run_id=personalization_run_id,
                quality_score=quality_score,
                company_insight=draft.get("company_insight", ""),
                data_grounding=draft.get("data_grounding", []),
                slop_warnings=slop_warnings,
                advisor_used=False,  # V2 has no advisor tool
                original_template_id=str(template.get("_id")),
                dimension_scores=None,  # Set only if critic ran; orchestrator
                                        # doesn't currently pipe dim_scores back
                                        # through submit_step. Could in a follow-up.
                previous_version=prev,
                last_rewrite_feedback=feedback,
            )
        except Exception as e:  # noqa: BLE001 — Mongo failure shouldn't crash session
            logger.error(
                "orchestrator upsert failed recipient=%s step=%d: %s",
                rid, step, e, exc_info=True,
            )
            return {"ok": False, "reason": f"mongo_error: {type(e).__name__}"}

        if ok:
            memory.record_accepted(
                step=step,
                subject=resolved_subject,
                content=resolved_content,
                raw_content=draft.get("content", ""),
                score=quality_score,
                dimension_scores=None,
                slop_warnings=slop_warnings,
                company_insight=draft.get("company_insight", ""),
                data_grounding=draft.get("data_grounding", []),
            )
            logger.info(
                "orchestrator submit_step ACCEPTED recipient=%s step=%d "
                "score=%.2f slop_warnings=%d",
                rid, step, quality_score, len(slop_warnings),
            )
        return {"ok": bool(ok), "score": quality_score}

    def _skip_step(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        step = int(tool_input.get("step", 0))
        reason = (tool_input.get("reason") or "unspecified").strip()
        memory.record_skipped(step, reason)
        logger.warning(
            "orchestrator skip_step recipient=%s step=%d reason=%s",
            rid, step, reason,
        )
        return {"ok": True}

    # ------------------------------------------------------------
    # Tool handler dispatch
    # ------------------------------------------------------------

    def tool_handler(name: str, tool_input: Dict[str, Any], tool_use_id: str) -> ToolResult:
        memory.log_decision(name, {"input_keys": sorted(tool_input.keys())})

        if name == "list_recipient_gaps":
            gaps = list_recipient_gaps(
                recipient_id=rid,
                sequence_id=str(sequence_id),
                template_steps=all_template_steps,
            )
            # Filter gaps against active_steps so the orchestrator only sees
            # steps it's supposed to work on in this run.
            filtered = {
                "done": sorted(s for s in gaps["done"] if s in active_steps),
                "needs_rewrite": sorted(
                    s for s in gaps["needs_rewrite"] if s in active_steps
                ),
                "missing": sorted(s for s in gaps["missing"] if s in active_steps),
            }
            return ToolResult(content=json.dumps(filtered))

        if name == "get_recipient_brief":
            result = handle_get_recipient_brief(
                tool_input=tool_input,
                memory=memory,
                budget=budget,
                cost_tracker=cost_tracker,
                recipient=recipient,
                researcher_system_prompt=researcher_prompt,
            )
            return ToolResult(content=json.dumps(result))

        if name == "dispatch_researcher":
            result = handle_dispatch_researcher(
                tool_input=tool_input,
                memory=memory,
                budget=budget,
                cost_tracker=cost_tracker,
                recipient=recipient,
                system_prompt=researcher_prompt,
            )
            return ToolResult(content=json.dumps(result))

        if name == "dispatch_writer":
            result = handle_dispatch_writer(
                tool_input=tool_input,
                memory=memory,
                budget=budget,
                cost_tracker=cost_tracker,
                recipient_summary=recipient_summary,
                available_merge_keys=available_merge_keys,
                feedback=feedback,
                quality_example=quality_example,
                writer_system_prompt=writer_prompt,
            )
            return ToolResult(content=json.dumps(result))

        if name == "dispatch_critic":
            result = handle_dispatch_critic(
                tool_input=tool_input,
                memory=memory,
                budget=budget,
                cost_tracker=cost_tracker,
                recipient_summary=recipient_summary,
                critic_system_prompt=critic_prompt,
            )
            return ToolResult(content=json.dumps(result))

        if name == "read_draft":
            draft_id = tool_input.get("draft_id")
            fields = tool_input.get("fields") or []
            if not draft_id or draft_id not in memory.drafts:
                return ToolResult(
                    content=json.dumps({"ok": False, "reason": "unknown draft_id"}),
                    is_error=True,
                )
            out = read_draft_fields(memory.drafts[draft_id], fields)
            # Cap content to 4K chars so we don't blow up the orchestrator's
            # context if it asks for the full body.
            if "content" in out and isinstance(out["content"], str):
                out["content"] = out["content"][:4000]
            return ToolResult(content=json.dumps(out, default=str))

        if name == "submit_step":
            result = _submit_step(tool_input)
            # Don't signal terminal — orchestrator may submit multiple steps.
            return ToolResult(content=json.dumps(result))

        if name == "skip_step":
            result = _skip_step(tool_input)
            return ToolResult(content=json.dumps(result))

        return ToolResult(
            content=f"[unknown tool: {name}]",
            is_error=True,
        )

    # ------------------------------------------------------------
    # Run the orchestrator loop
    # ------------------------------------------------------------

    initial_task = _build_orchestrator_task(
        recipient=recipient,
        sequence_doc=sequence_doc,
        active_steps=sorted(active_steps),
        is_rewrite=is_rewrite,
        feedback=feedback,
        budget=budget,
    )
    messages = [{"role": "user", "content": initial_task}]

    logger.info(
        "orchestrator_session START recipient=%s sequence=%s active_steps=%s "
        "is_rewrite=%s budget_usd=%.2f",
        rid, sequence_id, sorted(active_steps), is_rewrite, budget.max_usd,
    )

    # Custom per-turn guard: before the model issues each turn, check the
    # budget. call_with_tools_loop itself doesn't know about our Budget;
    # we cap via max_turns here and also let the tool_handler refuse new
    # dispatches when budget is exhausted (future enhancement — today we
    # rely on max_turns and spend tracking after each call).
    loop_result = call_with_tools_loop(
        system_prompt=orchestrator_prompt,
        messages=messages,
        tools=ORCHESTRATOR_TOOLS,
        model=JUDGE_MODEL_FINAL,  # Opus as orchestrator
        max_turns=budget.max_turns,
        tool_handler=tool_handler,
        max_tokens_per_turn=4096,
        temperature=0.2,
    )

    # Record orchestrator's own cost (the Opus turns).
    cost_tracker.record(
        "orchestrator",
        JUDGE_MODEL_FINAL,
        loop_result.usage.input_tokens,
        loop_result.usage.output_tokens,
    )
    budget.spend(usd_for_tokens(
        JUDGE_MODEL_FINAL,
        loop_result.usage.input_tokens,
        loop_result.usage.output_tokens,
    ))
    for _ in range(loop_result.usage.turns_used):
        budget.tick_turn()

    # Finalize: any steps in active_steps that weren't resolved get skipped
    # with budget_exhausted (if budget died) or persistent_unknown (if the
    # orchestrator just ended its turn without processing them).
    unresolved = [
        s for s in sorted(active_steps)
        if s not in memory.accepted and s not in memory.skipped
    ]
    if unresolved:
        exhausted_reason = budget.reason_exhausted() or "orchestrator_ended_early"
        for s in unresolved:
            memory.record_skipped(s, exhausted_reason)
            logger.warning(
                "orchestrator unresolved step=%d recipient=%s reason=%s",
                s, rid, exhausted_reason,
            )

    logger.info(
        "orchestrator_session END recipient=%s submitted=%s skipped=%s "
        "turns=%d cost=$%.4f exhausted=%s stop_reason=%s",
        rid, sorted(memory.accepted.keys()), dict(memory.skipped),
        budget.turn_count, budget.cost_usd, budget.reason_exhausted(),
        loop_result.stop_reason,
    )

    return OrchestratorResult(
        any_step_succeeded=bool(memory.accepted),
        steps_submitted=sorted(memory.accepted.keys()),
        steps_skipped=dict(memory.skipped),
        total_cost_usd=round(budget.cost_usd, 4),
        turns_used=budget.turn_count,
        budget_exhausted=budget.reason_exhausted(),
    )


def _build_orchestrator_task(
    recipient: Dict[str, Any],
    sequence_doc: Dict[str, Any],
    active_steps: List[int],
    is_rewrite: bool,
    feedback: Optional[str],
    budget: Budget,
) -> str:
    """Build the initial user message for the orchestrator.

    Keeps it compact — the orchestrator's system prompt holds the rules;
    this is just "here's what to work on." No recipient summary or
    template text in the orchestrator's context (those live in the writer
    sub-agent's task).
    """
    rid = str(recipient["_id"])
    name_parts = [recipient.get("first_name", ""), recipient.get("last_name", "")]
    full_name = " ".join(p for p in name_parts if p).strip() or "(unknown)"
    company = recipient.get("business_name") or recipient.get("company") or "(unknown)"

    lines = [
        "## Your task",
        f"- Recipient id: {rid}",
        f"- Recipient name: {full_name}",
        f"- Company: {company}",
        f"- Sequence id: {sequence_doc.get('_id')!s}",
        f"- Sequence name: {sequence_doc.get('name', '(unknown)')!r}",
        f"- Active steps to resolve: {active_steps}",
        f"- Mode: {'rewrite' if is_rewrite else 'initial_batch'}",
    ]
    if feedback:
        lines.append(f"- User feedback for this rewrite: {feedback!r}")
    lines.append(
        f"- Budget: max_turns={budget.max_turns}, max_usd=${budget.max_usd:.2f}, "
        f"max_sec={budget.max_sec}"
    )
    lines.append(
        "\nStart by calling list_recipient_gaps. Then get_recipient_brief. "
        "Then dispatch writers for each active step, submit or skip based "
        "on your heuristics. End your turn when all active steps have a "
        "resolution."
    )
    return "\n".join(lines)
