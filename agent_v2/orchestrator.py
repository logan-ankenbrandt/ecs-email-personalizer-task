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
from utils.research import _extract_website, build_recipient_summary

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
    # Use _extract_website (utils.research) to handle the v1/v2 schema
    # split where custom_fields is a list of {key, value} in production.
    # Bug: the previous inline dict access crashed with
    # "'list' object has no attribute 'get'" on real recipient docs.
    has_website = bool(_extract_website(recipient))
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
            # A.3: special case — if the orchestrator is submitting at the
            # writer-self-validated baseline (0.85) against a higher prior,
            # the delta isn't a real regression. Nudge the orchestrator
            # toward calling the critic for a real score instead of
            # retrying the writer (which produces the same 0.85 default).
            SELF_VALIDATED_BASELINE = 0.85
            if abs(quality_score - SELF_VALIDATED_BASELINE) < 0.01:
                logger.info(
                    "orchestrator submit_step writer_self_validated_baseline_regression_avoided "
                    "recipient=%s step=%d prev=%.2f baseline=%.2f — suggesting critic",
                    rid, step, prev_score, quality_score,
                )
                return {
                    "ok": False,
                    "reason": (
                        f"writer-self-validated baseline {quality_score:.2f} is "
                        f"below prior {prev_score:.2f}. Call dispatch_critic on "
                        f"this draft_id to get a real score, then retry "
                        f"submit_step with the critic's overall_score."
                    ),
                }
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

        # Round 4.6 A: cross-step proof-point recycling detector. Compare
        # this draft's proof signatures against every earlier accepted
        # step. Any shared signature (e.g. both steps use "4x" + "90
        # days") is a recycling defect that reads like template filler
        # by the time the sequence reader hits step 3. Flag on the LATER
        # doc so the offending rewrite surfaces in the UI.
        from agent_v2.memory import extract_proof_signatures

        this_sigs = extract_proof_signatures(draft.get("content", "") or "")
        if this_sigs and memory.accepted:
            for earlier_step in sorted(memory.accepted.keys()):
                if earlier_step >= step:
                    continue
                earlier = memory.accepted[earlier_step]
                earlier_sigs = extract_proof_signatures(earlier.raw_content or "")
                shared = this_sigs & earlier_sigs
                if shared:
                    logger.warning(
                        "orchestrator proof_recycled recipient=%s step=%d "
                        "recycled_from=%d shared=%s",
                        rid, step, earlier_step, sorted(shared),
                    )
                    slop_warnings.append({
                        "pattern_type": "proof_recycled",
                        "email_position": step,
                        "field": "content",
                        "excerpt": (
                            f"[recycled proof-point signatures {sorted(shared)} "
                            f"also used in step {earlier_step}]"
                        ),
                        "severity": "hard_fail",
                        "issue": (
                            f"Proof point(s) {sorted(shared)} already appear "
                            f"in step {earlier_step}. Reader sees the same "
                            f"stat twice. Pick a different proof point."
                        ),
                    })

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

        # Tier B.3: doom-loop detection. If the same tool + identical input
        # is about to fire for the 3rd time in a row, intervene with a
        # tool-specific corrective message so the orchestrator stops burning
        # budget on pathological retries. Not enforced on small/read-only
        # tools (they're cheap enough that repeated calls don't hurt).
        _DOOM_CHECKED = {
            "dispatch_writer",
            "dispatch_critic",
            "dispatch_researcher",
            "get_recipient_brief",
        }
        if name in _DOOM_CHECKED and memory.is_doom_loop(name, tool_input):
            logger.warning(
                "orchestrator doom_loop_detected recipient=%s tool=%s "
                "input_keys=%s — blocking and advising change",
                rid, name, sorted(tool_input.keys()),
            )
            if name == "dispatch_writer":
                advice = (
                    "DOOM LOOP: you've dispatched writer with the same input "
                    "3x. Do NOT retry with identical arguments. Either "
                    "(1) add specific `constraints` describing what should "
                    "change, or (2) call dispatch_critic on the existing "
                    "draft_id to get a real score, or (3) call skip_step "
                    "with reason='persistent_quality_failure'."
                )
            elif name == "dispatch_critic":
                advice = (
                    "DOOM LOOP: you've critiqued the same draft_id 3x. Critic "
                    "is deterministic; identical input returns identical "
                    "verdict. Call submit_step with the score you already "
                    "have, or skip_step."
                )
            elif name == "dispatch_researcher":
                advice = (
                    "DOOM LOOP: researcher already ran. Use get_recipient_brief "
                    "to fetch the cached result; do not re-dispatch."
                )
            else:  # get_recipient_brief
                advice = (
                    "DOOM LOOP: you've called get_recipient_brief 3x in a row. "
                    "The brief is cached after the first call. Move on to "
                    "dispatch_writer or another step."
                )
            return ToolResult(content=advice, is_error=True)

        # Record the call BEFORE dispatch so the next check sees this one
        # in the ring buffer.
        if name in _DOOM_CHECKED:
            memory.record_tool_call(name, tool_input)

        if name == "list_recipient_gaps":
            gaps = list_recipient_gaps(
                recipient_id=rid,
                sequence_id=str(sequence_id),
                template_steps=all_template_steps,
            )
            # A.1: in rewrite mode (user explicitly asked for regeneration via
            # scope=recipient or scope=all), every step in active_steps is
            # work to do. Previously the orchestrator correctly categorized
            # high-scoring prior docs as "done" and skipped them, but the
            # user's intent on a scope=recipient click is "regenerate ALL of
            # this recipient's emails." Promote done -> needs_rewrite so the
            # orchestrator treats the full active set as work. Prior scores
            # are still accessible via the regression guard in _submit_step.
            if is_rewrite:
                filtered = {
                    "done": [],
                    "needs_rewrite": sorted(active_steps),
                    "missing": sorted(
                        s for s in gaps["missing"] if s in active_steps
                    ),
                }
            else:
                # Initial batch: categorize normally so the orchestrator can
                # skip already-done steps instead of redoing them.
                filtered = {
                    "done": sorted(
                        s for s in gaps["done"] if s in active_steps
                    ),
                    "needs_rewrite": sorted(
                        s for s in gaps["needs_rewrite"] if s in active_steps
                    ),
                    "missing": sorted(
                        s for s in gaps["missing"] if s in active_steps
                    ),
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
    # Tier B.1 compaction + Tier E.1 budget telemetry: combined per-turn
    # hook. Prunes stale tool_result payloads AND appends a structured
    # session_status block to the last user message so the orchestrator
    # can pace itself as budget depletes. Modeled on Claude Code's
    # taskBudget.remaining surfacing.
    from agent_v2.memory import compact_messages

    def _pre_turn_hook(msgs: List[Dict[str, Any]]) -> None:
        # Compact first so downstream token estimates reflect the trimmed
        # message list.
        compacted = compact_messages(msgs)
        if compacted:
            logger.info(
                "orchestrator compact_messages recipient=%s compacted=%d",
                rid, compacted,
            )
        # Build session_status and attach to the last user message. Only
        # fires after the first orchestrator turn (when there's at least
        # one tool_result user message to attach to).
        if not msgs:
            return
        # Find the last user message — that's the one the model reads
        # before its next reply.
        last_user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return

        turns_used = budget.turn_count
        turns_remaining = max(0, budget.max_turns - turns_used)
        usd_spent = round(budget.cost_usd, 3)
        usd_remaining = round(max(0.0, budget.max_usd - budget.cost_usd), 3)
        steps_resolved = sorted(list(memory.accepted.keys()) + list(memory.skipped.keys()))
        steps_pending = sorted([s for s in active_steps if s not in memory.accepted and s not in memory.skipped])
        status_parts = [
            "<session_status>",
            f"budget_spent_usd: {usd_spent}",
            f"budget_remaining_usd: {usd_remaining}",
            f"turns_used: {turns_used}",
            f"turns_remaining: {turns_remaining}",
            f"steps_resolved: {steps_resolved}",
            f"steps_pending: {steps_pending}",
        ]
        if usd_remaining < 0.10 or turns_remaining <= 2:
            status_parts.append(
                "URGENT: You are near budget exhaustion. For each pending "
                "step, either submit the best available draft now or call "
                "skip_step. Do not dispatch new writer or critic calls."
            )
        status_parts.append("</session_status>")
        status_block = "\n".join(status_parts)

        # Append to the last user message's content. Handles both
        # list-of-blocks and string content shapes.
        last_user = msgs[last_user_idx]
        current = last_user.get("content", "")
        if isinstance(current, list):
            # Tool-result list: append a text block.
            new_content = list(current)
            new_content.append({"type": "text", "text": "\n\n" + status_block})
            msgs[last_user_idx] = {**last_user, "content": new_content}
        elif isinstance(current, str):
            msgs[last_user_idx] = {
                **last_user,
                "content": current.rstrip() + "\n\n" + status_block,
            }

    loop_result = call_with_tools_loop(
        system_prompt=orchestrator_prompt,
        messages=messages,
        tools=ORCHESTRATOR_TOOLS,
        model=JUDGE_MODEL_FINAL,  # Opus as orchestrator
        max_turns=budget.max_turns,
        tool_handler=tool_handler,
        max_tokens_per_turn=4096,
        temperature=0.2,
        pre_turn_hook=_pre_turn_hook,
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
