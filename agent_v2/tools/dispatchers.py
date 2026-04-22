"""Orchestrator-side handlers for the dispatch_* tools.

Each handler invokes the corresponding sub-agent, records cost, updates
RecipientMemory, and returns a SUMMARY (not raw draft text) for the
orchestrator to reason over. Keeping drafts out of the orchestrator's
context window is a deliberate Claude-Code-style pattern: the
orchestrator coordinates; it does not read copy.

Only `read_draft` (handled in orchestrator.py, not here) returns raw
draft fields, and only for the fields the orchestrator explicitly asks for.
"""

import json
import logging
from typing import Any, Dict, Optional

from config import RESEARCH_MODEL, WRITER_MODEL, JUDGE_MODEL_FINAL
from utils.cost import CostAccumulator, usd_for_tokens

from agent_v2.budget import Budget
from agent_v2.loop import TokenUsage
from agent_v2.memory import RecipientMemory
from agent_v2.subagents.critic import run_critic
from agent_v2.subagents.researcher import run_researcher
from agent_v2.subagents.writer import run_writer
from agent_v2.tools.drafts_store import new_draft_id

logger = logging.getLogger(__name__)


def _record_sub_agent_cost(
    cost_tracker: CostAccumulator,
    budget: Budget,
    phase: str,
    model: str,
    usage,
) -> None:
    """Record sub-agent cost into both the pipeline-level CostAccumulator
    and the orchestrator's per-recipient Budget.

    Accepts either a TokenUsage dataclass (from call_with_tools_loop) or a
    plain dict {input_tokens, output_tokens} (from generate_structured).
    """
    if isinstance(usage, TokenUsage):
        in_t = usage.input_tokens
        out_t = usage.output_tokens
    elif isinstance(usage, dict):
        in_t = int(usage.get("input_tokens", 0) or 0)
        out_t = int(usage.get("output_tokens", 0) or 0)
    else:
        in_t = out_t = 0
    cost_tracker.record(phase, model, in_t, out_t)
    budget.spend(usd_for_tokens(model, in_t, out_t))


def handle_dispatch_researcher(
    tool_input: Dict[str, Any],
    memory: RecipientMemory,
    budget: Budget,
    cost_tracker: CostAccumulator,
    recipient: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, Any]:
    """Force a fresh research pass. Updates memory.brief on success.

    Returns the orchestrator-facing summary: {brief_id, brief_summary,
    vertical, cached: False, sources, ok}.
    """
    focus = tool_input.get("focus")
    brief, usage = run_researcher(
        system_prompt=system_prompt,
        recipient=recipient,
        focus=focus,
    )
    _record_sub_agent_cost(cost_tracker, budget, "researcher", RESEARCH_MODEL, usage)

    if not brief:
        return {
            "ok": False,
            "brief_id": None,
            "reason": "researcher_did_not_submit",
        }

    memory.brief = brief
    memory.brief_sources = brief.get("sources", []) or []
    summary = _summarize_brief(brief)
    return {
        "ok": True,
        "brief_id": memory.brief_id,
        "brief_summary": summary,
        "vertical": brief.get("vertical"),
        "cached": False,
        "sources": memory.brief_sources,
    }


def handle_get_recipient_brief(
    tool_input: Dict[str, Any],
    memory: RecipientMemory,
    budget: Budget,
    cost_tracker: CostAccumulator,
    recipient: Dict[str, Any],
    researcher_system_prompt: str,
) -> Dict[str, Any]:
    """Return cached brief if present, else run researcher once.

    The orchestrator calls this near the start of every session. Subsequent
    writer dispatches reuse the cached result.
    """
    if memory.brief:
        return {
            "ok": True,
            "brief_id": memory.brief_id,
            "brief_summary": _summarize_brief(memory.brief),
            "vertical": memory.brief.get("vertical"),
            "cached": True,
            "sources": memory.brief_sources,
        }
    return handle_dispatch_researcher(
        tool_input={},
        memory=memory,
        budget=budget,
        cost_tracker=cost_tracker,
        recipient=recipient,
        system_prompt=researcher_system_prompt,
    )


def handle_dispatch_writer(
    tool_input: Dict[str, Any],
    memory: RecipientMemory,
    budget: Budget,
    cost_tracker: CostAccumulator,
    recipient_summary: str,
    available_merge_keys: Optional[list],
    feedback: Optional[str],
    quality_example: Optional[str],
    writer_system_prompt: str,
) -> Dict[str, Any]:
    """Dispatch the writer for one step. Stores the draft in memory.drafts
    under a fresh draft_id. Returns the orchestrator-facing summary.
    """
    step = int(tool_input.get("step", 1))
    brief_id = tool_input.get("brief_id")
    constraints = tool_input.get("constraints") or None
    orchestrator_prior_summary = tool_input.get("prior_summary") or None

    if brief_id and brief_id != memory.brief_id:
        logger.warning(
            "handle_dispatch_writer: unexpected brief_id=%r (expected %r)",
            brief_id, memory.brief_id,
        )

    # Build the canonical prior_summary from memory + augment with the
    # orchestrator's free-form notes if it provided any.
    memory_summary = memory.prior_summary_for_step(step)
    if orchestrator_prior_summary and memory_summary:
        prior_summary = f"{memory_summary}\n\n(orchestrator notes) {orchestrator_prior_summary}"
    else:
        prior_summary = memory_summary or orchestrator_prior_summary

    draft, usage, validation_passed = run_writer(
        system_prompt=writer_system_prompt,
        brief=memory.brief,
        recipient_summary=recipient_summary,
        step=step,
        prior_summary=prior_summary,
        constraints=constraints,
        feedback=feedback,
        available_merge_keys=available_merge_keys,
        quality_example=quality_example,
    )
    _record_sub_agent_cost(cost_tracker, budget, "writer", WRITER_MODEL, usage)

    if not draft:
        return {
            "ok": False,
            "draft_id": None,
            "validation_passed": False,
            "reason": "writer_did_not_submit",
        }

    draft_id = new_draft_id()
    memory.drafts[draft_id] = {**draft, "step": step}

    subject = draft.get("subject", "") or ""
    return {
        "ok": True,
        "draft_id": draft_id,
        "validation_passed": validation_passed,
        "word_count": int(draft.get("word_count", 0)),
        "subject_preview": subject[:80],
    }


def handle_dispatch_critic(
    tool_input: Dict[str, Any],
    memory: RecipientMemory,
    budget: Budget,
    cost_tracker: CostAccumulator,
    recipient_summary: str,
    critic_system_prompt: str,
) -> Dict[str, Any]:
    """Run Opus critic on a stored draft."""
    draft_id = tool_input.get("draft_id")
    step = int(tool_input.get("step", 1))
    if not draft_id or draft_id not in memory.drafts:
        return {"ok": False, "reason": f"unknown draft_id {draft_id!r}"}

    draft = memory.drafts[draft_id]
    verdict, usage = run_critic(
        system_prompt=critic_system_prompt,
        draft=draft,
        step=step,
        brief=memory.brief,
        recipient_summary=recipient_summary,
    )
    _record_sub_agent_cost(cost_tracker, budget, "critic", JUDGE_MODEL_FINAL, usage)

    return {
        "ok": True,
        "score": float(verdict.get("overall_score", 0.0)),
        "dim_scores": verdict.get("dimension_scores", {}) or {},
        "issues": verdict.get("issues", []) or [],
        "should_refine": bool(verdict.get("should_refine", False)),
        "swap_test_result": verdict.get("swap_test_result", {}),
        "verdict_draft_id": draft_id,  # so orchestrator can pair verdict with draft
    }


def _summarize_brief(brief: Dict[str, Any]) -> str:
    """Compact one-paragraph summary of the brief for orchestrator context.

    The orchestrator doesn't need the full brief — it's given to writers
    directly. The orchestrator just needs enough to make dispatch decisions.
    """
    parts = []
    if brief.get("vertical") and brief["vertical"] != "general":
        parts.append(f"vertical: {brief['vertical']}")
    if brief.get("team_size"):
        parts.append(f"team: ~{brief['team_size']}")
    if brief.get("differentiator"):
        parts.append(f"differentiator: {brief['differentiator'][:120]}")
    if brief.get("markets"):
        parts.append(f"markets: {', '.join(brief['markets'][:4])}")
    metrics = brief.get("notable_metrics", [])
    if metrics:
        parts.append(f"metrics: {'; '.join(metrics[:3])}")
    return " | ".join(parts) if parts else "[thin brief, limited data]"
