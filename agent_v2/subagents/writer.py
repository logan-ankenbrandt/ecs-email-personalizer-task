"""Writer sub-agent: Sonnet with internal validate_draft loop.

The writer runs its own tool-use loop: calls validate_draft, gets a
list of issues, rewrites, calls validate_draft again. Only when the list
is empty does it call submit_draft. This replaces the V1 pattern where
the judge/refine loop fired externally — here the writer self-corrects
structurally before the orchestrator ever sees the draft.

Output: dict with subject, content, company_insight, data_grounding,
word_count, validation_passed. The content has already been run through
sanitize_punctuation + enforce_signature, so the orchestrator receives a
structurally clean draft.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from config import WRITER_MODEL

from agent_v2.loop import LoopResult, ToolResult, TokenUsage, call_with_tools_loop
from agent_v2.schemas import WRITER_TOOLS
from agent_v2.tools.validate_draft import validate_draft

logger = logging.getLogger(__name__)


def _build_writer_task(
    brief: Optional[Dict[str, Any]],
    recipient_summary: str,
    step: int,
    prior_summary: Optional[str],
    constraints: Optional[str],
    feedback: Optional[str],
    available_merge_keys: Optional[List[str]],
    quality_example: Optional[str],
) -> str:
    """Build the writer's user prompt.

    The system prompt (writer.md) holds the standing rules + calibration
    anchor. The user prompt is the per-step task: recipient + brief +
    constraints + prior-step summary.
    """
    lines: List[str] = [f"## Task: write email {step} in the sequence\n"]

    if brief:
        lines.append("## Company brief (from researcher)\n")
        vertical = brief.get("vertical")
        if vertical and vertical != "general":
            lines.append(f"- Vertical: {vertical}")
        if brief.get("team_size"):
            lines.append(f"- Team size (approx): {brief['team_size']}")
        if brief.get("differentiator"):
            lines.append(f"- Differentiator: {brief['differentiator']}")
        if brief.get("markets"):
            lines.append(f"- Markets: {', '.join(brief['markets'])}")
        if brief.get("notable_metrics"):
            lines.append(
                f"- Notable metrics: {'; '.join(brief['notable_metrics'])}"
            )
        lines.append("")
    else:
        lines.append(
            "## Company brief\n\n"
            "No research brief available. Write from the recipient context "
            "only. Do not mention that research was unavailable.\n"
        )

    lines.append("## Recipient context\n")
    lines.append(recipient_summary.strip())
    lines.append("")

    if prior_summary:
        lines.append("## Earlier accepted steps (do NOT recycle)\n")
        lines.append(prior_summary)
        lines.append(
            "\nUse a DIFFERENT opener framing and a DIFFERENT proof point "
            "than any listed above. If the same metric is relevant, frame "
            "it from a new angle.\n"
        )

    if constraints:
        lines.append("## Orchestrator constraints for this draft\n")
        lines.append(constraints)
        lines.append("")

    if feedback:
        lines.append("## User feedback (hard constraint, overrides other rules)\n")
        lines.append(feedback)
        lines.append("")

    if available_merge_keys:
        keys_list = ", ".join(f"{{{{{k}}}}}" for k in sorted(available_merge_keys))
        lines.append(
            f"## Merge fields available\n\n"
            f"ONLY these merge fields are populated and safe to use as "
            f"placeholders: {keys_list}. Do NOT emit any other "
            f"{{{{placeholder}}}}; use the literal value instead.\n"
        )

    if quality_example:
        # The writer system prompt already includes a calibration anchor,
        # but re-surfacing it here cements the target for this specific step.
        lines.append("## Reference: a 0.92-quality email for a similar recipient\n")
        lines.append(quality_example.strip()[:3000])
        lines.append("")

    lines.append(
        "## Next steps\n\n"
        "1. Write the subject and body.\n"
        "2. Call validate_draft(subject, content, step) to check structure.\n"
        "3. Fix any issues and re-validate (max 4 iterations).\n"
        "4. Call submit_draft with the final version.\n"
    )
    return "\n".join(lines)


def _strip_html_for_wordcount(text: str) -> int:
    import re
    if not text:
        return 0
    plain = re.sub(r"<[^>]+>", " ", text)
    return len(plain.split())


def run_writer(
    system_prompt: str,
    brief: Optional[Dict[str, Any]],
    recipient_summary: str,
    step: int,
    prior_summary: Optional[str] = None,
    constraints: Optional[str] = None,
    feedback: Optional[str] = None,
    available_merge_keys: Optional[List[str]] = None,
    quality_example: Optional[str] = None,
    max_turns: int = 8,
    temperature: float = 0.3,
) -> Tuple[Optional[Dict[str, Any]], TokenUsage, bool]:
    """Execute one writer session for one step.

    Returns (draft, usage, validation_passed). `draft` keys:
      subject, content, company_insight, data_grounding, word_count.
      content has been run through sanitize_punctuation + enforce_signature.
    `validation_passed` is True when the writer's final submit_draft was
    preceded by a passing validate_draft call in the same session.
    """
    from utils.sanitize import enforce_signature, sanitize_punctuation

    # Track validation state so we can tell the orchestrator whether the
    # writer self-cleared structural checks.
    state = {
        "last_validation_passed": False,
        "validate_calls": 0,
    }

    def handle_tool(name: str, tool_input: Dict[str, Any], tool_use_id: str) -> ToolResult:
        if name == "validate_draft":
            state["validate_calls"] += 1
            issues = validate_draft(
                subject=tool_input.get("subject", ""),
                content=tool_input.get("content", ""),
                step=int(tool_input.get("step", step)),
            )
            state["last_validation_passed"] = not issues
            if not issues:
                return ToolResult(
                    content='{"issues": [], "passed": true}',
                )
            # Serialize the issue list for the model to read.
            import json
            return ToolResult(
                content=json.dumps({"issues": issues, "passed": False}, default=str),
            )

        if name == "submit_draft":
            # Apply post-processing sanitizers. The validator ran on the raw
            # output; sanitize here defensively so a writer that skipped
            # validate_draft (or submitted despite issues on iteration 4)
            # still gets em-dashes stripped and signature enforced.
            subject = sanitize_punctuation(tool_input.get("subject", "") or "")
            content = sanitize_punctuation(tool_input.get("content", "") or "")
            content = enforce_signature(content)
            payload = {
                "subject": subject,
                "content": content,
                "company_insight": tool_input.get("company_insight", "") or "",
                "data_grounding": tool_input.get("data_grounding", []) or [],
                "word_count": int(
                    tool_input.get("word_count")
                    or _strip_html_for_wordcount(content)
                ),
            }
            return ToolResult(
                content="draft accepted",
                is_terminal=True,
                payload=payload,
            )

        return ToolResult(content=f"[unknown tool: {name}]", is_error=True)

    user_task = _build_writer_task(
        brief=brief,
        recipient_summary=recipient_summary,
        step=step,
        prior_summary=prior_summary,
        constraints=constraints,
        feedback=feedback,
        available_merge_keys=available_merge_keys,
        quality_example=quality_example,
    )
    messages = [{"role": "user", "content": user_task}]

    result: LoopResult = call_with_tools_loop(
        system_prompt=system_prompt,
        messages=messages,
        tools=WRITER_TOOLS,
        model=WRITER_MODEL,
        max_turns=max_turns,
        tool_handler=handle_tool,
        max_tokens_per_turn=4096,
        temperature=temperature,
    )

    if result.stop_reason == "tool_terminal" and isinstance(result.terminal_payload, dict):
        draft = result.terminal_payload
        # validation_passed is True only if the writer called validate_draft
        # at least once AND the last call returned passed=true AND the
        # submit payload matches (we can't strictly verify match here, so we
        # trust last_validation_passed).
        validation_passed = bool(state["last_validation_passed"]) and state["validate_calls"] > 0
        logger.info(
            "writer submitted: step=%d validate_calls=%d validation_passed=%s "
            "word_count=%d",
            step, state["validate_calls"], validation_passed,
            draft.get("word_count", 0),
        )
        return draft, result.usage, validation_passed

    logger.warning(
        "writer did not submit: step=%d stop_reason=%s turns=%d",
        step, result.stop_reason, result.usage.turns_used,
    )
    return None, result.usage, False
