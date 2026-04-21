"""Sonnet refinement — rewrites flagged emails using judge feedback.

The refine prompt asks the model to fix ONLY the flagged sentences while
preserving everything else. Pattern ported from the rewriter's refinement_prompt.md.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import HARNESS_BUCKET, KNOWLEDGE_PREFIX, REFINE_MODEL
from utils.llm import generate_structured
from utils.s3 import load_knowledge

logger = logging.getLogger(__name__)


_REFINE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "content": {"type": "string"},
        "changes_made": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Brief notes on what was changed and why.",
        },
    },
    "required": ["subject", "content", "changes_made"],
}


_refinement_prompt_cache: Optional[str] = None


def _load_refinement_prompt() -> str:
    global _refinement_prompt_cache
    if _refinement_prompt_cache is None:
        _refinement_prompt_cache = load_knowledge(
            HARNESS_BUCKET,
            f"{KNOWLEDGE_PREFIX}/refinement_prompt.md",
        )
    return _refinement_prompt_cache


def refine_email(
    current_subject: str,
    current_content: str,
    issues: List[Dict[str, Any]],
    company_brief: str = "",
    user_feedback: Optional[str] = None,
    recipient_summary: Optional[str] = None,
    step: Optional[int] = None,
    step_role: Optional[str] = None,
    vertical: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    """Refine the email using the judge's specific issue feedback.

    Returns (refined_dict, token_counts). refined_dict has subject + content +
    changes_made. None if the refine LLM call failed.

    When `user_feedback` is set, it's appended as a hard constraint that must
    be preserved even if it conflicts with the judge's issues. Without this,
    a user's "don't mention the company" feedback could be silently undone by
    refine chasing the judge's tone critique.

    T3.2: when `recipient_summary`, `step`, `step_role`, or `vertical` are
    passed, they're rendered at the top of the refinement prompt so the
    refiner can re-ground claims and enforce CTA tier rules per email
    position. Without these, the refiner had only the current draft + issues
    and couldn't regenerate grounded content.
    """
    if not issues:
        # Nothing to refine; return the original.
        return {
            "subject": current_subject,
            "content": current_content,
            "changes_made": ["[no issues to fix]"],
        }, {"input_tokens": 0, "output_tokens": 0}

    base_prompt = _load_refinement_prompt()

    # Format issues as a numbered list with excerpt + suggestion
    issues_text_parts = []
    for i, issue in enumerate(issues, start=1):
        slop = issue.get("slop_type") or "general"
        excerpt = issue.get("excerpt", "")[:200]
        feedback = issue.get("issue", "")
        suggestion = issue.get("suggestion", "")
        issues_text_parts.append(
            f"{i}. [{slop}] EXCERPT: {excerpt!r}\n"
            f"   ISSUE: {feedback}\n"
            f"   SUGGESTION: {suggestion}"
        )
    issues_text = "\n\n".join(issues_text_parts)

    # User-feedback block sits AFTER the issues list so it reads as a
    # priority constraint. Kept short to avoid diluting the judge issues.
    user_feedback_block = ""
    if user_feedback and user_feedback.strip():
        user_feedback_block = (
            f"\n## User constraint (MUST preserve across this refinement)\n\n"
            f"{user_feedback.strip()}\n\n"
            f"Even if fixing a judge issue would conflict with this "
            f"constraint, the constraint takes precedence.\n"
        )

    # T3.2: optional context header so the refiner can re-ground claims and
    # enforce CTA tier. Before this, refiner flew blind on recipient + step.
    header_bits: List[str] = []
    if step is not None:
        header_bits.append(f"Email position: {step}")
    if step_role:
        header_bits.append(f"Strategic role: {step_role}")
    if vertical:
        header_bits.append(f"Vertical: {vertical}")
    context_header = ""
    if header_bits:
        context_header = (
            "## Refinement context\n\n" + "\n".join(header_bits) + "\n\n"
        )
    recipient_block = ""
    if recipient_summary:
        recipient_block = (
            f"## Recipient context\n\n{recipient_summary}\n\n"
        )

    prompt = (
        f"{base_prompt}\n\n"
        f"---\n\n"
        f"{context_header}"
        f"{recipient_block}"
        f"## Current draft\n\n"
        f"### Subject\n{current_subject}\n\n"
        f"### Body\n{current_content}\n\n"
        f"### Company context (used during original writing)\n"
        f"{company_brief or '[no company brief]'}\n\n"
        f"## Issues to fix (rewrite ONLY these; preserve everything else)\n\n"
        f"{issues_text}\n"
        f"{user_feedback_block}"
        f"\n## Instructions\n\n"
        f"Return the refined subject + body. Fix every flagged issue. "
        f"Do not change sentences that weren't flagged. Keep merge field "
        f"placeholders ({{{{first_name}}}}, etc.) UNRESOLVED."
    )

    try:
        # Refine: temp=0.2 so we get light variation when rewording but not
        # so much variance that the rewrite randomly regresses.
        result = generate_structured(
            prompt=prompt,
            schema=_REFINE_OUTPUT_SCHEMA,
            model=REFINE_MODEL,
            temperature=0.2,
        )
        est_input = len(prompt) // 4
        est_output = len(result.get("content", "")) // 4 + 200
        return result, {"input_tokens": est_input, "output_tokens": est_output}
    except Exception as e:
        logger.warning("Refine LLM call failed: %s", e)
        return None, {"input_tokens": 0, "output_tokens": 0}
