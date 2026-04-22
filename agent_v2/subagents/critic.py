"""Critic sub-agent: single-shot Opus evaluation of a draft.

Uses generate_structured (single forced tool call, no multi-turn loop).
Cheaper and simpler than a tool-use loop: the critic never needs to
iterate, it just reads the draft and emits a verdict.

Returns (verdict, usage) where verdict has keys:
  - overall_score: float 0-1
  - dimension_scores: 5-key dict
  - issues: list of {excerpt, slop_type, issue, suggestion}
  - should_refine: bool
  - swap_test_result: {sentences_tested, sentences_passing, sentences_failing}
"""

import logging
from typing import Any, Dict, Optional, Tuple

from config import JUDGE_MODEL_FINAL
from utils.llm import generate_structured

from agent_v2.schemas import CRITIC_OUTPUT_SCHEMA

logger = logging.getLogger(__name__)


def _build_critic_task(
    draft: Dict[str, Any],
    step: int,
    brief: Optional[Dict[str, Any]],
    recipient_summary: str,
) -> str:
    """User prompt for the critic: the draft plus its context."""
    subject = draft.get("subject", "")
    content = draft.get("content", "")
    lines = [
        f"## Draft to evaluate\n",
        f"### Email position: step {step} of the sequence\n",
        f"### Subject\n{subject}\n",
        f"### Body\n{content}\n",
        f"### Recipient context\n{recipient_summary.strip()}\n",
    ]
    if brief:
        lines.append("### Company brief (used during drafting)\n")
        if brief.get("differentiator"):
            lines.append(f"- Differentiator: {brief['differentiator']}")
        if brief.get("vertical"):
            lines.append(f"- Vertical: {brief['vertical']}")
        if brief.get("notable_metrics"):
            lines.append(f"- Notable metrics: {'; '.join(brief['notable_metrics'])}")
        lines.append("")

    writer_insight = draft.get("company_insight", "")
    grounding = draft.get("data_grounding", [])
    if writer_insight or grounding:
        lines.append("### Writer's self-reported grounding\n")
        if writer_insight:
            lines.append(f"Insight: {writer_insight}")
        if grounding:
            lines.append("Grounded claims:")
            for i, g in enumerate(grounding, 1):
                lines.append(f"  {i}. {g}")
        lines.append(
            "\nVerify these claims. If a claim does not appear in the body, "
            "or appears in a way that still fails the swap test, flag it."
        )
    return "\n".join(lines)


def run_critic(
    system_prompt: str,
    draft: Dict[str, Any],
    step: int,
    brief: Optional[Dict[str, Any]],
    recipient_summary: str,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Single-shot Opus critic. Returns (verdict, token_counts).

    token_counts: {input_tokens, output_tokens} for CostAccumulator.record.
    Falls back to a permissive verdict on LLM failure so the orchestrator
    can decide to submit-with-warning vs skip.
    """
    try:
        verdict, tokens = generate_structured(
            prompt=_build_critic_task(draft, step, brief, recipient_summary),
            schema=CRITIC_OUTPUT_SCHEMA,
            model=JUDGE_MODEL_FINAL,
            max_tokens=8192,
            system=system_prompt,
            return_usage=True,
        )
        logger.info(
            "critic verdict: step=%d score=%.2f should_refine=%s issues=%d",
            step, verdict.get("overall_score", 0.0),
            verdict.get("should_refine", False),
            len(verdict.get("issues", []) or []),
        )
        return verdict, tokens
    except Exception as e:  # noqa: BLE001
        logger.warning("critic LLM call failed: %s. Returning permissive default.", e)
        return (
            {
                "overall_score": 0.0,
                "dimension_scores": {
                    "personalization_depth": 0.0,
                    "slop_absence": 0.0,
                    "tone_authenticity": 0.0,
                    "structural_compliance": 0.0,
                    "segment_specificity": 0.0,
                },
                "issues": [
                    {
                        "excerpt": "[critic unavailable]",
                        "slop_type": None,
                        "issue": f"Critic LLM call failed: {type(e).__name__}",
                        "suggestion": "Orchestrator should decide to retry or skip.",
                    }
                ],
                "should_refine": True,
                "swap_test_result": {
                    "sentences_tested": 0,
                    "sentences_passing": 0,
                    "sentences_failing": 0,
                },
            },
            {"input_tokens": 0, "output_tokens": 0},
        )
