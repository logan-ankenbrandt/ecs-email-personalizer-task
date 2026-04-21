"""Opus 4.6 judge for personalized email quality.

Loads the email-rewriter assessment_prompt + rubric (which is the proven
0.92-vs-0.49 calibration framework) from S3 and scores each draft against
5 dimensions: personalization_depth, slop_absence, tone_authenticity,
structural_compliance, segment_specificity.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from config import HARNESS_BUCKET, JUDGE_MODEL, KNOWLEDGE_PREFIX
from utils.llm import generate_structured
from utils.s3 import load_knowledge

logger = logging.getLogger(__name__)


# JSON schema for the judge's output. Matches the rewriter's assessor format.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "number"},
        "dimension_scores": {
            "type": "object",
            "properties": {
                "personalization_depth": {"type": "number"},
                "slop_absence": {"type": "number"},
                "tone_authenticity": {"type": "number"},
                "structural_compliance": {"type": "number"},
                "segment_specificity": {"type": "number"},
            },
            "required": [
                "personalization_depth", "slop_absence", "tone_authenticity",
                "structural_compliance", "segment_specificity",
            ],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "excerpt": {"type": "string"},
                    "slop_type": {"type": ["string", "null"]},
                    "issue": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["excerpt", "issue", "suggestion"],
            },
        },
        "swap_test_result": {
            "type": "object",
            "properties": {
                "sentences_tested": {"type": "integer"},
                "sentences_passing": {"type": "integer"},
                "sentences_failing": {"type": "integer"},
            },
        },
        "should_refine": {"type": "boolean"},
    },
    "required": ["overall_score", "dimension_scores", "issues", "should_refine"],
}


_assessment_prompt_cache: Optional[str] = None


def _load_assessment_prompt() -> str:
    """Load the email-rewriter assessment_prompt.md (cached for the run)."""
    global _assessment_prompt_cache
    if _assessment_prompt_cache is None:
        _assessment_prompt_cache = load_knowledge(
            HARNESS_BUCKET,
            f"{KNOWLEDGE_PREFIX}/assessment_prompt.md",
        )
    return _assessment_prompt_cache


def judge_email(
    subject: str,
    content: str,
    company_brief: str = "",
    recipient_summary: str = "",
    model: Optional[str] = None,
    data_grounding: Optional[List[str]] = None,
    company_insight: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Score a draft against the rewriter rubric. Returns (judgment, token_counts).

    judgment dict keys: overall_score, dimension_scores, issues, swap_test_result,
    should_refine. See _JUDGE_SCHEMA.

    T3.1: when `data_grounding` and `company_insight` (both writer-produced)
    are passed, the judge is told to VERIFY these self-reported claims against
    the actual copy. Prevents writers from claiming groundedness they don't
    have — the judge can flag fabricated-feeling facts.

    Pass `model` to override the default (JUDGE_MODEL from config). Pipeline
    uses Sonnet for intermediate judging and Opus for the final iteration
    to cut per-step Opus calls from 2+ to 1.

    Falls back to a default failing judgment ({overall_score: 0.0, should_refine: True})
    if the LLM call fails so the caller can decide to refine or skip.
    """
    base_prompt = _load_assessment_prompt()

    # T3.1: optional writer self-report section so the judge can verify
    # rather than score blind.
    self_report_section = ""
    if data_grounding or company_insight:
        lines = ["### Writer's self-reported grounding"]
        if company_insight:
            lines.append(f"\nCompany insight (writer-stated): {company_insight}")
        if data_grounding:
            lines.append("\nFacts the writer says are grounded in real data:")
            for i, fact in enumerate(data_grounding, start=1):
                lines.append(f"  {i}. {fact}")
        lines.append(
            "\nVerify these claims against the body. If a fact the writer "
            "says is grounded does not appear in the copy, or appears in a way "
            "that still fails the swap test, flag it as an issue with "
            "slop_type='ungrounded_claim'."
        )
        self_report_section = "\n".join(lines) + "\n\n"

    prompt = (
        f"{base_prompt}\n\n"
        f"---\n\n"
        f"## Draft to evaluate\n\n"
        f"### Subject\n{subject}\n\n"
        f"### Body\n{content}\n\n"
        f"### Company context (used during writing)\n{company_brief or '[no company brief available]'}\n\n"
        f"### Recipient context\n{recipient_summary or '[no recipient context available]'}\n\n"
        f"{self_report_section}"
    )
    try:
        judgment = generate_structured(
            prompt=prompt,
            schema=_JUDGE_SCHEMA,
            model=model or JUDGE_MODEL,
        )
        # generate_structured doesn't return token counts directly, so we fake
        # an estimate based on prompt length. TODO: extend generate_structured to
        # return usage. For now, downstream cost telemetry uses these estimates.
        est_input = len(prompt) // 4
        est_output = 500  # judge responses are typically short
        return judgment, {"input_tokens": est_input, "output_tokens": est_output}
    except Exception as e:
        logger.warning("Judge LLM call failed: %s. Returning failing default.", e)
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
                "issues": [{
                    "excerpt": "[judge unavailable]",
                    "slop_type": None,
                    "issue": f"Judge LLM call failed: {type(e).__name__}",
                    "suggestion": "Retry or skip this recipient.",
                }],
                "swap_test_result": {"sentences_tested": 0, "sentences_passing": 0, "sentences_failing": 0},
                "should_refine": True,
            },
            {"input_tokens": 0, "output_tokens": 0},
        )
