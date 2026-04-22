"""Per-recipient orchestrator session state.

Thread-local to the orchestrator call (each ThreadPoolExecutor worker
operates on its own recipient, its own memory). Not shared across
recipients. The memory caches the company brief (so multiple writer
dispatches for different steps reuse one research pass) and tracks
accepted drafts per step (so later steps can be told "don't recycle the
opener or proof points from step 1").

Reuse signals come from pipeline._extract_reuse_signals (already in V1),
which we import rather than duplicating.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class AcceptedDraft:
    """A draft that was submit_step'd and upserted to Mongo."""

    step: int
    subject: str           # post-sanitize, post-merge (what Mongo holds)
    content: str           # post-sanitize, post-merge
    raw_content: str       # pre-merge (for reuse signal extraction)
    score: float
    dimension_scores: Optional[Dict[str, float]]  # None if critic was skipped
    slop_warnings: List[Dict]
    company_insight: str
    data_grounding: List[Any]  # list of strings or {claim, source} dicts
    opener: str            # first sentence, for prior_summary
    proof_points: str      # semicolon-joined proof-point excerpts


_NUMERIC_PROOF_RE = re.compile(
    r"\b\d+x(?:['’]?d)?\b|\b\d+%|\b\d+\s+(?:days|weeks|months)\b"
    r"|\b\d+\s+(?:placements|contacts|meetings|hires|searches)\b",
    re.IGNORECASE,
)


def _extract_first_sentence(text: str) -> str:
    """First terminated sentence, or first 160 chars if no terminator."""
    if not text:
        return ""
    # Strip HTML to reason about plain text.
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", plain, maxsplit=1)
    first = parts[0] if parts else plain[:160]
    return first[:200]


def _extract_proof_points(text: str) -> str:
    """Semicolon-joined numeric markers ("4x'd", "90 days", "673 placements")."""
    if not text:
        return ""
    plain = re.sub(r"<[^>]+>", " ", text)
    seen: Set[str] = set()
    ordered: List[str] = []
    for m in _NUMERIC_PROOF_RE.finditer(plain):
        val = m.group(0).strip()
        key = val.lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(val)
    return "; ".join(ordered[:8])


@dataclass
class RecipientMemory:
    """Session-scoped state for one orchestrator call.

    Fields:
      recipient_id: the recipient being personalized.
      sequence_id: the email_sequences _id.
      brief: cached research brief (dict); None until get_recipient_brief
        first triggers a researcher pass.
      brief_id: short string id the orchestrator uses to pass brief to
        subsequent writer dispatches; constant per session.
      brief_sources: URLs fetched by the researcher (for audit).
      accepted: step -> AcceptedDraft for every successfully submitted step.
      skipped: step -> reason string for every skipped step.
      drafts: draft_id -> full draft payload (writer output). Kept so
        dispatch_critic and submit_step can look up by draft_id.
      decision_log: ordered list of orchestrator decisions (tool_name +
        summary), flushed to CloudWatch at end of session.
    """

    recipient_id: str
    sequence_id: str

    brief: Optional[Dict[str, Any]] = None
    brief_id: str = "brief-1"  # single brief per session, so this is constant
    brief_sources: List[str] = field(default_factory=list)

    accepted: Dict[int, AcceptedDraft] = field(default_factory=dict)
    skipped: Dict[int, str] = field(default_factory=dict)
    drafts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    decision_log: List[Dict[str, Any]] = field(default_factory=list)

    def record_accepted(
        self,
        step: int,
        subject: str,
        content: str,
        raw_content: str,
        score: float,
        dimension_scores: Optional[Dict[str, float]],
        slop_warnings: List[Dict],
        company_insight: str,
        data_grounding: List[Any],
    ) -> None:
        self.accepted[step] = AcceptedDraft(
            step=step,
            subject=subject,
            content=content,
            raw_content=raw_content,
            score=score,
            dimension_scores=dimension_scores,
            slop_warnings=slop_warnings,
            company_insight=company_insight,
            data_grounding=data_grounding,
            opener=_extract_first_sentence(raw_content),
            proof_points=_extract_proof_points(raw_content),
        )

    def record_skipped(self, step: int, reason: str) -> None:
        self.skipped[step] = reason

    def prior_summary_for_step(self, step: int) -> str:
        """Build a 'don't recycle' string from every earlier accepted step.

        Empty string if this is the first step to be written this session.
        Fed to dispatch_writer so the writer can avoid repeating the opener
        or proof point already used in an earlier step.
        """
        parts: List[str] = []
        for earlier_step in sorted(self.accepted.keys()):
            if earlier_step >= step:
                continue
            d = self.accepted[earlier_step]
            lines = [f"Step {earlier_step}:"]
            if d.opener:
                lines.append(f"  opener: {d.opener!r}")
            if d.proof_points:
                lines.append(f"  proof points: {d.proof_points}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def resolution_for(self, step: int) -> Optional[str]:
        """"accepted" / "skipped:<reason>" / None if neither."""
        if step in self.accepted:
            return "accepted"
        if step in self.skipped:
            return f"skipped:{self.skipped[step]}"
        return None

    def log_decision(self, tool_name: str, summary: Dict[str, Any]) -> None:
        self.decision_log.append({"tool": tool_name, **summary})
