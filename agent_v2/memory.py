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
    # Tier B.3: ring buffer of (tool_name, canonical_input) for the last
    # orchestrator tool dispatches. When the last 3 are identical we flag a
    # doom loop and intervene before burning more budget on it.
    recent_tool_calls: List[tuple] = field(default_factory=list)

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

    # ------------------------------------------------------------
    # Tier B.3: doom loop detection
    # ------------------------------------------------------------

    def record_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Add this tool call to the recent-calls ring buffer (max 5)."""
        import json as _json
        try:
            key = _json.dumps(tool_input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key = repr(tool_input)
        self.recent_tool_calls.append((tool_name, key))
        # Trim to last 5 (we only need last 3 for detection, keep a small
        # buffer for log forensics).
        if len(self.recent_tool_calls) > 5:
            self.recent_tool_calls = self.recent_tool_calls[-5:]

    def is_doom_loop(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """True when this call would be the 3rd consecutive identical call.

        Compares the PROPOSED call against the last 2 recorded. Detection
        fires before we record the 3rd, so the caller can intervene.
        """
        if len(self.recent_tool_calls) < 2:
            return False
        import json as _json
        try:
            key = _json.dumps(tool_input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key = repr(tool_input)
        proposed = (tool_name, key)
        return (
            self.recent_tool_calls[-1] == proposed
            and self.recent_tool_calls[-2] == proposed
        )


# ============================================================
# Round 4 / Tier B.1: message compaction
# ============================================================

# Tool result content that is safe to compact after the orchestrator has
# consumed it. The structured summary the orchestrator already saw on the
# initial response is sufficient; the raw tool_result content (full brief,
# full critic issues list, etc.) just bloats every subsequent API turn.
#
# list_recipient_gaps + read_draft are intentionally NOT compacted:
#   - list_recipient_gaps results are tiny (< 200 chars)
#   - read_draft is the orchestrator's deliberate escape hatch for
#     inspecting specific draft fields; compacting it defeats the point.
_COMPACTABLE_TOOLS = frozenset({
    "dispatch_researcher",
    "get_recipient_brief",
    "dispatch_writer",
    "dispatch_critic",
    "submit_step",
    "skip_step",
})

# Keep this many of the MOST RECENT assistant+user pairs untouched. Protects
# the orchestrator's short-term memory (the last couple of tool calls + their
# results are what it's actively reasoning about).
_COMPACT_TAIL_PAIRS = 2

# Don't bother compacting if the cumulative tool_result payload across
# compactable messages is under this. Keeps cheap runs untouched.
_COMPACT_MIN_CHARS = 4_000


def compact_messages(messages: List[Dict[str, Any]]) -> int:
    """Replace old tool_result content blocks with one-line summaries.

    Modifies `messages` in-place. Returns the count of tool_result blocks
    that were compacted. Safe to call between every API turn.

    Preserves:
      - All assistant messages (the orchestrator's own reasoning).
      - Last _COMPACT_TAIL_PAIRS assistant+user pairs (recent context).
      - tool_results whose originating tool is NOT in _COMPACTABLE_TOOLS
        (e.g. list_recipient_gaps, read_draft).
      - tool_results already marked [compacted:] (idempotent).

    Rules:
      1. Walk from the tail backwards. Skip the last N=_COMPACT_TAIL_PAIRS
         user messages (those are the recent tool results the orchestrator
         is currently reasoning about).
      2. For every older user message that contains tool_result blocks,
         replace `content` of compactable tool_results with a 1-line
         summary derived from the result's structured content.
    """
    if not messages or len(messages) < 4:
        # Not enough history to justify compaction — keep everything.
        return 0

    # Build a map of tool_use_id -> tool_name by scanning assistant
    # messages (each tool_use block carries both id and name).
    tool_name_by_id: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        for block in content:
            tu_id = getattr(block, "id", None) or (
                block.get("id") if isinstance(block, dict) else None
            )
            tu_name = getattr(block, "name", None) or (
                block.get("name") if isinstance(block, dict) else None
            )
            tu_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if tu_type == "tool_use" and tu_id and tu_name:
                tool_name_by_id[tu_id] = tu_name

    # Find user-message indices that contain tool_result blocks.
    user_tr_indices: List[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        has_tool_result = any(
            (isinstance(b, dict) and b.get("type") == "tool_result")
            for b in content
        )
        if has_tool_result:
            user_tr_indices.append(i)

    # Preserve the last _COMPACT_TAIL_PAIRS tool-result messages — those are
    # the orchestrator's recent input. Everything before that is eligible.
    if len(user_tr_indices) <= _COMPACT_TAIL_PAIRS:
        return 0
    eligible_indices = user_tr_indices[:-_COMPACT_TAIL_PAIRS]

    # Quick size gate: skip compaction if the cumulative eligible payload
    # is small.
    payload_chars = 0
    for idx in eligible_indices:
        for block in messages[idx].get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                val = block.get("content", "")
                if isinstance(val, str):
                    payload_chars += len(val)
    if payload_chars < _COMPACT_MIN_CHARS:
        return 0

    # Perform compaction.
    compacted_count = 0
    for idx in eligible_indices:
        content = messages[idx].get("content", []) or []
        new_content: List[Any] = []
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                new_content.append(block)
                continue
            tu_id = block.get("tool_use_id")
            tool_name = tool_name_by_id.get(tu_id, "")
            if tool_name not in _COMPACTABLE_TOOLS:
                new_content.append(block)
                continue
            raw = block.get("content", "")
            if isinstance(raw, str) and raw.startswith("[compacted:"):
                new_content.append(block)
                continue
            # Build a 1-line summary of the tool result. Preserve just
            # enough for the orchestrator to know what happened.
            summary = _summarize_tool_result(tool_name, raw)
            new_block = dict(block)
            new_block["content"] = summary
            new_content.append(new_block)
            compacted_count += 1
        messages[idx] = {**messages[idx], "content": new_content}

    return compacted_count


def _summarize_tool_result(tool_name: str, raw: Any) -> str:
    """Produce a 1-line '[compacted:]' summary of a tool result payload."""
    import json as _json

    # Try to parse structured JSON payloads; fall back to truncation.
    payload: Any = raw
    if isinstance(raw, str):
        try:
            payload = _json.loads(raw)
        except (ValueError, TypeError):
            payload = raw

    if tool_name == "dispatch_researcher" or tool_name == "get_recipient_brief":
        if isinstance(payload, dict):
            vert = payload.get("vertical") or "unknown"
            bid = payload.get("brief_id") or ""
            cached = payload.get("cached", False)
            return f"[compacted: {tool_name} brief_id={bid} vertical={vert} cached={cached}]"
    if tool_name == "dispatch_writer":
        if isinstance(payload, dict):
            did = payload.get("draft_id") or ""
            wc = payload.get("word_count", 0)
            vp = payload.get("validation_passed", False)
            return f"[compacted: dispatch_writer draft_id={did} word_count={wc} validation_passed={vp}]"
    if tool_name == "dispatch_critic":
        if isinstance(payload, dict):
            score = payload.get("score", 0)
            should_refine = payload.get("should_refine", False)
            n_issues = len(payload.get("issues", []) or [])
            return (
                f"[compacted: dispatch_critic score={score:.2f} "
                f"should_refine={should_refine} issues={n_issues}]"
                if isinstance(score, (int, float))
                else f"[compacted: dispatch_critic (unparseable score)]"
            )
    if tool_name == "submit_step":
        if isinstance(payload, dict):
            ok = payload.get("ok", False)
            score = payload.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "none"
            return f"[compacted: submit_step ok={ok} score={score_str}]"
    if tool_name == "skip_step":
        return f"[compacted: skip_step ok=true]"

    # Unknown structure: truncated fallback.
    s = raw if isinstance(raw, str) else str(raw)
    return f"[compacted: {tool_name} ({len(s)} chars)]"
