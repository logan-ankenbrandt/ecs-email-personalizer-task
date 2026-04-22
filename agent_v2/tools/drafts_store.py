"""Draft-id allocation for an orchestrator session.

Trivial wrapper over RecipientMemory.drafts that issues a uuid4-based
draft_id per persisted draft and supports surgical reads. Scoped to one
session (one RecipientMemory) so the global DraftStore is not needed.
"""

import uuid
from typing import Any, Dict, Iterable, List


def new_draft_id() -> str:
    """Short-ish unique id for a session-local draft reference."""
    return f"draft-{uuid.uuid4().hex[:10]}"


def read_draft_fields(
    draft: Dict[str, Any],
    fields: Iterable[str],
) -> Dict[str, Any]:
    """Return only the requested fields. Unknown fields become None.

    Supports the synthetic fields the orchestrator's read_draft tool exposes:
      - first_sentence: first sentence of content (HTML-stripped)
      - last_sentence: last sentence of content (HTML-stripped)
      - word_count: len(content.split()) post-HTML-strip
    Plus any raw draft key (subject, content, company_insight,
    data_grounding, ...).
    """
    import re

    def _strip_html(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

    plain_content = _strip_html(draft.get("content", "") or "")

    out: Dict[str, Any] = {}
    for f in fields:
        if f == "first_sentence":
            parts = re.split(r"(?<=[.!?])\s+", plain_content, maxsplit=1)
            out[f] = parts[0] if parts else plain_content[:160]
        elif f == "last_sentence":
            parts = re.split(r"(?<=[.!?])\s+", plain_content)
            parts = [p for p in parts if p.strip()]
            out[f] = parts[-1] if parts else ""
        elif f == "word_count":
            out[f] = len(plain_content.split())
        else:
            out[f] = draft.get(f)
    return out
