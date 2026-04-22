"""list_recipient_gaps tool: categorize template steps vs Mongo state.

Categories returned to the orchestrator:
  done:         personalized doc exists AND quality_score >= QUALITY_THRESHOLD
                AND review_status is not needs_revision.
  needs_rewrite: personalized doc exists BUT below threshold OR flagged.
  missing:      template has this step but no personalized doc at all.

Mirrors the P1.1 fix in pipeline._resolve_targets_from_scope but scoped
per-recipient and with three-way categorization so the orchestrator can
prioritize (missing steps are the highest-priority; needs_rewrite is
lower; done is no-op).
"""

import logging
from typing import Any, Dict, Iterable, List, Set

from config import QUALITY_THRESHOLD

logger = logging.getLogger(__name__)


def list_recipient_gaps(
    recipient_id: str,
    sequence_id: str,
    template_steps: Iterable[int],
) -> Dict[str, List[int]]:
    """Categorize each template step for this recipient.

    Args:
        recipient_id: Mongo recipient _id as string.
        sequence_id: email_sequences _id as string.
        template_steps: all step numbers that the template defines.

    Returns:
        {"done": [...], "needs_rewrite": [...], "missing": [...]}
        with each list sorted ascending.
    """
    from utils.mongo import _get_primary_db

    db = _get_primary_db()
    template_set: Set[int] = {int(s) for s in template_steps}
    cursor = db.personalized_sequence_emails.find(
        {
            "email_sequence_id": str(sequence_id),
            "recipient_id": str(recipient_id),
        },
        {"step": 1, "quality_score": 1, "review_status": 1, "_id": 0},
    )

    done: List[int] = []
    needs_rewrite: List[int] = []
    seen_steps: Set[int] = set()

    for doc in cursor:
        step = int(doc.get("step", 0))
        if step not in template_set:
            # Stray doc for a step not in the template — ignore.
            continue
        seen_steps.add(step)
        score = doc.get("quality_score")
        review = doc.get("review_status")
        if review == "needs_revision":
            needs_rewrite.append(step)
        elif isinstance(score, (int, float)) and score < QUALITY_THRESHOLD:
            needs_rewrite.append(step)
        else:
            done.append(step)

    missing = sorted(template_set - seen_steps)
    logger.info(
        "list_recipient_gaps recipient=%s sequence=%s done=%s "
        "needs_rewrite=%s missing=%s",
        recipient_id, sequence_id, sorted(done), sorted(needs_rewrite), missing,
    )
    return {
        "done": sorted(done),
        "needs_rewrite": sorted(needs_rewrite),
        "missing": missing,
    }
