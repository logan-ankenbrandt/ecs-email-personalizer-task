"""MongoDB helpers for the email-personalizer task.

Dual-cluster split mirrors sequence-architect-task:
- READ cluster (cluster0.725a4j4): copy_generator_runs, sometimes recipients
- PRIMARY cluster (34.29.235.153): email_sequences, sequence_emails,
  personalized_sequence_emails, recipients (canonical)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import MONGO_URI, MONGO_DB, PRIMARY_MONGO_URI

logger = logging.getLogger(__name__)

_read_client: Optional[MongoClient] = None
_primary_client: Optional[MongoClient] = None


def _get_read_db():
    """READ cluster — copy_generator_runs + read-only joins."""
    global _read_client
    if _read_client is None:
        _read_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _read_client[MONGO_DB]


def _get_primary_db():
    """PRIMARY cluster — email_sequences, sequence_emails,
    personalized_sequence_emails, recipients."""
    global _primary_client
    if _primary_client is None:
        _primary_client = MongoClient(PRIMARY_MONGO_URI, serverSelectionTimeoutMS=5000)
    return _primary_client[MONGO_DB]


# ============================================================
# Sequence + template lookups
# ============================================================

def load_sequence(sequence_id: str) -> Optional[Dict[str, Any]]:
    """Load the parent email_sequences doc."""
    db = _get_primary_db()
    return db.email_sequences.find_one({"_id": ObjectId(sequence_id)})


def load_template_emails(sequence_id: str) -> List[Dict[str, Any]]:
    """Load all sequence_emails for a sequence, sorted by step."""
    db = _get_primary_db()
    cursor = db.sequence_emails.find(
        {"email_sequence_id": sequence_id}
    ).sort("step", 1)
    return list(cursor)


# ============================================================
# Recipient query
# ============================================================

def query_recipients(
    organization_id: str,
    tags: List[str],
    exclude_tags: List[str],
    operation: str = "and",
    max_recipients: int = 0,
) -> List[Dict[str, Any]]:
    """Find recipients matching the targeting query.

    Cluster: recipients live on the READ cluster (cluster0.725a4j4) —
    verified directly: the PRIMARY cluster returns 0 recipients for an org's
    tags while READ returns the real count. cold-api's recipient_service
    uses `clients.read_motor_db.recipients` for the same reason. An earlier
    version of this function used _get_primary_db() and the ECS task found
    0 recipients every time.

    Filter: excludes Unsubscribed so the personalizer works on the same set
    the cost-estimate endpoint counted.
    """
    db = _get_read_db()
    match: Dict[str, Any] = {
        "organization_id": organization_id,
        "status": {"$ne": "Unsubscribed"},
    }
    if operation == "and":
        match["tags"] = {"$all": tags}
    else:
        match["tags"] = {"$in": tags}
    if exclude_tags:
        match["tags"] = {**match.get("tags", {}), "$nin": exclude_tags}
    cursor = db.recipients.find(match)
    if max_recipients > 0:
        cursor = cursor.limit(max_recipients)
    return list(cursor)


# ============================================================
# Personalized email upserts
# ============================================================

def upsert_personalized_email(
    email_sequence_id: str,
    recipient_id: str,
    step: int,
    subject: str,
    content: str,
    personalization_run_id: str,
    quality_score: Optional[float] = None,
    company_insight: Optional[str] = None,
    data_grounding: Optional[List[str]] = None,
    slop_warnings: Optional[List[Dict[str, Any]]] = None,
    advisor_used: bool = False,
    original_template_id: Optional[str] = None,
    dimension_scores: Optional[Dict[str, float]] = None,
    previous_version: Optional[Dict[str, Any]] = None,
    last_rewrite_feedback: Optional[str] = None,
) -> bool:
    """Upsert a per-recipient personalized email. Idempotent on the unique
    compound index (email_sequence_id, recipient_id, step).

    When `previous_version` is passed (targeted-rewrite mode), the existing
    doc's snapshot is appended to `rewrite_history[]` in the same atomic
    update that sets the new content.
    """
    db = _get_primary_db()
    doc = {
        "email_sequence_id": email_sequence_id,
        "recipient_id": str(recipient_id),
        "step": step,
        "subject": subject,
        "content": content,
        "personalization_run_id": personalization_run_id,
        "original_template_id": original_template_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality_score": quality_score,
        "company_insight": company_insight,
        "data_grounding": data_grounding or [],
        "slop_warnings": slop_warnings or [],
        "advisor_used": advisor_used,
        "dimension_scores": dimension_scores,
    }
    # Bug H8: persist the feedback that produced this version on the live doc
    # so the UI can query it without walking rewrite_history[].
    if last_rewrite_feedback is not None:
        doc["last_rewrite_feedback"] = last_rewrite_feedback
    update_ops: Dict[str, Any] = {"$set": doc}
    if previous_version:
        update_ops["$push"] = {"rewrite_history": previous_version}
    try:
        result = db.personalized_sequence_emails.update_one(
            {
                "email_sequence_id": email_sequence_id,
                "recipient_id": str(recipient_id),
                "step": step,
            },
            update_ops,
            upsert=True,
        )
        return result.upserted_id is not None or result.modified_count > 0
    except PyMongoError as e:
        logger.error("upsert_personalized_email failed: %s", e)
        return False


# ============================================================
# Personalization run progress (on the parent email_sequences doc)
# ============================================================

def init_personalization_run(
    sequence_id: str,
    personalization_run_id: str,
    total: int,
    is_rewrite: bool = False,
) -> None:
    """Initialize the personalization fields on the email_sequences doc.

    For full-batch runs (is_rewrite=False): overwrites `personalization_progress`
    with fresh state. For rewrites (is_rewrite=True): only updates the matching
    `rewrite_runs.$` subdoc so the prior full-batch completion counters stay
    visible (Bug C3). A rewrite of 5 emails must not wipe "300/300 completed"
    from the parent doc.
    """
    db = _get_primary_db()
    now_iso = datetime.now(timezone.utc).isoformat()

    if not is_rewrite:
        # Full-batch: safe to overwrite the top-level progress.
        db.email_sequences.update_one(
            {"_id": ObjectId(sequence_id)},
            {
                "$set": {
                    "personalization_run_id": personalization_run_id,
                    "personalization_progress": {
                        "status": "running",
                        "total": total,
                        "completed": 0,
                        "failed": 0,
                    },
                    "personalization_started_at": now_iso,
                    "personalization_completed_at": None,
                    "personalization_error": None,
                }
            },
        )
    else:
        # Rewrite: only touch the matching rewrite_runs[] entry. Leave
        # personalization_progress intact so the UI continues to show the
        # prior full-batch completion state.
        db.email_sequences.update_one(
            {
                "_id": ObjectId(sequence_id),
                "rewrite_runs.run_id": personalization_run_id,
            },
            {
                "$set": {
                    "rewrite_runs.$.status": "running",
                    "rewrite_runs.$.total": total,
                    "rewrite_runs.$.completed": 0,
                    "rewrite_runs.$.failed": 0,
                    "rewrite_runs.$.started_at": now_iso,
                }
            },
        )


def update_personalization_progress(
    sequence_id: str,
    completed: int,
    failed: int,
    is_rewrite: bool = False,
    personalization_run_id: Optional[str] = None,
) -> None:
    """Update running counters. For full-batch: writes to
    `personalization_progress`. For rewrites: writes to the matching
    `rewrite_runs.$` entry so the parent counters stay intact (Bug C3).
    """
    db = _get_primary_db()
    if not is_rewrite:
        db.email_sequences.update_one(
            {"_id": ObjectId(sequence_id)},
            {
                "$set": {
                    "personalization_progress.completed": completed,
                    "personalization_progress.failed": failed,
                }
            },
        )
    elif personalization_run_id:
        db.email_sequences.update_one(
            {
                "_id": ObjectId(sequence_id),
                "rewrite_runs.run_id": personalization_run_id,
            },
            {
                "$set": {
                    "rewrite_runs.$.completed": completed,
                    "rewrite_runs.$.failed": failed,
                }
            },
        )


def finalize_personalization_run(
    sequence_id: str,
    completed: int,
    failed: int,
    metadata: Dict[str, Any],
    error: Optional[str] = None,
    personalization_run_id: Optional[str] = None,
    is_rewrite: bool = False,
) -> None:
    """Mark the personalization run as completed (or failed) with metadata.

    For full-batch runs: writes to the top-level `personalization_progress`
    + `personalization_metadata` fields on the parent sequence doc.

    For rewrites (is_rewrite=True, Bug C3): only updates the matching
    `rewrite_runs.$` subdoc. The parent `personalization_progress` state is
    preserved so the UI keeps showing the prior full-batch completion counts.
    """
    db = _get_primary_db()
    status = "failed" if error else "completed"
    now_iso = datetime.now(timezone.utc).isoformat()

    if not is_rewrite:
        db.email_sequences.update_one(
            {"_id": ObjectId(sequence_id)},
            {
                "$set": {
                    "personalization_progress.status": status,
                    "personalization_progress.completed": completed,
                    "personalization_progress.failed": failed,
                    "personalization_completed_at": now_iso,
                    "personalization_metadata": metadata,
                    "personalization_error": error,
                }
            },
        )

    # Mark the matching rewrite_runs entry terminal (for rewrites this is the
    # only write; for full-batch runs the match is a no-op because no entry
    # exists with this run_id).
    if personalization_run_id:
        db.email_sequences.update_one(
            {
                "_id": ObjectId(sequence_id),
                "rewrite_runs.run_id": personalization_run_id,
            },
            {
                "$set": {
                    "rewrite_runs.$.status": status,
                    "rewrite_runs.$.completed": completed,
                    "rewrite_runs.$.failed": failed,
                    "rewrite_runs.$.completed_at": now_iso,
                    "rewrite_runs.$.metadata": metadata,
                    "rewrite_runs.$.error": error,
                }
            },
        )
    logger.info(
        "Personalization %s for sequence %s: completed=%d failed=%d (rewrite=%s)",
        status, sequence_id, completed, failed, is_rewrite,
    )
