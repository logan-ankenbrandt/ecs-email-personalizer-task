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
) -> bool:
    """Upsert a per-recipient personalized email. Idempotent on the unique
    compound index (email_sequence_id, recipient_id, step)."""
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
    }
    try:
        result = db.personalized_sequence_emails.update_one(
            {
                "email_sequence_id": email_sequence_id,
                "recipient_id": str(recipient_id),
                "step": step,
            },
            {"$set": doc},
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
) -> None:
    """Initialize the personalization fields on the email_sequences doc."""
    db = _get_primary_db()
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
                "personalization_started_at": datetime.now(timezone.utc).isoformat(),
                "personalization_completed_at": None,
                "personalization_error": None,
            }
        },
    )


def update_personalization_progress(
    sequence_id: str,
    completed: int,
    failed: int,
) -> None:
    """Update the running counters on the parent doc. Called every CHECKPOINT_EVERY_N."""
    db = _get_primary_db()
    db.email_sequences.update_one(
        {"_id": ObjectId(sequence_id)},
        {
            "$set": {
                "personalization_progress.completed": completed,
                "personalization_progress.failed": failed,
            }
        },
    )


def finalize_personalization_run(
    sequence_id: str,
    completed: int,
    failed: int,
    metadata: Dict[str, Any],
    error: Optional[str] = None,
) -> None:
    """Mark the personalization run as completed (or failed) with metadata."""
    db = _get_primary_db()
    status = "failed" if error else "completed"
    db.email_sequences.update_one(
        {"_id": ObjectId(sequence_id)},
        {
            "$set": {
                "personalization_progress.status": status,
                "personalization_progress.completed": completed,
                "personalization_progress.failed": failed,
                "personalization_completed_at": datetime.now(timezone.utc).isoformat(),
                "personalization_metadata": metadata,
                "personalization_error": error,
            }
        },
    )
    logger.info(
        "Personalization %s for sequence %s: completed=%d failed=%d",
        status, sequence_id, completed, failed,
    )
