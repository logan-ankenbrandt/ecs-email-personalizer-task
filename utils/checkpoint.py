"""S3-backed checkpoint for the personalizer task.

Tracks completed_recipient_ids so a crashed run can resume without
re-personalizing recipients we already finished. Written every
CHECKPOINT_EVERY_N completions.
"""

import json
import logging
from typing import List, Optional, Set

from utils.s3 import write_json, read_file_optional

logger = logging.getLogger(__name__)


def _checkpoint_key(personalization_run_id: str) -> str:
    return f"runs/{personalization_run_id}/email-personalizer/checkpoint.json"


def load_checkpoint(bucket: str, personalization_run_id: str) -> Set[str]:
    """Return the set of recipient_ids already completed in this run."""
    key = _checkpoint_key(personalization_run_id)
    raw = read_file_optional(bucket, key)
    if not raw:
        return set()
    try:
        data = json.loads(raw)
        completed = data.get("completed_recipient_ids", [])
        logger.info(
            "Resumed checkpoint %s: %d recipients already completed",
            key, len(completed),
        )
        return set(completed)
    except Exception as e:
        logger.warning("Failed to parse checkpoint %s: %s", key, e)
        return set()


def write_checkpoint(
    bucket: str,
    personalization_run_id: str,
    completed_recipient_ids: List[str],
    failed_recipient_ids: List[str],
) -> None:
    """Persist the current set of completed recipient_ids."""
    key = _checkpoint_key(personalization_run_id)
    data = {
        "personalization_run_id": personalization_run_id,
        "completed_recipient_ids": list(completed_recipient_ids),
        "failed_recipient_ids": list(failed_recipient_ids),
        "completed_count": len(completed_recipient_ids),
        "failed_count": len(failed_recipient_ids),
    }
    try:
        write_json(bucket, key, data)
    except Exception as e:
        logger.warning("Failed to write checkpoint: %s", e)
