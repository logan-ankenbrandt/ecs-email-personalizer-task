"""Resolve {{first_name}}, {{company_name}}, etc. placeholders in
personalized copy using recipient data.

The personalizer-task pre-resolves merge fields BEFORE writing to
personalized_sequence_emails so that message_scheduler can skip
content_updater entirely for personalized rows. This keeps the
end-to-end flow simple (look up personalized → use directly).
"""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Standard merge fields recognized by the platform. Values come from the
# recipient doc (first_name, last_name, email) + custom_fields[] array
# (anything else like website, business_name, etc.).
_DEFAULT_BLANK = ""


def _extract_custom_field(recipient: Dict[str, Any], key: str) -> Optional[str]:
    """Look up a key in the recipient's custom_fields array."""
    custom = recipient.get("custom_fields") or []
    if isinstance(custom, list):
        for cf in custom:
            if isinstance(cf, dict) and cf.get("key") == key:
                val = cf.get("value")
                if val is not None:
                    return str(val)
    return None


def build_merge_dict(recipient: Dict[str, Any]) -> Dict[str, str]:
    """Build the standard merge field substitution dict from a recipient doc.

    Includes top-level fields (first_name, last_name, email, business_name,
    title, city, state, industry, company_size) AND every key in custom_fields.
    """
    merge: Dict[str, str] = {}

    # Top-level recognized fields
    for key in (
        "first_name", "last_name", "email", "business_name", "title",
        "city", "state", "industry", "company_size", "phone",
    ):
        val = recipient.get(key)
        if val is not None:
            merge[key] = str(val)

    # Aliases for common variants
    if "business_name" in merge:
        merge.setdefault("company", merge["business_name"])
        merge.setdefault("company_name", merge["business_name"])
    if "first_name" in merge:
        merge.setdefault("firstName", merge["first_name"])
    if "last_name" in merge:
        merge.setdefault("lastName", merge["last_name"])

    # Custom fields verbatim
    custom = recipient.get("custom_fields") or []
    if isinstance(custom, list):
        for cf in custom:
            if isinstance(cf, dict):
                k = cf.get("key")
                v = cf.get("value")
                if k and v is not None:
                    merge[str(k)] = str(v)

    return merge


def resolve_merge_fields(text: str, merge_dict: Dict[str, str]) -> str:
    """Replace {{key}} placeholders with values from merge_dict.

    Unknown keys are replaced with empty string (safer than leaving the
    placeholder visible to the recipient). Logs at INFO when this happens.
    """
    if not text:
        return text

    unknown_keys = set()

    def _sub(match: re.Match) -> str:
        key = match.group(1).strip()
        if key in merge_dict:
            return merge_dict[key]
        unknown_keys.add(key)
        return _DEFAULT_BLANK

    # Match {{ anything }} including spaces inside braces
    result = re.sub(r"\{\{([^}]+)\}\}", _sub, text)

    if unknown_keys:
        logger.warning(
            "resolve_merge_fields: unresolved keys replaced with empty: %s",
            sorted(unknown_keys),
        )

    return result
