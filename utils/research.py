"""Per-recipient research helpers.

The writer gets web_fetch as a tool, so it can fetch on its own. This
helper exists for the case where we want to PRE-RESEARCH a recipient's
company outside the writer loop (e.g., to build a company_brief that
seeds the writer's prompt and reduces wasted writer turns).

Most of the time the writer will fetch on its own, so this is optional.
"""

import logging
from typing import Any, Dict, Optional

from anthropic import Anthropic

from config import RESEARCH_MODEL, get_api_key
from utils.web_fetch import fetch_url_cached, normalize_url

logger = logging.getLogger(__name__)


def _extract_website(recipient: Dict[str, Any]) -> Optional[str]:
    """Pull a website URL from a recipient doc (custom_fields[].key='website')."""
    custom = recipient.get("custom_fields") or []
    if isinstance(custom, list):
        for cf in custom:
            if isinstance(cf, dict) and cf.get("key") == "website":
                val = cf.get("value")
                if val and isinstance(val, str):
                    return val
    for key in ("website", "company_website", "domain"):
        val = recipient.get(key)
        if val and isinstance(val, str):
            return val
    return None


def build_recipient_summary(recipient: Dict[str, Any]) -> str:
    """Build a short text summary of the recipient from their meta fields.

    Used to seed the writer's prompt with structured recipient context
    even if their company website is unavailable.
    """
    parts = []
    name = recipient.get("first_name", "")
    if recipient.get("last_name"):
        name = f"{name} {recipient['last_name']}".strip()
    if name:
        parts.append(f"Name: {name}")
    if recipient.get("title"):
        parts.append(f"Title: {recipient['title']}")
    if recipient.get("business_name"):
        parts.append(f"Company: {recipient['business_name']}")
    if recipient.get("city") or recipient.get("state"):
        loc = ", ".join(filter(None, [recipient.get("city"), recipient.get("state")]))
        parts.append(f"Location: {loc}")
    if recipient.get("industry"):
        parts.append(f"Industry: {recipient['industry']}")
    if recipient.get("company_size"):
        parts.append(f"Company size: {recipient['company_size']}")

    custom = recipient.get("custom_fields") or []
    custom_summary = []
    if isinstance(custom, list):
        for cf in custom:
            if isinstance(cf, dict):
                k, v = cf.get("key"), cf.get("value")
                if k and v and k not in ("website", "domain"):
                    custom_summary.append(f"{k}: {v}")
    if custom_summary:
        parts.append("Additional fields: " + "; ".join(custom_summary[:8]))

    if not parts:
        return "[no recipient meta available]"
    return "\n".join(parts)


def build_company_brief(recipient: Dict[str, Any]) -> str:
    """Pre-fetch + summarize the recipient's company website.

    Returns "" if no website on file or fetch failed. Used to seed the
    writer's prompt with company context, reducing writer turns.
    """
    url = _extract_website(recipient)
    if not url:
        return ""
    url = normalize_url(url)
    raw = fetch_url_cached(url, max_chars=4000)
    if not raw:
        return ""

    biz = recipient.get("business_name") or url
    prompt = (
        f"Summarize this website into 3-5 bullets that would help a cold "
        f"email writer ground claims about this company. Focus on: what they "
        f"do, who they serve, specific numbers/data points, named clients, "
        f"differentiators. Skip marketing fluff. Return plain markdown bullets.\n\n"
        f"Source: {url}\n\n"
        f"---\n{raw}\n---"
    )
    try:
        client = Anthropic(api_key=get_api_key())
        response = client.messages.create(
            model=RESEARCH_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text") and b.text).strip()
        if text:
            return f"### {biz} ({url})\n{text}"
        return ""
    except Exception as e:
        logger.warning("build_company_brief failed for %s: %s", url, type(e).__name__)
        return ""
