"""Per-recipient research helpers.

The writer gets web_fetch as a tool, so it can fetch on its own. This
helper exists for the case where we want to PRE-RESEARCH a recipient's
company outside the writer loop (e.g., to build a company_brief that
seeds the writer's prompt and reduces wasted writer turns).

Most of the time the writer will fetch on its own, so this is optional.
"""

import logging
from typing import Any, Dict, Iterable, Optional

from anthropic import Anthropic

from config import RESEARCH_MODEL, get_api_key
from utils.web_fetch import fetch_url_cached, normalize_url

logger = logging.getLogger(__name__)


# Priority-ordered list of custom_field keys that might hold a company URL.
# Production data from lead_finder enrichment writes to `company_website` /
# `company_domain`; older imports may use `website` / `domain`. We check all.
_WEBSITE_CF_KEYS: tuple = (
    "company_website",
    "company_domain",
    "current_employer_website",
    "current_employer_domain",
    "website",
    "domain",
)


# Personal-email domains to skip when falling back to email-based company URL.
# These should NEVER be treated as the recipient's company website — fetching
# gmail.com as a "company brief" produces garbage.
_PERSONAL_EMAIL_DOMAINS: frozenset = frozenset({
    # Google
    "gmail.com", "googlemail.com",
    # Yahoo
    "yahoo.com", "yahoo.co.uk", "yahoo.ca", "yahoo.fr", "yahoo.de", "yahoo.es",
    "ymail.com", "rocketmail.com",
    # Microsoft
    "outlook.com", "outlook.co.uk", "hotmail.com", "hotmail.co.uk",
    "live.com", "live.co.uk", "msn.com",
    # Apple
    "icloud.com", "me.com", "mac.com",
    # Other major consumer ISPs
    "aol.com", "verizon.net", "att.net", "comcast.net", "sbcglobal.net",
    "bellsouth.net", "cox.net", "charter.net",
    # Privacy-focused providers
    "protonmail.com", "proton.me", "pm.me",
    "tutanota.com", "tutamail.com",
    "duck.com",
    # Other consumer
    "gmx.com", "gmx.net", "gmx.de",
    "mail.com", "zoho.com", "fastmail.com", "hey.com", "rediffmail.com",
    # Russia / China
    "yandex.com", "yandex.ru", "mail.ru",
    "qq.com", "163.com", "126.com", "sina.com", "sina.cn", "foxmail.com",
})


def _get_cf(custom: Any, key: str) -> Optional[str]:
    """Pull a custom_fields entry by key. Returns None when missing or empty."""
    if not isinstance(custom, list):
        return None
    for cf in custom:
        if isinstance(cf, dict) and cf.get("key") == key:
            val = cf.get("value")
            if val is not None and isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _get_field(recipient: Dict[str, Any], *keys: str) -> Optional[str]:
    """Return the first non-empty value found among the given keys, checking
    top-level fields first, then custom_fields[]. Used to tolerate the v1/v2
    schema split where some installs store fields top-level and others in
    custom_fields."""
    custom = recipient.get("custom_fields") or []
    for key in keys:
        # Top-level
        val = recipient.get(key)
        if val is not None and isinstance(val, str) and val.strip():
            return val.strip()
        # Custom fields
        cf_val = _get_cf(custom, key)
        if cf_val:
            return cf_val
    return None


def _domain_from_email(email: Optional[str]) -> Optional[str]:
    """Parse the domain out of an email address. Returns None if not present
    or if the domain is in the personal-email skip list."""
    if not email or not isinstance(email, str) or "@" not in email:
        return None
    domain = email.split("@", 1)[1].strip().lower()
    if not domain or "." not in domain:
        return None
    if domain in _PERSONAL_EMAIL_DOMAINS:
        return None
    return domain


def _extract_website(recipient: Dict[str, Any]) -> Optional[str]:
    """Pull a website URL from a recipient doc.

    Three-tier resolution:
      1. custom_fields[] for any of _WEBSITE_CF_KEYS (production lead-finder
         data uses `company_website` / `company_domain`).
      2. Top-level recipient fields by the same key list.
      3. Fall back to the recipient's email domain (minus personal-email
         providers). Jonathan at theultimatellc.com → https://theultimatellc.com.

    Returns None when nothing resolves. Caller treats None as "no grounding"
    and runs the degraded-quality tiered-threshold path.
    """
    # Tiers 1 and 2 combined by _get_field — it walks custom_fields then top-level.
    raw = _get_field(recipient, *_WEBSITE_CF_KEYS)
    if raw:
        # If what we pulled is a bare domain, add the scheme. normalize_url
        # handles the rest (trailing slashes, etc.).
        if not raw.startswith(("http://", "https://")):
            raw = f"https://{raw}"
        return raw

    # Tier 3: email domain fallback.
    domain = _domain_from_email(recipient.get("email"))
    if domain:
        return f"https://{domain}"

    return None


def _format_experience_years(months_str: Optional[str]) -> Optional[str]:
    """Convert a `total_experience` months string into human-readable years."""
    if not months_str:
        return None
    try:
        months = int(months_str)
    except (TypeError, ValueError):
        return None
    if months < 12:
        return "< 1 year"
    years = round(months / 12)
    return f"{years} year{'s' if years != 1 else ''}"


# Custom_fields keys that are internal IDs or noise and should NOT appear in
# the writer's recipient summary. Mirrors INTERNAL_KEYS from
# cold-ui/components/recipients/modals/RecipientDetailDialog.tsx.
_SUMMARY_EXCLUDE_CF_KEYS: frozenset = frozenset({
    "saleshandy_id", "saleshandy_status", "lead_finder_run_id",
    "needs_email_enrichment", "update_time", "current_employer_id",
    "region_latitude", "region_longitude", "country_code",
    "profile_pic", "company_logo_url", "picture_url",
    "company_id", "total_experience",
    # Already surfaced explicitly below; don't duplicate.
    "job_title", "current_title", "title",
    "department",
    "location_city", "location_state", "location_country",
    "city", "region", "country",
    "company_website", "company_domain",
    "current_employer_website", "current_employer_domain",
    "website", "domain",
})


def build_recipient_summary(recipient: Dict[str, Any]) -> str:
    """Build a short text summary of the recipient from their meta fields.

    Reads top-level AND custom_fields[] so lead_finder-enriched recipients
    (which write to custom_fields) aren't rendered as blank.
    """
    parts: list = []
    custom = recipient.get("custom_fields") or []

    # Name (top-level only in this schema)
    name = recipient.get("first_name", "")
    if recipient.get("last_name"):
        name = f"{name} {recipient['last_name']}".strip()
    if name:
        parts.append(f"Name: {name}")

    # Title — check v2 (job_title) then v1 (current_title) then legacy
    title = _get_field(recipient, "job_title", "current_title", "title")
    if title:
        parts.append(f"Title: {title}")

    department = _get_field(recipient, "department")
    if department:
        parts.append(f"Department: {department}")

    # Company — top-level `company` first, then `business_name`, then domain
    company = _get_field(recipient, "company", "business_name")
    if company:
        parts.append(f"Company: {company}")

    # Location — v2 keys first, then v1 fallbacks
    city = _get_field(recipient, "location_city", "city")
    state = _get_field(recipient, "location_state", "region")
    country = _get_field(recipient, "location_country", "country")
    loc_bits = [b for b in (city, state, country) if b]
    if loc_bits:
        parts.append(f"Location: {', '.join(loc_bits)}")

    # Industry + company size
    industry = _get_field(recipient, "industry")
    if industry:
        parts.append(f"Industry: {industry}")
    company_size = _get_field(recipient, "company_size")
    if company_size:
        parts.append(f"Company size: {company_size}")

    # Experience (months → years)
    exp_years = _format_experience_years(
        _get_field(recipient, "total_experience")
    )
    if exp_years:
        parts.append(f"Tenure: {exp_years}")

    # Any other useful custom_fields (filter internal IDs + already-surfaced).
    extra_bits: list = []
    if isinstance(custom, list):
        seen: set = set()
        for cf in custom:
            if not isinstance(cf, dict):
                continue
            k, v = cf.get("key"), cf.get("value")
            if not k or not v or not isinstance(v, str):
                continue
            if k in _SUMMARY_EXCLUDE_CF_KEYS:
                continue
            if k in seen:
                continue
            seen.add(k)
            extra_bits.append(f"{k}: {v}")
    if extra_bits:
        parts.append("Additional fields: " + "; ".join(extra_bits[:8]))

    if not parts:
        return "[no recipient meta available]"
    return "\n".join(parts)


def build_company_brief(recipient: Dict[str, Any]) -> str:
    """Pre-fetch + summarize the recipient's company website.

    Returns "" if no website can be resolved or if the fetch failed. Used to
    seed the writer's prompt with company context, reducing writer turns and
    grounding copy in specifics.
    """
    url = _extract_website(recipient)
    if not url:
        return ""
    url = normalize_url(url)
    raw = fetch_url_cached(url, max_chars=4000)
    if not raw:
        return ""

    # Prefer the explicit `company` field for the heading label; fall back to
    # business_name, then the URL.
    biz = (
        _get_field(recipient, "company", "business_name")
        or url
    )
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
