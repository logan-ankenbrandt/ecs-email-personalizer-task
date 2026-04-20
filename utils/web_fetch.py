"""HTTP fetch + HTML strip + filesystem cache.

Used by the P1 research pre-step. Ports the rewriter's web_fetch pattern
(see ~/.scripts/agentic-email-rewriter-v2.py) but synchronous (httpx) to
match the deterministic sync pipeline.
"""

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from config import WEB_FETCH_CACHE_DAYS, WEB_FETCH_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Filesystem cache. /tmp is ephemeral on Fargate per-task, but since each
# sequence-architect run is a single task instance, intra-run dedupe (e.g.
# multiple recipients sharing a domain) still pays off.
_CACHE_DIR = Path("/tmp/cold_web_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url: str) -> str:
    """Add https:// if missing. Strip trailing slash. Lowercase host."""
    url = url.strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    if not parsed.netloc:
        return url
    host = parsed.netloc.lower()
    path = (parsed.path or "/").rstrip("/") or "/"
    return f"{parsed.scheme}://{host}{path}"


def strip_html(html: str) -> str:
    """Convert HTML to plain text, removing scripts, styles, and tags.

    Not a full markdown converter — just enough to give an LLM readable
    text. Preserves paragraph breaks via double-newlines.
    """
    if not html:
        return ""
    # Drop script + style blocks (their text content is junk)
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    # Convert block-level closes to newlines so paragraphs separate
    html = re.sub(r"</(p|div|li|h\d|br|tr|article|section)>", "\n\n", html, flags=re.IGNORECASE)
    # Drop all remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    html = (
        html.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
    )
    # Collapse whitespace
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def _cache_path(url: str) -> Path:
    """Filesystem cache key per URL."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return _CACHE_DIR / f"{digest}.txt"


def fetch_url(url: str, timeout: Optional[int] = None, max_chars: int = 5000) -> str:
    """Fetch a URL, strip HTML, return plain text. No caching.

    Returns "" on any failure (timeout, 4xx/5xx, connection error). Failures
    are logged at WARNING; never raised. Callers degrade gracefully.
    """
    url = normalize_url(url)
    if not url:
        return ""
    timeout = timeout or WEB_FETCH_TIMEOUT_SECONDS
    try:
        # Many corporate sites block default user-agents. Look like a normal browser.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            r = client.get(url)
        if r.status_code >= 400:
            logger.warning("web_fetch %s -> HTTP %d", url, r.status_code)
            return ""
        text = strip_html(r.text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[truncated]"
        logger.info("web_fetch %s -> %d chars", url, len(text))
        return text
    except Exception as e:
        logger.warning("web_fetch %s failed: %s", url, type(e).__name__)
        return ""


def fetch_url_cached(url: str, ttl_days: Optional[int] = None, max_chars: int = 5000) -> str:
    """Fetch with filesystem cache. Re-fetches if cache is older than ttl_days."""
    url = normalize_url(url)
    if not url:
        return ""
    ttl_days = ttl_days if ttl_days is not None else WEB_FETCH_CACHE_DAYS
    path = _cache_path(url)
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < ttl_days:
            try:
                cached = path.read_text(encoding="utf-8")
                logger.info("web_fetch CACHE HIT %s (%.1fd old)", url, age_days)
                return cached
            except Exception:
                pass  # fall through to fresh fetch
    text = fetch_url(url, max_chars=max_chars)
    if text:
        try:
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            logger.warning("web_fetch cache write failed: %s", e)
    return text
