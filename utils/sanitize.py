"""Post-processing sanitizers — deterministic fixes the LLM keeps failing.

Round 3 Phase 1 (P1.2): log review of the last 24h found em_dash and
missing_withcold_link dominating slop violations (7 + 9 occurrences
respectively across 13 iter-level checkpoints). Sonnet 4.6 defaults to
em dashes and occasionally omits the signature block; the refiner then
burns iterations trying to fix what a regex solves in one line.

These helpers run AFTER the writer returns and BEFORE the judge/slop gate
sees the content. They are pure text transforms, no LLM calls.
"""

import re

# The canonical Cold signature block. Matches the structure the writer is
# instructed to emit in the system prompt so post-judge diffs show clean
# output whether the writer produced it or we injected it.
LOGAN_SIGNATURE_HTML = '<p>Logan<br><a href="https://withcold.com">Cold</a></p>'


def sanitize_punctuation(text: str) -> str:
    """Replace em / en dashes with commas.

    Both dashes are banned by the anti-slop rules. Sonnet re-introduces
    them on every refine pass; this step makes the fix deterministic.
    """
    if not text:
        return text
    return text.replace("\u2014", ",").replace("\u2013", ",")


# Detect whether content already contains a withcold.com link in ANY form:
# bare URL, inside an <a href> attribute, or in visible text. Kept case-
# insensitive because LLMs sometimes title-case URL hosts.
_WITHCOLD_RE = re.compile(r"withcold\.com", re.IGNORECASE)


def enforce_signature(content: str, signature_html: str = LOGAN_SIGNATURE_HTML) -> str:
    """Append the Cold signature block if `withcold.com` is missing.

    Non-destructive: if the writer already included the link in any form,
    we leave the content alone. Prevents the writer from shipping drafts
    that fail the slop gate's missing_withcold_link check.
    """
    if not content:
        # An empty body is a separate failure mode; don't paper over it by
        # returning just a signature. Let the caller surface the problem.
        return content
    if _WITHCOLD_RE.search(content):
        return content
    # Trim trailing whitespace so the injected block sits flush.
    return content.rstrip() + "\n" + signature_html
