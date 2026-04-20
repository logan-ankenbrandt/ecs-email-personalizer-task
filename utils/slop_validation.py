"""Programmatic slop validation for generated email copy.

Belt-and-suspenders for the Opus critic. Catches deterministic patterns
the LLM judge may miss (banned phrases, em dashes, signature deviations,
forced negation in regex form). Triggers granular_refine retries on hard
failures; remaining violations after retries become soft `quality_warnings`
on the promoted sequence (existing mongo.py pattern).

Patterns ported from ~/.scripts/rewriter_config_v2.py with minor additions
specific to the sequence-architect output shape (no recipient-specific
context to anchor swap tests).
"""

import logging
import re
from typing import List, NamedTuple

logger = logging.getLogger(__name__)


# ============================================================
# Pattern catalog (ported verbatim from rewriter_config_v2.py)
# ============================================================

# Regex slop patterns: (pattern_type, compiled_regex, description, severity)
# Severity: 'hard_fail' triggers granular_refine; 'deduction' is logged only.
# Currently all are hard_fail; deduction reserved for future soft checks.
SLOP_PATTERNS = [
    (
        "tricolon_list",
        re.compile(r"\b\w+,\s+\w+,\s+(?:and|or)\s+\w+\b", re.IGNORECASE),
        "Three-item comma list (tricolon)",
        "hard_fail",
    ),
    (
        "forced_negation_staccato",
        # Matches "Not X. Y." pattern where Y is a short noun phrase fragment
        re.compile(r"\bNot\s+a\s+\w+\s+(?:problem|issue)\.\s+A\s+\w+", re.IGNORECASE),
        "Forced negation staccato ('Not X problem. A Y problem.')",
        "hard_fail",
    ),
    (
        "forced_negation_inline",
        re.compile(r"\b(?:is|are|was|were)\s+not\s+(?:a|an)\s+\w+(?:\s+\w+)*[,;]\s*(?:it|they)\s+(?:is|are)", re.IGNORECASE),
        "Forced negation inline ('X is not Y, it is Z')",
        "hard_fail",
    ),
    (
        "without_dramatic",
        # "without [verb-ing]" used for dramatic contrast (positive after negative)
        re.compile(r"\bwithout\s+\w+ing\s+(?:your|the|a|an)\s+\w+", re.IGNORECASE),
        "Forced negation 'without X-ing' contrast",
        "hard_fail",
    ),
    (
        "template_cta_radar",
        re.compile(r"curious if this is on your radar", re.IGNORECASE),
        "Template CTA: 'curious if this is on your radar'",
        "hard_fail",
    ),
    (
        "template_cta_matches",
        re.compile(r"curious if this matches what you[''']re seeing", re.IGNORECASE),
        "Template CTA: 'curious if this matches what you're seeing' (verbatim)",
        "hard_fail",
    ),
    (
        "template_cta_resonates",
        re.compile(r"does (?:this|that) resonate", re.IGNORECASE),
        "Template CTA: 'does this resonate'",
        "hard_fail",
    ),
    (
        "template_compliment_impressed",
        re.compile(r"impressed by what you[''](?:ve|re)\s+(?:built|building|doing)", re.IGNORECASE),
        "Template compliment: 'impressed by what you've built'",
        "hard_fail",
    ),
    (
        "template_compliment_love",
        re.compile(r"love what you[''](?:re|ve)\s+(?:doing|built|building)", re.IGNORECASE),
        "Template compliment: 'love what you're doing'",
        "hard_fail",
    ),
    (
        "em_dash",
        re.compile(r"—"),
        "Em dash (—); use commas, periods, or parentheses",
        "hard_fail",
    ),
    (
        "en_dash",
        re.compile(r"–"),
        "En dash (–); use commas, periods, or parentheses",
        "hard_fail",
    ),
    (
        "transition_opener",
        re.compile(
            r"(?:^|\n|\.\s+)(?:Moreover|Furthermore|Additionally|Importantly|Notably|Significantly|Interestingly|Crucially|Ultimately),",
            re.IGNORECASE,
        ),
        "Transition-word opener (Moreover/Furthermore/Additionally/etc.)",
        "hard_fail",
    ),
    (
        "hedging",
        re.compile(r"\b(?:I think|it seems like|it appears that|might be|could potentially)\b", re.IGNORECASE),
        "Hedging language",
        "hard_fail",
    ),
    (
        "permission_seeking",
        re.compile(r"\b(?:I[''']?d love to|would love to|I[''']?d be happy to|would it be alright)", re.IGNORECASE),
        "Permission-seeking ('I'd love to', 'would love to')",
        "hard_fail",
    ),
]

# Banned phrases (substring match, case-insensitive)
BANNED_PHRASES = [
    "synergy", "leverage", "utilize", "facilitate", "holistic",
    "touching base", "circle back", "loop in", "ping you", "cutting-edge",
    "game-changer", "revolutionary", "next-level", "just checking in",
    "wanted to follow up", "hope you're doing well", "pick your brain",
    "low-hanging fruit", "move the needle", "I know you're busy",
    "reaching out because", "solutions", "offerings", "suite of services",
    "empower", "unlock", "supercharge", "turbocharge",
]

# Banned adjectives (word-boundary match, case-insensitive)
BANNED_ADJECTIVES = [
    "robust", "comprehensive", "streamlined", "delve", "actionable",
    "bespoke", "captivating", "groundbreaking", "holistic", "impactful",
    "innovative", "insightful", "meticulous", "nuanced", "pivotal",
    "seamless", "synergistic", "transformative", "unparalleled", "unwavering",
]


# ============================================================
# Result types
# ============================================================

class Violation(NamedTuple):
    pattern_type: str
    email_position: int
    field: str  # 'subject' or 'content'
    excerpt: str
    severity: str  # 'hard_fail' or 'deduction'

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type,
            "email_position": self.email_position,
            "field": self.field,
            "excerpt": self.excerpt[:200],
            "severity": self.severity,
            "issue": self._description(),
        }

    def _description(self) -> str:
        # Find the matching pattern's description
        for ptype, _re, desc, _sev in SLOP_PATTERNS:
            if ptype == self.pattern_type:
                return desc
        if self.pattern_type == "banned_phrase":
            return f"Banned phrase: {self.excerpt}"
        if self.pattern_type == "banned_adjective":
            return f"Banned adjective: {self.excerpt}"
        if self.pattern_type == "staccato_repetition":
            return "3+ consecutive short sentences (staccato repetition)"
        return self.pattern_type


class ValidationResult(NamedTuple):
    hard_fails: List[Violation]
    soft_warns: List[Violation]

    @property
    def is_clean(self) -> bool:
        return not self.hard_fails and not self.soft_warns


# ============================================================
# HTML stripping (so regexes don't match inside tags)
# ============================================================

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# Specialized checks
# ============================================================

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _check_staccato(text: str, position: int, field: str) -> List[Violation]:
    """Catch 3+ consecutive sentences each <8 words (staccato repetition)."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    short_run = 0
    violations = []
    for i, s in enumerate(sentences):
        word_count = len(s.split())
        if word_count < 8:
            short_run += 1
            if short_run >= 3:
                # Capture the run of short sentences
                run_start = max(0, i - 2)
                excerpt = " ".join(sentences[run_start:i + 1])
                violations.append(Violation(
                    pattern_type="staccato_repetition",
                    email_position=position,
                    field=field,
                    excerpt=excerpt,
                    severity="hard_fail",
                ))
                short_run = 0  # reset to avoid duplicate flagging on same run
        else:
            short_run = 0
    return violations


def _check_banned_phrases(text: str, position: int, field: str) -> List[Violation]:
    text_lower = text.lower()
    violations = []
    for phrase in BANNED_PHRASES:
        if phrase.lower() in text_lower:
            # Get the surrounding context
            idx = text_lower.find(phrase.lower())
            start = max(0, idx - 30)
            end = min(len(text), idx + len(phrase) + 30)
            excerpt = text[start:end]
            violations.append(Violation(
                pattern_type="banned_phrase",
                email_position=position,
                field=field,
                excerpt=excerpt,
                severity="hard_fail",
            ))
    return violations


def _check_banned_adjectives(text: str, position: int, field: str) -> List[Violation]:
    violations = []
    for adj in BANNED_ADJECTIVES:
        # Word-boundary match
        if re.search(r"\b" + re.escape(adj) + r"\b", text, re.IGNORECASE):
            match = re.search(r"\b" + re.escape(adj) + r"\b", text, re.IGNORECASE)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            excerpt = text[start:end]
            violations.append(Violation(
                pattern_type="banned_adjective",
                email_position=position,
                field=field,
                excerpt=excerpt,
                severity="hard_fail",
            ))
    return violations


# ============================================================
# Public API
# ============================================================

def validate_email(subject: str, content: str, position: int) -> List[Violation]:
    """Validate a single email against all slop patterns. Returns flat list."""
    violations = []
    for raw, field in [(subject or "", "subject"), (content or "", "content")]:
        text = _strip_html(raw)
        if not text:
            continue
        # Regex patterns
        for pattern_type, regex, _desc, severity in SLOP_PATTERNS:
            for match in regex.finditer(text):
                excerpt = match.group(0)
                # Pad with surrounding context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context_excerpt = text[start:end]
                violations.append(Violation(
                    pattern_type=pattern_type,
                    email_position=position,
                    field=field,
                    excerpt=context_excerpt,
                    severity=severity,
                ))
        # Specialized checks (only on body content; subject is too short)
        if field == "content":
            violations.extend(_check_staccato(text, position, field))
        violations.extend(_check_banned_phrases(text, position, field))
        violations.extend(_check_banned_adjectives(text, position, field))
    return violations


def validate_sequence(emails: List[dict]) -> ValidationResult:
    """Validate every email in a sequence. Returns split hard_fail / soft_warn lists."""
    hard_fails: List[Violation] = []
    soft_warns: List[Violation] = []
    for i, email in enumerate(emails, start=1):
        position = int(email.get("position") or i)
        subject = email.get("subject", "")
        content = email.get("content", "")
        for v in validate_email(subject, content, position):
            if v.severity == "hard_fail":
                hard_fails.append(v)
            else:
                soft_warns.append(v)
    if hard_fails or soft_warns:
        logger.warning(
            "validate_sequence: %d hard_fails + %d soft_warns across %d emails",
            len(hard_fails), len(soft_warns), len(emails),
        )
    else:
        logger.info("validate_sequence: clean (%d emails, 0 violations)", len(emails))
    return ValidationResult(hard_fails=hard_fails, soft_warns=soft_warns)
