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
    # C.4: process-mechanics leaks. The writer keeps emitting plumbing
    # details ("three-track system", automation steps, implementation
    # timelines) that Cash gold-standard never describes.
    (
        "process_leak_tracks",
        re.compile(r"\b(?:two|three|four)[\s-]tracks?\b", re.IGNORECASE),
        "Process mechanics leak: don't describe track counts",
        "hard_fail",
    ),
    (
        "process_leak_automation",
        re.compile(r"\b(?:picks them up automatically|fires when|activates net-new)\b", re.IGNORECASE),
        "Process mechanics leak: don't describe automation plumbing",
        "hard_fail",
    ),
    (
        "process_leak_timeline",
        re.compile(r"\b(?:ten|seven|five|fourteen)\s+days?\s+to\s+(?:build|launch|stand up)\b", re.IGNORECASE),
        "Process mechanics leak: don't describe implementation timelines",
        "hard_fail",
    ),
    # C.6: P.S. lines aren't in the Cash gold-standard template and read as
    # filler. Match P.S. at start-of-text OR after a sentence boundary since
    # HTML stripping collapses paragraph breaks to spaces before regex runs.
    (
        "postscript_line",
        re.compile(r"(?:^|[.!?]\s+)P\.?\s*S\.?[:.]?\s+", re.IGNORECASE),
        "P.S. line: not in the gold-standard template",
        "hard_fail",
    ),
    # Round 4 / Tier C.7: 9 new slop patterns identified by line-by-line
    # audit of the first V2 runs. Each targets a specific slop family the
    # writer keeps reaching for when it lacks a hard number to anchor its
    # diagnosis.
    (
        "consultant_gap_framing",
        re.compile(r"\bthe gap I see\b", re.IGNORECASE),
        "Consultant voice: 'the gap I see' framing. State the gap directly.",
        "hard_fail",
    ),
    (
        "forced_negation_on_its_own",
        re.compile(r"\bdoesn[\u2018\u2019']?t\s+\w+(?:\s+\w+){0,5}\s+on its own\b", re.IGNORECASE),
        "Forced negation: 'doesn't X on its own'. Diagnose without negating.",
        "hard_fail",
    ),
    (
        "universality_cant_match",
        re.compile(r"\bmost\s+\w+(?:\s+\w+)?\s+can[\u2018\u2019']?t\s+match\b", re.IGNORECASE),
        "Universality claim: 'most Xs can't match'. Drop the market comparison.",
        "hard_fail",
    ),
    (
        "vague_intensifier_real",
        re.compile(r"\ba\s+real\s+(?:edge|advantage|differentiator|asset|strength)\b", re.IGNORECASE),
        "Vague intensifier: 'a real [noun]'. Name the specific advantage.",
        "hard_fail",
    ),
    (
        "vague_intensifier_meaningful",
        re.compile(
            r"\b(?:a|that[\u2018\u2019']?s\s+a)\s+meaningful\s+(?:return|difference|impact|advantage|edge)\b",
            re.IGNORECASE,
        ),
        "Vague intensifier: 'a meaningful [noun]'. Give the number instead.",
        "hard_fail",
    ),
    (
        "consultant_firms_i_work_with",
        re.compile(r"\bthe\s+(?:firms|companies|agencies)\s+I\s+work\s+with\b", re.IGNORECASE),
        "Consultant voice: 'the firms I work with'. Use a singular proof point instead.",
        "hard_fail",
    ),
    (
        "fake_opinion_signal",
        re.compile(r"\bthat[\u2018\u2019']?s\s+the\s+signal\s+that\s+separates\b", re.IGNORECASE),
        "Fake-opinion connector: 'that's the signal that separates'. State what the signal means.",
        "hard_fail",
    ),
    (
        "vague_conservative_estimate",
        re.compile(
            r"\bat\s+a\s+conservative\s+(?:estimate|average|valuation|projection)\b",
            re.IGNORECASE,
        ),
        "Vague quantifier: 'at a conservative X'. Give the actual number or cut the sentence.",
        "hard_fail",
    ),
    (
        "forced_negation_not_generic",
        re.compile(r"\bnot\s+(?:a\s+)?generic\s+\w+", re.IGNORECASE),
        "Forced negation: 'not a generic X'. State what you ARE, not what you're not.",
        "hard_fail",
    ),
]

# Banned phrases (substring match, case-insensitive)
# A.2: removed "solutions" from this list. Substring match false-positives on
# legitimate company names (e.g., "FirstOption Workforce Solutions"). Plain-
# language "solutions" use is low-signal and the company-name false-positive
# rate outweighed the true-positive catch.
BANNED_PHRASES = [
    "synergy", "leverage", "utilize", "facilitate", "holistic",
    "touching base", "circle back", "loop in", "ping you", "cutting-edge",
    "game-changer", "revolutionary", "next-level", "just checking in",
    "wanted to follow up", "hope you're doing well", "pick your brain",
    "low-hanging fruit", "move the needle", "I know you're busy",
    "reaching out because", "offerings", "suite of services",
    "empower", "unlock", "supercharge", "turbocharge",
]

# Banned adjectives (word-boundary match, case-insensitive)
BANNED_ADJECTIVES = [
    "robust", "comprehensive", "streamlined", "delve", "actionable",
    "bespoke", "captivating", "groundbreaking", "holistic", "impactful",
    "innovative", "insightful", "meticulous", "nuanced", "pivotal",
    "seamless", "synergistic", "transformative", "unparalleled", "unwavering",
]

# T2.3: verbatim template phrases that ship as cold-email slop. Substring
# match. Separate from BANNED_PHRASES because these are full sentences /
# verbatim CTAs that should trigger their own issue category.
# A.3: removed "i'd love to" / "i would love to" — they're already covered
# by the `permission_seeking` regex at SLOP_PATTERNS, which handles smart
# quotes. Including them here caused double-flag (0.2 penalty for one phrase).
TEMPLATE_PHRASES = [
    "here is how it works", "here's how it works",
    "curious if this is on your radar",
    "curious if this matches what you're seeing",
    "does this resonate",
    "hope this finds you well",
    "i wanted to reach out",
    "are you the right person",
    "let me know when you're free",
    "when might be a good time",
    "worth a quick chat",
    "let me know if interested",
]

# C.3: consultant-voice / invented-terminology phrases. Substring match.
# These are patterns the copy-forensic audit surfaced that pass the judge
# but read as MBA-consulting voice, not peer-to-peer. Grow over time as new
# patterns are observed.
CONSULTANT_VOICE_PHRASES = [
    "relationship-reactive",
    "equity to deploy",
    "bd leak",
    "timeline compression",
    "surface conversations",
    "surfacing search mandates",
    "client development ceiling",
    "capacity constraint",
    # Round 4 / Tier C.8: substring-only additions catching what the V2
    # writer emitted in Jonathan's emails. These are phrase-level rather
    # than regex-level (i.e. too short/general to regex) — substring match
    # is sufficient.
    "the gap I see",
    "messaging angle",
    "the firms I work with",
    "at a conservative average",
]

# Structural bounds (T2.3)
MAX_WORDS_BODY = 165  # hard upper bound; ~150 is the target
MAX_CHARS_SUBJECT = 55  # hard upper bound; ~50 is the target

# C.1: sentence-length hard cap. Cash gold standard caps at 22 words.
# Personalizer routinely emits 35-45 word sentences. Judge never flags this.
# Single highest-leverage copy defect per copy-forensic audit.
SENTENCE_WORD_CAP = 25


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
        if self.pattern_type == "word_count_too_high":
            return "Email exceeds 165 words; cut to 150 max"
        if self.pattern_type == "subject_too_long":
            return "Subject exceeds 55 chars; 50 cap"
        if self.pattern_type == "missing_signature":
            return "Missing 'Logan' in signature"
        if self.pattern_type == "missing_withcold_link":
            return "Missing withcold.com link in signature"
        if self.pattern_type == "template_phrase":
            return f"Template phrase: {self.excerpt}"
        if self.pattern_type == "consultant_voice":
            return f"Consultant voice / invented terminology: {self.excerpt}"
        if self.pattern_type == "sentence_too_long":
            return "Sentence exceeds 25-word cap; Cash gold-standard caps at 22"
        return self.pattern_type

    def _suggestion(self) -> str:
        """Fix-oriented guidance a refiner can act on."""
        if self.pattern_type == "em_dash":
            return "Replace the em dash with a comma, period, or parentheses."
        if self.pattern_type == "en_dash":
            return "Replace the en dash with a comma, period, or parentheses."
        if self.pattern_type == "tricolon_list":
            return "Cut the three-item list to two items, or restructure as prose."
        if self.pattern_type == "staccato_repetition":
            return "Combine short sentences into a flowing sentence with subordinate clauses."
        if self.pattern_type == "transition_opener":
            return "Start the sentence with the actual content, not a transition word."
        if self.pattern_type == "hedging":
            return "State the claim directly without hedging."
        if self.pattern_type == "permission_seeking":
            return "Replace with a direct statement or question."
        if self.pattern_type.startswith("template_cta"):
            return "Replace with a CTA that references specific content from THIS email."
        if self.pattern_type.startswith("template_compliment"):
            return "Replace with a data-grounded observation about the recipient."
        if self.pattern_type.startswith("forced_negation"):
            if self.pattern_type == "forced_negation_on_its_own":
                return (
                    "Drop the 'doesn't X on its own' construction. State the "
                    "adjacent positive claim: what DOES move the needle, not "
                    "what doesn't."
                )
            if self.pattern_type == "forced_negation_not_generic":
                return (
                    "Delete the 'not a generic X' clause. If your positive "
                    "claim is specific enough, the contrast is implied."
                )
            return "State the positive claim as its own sentence, no 'not X, but Y' contrast."
        if self.pattern_type == "consultant_gap_framing":
            return (
                "Drop 'the gap I see for firms...'. State the gap directly: "
                "'The harder part at your size is [specific failure mode].'"
            )
        if self.pattern_type == "universality_cant_match":
            return (
                "Delete the 'most Xs can't match' clause. If the recipient's "
                "advantage is obvious from the facts, you don't need to rank "
                "it against the market."
            )
        if self.pattern_type == "vague_intensifier_real":
            return (
                "Replace 'a real [edge/advantage/...]' with the specific "
                "mechanism. If the edge is real, name what creates it."
            )
        if self.pattern_type == "vague_intensifier_meaningful":
            return (
                "Replace 'a meaningful [return/difference/...]' with an "
                "actual number (dollar figure, multiplier, or percentage)."
            )
        if self.pattern_type == "consultant_firms_i_work_with":
            return (
                "Replace 'the firms I work with typically...' with a "
                "singular proof point: 'One client [verb'd] [number].'"
            )
        if self.pattern_type == "fake_opinion_signal":
            return (
                "Drop 'that's the signal that separates...'. State what the "
                "signal means directly: 'A GC pulling permits has a project.'"
            )
        if self.pattern_type == "vague_conservative_estimate":
            return (
                "Replace 'at a conservative [estimate/average]' with the "
                "actual number. If no number exists, cut the sentence."
            )
        if self.pattern_type == "banned_phrase":
            return f"Remove the banned phrase '{self.excerpt.strip()}' entirely."
        if self.pattern_type == "banned_adjective":
            return "Replace the adjective with a concrete noun or verb, or cut it."
        if self.pattern_type == "word_count_too_high":
            return "Cut to under 150 words. Drop a paragraph, not individual words."
        if self.pattern_type == "subject_too_long":
            return "Shorten the subject to under 50 chars. Cut fluff, keep the data point."
        if self.pattern_type == "missing_signature":
            return "Add 'Logan' as the sign-off line."
        if self.pattern_type == "missing_withcold_link":
            return "Add the 'Cold' hyperlink (text 'Cold', href 'https://withcold.com') after the sign-off."
        if self.pattern_type == "template_phrase":
            return f"Replace '{self.excerpt.strip()}' with specific, grounded language."
        if self.pattern_type == "consultant_voice":
            return (
                "Replace the consultant-voice phrase with plain industry language. "
                "Use the recipient's words (placements, desk, referral network), "
                "not invented compound terms."
            )
        if self.pattern_type == "sentence_too_long":
            return (
                "Split this sentence. Cash gold-standard caps at 22 words; hard "
                "limit is 25. Break into two shorter sentences or cut subordinate "
                "clauses."
            )
        if self.pattern_type.startswith("process_leak"):
            return (
                "Remove the process-mechanics description. Reader cares about "
                "outcomes, not plumbing. State what you do (titles, geography), "
                "not how the system works internally."
            )
        if self.pattern_type == "postscript_line":
            return "Remove the P.S. line. Not in the gold-standard template."
        return "Rewrite the flagged excerpt."

    def to_judge_issue(self) -> dict:
        """Shape-compatible with judge.py's `issues[]` schema so the refine
        loop can consume programmatic violations alongside the LLM judge's
        own issues without translation.
        """
        return {
            "excerpt": self.excerpt[:200],
            "slop_type": self.pattern_type,
            "issue": self._description(),
            "suggestion": self._suggestion(),
        }


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


def _check_template_phrases(text: str, position: int, field: str) -> List[Violation]:
    """T2.3: verbatim template-phrase detection.

    A.4: normalize smart / typographic apostrophes to ASCII before matching
    so phrases like "here's how it works" still catch when the LLM outputs
    U+2019. Match is literal substring, so Unicode drift silently misses.
    """
    text_lower = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    violations: List[Violation] = []
    for phrase in TEMPLATE_PHRASES:
        if phrase in text_lower:
            idx = text_lower.find(phrase)
            start = max(0, idx - 20)
            end = min(len(text), idx + len(phrase) + 20)
            violations.append(Violation(
                pattern_type="template_phrase",
                email_position=position,
                field=field,
                excerpt=text[start:end],
                severity="hard_fail",
            ))
    return violations


def _check_consultant_voice(text: str, position: int, field: str) -> List[Violation]:
    """C.3: consultant-voice / invented-terminology detection.

    Substring match against CONSULTANT_VOICE_PHRASES. Same smart-quote
    normalization as _check_template_phrases for parity.
    """
    text_lower = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    violations: List[Violation] = []
    for phrase in CONSULTANT_VOICE_PHRASES:
        if phrase in text_lower:
            idx = text_lower.find(phrase)
            start = max(0, idx - 20)
            end = min(len(text), idx + len(phrase) + 20)
            violations.append(Violation(
                pattern_type="consultant_voice",
                email_position=position,
                field=field,
                excerpt=text[start:end],
                severity="hard_fail",
            ))
    return violations


def _check_sentence_length(text: str, position: int) -> List[Violation]:
    """C.1: flag any body sentence that exceeds SENTENCE_WORD_CAP words.

    Cash gold-standard caps at 22 words. 25-word cap gives a small buffer
    before a violation fires. Single highest-leverage copy defect per audit.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    violations: List[Violation] = []
    for s in sentences:
        wc = len(s.split())
        if wc > SENTENCE_WORD_CAP:
            violations.append(Violation(
                pattern_type="sentence_too_long",
                email_position=position,
                field="content",
                excerpt=f"[{wc} words] {s[:180]}",
                severity="hard_fail",
            ))
    return violations


def _check_structure(
    subject: str,
    content_text: str,
    position: int,
    raw_content: str = "",
) -> List[Violation]:
    """T2.3: structural checks that aren't pattern-level.

    - body word count ceiling (cut at 165, target 150)
    - subject character ceiling (cut at 55, target 50)
    - body must contain 'Logan' in the signature
    - body must contain a withcold.com link (hyperlink or bare)

    A.1: the withcold.com check looks at the RAW content (with HTML intact).
    _strip_html replaces `<a href="https://withcold.com">` with a space, so a
    stripped-text scan loses the URL even when the link is correctly present
    in the signature. Pass raw_content (the unstripped body) so href
    attributes survive.
    """
    violations: List[Violation] = []

    # Subject too long
    if subject and len(subject) > MAX_CHARS_SUBJECT:
        violations.append(Violation(
            pattern_type="subject_too_long",
            email_position=position,
            field="subject",
            excerpt=subject[:80],
            severity="hard_fail",
        ))

    if not content_text:
        return violations

    # Body word count
    words = content_text.split()
    if len(words) > MAX_WORDS_BODY:
        violations.append(Violation(
            pattern_type="word_count_too_high",
            email_position=position,
            field="content",
            excerpt=f"[{len(words)} words, max {MAX_WORDS_BODY}]",
            severity="hard_fail",
        ))

    # Signature: must contain 'Logan' somewhere in the body
    if "Logan" not in content_text:
        violations.append(Violation(
            pattern_type="missing_signature",
            email_position=position,
            field="content",
            excerpt="[signature missing]",
            severity="hard_fail",
        ))

    # Signature link: must reference withcold.com. Check RAW content (if
    # provided) so href attributes inside <a> tags survive HTML stripping.
    # Fall back to stripped text for backward compatibility with callers
    # that didn't pass raw_content.
    haystack = raw_content if raw_content else content_text
    if "withcold.com" not in haystack.lower():
        violations.append(Violation(
            pattern_type="missing_withcold_link",
            email_position=position,
            field="content",
            excerpt="[withcold.com link missing]",
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
    stripped_content = ""
    for raw, field in [(subject or "", "subject"), (content or "", "content")]:
        text = _strip_html(raw)
        if field == "content":
            stripped_content = text
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
            # C.1: sentence-length check only meaningful on body.
            violations.extend(_check_sentence_length(text, position))
        violations.extend(_check_banned_phrases(text, position, field))
        violations.extend(_check_banned_adjectives(text, position, field))
        violations.extend(_check_template_phrases(text, position, field))
        # C.3: consultant-voice check applies to both subject and body.
        violations.extend(_check_consultant_voice(text, position, field))

    # T2.3: structural checks run once per email (need both subject + body).
    # A.1: pass raw content so the withcold.com URL survives inside href attrs.
    violations.extend(
        _check_structure(
            _strip_html(subject or ""),
            stripped_content,
            position,
            raw_content=content or "",
        )
    )
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
