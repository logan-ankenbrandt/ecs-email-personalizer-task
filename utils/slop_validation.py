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
import statistics
from typing import List, NamedTuple

logger = logging.getLogger(__name__)


# ============================================================
# Grammar / rhythm rule helpers (R0-R7, see grammar.md)
# ============================================================

# Lightweight POS heuristics. No spaCy dependency. Word lists cover the
# tokens we care about for opener-class detection (R3, R4, R6).
_DET_TOKENS = {
    "the", "a", "an", "those", "that", "this", "these", "your", "our",
}
_PRON_TOKENS = {"i", "we", "you", "they", "it"}
# Verb word list used for two purposes:
#   1. R1 staccato exemption: <=3 word sentences with no verb match are
#      treated as intentional fragments (e.g. "Healthcare. Construction. IT.")
#      and excluded from the short-run counter.
#   2. R3/R4/R6 opener classification: VERB lookup for imperative / verb-led
#      sentence detection.
_VERB_TOKENS_HEURISTIC = {
    "is", "are", "was", "were", "run", "runs", "build", "builds",
    "has", "had", "do", "does",
}
_VERB_PRESENCE_RE = re.compile(
    r"\b(?:is|are|was|were|run|runs|build|builds|has|had|do|does)\b",
    re.IGNORECASE,
)
# Broader verb-shape heuristic for R7 (independent-clause detection).
# Catches common -s / -ed / -ing inflections and a small set of irregulars
# that the R1 exemption list deliberately excludes. This is intentionally
# loose (some adjectives ending in -ed will match) because R7 only fires
# on 3+ comma-joined clauses, so a one-off false match in a single clause
# is harmless.
_VERB_SHAPE_RE = re.compile(
    r"\b(?:\w{2,}(?:s|ed|ing)|is|are|was|were|be|been|being|has|had|have|"
    r"do|does|did|made|make|makes|run|runs|ran|build|builds|built|"
    r"send|sends|sent|got|get|gets|gives|give|gave|tell|tells|told|"
    r"work|works|worked|see|sees|saw|come|comes|came|go|goes|went|"
    r"keep|keeps|kept|put|puts|hold|holds|held)\b",
    re.IGNORECASE,
)

# FANBOYS coordinating-conjunction joiner (R2). Looks for ", and " family
# patterns. Subordinators (R2 second pass) count as flow indicators even
# when no FANBOYS joiner is present.
_FANBOYS_JOINER_RE = re.compile(
    r",\s+(?:and|but|so|yet|or|for|nor)\s+",
    re.IGNORECASE,
)
_SUBORDINATOR_RE = re.compile(
    r"\b(?:because|while|if|when|since|although|though|whereas|unless|"
    r"until|after|before)\b",
    re.IGNORECASE,
)
_SUBORDINATE_LEAD_TOKENS = {
    "if", "when", "while", "because", "since", "although", "though",
    "after", "before", "unless", "until", "whereas",
}

# Paragraph segmentation runs on the RAW (pre-strip) HTML so paragraph
# boundaries survive. _strip_html collapses all whitespace.
_PARAGRAPH_SPLIT_RE = re.compile(
    r"</p\s*>|<p[^>]*>|<br\s*/?>\s*<br\s*/?>|\n{2,}",
    re.IGNORECASE,
)

# Bare-name greeting (R0): a single titlecase word followed by a comma
# and nothing else on the line (e.g. "Logan,"). The accepted form is
# "Hi Logan," / "Hey Logan,".
_BARE_NAME_GREETING_RE = re.compile(r"^[A-Z][a-z]+,$")


def _split_paragraphs(raw_content: str) -> List[str]:
    """Split raw HTML body into per-paragraph stripped text.

    Splits on </p>, <p>, <br><br>, and blank lines BEFORE running
    _strip_html, since _strip_html collapses whitespace and would erase
    paragraph boundaries. Returns paragraphs in document order with empty
    strings dropped.
    """
    if not raw_content:
        return []
    chunks = _PARAGRAPH_SPLIT_RE.split(raw_content)
    paragraphs: List[str] = []
    for chunk in chunks:
        stripped = _strip_html(chunk) if chunk else ""
        if stripped:
            paragraphs.append(stripped)
    return paragraphs


def _first_token(sentence: str) -> str:
    """Return the first whitespace-separated token, with surrounding
    punctuation stripped. Empty string if no token."""
    parts = sentence.strip().split()
    if not parts:
        return ""
    return parts[0].strip(".,;:!?\"'()[]")


_GREETING_TOKENS = {"hi", "hey", "hello"}
_INTERJECTION_TOKENS = {"hi", "hey", "hello", "yes", "no", "okay", "ok", "sure"}
_QUANTIFIER_TOKENS = {"one", "two", "three", "four", "five", "six", "many", "few", "some", "most"}


def _classify_pos(token: str) -> str:
    """Coarse POS classification using word lists + suffix heuristics.

    Returns one of: 'DET', 'PRON', 'VERB', 'PROPN', 'NOUN', 'OTHER'.

    Conservative on the NOUN fallback: only commits to NOUN when the
    token has clear noun-ish shape (lowercase, no -ing/-ed/-ly suffix).
    Subordinators, interjections, quantifiers, and adverbs all return
    OTHER so the soft POS-class run check (R3, R4) doesn't pile false
    positives onto syntactically diverse sentences that happen to begin
    with non-DET/PRON/VERB tokens.
    """
    if not token:
        return "OTHER"
    lower = token.lower()
    if lower in _DET_TOKENS:
        return "DET"
    if lower in _PRON_TOKENS:
        return "PRON"
    if lower in _VERB_TOKENS_HEURISTIC:
        return "VERB"
    if lower in _SUBORDINATE_LEAD_TOKENS:
        return "OTHER"
    if lower in _INTERJECTION_TOKENS:
        return "OTHER"
    if lower in _QUANTIFIER_TOKENS:
        return "OTHER"
    if lower.endswith("ly") and len(lower) > 3:
        return "OTHER"
    if lower.endswith(("ing", "ed")) and len(lower) > 4:
        return "VERB"
    if lower.endswith("s") and len(lower) > 3 and not token[0].isupper():
        return "VERB"
    if token[:1].isupper() and not token.isupper():
        return "PROPN"
    return "NOUN"


def _opener_bigram_class(sentence: str) -> str:
    """Classify the first 2-token bigram of a sentence (R3).

    Returns one of 'DET_NOUN', 'PRON_VERB', 'PROPN_VERB', or 'OTHER'.
    """
    parts = [p.strip(".,;:!?\"'()[]") for p in sentence.strip().split()]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return "OTHER"
    a = _classify_pos(parts[0])
    b = _classify_pos(parts[1])
    if a == "DET" and b in {"NOUN", "PROPN"}:
        return "DET_NOUN"
    if a == "PRON" and b == "VERB":
        return "PRON_VERB"
    if a == "PROPN" and b == "VERB":
        return "PROPN_VERB"
    return "OTHER"


def _classify_first_sentence_pattern(sentence: str) -> str:
    """Classify the first sentence of a paragraph (R6).

    Returns one of: 'DECLARATIVE_NOUN_LEAD', 'SUBORDINATE_LEAD', 'QUESTION',
    'IMPERATIVE', 'FRAGMENT'.
    """
    s = sentence.strip()
    if not s:
        return "FRAGMENT"
    if s.endswith("?"):
        return "QUESTION"
    tokens = [t.strip(".,;:!?\"'()[]") for t in s.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return "FRAGMENT"
    first_lower = tokens[0].lower()
    has_verb = bool(_VERB_PRESENCE_RE.search(s))
    if not has_verb and len(tokens) <= 6:
        return "FRAGMENT"
    if first_lower in _SUBORDINATE_LEAD_TOKENS:
        return "SUBORDINATE_LEAD"
    # Comma within first 6 tokens signals a leading subordinate clause.
    head = " ".join(tokens[:6])
    if "," in head and first_lower not in _DET_TOKENS and first_lower not in _PRON_TOKENS:
        # Only treat as subordinate lead if the leading clause looks
        # like a clause, not a list. Check for a subordinator keyword.
        if _SUBORDINATOR_RE.search(head):
            return "SUBORDINATE_LEAD"
    if first_lower in _VERB_TOKENS_HEURISTIC:
        return "IMPERATIVE"
    if _classify_pos(tokens[0]) == "VERB" and tokens[0][:1].islower():
        return "IMPERATIVE"
    return "DECLARATIVE_NOUN_LEAD"


def _strip_leading_greeting(sentence: str) -> str:
    """Remove a leading "Hi/Hey/Hello <Name>," prefix from a sentence.

    _strip_html collapses paragraph boundaries, so the greeting paragraph
    ("Hi Logan,") gets concatenated into the first body sentence. This
    helper trims the greeting prefix so opener-class analysis sees the
    actual first body token. If no greeting prefix is present the
    sentence is returned unchanged.
    """
    s = sentence.strip()
    if not s:
        return s
    parts = s.split(None, 2)
    if not parts:
        return s
    if parts[0].lower() not in _GREETING_TOKENS:
        return s
    if len(parts) < 3:
        return ""
    if parts[1].endswith(","):
        return parts[2].strip()
    return s


def _is_skippable_paragraph_for_r6(paragraph: str) -> bool:
    """R6 skips paragraphs starting with digit + period or open quote."""
    s = paragraph.lstrip()
    if not s:
        return True
    if re.match(r"^\d+\.", s):
        return True
    if s[:1] in {"\"", "'", "“", "‘"}:
        return True
    return False


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
    # Round 4.5: widened from the v4 "most X can't match" pattern to cover
    # the broader "most [noun phrase] don't/can't/aren't/haven't/won't [verb]"
    # family. v6 output still emitted "most trades staffing firms don't
    # have" and "most outbound they receive comes from..." — both are
    # universality claims that the narrow regex missed.
    (
        "universality_most_negation",
        re.compile(
            r"\bmost\s+\w+(?:\s+\w+){0,4}\s+"
            r"(?:don[\u2018\u2019']?t|can[\u2018\u2019']?t|aren[\u2018\u2019']?t|"
            r"haven[\u2018\u2019']?t|won[\u2018\u2019']?t|isn[\u2018\u2019']?t|"
            r"doesn[\u2018\u2019']?t|never)\b",
            re.IGNORECASE,
        ),
        "Universality claim: 'most X don't/can't/never Y'. Drop the market comparison.",
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
    # Round 4.5: "generalist firm/agency/etc" market-comparison fallback
    # that v6 reached for 3 times across the sequence ("a generalist firm
    # can't", "than it does for a generalist firm", "generalist staffing
    # agencies"). The writer uses "generalist" when it wants to position
    # the recipient against a rival but doesn't have a specific named
    # competitor. Ban the shortcut.
    (
        "generalist_comparison",
        re.compile(
            r"\b(?:a|than|like|from|as)\s+generalist\s+"
            r"(?:firm|firms|staffing|agency|agencies|shop|shops|outfit|outfits|recruiter|recruiters)\b",
            re.IGNORECASE,
        ),
        "Market-comparison shortcut: 'a generalist [firm/agency/...]'. "
        "Name a specific alternative or drop the contrast.",
        "hard_fail",
    ),
    # Round 4.5: vague "different category/level/tier/class" phrasing that
    # v6 used in step 3 ("conversation starts at a different level") and
    # step 4 ("put you in a different category"). These phrases sound
    # grounded but are consultant-deck abstractions that lack specifics.
    (
        "vague_category_comparison",
        re.compile(
            r"\b(?:in|at|into|on)\s+a\s+different\s+(?:category|level|tier|class|bucket|league)\b",
            re.IGNORECASE,
        ),
        "Vague comparison: 'in/at a different [category/level/...]'. "
        "Name the specific trait that's different instead.",
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
        if self.pattern_type == "greeting_too_terse":
            return "Bare-name greeting; opens with just '<Name>,' instead of 'Hi <Name>,'"
        if self.pattern_type == "fanboys_density_low":
            return "Paragraph has 3+ sentences with no coordinating conjunction or subordinator"
        if self.pattern_type == "paragraph_opener_monotony":
            return "3+ paragraphs in a row open with the same grammatical structure"
        if self.pattern_type == "sentence_opener_repetition":
            return "3+ consecutive sentences share the same opener token or POS class"
        if self.pattern_type == "length_rhythm_flat":
            return "Sentence-length variance too low (flat rhythm)"
        if self.pattern_type == "paragraph_grammar_uniformity":
            return "3+ paragraphs share the same first-sentence syntactic pattern"
        if self.pattern_type == "comma_stack_parallel":
            return "3+ comma-joined parallel independent clauses with no conjunction"
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
        if self.pattern_type == "universality_most_negation":
            return (
                "Drop the 'most Xs don't/can't Y' clause. If the recipient's "
                "advantage is real, it doesn't need ranking against 'most' "
                "of anything."
            )
        if self.pattern_type == "generalist_comparison":
            return (
                "Replace 'a generalist firm' with a specific named competitor, "
                "or drop the comparison entirely. Generalist is the fallback "
                "when you don't have a real rival to point at."
            )
        if self.pattern_type == "vague_category_comparison":
            return (
                "Replace 'in a different category' with the specific trait "
                "that's different. 'Your recruiters have field experience' "
                "beats 'puts you in a different category.'"
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
        if self.pattern_type == "greeting_too_terse":
            return (
                "Open with 'Hi {first_name},' or 'Hey {first_name},' rather "
                "than just the name. The bare-name greeting reads mechanical."
            )
        if self.pattern_type == "fanboys_density_low":
            return (
                "This paragraph has three or more sentences with no "
                "coordinating conjunction. Merge two related sentences with "
                "'and', 'but', or 'so' to give it flow."
            )
        if self.pattern_type == "paragraph_opener_monotony":
            return (
                "Three paragraphs in a row open with the same grammatical "
                "structure. Rewrite at least one to start with a "
                "subordinate clause, a question, or a different subject."
            )
        if self.pattern_type == "sentence_opener_repetition":
            return (
                "Three sentences in a row start with the same word. Vary "
                "the openers, one can lead with a subordinate clause, an "
                "adverb, or a question."
            )
        if self.pattern_type == "length_rhythm_flat":
            return (
                "The sentence-length rhythm is too flat. Add one longer "
                "sentence (20+ words) with a subordinate clause, or break "
                "a medium sentence into a punchy 4-6 word follow-up to "
                "vary cadence."
            )
        if self.pattern_type == "paragraph_grammar_uniformity":
            return (
                "Every paragraph starts with the same grammatical pattern. "
                "Try opening one paragraph with a question, a conditional "
                "clause ('If...'), or a short imperative."
            )
        if self.pattern_type == "comma_stack_parallel":
            return (
                "Comma-stacked parallel clauses read as AI rhythm. Use "
                "conjunctions or break into separate sentences."
            )
        if self.pattern_type == "staccato_repetition":
            return (
                "Combine these short sentences with conjunctions (and/but/so) "
                "into one flowing sentence, or vary the lengths so no three "
                "in a row are below 12 words."
            )
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
    """R1: staccato_short_run_v2.

    Two triggers, both hard_fail:
      A. 3+ consecutive sentences each <=12 words.
      B. 2+ consecutive sentences each <=6 words.

    Sentences with <=3 words AND no verb match are exempted from short-run
    counting (handles intentional triplets like "Healthcare. Construction.
    IT.") so they neither contribute to nor break a run.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    violations: List[Violation] = []
    run_le12: List[int] = []  # indices contributing to trigger A
    run_le6: List[int] = []   # indices contributing to trigger B
    for i, s in enumerate(sentences):
        word_count = len(s.split())
        is_exempt = word_count <= 3 and not _VERB_PRESENCE_RE.search(s)
        if is_exempt:
            # Exempt sentence: pass through without resetting or extending.
            continue
        if word_count <= 12:
            run_le12.append(i)
            if word_count <= 6:
                run_le6.append(i)
            else:
                if len(run_le6) >= 2:
                    excerpt = " ".join(sentences[run_le6[0]:run_le6[-1] + 1])
                    violations.append(Violation(
                        pattern_type="staccato_repetition",
                        email_position=position,
                        field=field,
                        excerpt=excerpt,
                        severity="hard_fail",
                    ))
                run_le6 = []
            if len(run_le12) >= 3:
                excerpt = " ".join(sentences[run_le12[0]:run_le12[-1] + 1])
                violations.append(Violation(
                    pattern_type="staccato_repetition",
                    email_position=position,
                    field=field,
                    excerpt=excerpt,
                    severity="hard_fail",
                ))
                run_le12 = []
                run_le6 = []
        else:
            if len(run_le6) >= 2:
                excerpt = " ".join(sentences[run_le6[0]:run_le6[-1] + 1])
                violations.append(Violation(
                    pattern_type="staccato_repetition",
                    email_position=position,
                    field=field,
                    excerpt=excerpt,
                    severity="hard_fail",
                ))
            run_le12 = []
            run_le6 = []
    # Flush any remaining run_le6 at end of text.
    if len(run_le6) >= 2:
        excerpt = " ".join(sentences[run_le6[0]:run_le6[-1] + 1])
        violations.append(Violation(
            pattern_type="staccato_repetition",
            email_position=position,
            field=field,
            excerpt=excerpt,
            severity="hard_fail",
        ))
    return violations


def _check_greeting_too_terse(
    paragraphs: List[str], position: int,
) -> List[Violation]:
    """R0: bare-name greeting like "Logan," with no greeting word.

    Examines the first non-empty paragraph of the body. Triggers if it
    matches `^[A-Z][a-z]+,$`. Operates on paragraph segmentation rather
    than sentence-split because _strip_html collapses paragraph
    boundaries, which would concatenate the greeting with the next
    paragraph's text and miss the bare-name signal.
    """
    if not paragraphs:
        return []
    first = paragraphs[0].strip()
    if _BARE_NAME_GREETING_RE.match(first):
        return [Violation(
            pattern_type="greeting_too_terse",
            email_position=position,
            field="content",
            excerpt=first,
            severity="hard_fail",
        )]
    return []


def _check_fanboys_density(paragraphs: List[str], position: int) -> List[Violation]:
    """R2: paragraphs with >=3 sentences and zero coordinating-conjunction
    joiners (FANBOYS: and/but/so/yet/or/for/nor) AND no subordinator
    (because/while/if/...). Soft signal only."""
    violations: List[Violation] = []
    for para in paragraphs:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
        if len(sentences) < 3:
            continue
        if _FANBOYS_JOINER_RE.search(para):
            continue
        if _SUBORDINATOR_RE.search(para):
            continue
        violations.append(Violation(
            pattern_type="fanboys_density_low",
            email_position=position,
            field="content",
            excerpt=para[:200],
            severity="deduction",
        ))
    return violations


def _check_paragraph_opener_monotony(
    paragraphs: List[str], position: int,
) -> List[Violation]:
    """R3: 3+ paragraphs in a row whose first 2-token bigram shares the
    same opener class (DET_NOUN, PRON_VERB, PROPN_VERB). Skip greeting
    (first) and signature (last) paragraphs. Soft for runs of 3, hard for
    runs of 4+. Emits one violation per maximal run."""
    violations: List[Violation] = []
    if len(paragraphs) <= 2:
        return violations
    middle = paragraphs[1:-1]
    classes = []
    for para in middle:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
        first_sentence = sentences[0] if sentences else para
        classes.append(_opener_bigram_class(first_sentence))

    i = 0
    while i < len(classes):
        cls = classes[i]
        if cls == "OTHER":
            i += 1
            continue
        j = i
        while j + 1 < len(classes) and classes[j + 1] == cls:
            j += 1
        run_len = j - i + 1
        if run_len >= 3:
            severity = "hard_fail" if run_len >= 4 else "deduction"
            excerpt = " | ".join(p[:80] for p in middle[i:j + 1])
            violations.append(Violation(
                pattern_type="paragraph_opener_monotony",
                email_position=position,
                field="content",
                excerpt=f"[{cls} x{run_len}] {excerpt}",
                severity=severity,
            ))
        i = j + 1
    return violations


def _check_sentence_opener_repetition(
    text: str, position: int, field: str,
) -> List[Violation]:
    """R4: 3+ consecutive sentences sharing the same opener.

    Hard fail when the lowercase first token is identical across 3+ in a
    row. Soft (deduction) when the first-token POS class matches but the
    tokens differ. Only one violation per maximal run; hard takes priority
    over soft for the same run."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    # _strip_html collapses paragraph boundaries, so the greeting paragraph
    # ("Hi Logan,") merges into the first body sentence. Strip a leading
    # greeting-token + name + comma prefix so the actual first sentence's
    # opener is what gets classified.
    sentences = [_strip_leading_greeting(s) if i == 0 else s
                 for i, s in enumerate(sentences)]
    sentences = [s for s in sentences if s]
    if len(sentences) < 3:
        return []
    tokens = [_first_token(s).lower() for s in sentences]
    pos_classes = [_classify_pos(t) for t in tokens]
    violations: List[Violation] = []

    i = 0
    while i < len(sentences):
        if not tokens[i]:
            i += 1
            continue
        # Literal-token run.
        j = i
        while j + 1 < len(tokens) and tokens[j + 1] == tokens[i] and tokens[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= 3:
            excerpt = " ".join(sentences[i:j + 1])
            violations.append(Violation(
                pattern_type="sentence_opener_repetition",
                email_position=position,
                field=field,
                excerpt=f"[token='{tokens[i]}' x{run_len}] {excerpt[:200]}",
                severity="hard_fail",
            ))
            i = j + 1
            continue
        # POS-class run (soft) when literal run did not fire.
        cls = pos_classes[i]
        if cls in {"OTHER", ""}:
            i += 1
            continue
        k = i
        while k + 1 < len(pos_classes) and pos_classes[k + 1] == cls:
            k += 1
        pos_run = k - i + 1
        if pos_run >= 3:
            excerpt = " ".join(sentences[i:k + 1])
            violations.append(Violation(
                pattern_type="sentence_opener_repetition",
                email_position=position,
                field=field,
                excerpt=f"[POS={cls} x{pos_run}] {excerpt[:200]}",
                severity="deduction",
            ))
            i = k + 1
            continue
        i += 1
    return violations


def _check_length_rhythm_flat(text: str, position: int) -> List[Violation]:
    """R5: across the body, sentence count >=5 AND mean word count in
    [8,14] AND population stdev < 4. Soft signal only."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    counts = [len(s.split()) for s in sentences]
    if len(counts) < 5:
        return []
    mean = sum(counts) / len(counts)
    if mean < 8 or mean > 14:
        return []
    sd = statistics.pstdev(counts)
    if sd >= 4:
        return []
    return [Violation(
        pattern_type="length_rhythm_flat",
        email_position=position,
        field="content",
        excerpt=f"[n={len(counts)}, mean={mean:.1f}, stdev={sd:.2f}]",
        severity="deduction",
    )]


def _check_paragraph_grammar_uniformity(
    paragraphs: List[str], position: int,
) -> List[Violation]:
    """R6: 3+ paragraphs sharing the same first-sentence syntactic pattern.

    Skip greeting (first) and signature (last) paragraphs. Skip paragraphs
    starting with digit + period or open quote. Soft v1."""
    if len(paragraphs) <= 2:
        return []
    middle = paragraphs[1:-1]
    indexed: List[tuple] = []  # list of (orig_index, pattern, paragraph)
    for idx, para in enumerate(middle):
        if _is_skippable_paragraph_for_r6(para):
            continue
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
        first_sentence = sentences[0] if sentences else para
        pattern = _classify_first_sentence_pattern(first_sentence)
        indexed.append((idx, pattern, para))

    violations: List[Violation] = []
    if len(indexed) < 3:
        return violations
    i = 0
    while i < len(indexed):
        pattern = indexed[i][1]
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == pattern:
            j += 1
        run_len = j - i + 1
        if run_len >= 3:
            excerpt = " | ".join(p[:80] for _, _, p in indexed[i:j + 1])
            violations.append(Violation(
                pattern_type="paragraph_grammar_uniformity",
                email_position=position,
                field="content",
                excerpt=f"[{pattern} x{run_len}] {excerpt}",
                severity="deduction",
            ))
        i = j + 1
    return violations


def _check_comma_stack_parallel(
    text: str, position: int, field: str,
) -> List[Violation]:
    """R7: 3+ comma-joined parallel independent clauses without a
    coordinating conjunction within a single sentence. Hard fail.

    Distinct from `tricolon_list` which requires explicit "and"/"or".
    Heuristic: split sentence on commas, keep clauses that contain a verb
    match. If we find 3+ verb-bearing clauses AND no FANBOYS joiner appears
    in the sentence, flag it."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    violations: List[Violation] = []
    for s in sentences:
        if _FANBOYS_JOINER_RE.search(s):
            continue
        if "," not in s:
            continue
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) < 3:
            continue
        verb_clauses = [p for p in parts if _VERB_SHAPE_RE.search(p)]
        if len(verb_clauses) < 3:
            continue
        violations.append(Violation(
            pattern_type="comma_stack_parallel",
            email_position=position,
            field=field,
            excerpt=s[:200],
            severity="hard_fail",
        ))
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
            # Grammar / rhythm rules (R4, R5, R7 operate on flat text).
            violations.extend(_check_sentence_opener_repetition(text, position, field))
            violations.extend(_check_length_rhythm_flat(text, position))
            violations.extend(_check_comma_stack_parallel(text, position, field))
            # Paragraph-aware grammar rules (R0, R2, R3, R6) need the RAW
            # HTML because _strip_html collapses paragraph boundaries.
            paragraphs = _split_paragraphs(content or "")
            violations.extend(_check_greeting_too_terse(paragraphs, position))
            violations.extend(_check_fanboys_density(paragraphs, position))
            violations.extend(_check_paragraph_opener_monotony(paragraphs, position))
            violations.extend(_check_paragraph_grammar_uniformity(paragraphs, position))
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
