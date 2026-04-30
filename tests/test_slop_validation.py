"""Tests for the 8 grammar / rhythm validator rules (R0-R7).

Three cases per rule (positive, negative, edge) plus an integration test
on the verbatim Pat email body. R1 has an extra edge case to cover both
the FANBOYS-internal scenario and the verb-less fragment exemption.

Run from any cwd:
    pytest /Users/loganankenbrandt/cold/processes/email-personalizer-task/tests/ -v
"""

from utils.slop_validation import (
    Violation,
    _check_comma_stack_parallel,
    _check_fabricated_data_grounding,
    _check_fanboys_density,
    _check_forbidden_outcome_claims_body,
    _check_greeting_too_terse,
    _check_length_rhythm_flat,
    _check_paragraph_grammar_uniformity,
    _check_paragraph_opener_monotony,
    _check_sentence_opener_repetition,
    _check_staccato,
    _check_unsourced_platform_stats,
    validate_email,
)


# ============================================================
# R0: greeting_too_terse
# ============================================================

def test_r0_positive_bare_name():
    paragraphs = ["Pat,", "Body sentence one. Body sentence two."]
    violations = _check_greeting_too_terse(paragraphs, position=1)
    assert len(violations) == 1
    assert violations[0].pattern_type == "greeting_too_terse"
    assert violations[0].severity == "hard_fail"


def test_r0_negative_hi_greeting():
    paragraphs = ["Hi Pat,", "Body sentence one. Body sentence two."]
    violations = _check_greeting_too_terse(paragraphs, position=1)
    assert violations == []


def test_r0_edge_hey_greeting():
    paragraphs = ["Hey Pat,", "Body sentence one. Body sentence two."]
    violations = _check_greeting_too_terse(paragraphs, position=1)
    assert violations == []


# ============================================================
# R1: staccato_short_run_v2
# ============================================================

def test_r1_positive_three_short_in_a_row():
    # Three consecutive sentences of 8, 7, 11 words. All > 3 words so no
    # exemption applies; all <= 12 words so trigger A fires.
    text = (
        "We rebuilt the pipeline last week with the team. "
        "It runs faster than before in the morning. "
        "The throughput numbers were higher than every prior week."
    )
    # word counts: 11 / 8 / 10 -> all <=12, three in a row -> trigger A.
    violations = _check_staccato(text, position=1, field="content")
    assert any(v.pattern_type == "staccato_repetition" and v.severity == "hard_fail"
               for v in violations)


def test_r1_negative_varied_lengths():
    # Counts approx 14 / 22 / 9 / 18.  No 3-in-a-row <=12, no 2-in-a-row <=6.
    text = (
        "The construction clients you serve trust the firm to find executives every quarter without fail. "
        "Those clients are fielding project delays right now because the GCs cannot staff the field crews fast enough in this market today. "
        "The campaigns run quietly in the background for you. "
        "Your team stays focused on the retained search work that already produces predictable revenue every month for partners."
    )
    violations = _check_staccato(text, position=1, field="content")
    assert violations == []


def test_r1_edge_with_fanboys_internal():
    # 5/5/5 word sentences, all >3 (no fragment exemption), all <=12 ->
    # trigger A fires regardless of leading FANBOYS conjunctions. The
    # FANBOYS check only applies to R2 (paragraph density) and R7
    # (comma-stack parallel), NOT to R1 staccato length detection.
    text = (
        "And the team grew quickly. "
        "So we hired more people. "
        "But the results varied widely."
    )
    violations = _check_staccato(text, position=1, field="content")
    assert any(v.pattern_type == "staccato_repetition" and v.severity == "hard_fail"
               for v in violations)


def test_r1_edge_verbless_fragments():
    # "Healthcare. Construction. IT." -- each sentence is 1 word with no
    # verb regex match, so all three are exempted from the short-run
    # counter and no violation fires.
    text = "Healthcare. Construction. IT."
    violations = _check_staccato(text, position=1, field="content")
    assert violations == []


# ============================================================
# R2: fanboys_density_low
# ============================================================

def test_r2_positive_no_conjunctions():
    # 4 sentences. No ", and " (or any FANBOYS) joiner. No subordinator
    # keyword (because/while/if/when/since/although/though/whereas/unless/
    # until/after/before).
    para = (
        "The campaigns run in the background. "
        "Your team stays focused on retained search. "
        "The messaging is built around construction hiring. "
        "The system targets EPC firms across Houston."
    )
    violations = _check_fanboys_density([para], position=1)
    assert len(violations) == 1
    assert violations[0].pattern_type == "fanboys_density_low"
    assert violations[0].severity == "deduction"


def test_r2_negative_with_fanboys_joiner():
    # 3 sentences but one of them has a ", and " joiner -> validator skips.
    para = (
        "The campaigns run in the background, and the team stays focused. "
        "The messaging is built around construction hiring. "
        "The system targets EPC firms across Houston."
    )
    violations = _check_fanboys_density([para], position=1)
    assert violations == []


def test_r2_edge_with_subordinator_only():
    # 3 sentences, no FANBOYS joiner, but contains "because" / "while"
    # subordinators -> validator counts these as flow indicators and skips.
    para = (
        "The campaigns run in the background because the team stays focused. "
        "The messaging is built around construction hiring. "
        "The system targets EPC firms while supporting other work."
    )
    violations = _check_fanboys_density([para], position=1)
    assert violations == []


# ============================================================
# R3: paragraph_opener_monotony
# ============================================================

def test_r3_positive_four_det_noun_paragraphs():
    # Greeting + 4 middle paragraphs all opening "The X..." (DET_NOUN
    # bigram class) + signature. Run length 4 -> hard_fail per spec.
    #
    # Note: nouns are intentionally singular and do not end in -s/-ed/-ing.
    # The `_classify_pos` heuristic sends any lowercase word ending in -s
    # (len>3) to VERB, which would flip the bigram to DET+VERB -> OTHER and
    # break the run. Singular forms keep the bigram class as DET_NOUN.
    paragraphs = [
        "Hi Pat,",
        "The team grew steadily over the year.",
        "The plan worked exactly as designed.",
        "The deal closed faster than expected.",
        "The outcome held throughout the quarter.",
        "Logan",
    ]
    violations = _check_paragraph_opener_monotony(paragraphs, position=1)
    assert len(violations) == 1
    assert violations[0].pattern_type == "paragraph_opener_monotony"
    assert violations[0].severity == "hard_fail"


def test_r3_negative_mixed_openers():
    # Greeting + 4 middle paragraphs with different opener classes:
    #   "The team grew..."           DET_NOUN
    #   "We launched the campaign."  PRON_VERB
    #   "Healthcare grew fast."      PROPN_VERB
    #   "Quietly, things shifted."   OTHER (lower-case adverb -> OTHER class)
    # No 3-in-a-row of the same class -> no violation.
    paragraphs = [
        "Hi Pat,",
        "The team grew steadily over the year.",
        "We launched the campaign last quarter.",
        "Healthcare grew fast in that market.",
        "Quickly, things shifted across the region.",
        "Logan",
    ]
    violations = _check_paragraph_opener_monotony(paragraphs, position=1)
    assert violations == []


def test_r3_edge_other_class_run_skipped():
    # 5 paragraphs (greeting + 3 middle + sig). Middle paragraphs all open
    # "Pat, the X" -> first bigram is PROPN+DET -> OTHER class. R3 skips
    # OTHER runs entirely (`if cls == "OTHER": continue`). No violation.
    paragraphs = [
        "Hi Pat,",
        "Pat, the team grew over the year significantly.",
        "Pat, the plan worked as designed for them.",
        "Pat, the results came in faster than expected this time.",
        "Logan",
    ]
    violations = _check_paragraph_opener_monotony(paragraphs, position=1)
    assert violations == []


# ============================================================
# R4: sentence_opener_repetition
# ============================================================

def test_r4_positive_literal_repeat_the():
    # Three sentences all leading with "The". Identical lowercase first
    # token -> hard_fail.
    text = (
        "The campaigns run smoothly. "
        "The team stays focused. "
        "The messaging fits perfectly."
    )
    violations = _check_sentence_opener_repetition(text, position=1, field="content")
    assert len(violations) == 1
    assert violations[0].pattern_type == "sentence_opener_repetition"
    assert violations[0].severity == "hard_fail"
    assert "[token='the' x3]" in violations[0].excerpt


def test_r4_negative_varied_openers():
    # Three sentences with different opener tokens AND different POS
    # classes (subordinator OTHER, DET, subordinator OTHER) -> no run.
    text = (
        "If we look at this, things change quickly. "
        "The data shows growth across the board. "
        "Because of that, we expanded coverage in May."
    )
    violations = _check_sentence_opener_repetition(text, position=1, field="content")
    assert violations == []


def test_r4_edge_pos_class_run_soft():
    # Three sentences leading with three different lowercase NOUN-class
    # words ("rhythm", "momentum", "traction"). Literal tokens differ so
    # the hard literal-token branch does NOT fire; the POS class for each
    # is NOUN x3 so the soft (deduction) POS-class branch fires.
    #
    # Spec / validator notes:
    # - The original task brief described an "If/If/If" deliberate-
    #   parallelism case as soft. The validator (per its docstring) treats
    #   identical literal tokens as hard_fail unconditionally and only
    #   emits the soft signal when literal tokens differ but POS classes
    #   match. This test exercises the actual soft-signal path.
    # - `_check_sentence_opener_repetition` lowercases the first token
    #   before running `_classify_pos`, which kills the PROPN code path
    #   (PROPN requires a leading capital letter). Therefore the only
    #   POS classes that can drive the soft branch in practice are DET,
    #   PRON, VERB, and NOUN. We use NOUN here.
    text = (
        "Rhythm carried the campaign through the busy season. "
        "Momentum built across both inbound and outbound work. "
        "Traction held steady in every market we covered."
    )
    violations = _check_sentence_opener_repetition(text, position=1, field="content")
    assert len(violations) == 1
    assert violations[0].pattern_type == "sentence_opener_repetition"
    assert violations[0].severity == "deduction"
    assert "[POS=NOUN x3]" in violations[0].excerpt


# ============================================================
# R5: length_rhythm_flat
# ============================================================

def test_r5_positive_flat_band():
    # 5 sentences, word counts [9, 11, 10, 12, 11]:
    #   mean = 10.6 (within [8, 14])
    #   pstdev approx 1.02 (< 4)
    text = (
        "The team rebuilt the pipeline last week with help. "        # 11
        "Throughput numbers came in higher than every prior week. "  # 9
        "Clients responded faster than they did last quarter today. "  # 9
        "We tracked every signal from the staging environment carefully. "  # 9
        "Results matched the projection within a small margin of error."  # 11
    )
    violations = _check_length_rhythm_flat(text, position=1)
    assert len(violations) == 1
    assert violations[0].pattern_type == "length_rhythm_flat"
    assert violations[0].severity == "deduction"


def test_r5_negative_varied_rhythm():
    # 5 sentences with widely varied lengths [22, 6, 14, 9, 25]. stdev > 4.
    text = (
        "The construction clients you have served for many decades trust this firm to find their next vice president of engineering or project executive every cycle. "  # 28
        "That relationship is the asset. "  # 5
        "Those clients are fielding project delays today because their general contractors cannot staff field crews. "  # 15
        "The campaigns run quietly in the background. "  # 7
        "Your team stays focused on the retained search work that already produces predictable revenue every month for both new and existing partners across the region."  # 27
    )
    violations = _check_length_rhythm_flat(text, position=1)
    assert violations == []


def test_r5_edge_below_sentence_threshold():
    # Only 4 sentences -> below the n>=5 threshold.
    text = (
        "The team rebuilt the pipeline. "
        "Throughput numbers improved. "
        "Clients responded faster. "
        "Results matched the projection."
    )
    violations = _check_length_rhythm_flat(text, position=1)
    assert violations == []


# ============================================================
# R6: paragraph_grammar_uniformity
# ============================================================

def test_r6_positive_three_declarative_noun_lead():
    # 5 paragraphs: greeting + 3 middle DECLARATIVE_NOUN_LEAD + signature.
    # Each middle first sentence: starts with "The" (DET) and contains a
    # verb from the regex (is/are/runs/build/etc). Length > 6 tokens.
    paragraphs = [
        "Hi Pat,",
        "The team is focused on construction hiring this quarter without exception.",
        "The plan is built around outbound to EPC firms across the Houston region.",
        "The system runs in the background while your recruiters stay focused on retained search.",
        "Logan",
    ]
    violations = _check_paragraph_grammar_uniformity(paragraphs, position=1)
    assert len(violations) == 1
    assert violations[0].pattern_type == "paragraph_grammar_uniformity"
    assert violations[0].severity == "deduction"


def test_r6_negative_mixed_patterns():
    # 5 paragraphs with mixed first-sentence patterns:
    #   "The team is focused..."           DECLARATIVE_NOUN_LEAD
    #   "If we look at the numbers..."     SUBORDINATE_LEAD
    #   "What does the rhythm look like?"  QUESTION
    paragraphs = [
        "Hi Pat,",
        "The team is focused on construction hiring this quarter without exception.",
        "If we look at the numbers, the picture is clear.",
        "What does the rhythm look like across the region?",
        "Logan",
    ]
    violations = _check_paragraph_grammar_uniformity(paragraphs, position=1)
    assert violations == []


def test_r6_edge_below_paragraph_threshold():
    # 4 paragraphs total -> middle is paragraphs[1:-1] = 2 paragraphs.
    # R6 requires len(indexed) >= 3 to fire.
    paragraphs = [
        "Hi Pat,",
        "The team is focused on construction hiring this quarter overall.",
        "The plan is built around outbound to EPC firms across Houston.",
        "Logan",
    ]
    violations = _check_paragraph_grammar_uniformity(paragraphs, position=1)
    assert violations == []


# ============================================================
# R7: comma_stack_parallel
# ============================================================

def test_r7_positive_three_comma_joined_clauses():
    # Three comma-joined clauses each containing a verb (run/stays/fits).
    # No FANBOYS joiner present -> hard_fail.
    text = "The campaigns run, the team stays focused, the messaging fits."
    violations = _check_comma_stack_parallel(text, position=1, field="content")
    assert len(violations) == 1
    assert violations[0].pattern_type == "comma_stack_parallel"
    assert violations[0].severity == "hard_fail"


def test_r7_negative_with_fanboys_joiners():
    # Same shape but with explicit FANBOYS joiners ("and", "so") ->
    # _FANBOYS_JOINER_RE matches -> validator skips this sentence.
    text = "The campaigns run, and the team stays focused, so the messaging fits."
    violations = _check_comma_stack_parallel(text, position=1, field="content")
    assert violations == []


def test_r7_edge_tricolon_noun_list():
    # "Healthcare, construction, and IT." has a " and " FANBOYS joiner so
    # R7 skips it. (The tricolon_list regex pattern in SLOP_PATTERNS
    # handles the noun-list case independently; this test only verifies
    # R7's own output.)
    text = "Healthcare, construction, and IT."
    violations = _check_comma_stack_parallel(text, position=1, field="content")
    assert violations == []


# ============================================================
# Integration: full validate_email on the verbatim Pat email body
# ============================================================

PAT_EMAIL_BODY = (
    "Pat,\n\n"
    "The construction clients you've served for decades trust HPB to find their next VP of Engineering or project executive. "
    "That relationship is the asset. "
    "Those same clients are fielding project delays because their GCs can't staff field crews fast enough in the Houston market.\n\n"
    "What I build is an outbound system that targets hiring managers at EPC firms and GCs across Houston. "
    "The campaigns run in the background. "
    "Your team stays focused on retained search.\n\n"
    "The messaging is built around construction and industrial hiring. "
    "It reads like it came from someone who knows the difference between a superintendent and a project manager.\n\n"
    "One client doubled their inbound engagement from construction hiring managers within a single quarter. "
    "Same team, same headcount on the business development side.\n\n"
    "If you want to see what a campaign targeting Houston EPC firms looks like, I can send a sample.\n\n"
    "Logan\n"
    "Cold"
)


def test_integration_pat_email_full_validate():
    """Feed the verbatim Pat email body through validate_email and assert
    the rule mix called for in the team-lead's revised expectations:
      hard_fails:  R0 (greeting_too_terse), R1 (staccato_repetition)
      deductions:  R2 (fanboys_density_low),
                   R4 (sentence_opener_repetition POS-class),
                   R6 (paragraph_grammar_uniformity)
    """
    violations = validate_email(subject="", content=PAT_EMAIL_BODY, position=1)
    pattern_severity_pairs = {(v.pattern_type, v.severity) for v in violations}

    # Hard fails
    assert ("greeting_too_terse", "hard_fail") in pattern_severity_pairs, (
        "expected R0 hard_fail (bare-name greeting) on Pat email"
    )
    assert ("staccato_repetition", "hard_fail") in pattern_severity_pairs, (
        "expected R1 hard_fail (staccato run) on Pat email"
    )

    # Deductions / soft signals
    assert ("fanboys_density_low", "deduction") in pattern_severity_pairs, (
        "expected R2 deduction (no FANBOYS / no subordinator paragraph)"
    )
    assert ("sentence_opener_repetition", "deduction") in pattern_severity_pairs, (
        "expected R4 deduction (POS-class run) on Pat email"
    )
    assert ("paragraph_grammar_uniformity", "deduction") in pattern_severity_pairs, (
        "expected R6 deduction (uniform DECLARATIVE_NOUN_LEAD paragraphs)"
    )

    # Sanity: every violation conforms to the Violation NamedTuple shape.
    assert all(isinstance(v, Violation) for v in violations)
    assert all(v.severity in {"hard_fail", "deduction"} for v in violations)


# ============================================================
# R8: forbidden_outcome_claim_body
# ============================================================

def test_r8_positive_4xd_their_pipeline():
    # "4x'd their pipeline" is the canonical fabricated outcome claim
    # observed live (Pat step 1, Jonathan step 1).
    text = "One client 4x'd their pipeline in 90 days with the same team."
    violations = _check_forbidden_outcome_claims_body(
        text, position=1, field="content",
    )
    assert len(violations) == 1
    assert violations[0].pattern_type == "forbidden_outcome_claim_body"
    assert violations[0].severity == "hard_fail"


def test_r8_positive_doubled_their_pipeline():
    # "doubled their pipeline" is the verbatim multiplier-form claim.
    text = "One client doubled their pipeline within a single quarter."
    violations = _check_forbidden_outcome_claims_body(
        text, position=1, field="content",
    )
    assert len(violations) == 1
    assert violations[0].pattern_type == "forbidden_outcome_claim_body"
    assert violations[0].severity == "hard_fail"


def test_r8_positive_same_bd_team():
    # "same BD team" is one of the literal forbidden phrases per claims.md.
    text = "Same BD team, same headcount, no new hires required."
    violations = _check_forbidden_outcome_claims_body(
        text, position=1, field="content",
    )
    assert len(violations) == 1
    assert violations[0].pattern_type == "forbidden_outcome_claim_body"
    assert violations[0].severity == "hard_fail"


def test_r8_negative_approved_greenbox_phrasing():
    # The approved Greenbox phrasing ("conversion rate roughly doubled
    # over a 12-week window") describes a conversion rate, NOT pipeline /
    # sales / revenue / conversions, and there is no "4x'd their pipeline"
    # form. Should produce zero violations.
    text = (
        "Their conversion rate roughly doubled over a 12-week window, "
        "which is the kind of result we are aiming for here."
    )
    violations = _check_forbidden_outcome_claims_body(
        text, position=1, field="content",
    )
    assert violations == []


def test_r8_edge_4x_unrelated_to_pipeline():
    # "4x bigger team" / "4x the volume" are NOT claims about pipeline,
    # so the regex (which requires "their|your pipeline" after the
    # multiplier) must not fire.
    text = "Their team is 4x bigger than ours and ships 4x the volume."
    violations = _check_forbidden_outcome_claims_body(
        text, position=1, field="content",
    )
    assert violations == []


# ============================================================
# R9: fabricated_data_grounding
# ============================================================

def test_r9_positive_approved_proof_point_annotation():
    # Live MongoDB record: Pat step 1 fabricated this exact entry.
    data_grounding = [
        "50-seat OSHA Training Center at Kansas City headquarters",
        "75 years combined field construction experience among recruiters",
        "One client 4x'd their pipeline in 90 days (approved proof point)",
    ]
    violations = _check_fabricated_data_grounding(
        data_grounding, position=1,
    )
    # The third entry hits BOTH the forbidden-outcome-claim regex AND the
    # proof-point annotation, but the implementation deduplicates so we
    # see exactly one violation per entry.
    assert len(violations) == 1
    assert violations[0].pattern_type == "fabricated_data_grounding"
    assert violations[0].severity == "hard_fail"
    assert violations[0].field == "data_grounding"


def test_r9_positive_approved_proof_point_framing_variant():
    # The "(approved proof point framing)" variant has been observed in
    # the wild. The regex matches the umbrella "(approved/verified
    # proof/data point...)" pattern.
    data_grounding = [
        "30+ years in market (referral network ceiling threshold)",
        "Some abstract result (approved proof point framing)",
    ]
    violations = _check_fabricated_data_grounding(
        data_grounding, position=1,
    )
    assert len(violations) == 1
    assert violations[0].pattern_type == "fabricated_data_grounding"
    assert violations[0].severity == "hard_fail"


def test_r9_positive_verified_proof_point_variant():
    # "(verified proof point)" is the second umbrella variant. Same regex
    # branch.
    data_grounding = [
        "Headquartered in Houston (verified proof point)",
    ]
    violations = _check_fabricated_data_grounding(
        data_grounding, position=1,
    )
    assert len(violations) == 1
    assert violations[0].pattern_type == "fabricated_data_grounding"


def test_r9_negative_legitimate_external_facts():
    # Real externally-sourced facts about the prospect. None of them
    # match the forbidden outcome regexes; none contain a proof-point
    # annotation. Zero violations.
    data_grounding = [
        "50-seat OSHA Training Center at Kansas City headquarters",
        "75 years combined field construction experience among recruiters",
        "30+ years in market (referral network ceiling threshold)",
        "San Antonio market (Jonathan's location)",
    ]
    violations = _check_fabricated_data_grounding(
        data_grounding, position=1,
    )
    assert violations == []


def test_r9_edge_data_grounding_is_none():
    # Graceful degradation: legacy callers pass nothing. Must return [].
    violations = _check_fabricated_data_grounding(None, position=1)
    assert violations == []


def test_r9_edge_data_grounding_empty_list():
    # Empty list also gracefully returns [].
    violations = _check_fabricated_data_grounding([], position=1)
    assert violations == []


# ============================================================
# Integration: validate_email threads data_grounding through R9
# ============================================================

def test_validate_email_threads_data_grounding():
    # Body is clean. data_grounding contains a fabricated proof point.
    # The end-to-end call path through validate_email must surface the
    # R9 violation.
    clean_body = (
        "<p>Hi Pat,</p>"
        "<p>Saw the OSHA Training Center note on your site, that is a real "
        "differentiator for industrial accounts in this market. "
        "If you are open to it, I can send a sample campaign next week.</p>"
        "<p>Logan</p>"
        '<p><a href="https://withcold.com">Cold</a></p>'
    )
    data_grounding = [
        "OSHA Training Center at Kansas City HQ",
        "One client 4x'd their pipeline in 90 days (approved proof point)",
    ]
    violations = validate_email(
        subject="Quick note on your training center",
        content=clean_body,
        position=1,
        data_grounding=data_grounding,
    )
    pattern_severity_pairs = {(v.pattern_type, v.severity) for v in violations}
    assert ("fabricated_data_grounding", "hard_fail") in pattern_severity_pairs


def test_validate_email_backward_compat_no_data_grounding():
    # Legacy three-arg call site must still work. No data_grounding kwarg
    # means R9 simply produces zero violations of its own.
    clean_body = (
        "<p>Hi Pat,</p>"
        "<p>Saw the OSHA Training Center note on your site, that is a real "
        "differentiator for industrial accounts in this market. "
        "If you are open to it, I can send a sample campaign next week.</p>"
        "<p>Logan</p>"
        '<p><a href="https://withcold.com">Cold</a></p>'
    )
    violations = validate_email(
        subject="Quick note on your training center",
        content=clean_body,
        position=1,
    )
    pattern_types = {v.pattern_type for v in violations}
    assert "fabricated_data_grounding" not in pattern_types


# ============================================================
# R8 Round 5: word-order evasions (multiplier-after-noun, no possessive)
# ============================================================

def test_r8_positive_multiplier_after_noun():
    # Live evidence (Pat step 2 body): the writer reordered the multiplier
    # to follow the noun phrase to evade the original prefix-form regex.
    text = "One client saw their pipeline grow 4x in 90 days through managed outbound."
    violations = _check_forbidden_outcome_claims_body(
        text, position=1, field="content",
    )
    assert any(
        v.pattern_type == "forbidden_outcome_claim_body"
        and v.severity == "hard_fail"
        for v in violations
    )


def test_r8_positive_4xd_pipeline_no_possessive():
    # Live evidence: "4x'd pipeline in 90 days" without "their"/"your"
    # slipped past the original possessive-required regex.
    text = "One client 4x'd pipeline in 90 days."
    violations = _check_forbidden_outcome_claims_body(
        text, position=1, field="content",
    )
    assert any(
        v.pattern_type == "forbidden_outcome_claim_body"
        and v.severity == "hard_fail"
        for v in violations
    )


def test_r8_negative_approved_greenbox_conversion_phrasing():
    # The approved Greenbox citable phrasing describes a conversion rate,
    # not pipeline / sales / revenue / conversions doubling. No forbidden
    # outcome regex should match.
    text = "Their conversion rate roughly doubled over a 12-week window."
    violations = _check_forbidden_outcome_claims_body(
        text, position=1, field="content",
    )
    assert violations == []


# ============================================================
# R9 Round 5: colon-prefix proof-point annotations
# ============================================================

def test_r9_positive_colon_prefix_approved_proof_point():
    # Live evidence (Pat step 2 data_grounding): the writer switched to
    # a colon separator to evade the parens-only regex.
    data_grounding = [
        "Approved proof point: one client 4x'd pipeline in 90 days",
    ]
    violations = _check_fabricated_data_grounding(
        data_grounding, position=1,
    )
    assert len(violations) == 1
    assert violations[0].pattern_type == "fabricated_data_grounding"
    assert violations[0].severity == "hard_fail"


def test_r9_negative_unrelated_approved_phrase():
    # "Approved by the QA team for shipment" is a different domain entirely:
    # no "proof point" / "data point" / "infrastructure point" token after
    # "approved", so the prefix regex must not fire.
    data_grounding = [
        "Approved by the QA team for shipment",
    ]
    violations = _check_fabricated_data_grounding(
        data_grounding, position=1,
    )
    assert violations == []


# ============================================================
# R10: unsourced_platform_stat
# ============================================================

def test_r10_positive_percentage_and_volume_combined():
    # The writer asserts both a percentage stat and an emails-sent
    # volume in the same sentence with no Greenbox citation. Two
    # violations expected: one from the "<num>% delivery|conversion|..."
    # pattern, one from the "<num> million emails|sends|messages" pattern.
    text = "We hit 98.2% delivery on 1.7 million emails last quarter."
    violations = _check_unsourced_platform_stats(
        text, position=1, field="content",
    )
    assert len(violations) == 2
    assert all(
        v.pattern_type == "unsourced_platform_stat"
        and v.severity == "hard_fail"
        for v in violations
    )


def test_r10_positive_percentage_alone():
    # A single delivery-rate percentage with no Greenbox token nearby.
    text = "We hit 98.2% delivery rate across our infrastructure."
    violations = _check_unsourced_platform_stats(
        text, position=1, field="content",
    )
    assert len(violations) == 1
    assert violations[0].pattern_type == "unsourced_platform_stat"
    assert violations[0].severity == "hard_fail"


def test_r10_positive_keyword_before_percent_word_order():
    # Live evidence (Jonathan step 3): the writer evaded the original
    # percent-then-keyword regex by reordering to keyword-then-percent
    # ("Delivery runs at 98.2%"). The new regex with a 0-30 char glue
    # window between the keyword and the percentage catches it.
    text = "Delivery runs at 98.2% across the infrastructure."
    violations = _check_unsourced_platform_stats(
        text, position=1, field="content",
    )
    assert len(violations) == 1
    assert violations[0].pattern_type == "unsourced_platform_stat"
    assert violations[0].severity == "hard_fail"


def test_r10_negative_greenbox_gate_present():
    # The literal "Greenbox" token whitelists the entire text. Two stat
    # patterns appear (14.5% conversion and 29.5% conversion) but R10
    # short-circuits when Greenbox is present.
    text = "Greenbox saw a 14.5% to 29.5% conversion rate over 12 weeks."
    violations = _check_unsourced_platform_stats(
        text, position=1, field="content",
    )
    assert violations == []


def test_r10_negative_non_stat_number():
    # "75 years of combined construction experience" is a domain fact
    # about the prospect, not a percentage or volume stat. R10 must not
    # match because the regex requires a percent sign or a million/thousand
    # quantifier next to emails/sends/messages.
    text = "75 years of combined construction experience across the firm."
    violations = _check_unsourced_platform_stats(
        text, position=1, field="content",
    )
    assert violations == []


# ============================================================
# Integration: R8 word-order + R10 stats via validate_email
# ============================================================

PAT_STEP_2_BODY_FRAGMENT = (
    "<p>Hi Pat,</p>"
    "<p>The construction clients you serve are fielding project delays "
    "right now because GCs cannot staff field crews fast enough. "
    "One client saw their pipeline grow 4x in 90 days through managed "
    "outbound.</p>"
    "<p>If you want to see what a campaign looks like, I can send a sample.</p>"
    "<p>Logan</p>"
    '<p><a href="https://withcold.com">Cold</a></p>'
)


def test_integration_pat_step_2_word_order_evasion_caught():
    # End-to-end: validate_email on the verbatim Pat step 2 body must
    # surface the new multiplier-after-noun R8 hard_fail.
    violations = validate_email(
        subject="Houston EPC hiring",
        content=PAT_STEP_2_BODY_FRAGMENT,
        position=2,
    )
    pattern_severity_pairs = {(v.pattern_type, v.severity) for v in violations}
    assert ("forbidden_outcome_claim_body", "hard_fail") in pattern_severity_pairs


JONATHAN_STEP_3_BODY_FRAGMENT = (
    "<p>Hi Jonathan,</p>"
    "<p>The infrastructure side of the platform is where we put most of "
    "the engineering effort. "
    "Delivery runs at 98.2% across 1.7 million emails sent.</p>"
    "<p>Worth a sample if you want to see how this would look for your "
    "San Antonio campaigns.</p>"
    "<p>Logan</p>"
    '<p><a href="https://withcold.com">Cold</a></p>'
)


def test_integration_jonathan_step_3_unsourced_stats_caught():
    # End-to-end: validate_email on the verbatim Jonathan step 3 body
    # must surface TWO R10 hard_fails: one from the keyword-before-
    # percent pattern ("Delivery runs at 98.2%"), one from the volume
    # pattern ("1.7 million emails"). No Greenbox token whitelists
    # the body, so both fire.
    violations = validate_email(
        subject="San Antonio outbound",
        content=JONATHAN_STEP_3_BODY_FRAGMENT,
        position=3,
    )
    r10_violations = [
        v for v in violations
        if v.pattern_type == "unsourced_platform_stat"
        and v.severity == "hard_fail"
    ]
    assert len(r10_violations) == 2, (
        f"expected 2 R10 hard_fails (percentage + volume) on Jonathan "
        f"step 3, got {len(r10_violations)}: "
        f"{[v.excerpt for v in r10_violations]}"
    )
