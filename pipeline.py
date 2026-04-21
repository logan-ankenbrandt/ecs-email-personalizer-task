"""Per-recipient personalization orchestrator.

For each recipient × step:
  1. Build per-recipient context (recipient summary + pre-fetched company brief)
  2. Sonnet writer (with web_fetch + advisor tools, multi-turn)
  3. Opus judge (5-dimension rubric)
  4. Sonnet refine (max MAX_REFINE_LOOPS times if score < QUALITY_THRESHOLD)
  5. Programmatic slop validation
  6. Resolve merge fields ({{first_name}} -> recipient.first_name etc.)
  7. Upsert into personalized_sequence_emails

Recipient-level parallelism via ThreadPoolExecutor.
Per-recipient steps are serial (each step builds on the prior step's tone).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from bson import ObjectId

from config import (
    HARNESS_BUCKET,
    JUDGE_MODEL,
    JUDGE_MODEL_FINAL,
    KNOWLEDGE_PREFIX,
    MAX_REFINE_LOOPS,
    PROMPT_PREFIX,
    QUALITY_HARD_FLOOR,
    QUALITY_HARD_FLOOR_NO_RESEARCH,
    QUALITY_THRESHOLD,
    QUALITY_THRESHOLD_NO_RESEARCH,
    REFINE_MODEL,
    RESEARCH_MODEL,
    S3_BUCKET,
    WRITER_MODEL,
)
from utils.checkpoint import load_checkpoint, write_checkpoint
from utils.cost import CostAccumulator
from utils.judge import judge_email
from utils.merge_fields import build_merge_dict, resolve_merge_fields
from utils.mongo import (
    finalize_personalization_run,
    init_personalization_run,
    load_sequence,
    load_template_emails,
    query_recipients,
    update_personalization_progress,
    upsert_personalized_email,
)
from utils.refine import refine_email
from utils.research import build_company_brief, build_recipient_summary
from utils.s3 import load_knowledge
from utils.slop_validation import validate_email
from utils.writer import write_personalized_email

logger = logging.getLogger(__name__)


CHECKPOINT_EVERY_N = 50  # write checkpoint after this many recipients complete

# T1.1: the full ruleset lives in six S3 files that were never loaded.
# Concatenating them into the system prompt recovers ~10KB of quality rules
# the LLM was previously making decisions without.
_RULE_FILES: tuple = (
    "alignment.md",
    "formatting.md",
    "quality.md",
    "slop.md",
    "structure.md",
    "tone.md",
)

# T1.3: port of rewriter_config_v2.STEP_ROLES. Extended to cover email 1 so
# the personalizer handles initial opener too (the rewriter only targeted
# follow-ups 2-4). Gives the writer tactical framing per email position.
STEP_ROLES: Dict[int, str] = {
    1: (
        "Opener (Email 1 of N). Introduce why you're reaching out. Data-grounded "
        "hook, NOT a compliment. Tier 1 CTA: passive, curiosity-based. No "
        "meeting ask. Under 125 words preferred."
    ),
    2: (
        "Value Bridge (Email 2 of N). Explain how the system works. Show what "
        "it looks like in practice for a firm like theirs. Tier 1 CTA: passive, "
        "curiosity-based. No meeting ask."
    ),
    3: (
        "ROI Proof (Email 3 of N). Show the math. Make the investment feel "
        "obvious. Tier 2 CTA: explicit ask for a call."
    ),
    4: (
        "Breakup (Email 4 of N). Last touch. Gentle urgency through timing "
        "truth, not artificial scarcity. Tier 2 CTA: explicit ask."
    ),
}

# T1.4: port of rewriter_config_v2.VERTICAL_DETECT. Keyword matches against
# the template text tell us which vertical the sequence targets so we can
# inject vertical-specific vocabulary into the writer prompt.
VERTICAL_DETECT: Dict[str, List[str]] = {
    "healthcare": [
        "DONs", "clinical managers", "facility relationships",
        "credentialing", "SNF", "home health",
    ],
    "it_tech": [
        "VPs of Engineering", "CTOs", "IT directors",
        "contract technical talent", "developers", "technical hiring",
    ],
    "exec_search": [
        "CEOs, CHROs", "board chairs", "retained search",
        "search engagements", "executive search",
    ],
    "hr_peo": [
        "business owners and CFOs", "compliance gaps", "payroll pain",
        "PEO", "hired HR internally",
    ],
    "va_offshore": [
        "remote hiring", "timezones", "quality control",
        "US business owners", "VA agency",
    ],
    "light_industrial": [
        "operations managers", "warehouse directors", "plant managers",
        "distribution centers", "light industrial",
    ],
    "construction_trades": [
        "superintendents", "project managers", "EPC",
        "construction staffing", "skilled trades", "GC", "subcontractor",
    ],
}

# Preferred vocabulary per vertical. When a vertical is detected the writer
# is instructed to use these exact words and avoid the copywriter equivalents.
VERTICAL_VOCAB: Dict[str, List[str]] = {
    "light_industrial": [
        "placements (not sales)", "desk (not portfolio)",
        "hiring managers (not prospects)", "referral network (not inbound pipeline)",
        "clients (not accounts)", "ops managers", "plant HR",
    ],
    "healthcare": [
        "DONs (not Directors)", "credentialing (not onboarding)",
        "facility (not client)", "shifts (not jobs)",
    ],
    "it_tech": [
        "contract-to-hire (not temp)", "req (not opening)",
        "hiring managers (not prospects)", "bench (not pool)",
    ],
    "exec_search": [
        "retained search (not recruiting)", "engagement (not project)",
        "search (not hire)", "CEO/CHRO (not leadership)",
    ],
    "hr_peo": [
        "compliance gap (not risk)", "payroll (not processing)",
        "book of business (not portfolio)",
    ],
    "va_offshore": [
        "virtual assistant (not remote worker)",
        "timezone overlap (not coverage)", "managed VA (not outsourced)",
    ],
    "construction_trades": [
        "superintendent (not supervisor)", "craftsmen (not workers)",
        "project (not job)", "GC (not client)", "subcontractor (not vendor)",
    ],
}


def detect_vertical(template_text: str) -> Optional[str]:
    """Match the template body against VERTICAL_DETECT keyword lists.

    Returns the first vertical whose keywords appear in the template, or
    None if nothing matches. Case-insensitive.
    """
    if not template_text:
        return None
    lower = template_text.lower()
    for vertical, keywords in VERTICAL_DETECT.items():
        for kw in keywords:
            if kw.lower() in lower:
                return vertical
    return None


def _load_system_prompt() -> str:
    """Load the rewriter harness system prompt from S3.

    T1.1: Concatenates the base system.md with every rule file under
    prompt/rules/ so the LLM sees the full ruleset, not the abbreviated
    summary. The rule files aren't inlined at build time because the
    personalizer loads from S3 at runtime; fetch + join per cold start.
    """
    base = load_knowledge(HARNESS_BUCKET, f"{PROMPT_PREFIX}/system.md")
    rule_sections: List[str] = []
    for fn in _RULE_FILES:
        try:
            body = load_knowledge(HARNESS_BUCKET, f"{PROMPT_PREFIX}/rules/{fn}")
            if body and body.strip():
                rule_sections.append(f"# Rule: {fn[:-3]}\n\n{body.strip()}")
        except Exception as e:
            # Missing rule file is non-fatal — log and fall through with what
            # we have. Prevents a harness config drift from crashing runs.
            logger.warning("Skipped rule file %s: %s", fn, type(e).__name__)
    if not rule_sections:
        return base
    return base.rstrip() + "\n\n---\n\n" + "\n\n---\n\n".join(rule_sections)


# T1.2: the gold-standard Cash/FirstOption email annotated by the harness.
# Writer reads it once per cold start as a 0.92 calibration anchor. ~800
# tokens per writer call; cost is trivial against the quality uplift.
_quality_example_cache: Optional[str] = None


def _load_quality_example() -> str:
    """Fetch knowledge/quality/examples/high-quality.md once per pipeline.

    Cached at module level so all recipients share the same fetch. Empty
    string if the file is missing (downgraded silently — the rest of the
    prompt still works, just without the calibration anchor).
    """
    global _quality_example_cache
    if _quality_example_cache is not None:
        return _quality_example_cache
    try:
        body = load_knowledge(
            HARNESS_BUCKET,
            f"{KNOWLEDGE_PREFIX}/quality/examples/high-quality.md",
        )
        _quality_example_cache = body or ""
    except Exception as e:
        logger.warning("Could not load quality example: %s", type(e).__name__)
        _quality_example_cache = ""
    return _quality_example_cache


def _build_writer_user_prompt(
    template_subject: str,
    template_content: str,
    step: int,
    recipient_summary: str,
    company_brief: str,
    sequence_name: str,
    available_merge_keys: Optional[Set[str]] = None,
    enable_web_fetch: bool = True,
    previous_subject: Optional[str] = None,
    previous_content: Optional[str] = None,
    previous_score: Optional[float] = None,
    vertical: Optional[str] = None,
    quality_example: Optional[str] = None,
) -> str:
    """Build the per-recipient user prompt for the writer LLM.

    When previous_subject / previous_content are passed (rewrite mode), the
    writer sees the current live personalized version and is instructed to
    improve it rather than regenerate from scratch. This makes rewrites
    monotonically iterate instead of being independent draws.

    When vertical is passed (T1.4), a vocabulary-guidance section is injected
    so the writer uses industry-specific words (e.g., "desk" not "portfolio"
    for staffing). When quality_example is passed (T1.2), the gold-standard
    reference is prepended as a calibration anchor.
    """
    brief_block = (
        company_brief
        if company_brief
        else (
            "[no pre-fetched brief and no company website on file. DO NOT attempt "
            "web_fetch — guessing URLs wastes your turn budget. Write from the "
            "recipient context above using the recipient's real name/title/location.]"
        )
    )
    fetch_hint = (
        "You may use web_fetch to research the recipient's website (only if a real "
        "URL appears in the recipient context). "
        if enable_web_fetch
        else "web_fetch is disabled for this recipient (no known website). "
    )
    if available_merge_keys:
        keys_list = ", ".join(f"{{{{{k}}}}}" for k in sorted(available_merge_keys))
        merge_constraint = (
            f"\n\n## Merge fields available for this recipient\n\n"
            f"ONLY these merge fields are populated and safe to use as placeholders:\n"
            f"{keys_list}\n\n"
            f"Do NOT use any other merge field. If you need to reference the company, "
            f"location, or vertical and those keys are not in the list above, write "
            f"the literal value from your research (or a generic fallback if research "
            f"is empty). Never output {{{{company}}}}, {{{{trade_vertical}}}}, or any "
            f"field not listed above — unresolved placeholders produce broken subjects "
            f"like \"Before 's next bid cycle\"."
        )
    else:
        merge_constraint = ""

    # Rewrite mode: writer sees the existing personalized version and is
    # told to iterate on it. This turns rewrites from "regenerate from
    # template" into "improve version N to produce N+1" — the difference
    # between monotonic improvement and random resampling.
    previous_block = ""
    task_framing = (
        f"Rewrite this email FOR THIS RECIPIENT. Keep the strategic intent of the "
        f"template (this is email {step}; preserve the role it plays in the sequence) "
        f"but ground every claim in specifics about the recipient's company. "
    )
    if previous_subject is not None and previous_content is not None:
        score_note = (
            f" The prior version scored {previous_score:.2f}/1.00. Your goal is to "
            f"keep what earned that score and address the issues that kept it from "
            f"being higher."
            if previous_score is not None
            else ""
        )
        previous_block = (
            f"## Previous personalized version\n\n"
            f"### Subject\n{previous_subject}\n\n"
            f"### Body\n{previous_content}\n\n"
            f"---\n\n"
        )
        task_framing = (
            f"Rewrite the previous personalized version above (NOT the generic "
            f"template).{score_note} Preserve the parts that work; fix the parts "
            f"that don't. Keep grounding in specifics about the recipient's "
            f"company. "
        )

    # T1.3: strategic framing by email position.
    step_role = STEP_ROLES.get(step)
    step_role_block = (
        f"## Strategic role for this email\n\n{step_role}\n\n---\n\n"
        if step_role
        else ""
    )

    # T1.4: vertical vocabulary injection. Only adds weight when detection
    # actually matched — no vertical = no vocabulary constraint.
    vocab_block = ""
    if vertical and vertical in VERTICAL_VOCAB:
        vocab_lines = "\n".join(f"- {w}" for w in VERTICAL_VOCAB[vertical])
        vocab_block = (
            f"## Vertical vocabulary ({vertical})\n\n"
            f"Use the words on the LEFT, not the copywriter equivalents on the "
            f"right:\n\n{vocab_lines}\n\n---\n\n"
        )

    # T1.2: calibration anchor. When a gold-standard example loaded,
    # prepend it as a reference so the writer sees what 0.92 looks like.
    calibration_block = ""
    if quality_example:
        calibration_block = (
            f"## Reference: a 0.92-quality email for a similar recipient\n\n"
            f"{quality_example.strip()}\n\n"
            f"Use this as a CALIBRATION ANCHOR. Your output should match this "
            f"quality bar:\n"
            f"- Every sentence must fail the company swap test.\n"
            f"- Company research shapes the VALUE PROPOSITION, not just the opener.\n"
            f"- Name specific mechanisms at this recipient's scale, not generic "
            f"industry observations.\n"
            f"- Tiered CTAs by email position (per the strategic role above).\n\n"
            f"---\n\n"
        )

    return (
        f"You are personalizing email {step} of a sequence ({sequence_name!r}).\n\n"
        f"{step_role_block}"
        f"{vocab_block}"
        f"{calibration_block}"
        f"## Template (the baseline written for the segment)\n\n"
        f"### Subject\n{template_subject}\n\n"
        f"### Body\n{template_content}\n\n"
        f"---\n\n"
        f"{previous_block}"
        f"## Recipient context\n\n{recipient_summary}\n\n"
        f"## Company research\n\n"
        f"{brief_block}\n\n"
        f"---\n\n"
        f"## Task\n\n"
        f"{task_framing}"
        f"Reference real data points ('673 placements', 'San Antonio DMA', 'plant HR directors'). "
        f"Use the swap test: if your sentence still works for any other company in the same "
        f"segment, rewrite it.\n\n"
        f"{fetch_hint}Submit via submit_personalized_email when done. "
        f"Keep merge field placeholders like {{{{first_name}}}} UNRESOLVED."
        f"{merge_constraint}"
    )


class PersonalizerPipeline:
    """Per-recipient personalizer for one email_sequence.

    Two modes:
      - Full batch (default): resolve recipients via sequence.tags, iterate all
        template_emails for each recipient.
      - Targeted rewrite: when `targets` is passed, skip recipient query and
        only process the specific (recipient_id, step) pairs. When `feedback`
        is also passed, inject it into the writer prompt so the model knows
        what to fix.
    """

    def __init__(
        self,
        org_id: str,
        sequence_id: str,
        personalization_run_id: str,
        concurrency: int = 5,
        max_recipients: int = 0,
        resume: bool = False,
        targets: Optional[List[Dict[str, Any]]] = None,
        feedback: Optional[str] = None,
        rewrite_scope: Optional[str] = None,
        rewrite_recipient_id: Optional[str] = None,
    ):
        self.org_id = org_id
        self.sequence_id = sequence_id
        self.personalization_run_id = personalization_run_id
        self.concurrency = concurrency
        self.max_recipients = max_recipients
        self.resume = resume
        self.targets = targets
        self.feedback = feedback
        self.rewrite_scope = rewrite_scope
        self.rewrite_recipient_id = rewrite_recipient_id

        self.cost = CostAccumulator()
        self.system_prompt = _load_system_prompt()

        # Will be loaded in run()
        self.sequence_doc: Optional[Dict[str, Any]] = None
        self.template_emails: List[Dict[str, Any]] = []
        self.recipients: List[Dict[str, Any]] = []
        self.completed_ids: Set[str] = set()
        self.failed_ids: Set[str] = set()
        # In targeted mode, maps recipient_id -> set of steps to run.
        self._target_steps_by_rid: Dict[str, Set[int]] = {}

    @property
    def is_rewrite(self) -> bool:
        """True when this run is a targeted rewrite (explicit targets or
        bulk scope), False for full-batch personalization. Used to gate
        Mongo writes that would otherwise clobber parent-doc full-batch state.
        """
        return bool(self.targets) or self.rewrite_scope in ("recipient", "all")

    def run(self) -> Dict[str, Any]:
        """Execute the personalization run. Returns final metadata."""
        # 1. Load sequence + template emails
        self.sequence_doc = load_sequence(self.sequence_id)
        if not self.sequence_doc:
            raise ValueError(f"Sequence {self.sequence_id} not found")
        self.template_emails = load_template_emails(self.sequence_id)
        if not self.template_emails:
            raise ValueError(f"Sequence {self.sequence_id} has no sequence_emails docs")
        logger.info(
            "Loaded sequence %r with %d template emails",
            self.sequence_doc.get("name"), len(self.template_emails),
        )

        # 2a. Bulk-scope rewrite: resolve targets from existing personalized
        # docs at runtime (avoids shipping a 90KB targets JSON through ECS).
        # Runs BEFORE the explicit-targets branch so the rest of the pipeline
        # proceeds identically in both cases.
        if self.rewrite_scope in ("recipient", "all"):
            self.targets = self._resolve_targets_from_scope()
            if not self.targets:
                raise ValueError(
                    f"Scope '{self.rewrite_scope}' resolved zero targets for "
                    f"sequence {self.sequence_id}"
                )
            logger.info(
                "Bulk rewrite scope=%s: resolved %d targets (%d unique recipients)",
                self.rewrite_scope,
                len(self.targets),
                len({t["recipient_id"] for t in self.targets}),
            )

        # 2b. Resolve recipients. Targeted mode: fetch by ID list. Full mode:
        #    query by sequence.tags.
        if self.targets:
            target_rids = sorted({t["recipient_id"] for t in self.targets})
            self.recipients = self._fetch_recipients_by_ids(target_rids)
            if not self.recipients:
                raise ValueError(
                    f"Targeted rewrite: no recipients found for ids={target_rids}"
                )
            # Bug H1: some recipients may have been deleted between rewrite
            # creation and ECS task start. Log the missing ids and record them
            # as failed so the final metadata reflects reality instead of
            # silently dropping them.
            found_rids = {str(r["_id"]) for r in self.recipients}
            missing_rids = set(target_rids) - found_rids
            if missing_rids:
                logger.warning(
                    "Targeted rewrite: %d of %d requested recipients not found "
                    "(deleted?): %s",
                    len(missing_rids), len(target_rids), sorted(missing_rids)[:10],
                )
                self.failed_ids.update(missing_rids)
            # Per-recipient step filter (used inside _personalize_one_recipient).
            self._target_steps_by_rid = {}
            for t in self.targets:
                rid = t["recipient_id"]
                step = int(t["step"])
                self._target_steps_by_rid.setdefault(rid, set()).add(step)
            logger.info(
                "Targeted rewrite: %d recipients, %d (recipient,step) pairs, "
                "feedback=%s",
                len(self.recipients), len(self.targets),
                "yes" if self.feedback else "no",
            )
        else:
            tags = self.sequence_doc.get("tags", [])
            exclude_tags = self.sequence_doc.get("exclude_tags", [])
            operation = self.sequence_doc.get("tag_operation", "and")
            self.recipients = query_recipients(
                organization_id=self.org_id,
                tags=tags,
                exclude_tags=exclude_tags,
                operation=operation,
                max_recipients=self.max_recipients,
            )
            if not self.recipients:
                raise ValueError(f"No recipients match sequence {self.sequence_id} targeting")
            logger.info(
                "Matched %d recipients for tags=%s op=%s exclude=%s",
                len(self.recipients), tags, operation, exclude_tags,
            )

        # 3. Resume support: skip recipients already completed in a prior attempt
        if self.resume:
            self.completed_ids = load_checkpoint(S3_BUCKET, self.personalization_run_id)
            if self.completed_ids:
                self.recipients = [
                    r for r in self.recipients
                    if str(r["_id"]) not in self.completed_ids
                ]
                logger.info(
                    "Resume: %d already done, %d remaining",
                    len(self.completed_ids), len(self.recipients),
                )

        # 4. Initialize tracking on the parent doc
        # Bug C3: rewrite runs must NOT overwrite the parent sequence's
        # personalization_progress counters. The init helper splits behavior
        # based on is_rewrite: full-batch resets the top-level state, rewrites
        # only update the matching rewrite_runs[] entry.
        init_personalization_run(
            sequence_id=self.sequence_id,
            personalization_run_id=self.personalization_run_id,
            total=len(self.recipients) + len(self.completed_ids),
            is_rewrite=self.is_rewrite,
        )

        # 5. Process recipients in parallel
        self._process_recipients_parallel()

        # 6. Finalize
        metadata = self.cost.summary()
        metadata["recipients_total"] = len(self.recipients) + len(self.completed_ids)
        metadata["recipients_completed"] = len(self.completed_ids)
        metadata["recipients_failed"] = len(self.failed_ids)
        finalize_personalization_run(
            sequence_id=self.sequence_id,
            completed=len(self.completed_ids),
            failed=len(self.failed_ids),
            metadata=metadata,
            personalization_run_id=self.personalization_run_id,
            is_rewrite=self.is_rewrite,
        )
        write_checkpoint(
            bucket=S3_BUCKET,
            personalization_run_id=self.personalization_run_id,
            completed_recipient_ids=list(self.completed_ids),
            failed_recipient_ids=list(self.failed_ids),
        )
        logger.info(
            "Personalization run complete: completed=%d failed=%d total_cost=$%.2f",
            len(self.completed_ids), len(self.failed_ids), metadata["_total_usd"],
        )
        return metadata

    def _process_recipients_parallel(self) -> None:
        """Run per-recipient personalization across a thread pool."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._personalize_one_recipient, recipient): recipient
                for recipient in self.recipients
            }
            done_count = 0
            for future in as_completed(futures):
                recipient = futures[future]
                rid = str(recipient["_id"])
                try:
                    success = future.result()
                except Exception as e:
                    logger.error(
                        "Recipient %s personalization crashed: %s",
                        rid, e, exc_info=True,
                    )
                    success = False

                if success:
                    self.completed_ids.add(rid)
                else:
                    self.failed_ids.add(rid)

                done_count += 1
                if done_count % CHECKPOINT_EVERY_N == 0:
                    update_personalization_progress(
                        sequence_id=self.sequence_id,
                        completed=len(self.completed_ids),
                        failed=len(self.failed_ids),
                        is_rewrite=self.is_rewrite,
                        personalization_run_id=self.personalization_run_id,
                    )
                    write_checkpoint(
                        bucket=S3_BUCKET,
                        personalization_run_id=self.personalization_run_id,
                        completed_recipient_ids=list(self.completed_ids),
                        failed_recipient_ids=list(self.failed_ids),
                    )
                    logger.info(
                        "Checkpoint at %d/%d (cost so far: $%.2f)",
                        done_count, len(self.recipients), self.cost.usd_so_far_running(),
                    )

    def _personalize_one_recipient(self, recipient: Dict[str, Any]) -> bool:
        """Personalize all template steps for one recipient.

        Returns True if at least one step was successfully written.
        """
        rid = str(recipient["_id"])
        recipient_summary = build_recipient_summary(recipient)
        # Pre-fetch + summarize the company website ONCE, share across all steps
        company_brief = build_company_brief(recipient)
        if company_brief:
            self.cost.record("research", RESEARCH_MODEL, len(company_brief) // 4, 600)

        merge_dict = build_merge_dict(recipient)
        any_step_succeeded = False

        # Targeted-rewrite: restrict to the steps in this recipient's target set.
        target_steps = self._target_steps_by_rid.get(rid) if self.targets else None

        for template in self.template_emails:
            step = template.get("step", 1)
            if target_steps is not None and step not in target_steps:
                continue
            template_subject = template.get("subject", "")
            template_content = template.get("content", "")
            template_id = str(template.get("_id"))

            try:
                ok = self._personalize_one_step(
                    recipient=recipient,
                    rid=rid,
                    step=step,
                    template_subject=template_subject,
                    template_content=template_content,
                    template_id=template_id,
                    recipient_summary=recipient_summary,
                    company_brief=company_brief,
                    merge_dict=merge_dict,
                )
                # Fix 6: retry-once on transient writer failures (max_turns miss,
                # transient LLM error). Hard-floor rejections are also retried —
                # a re-run sometimes produces a better draft — but the cost is
                # bounded because retry happens at most once per step.
                if not ok:
                    logger.info(
                        "Retrying step %d for recipient %s (first attempt failed)",
                        step, rid,
                    )
                    ok = self._personalize_one_step(
                        recipient=recipient,
                        rid=rid,
                        step=step,
                        template_subject=template_subject,
                        template_content=template_content,
                        template_id=template_id,
                        recipient_summary=recipient_summary,
                        company_brief=company_brief,
                        merge_dict=merge_dict,
                    )
                if ok:
                    any_step_succeeded = True
            except Exception as e:
                logger.error(
                    "Step %d for recipient %s crashed: %s",
                    step, rid, e, exc_info=True,
                )
                continue

        return any_step_succeeded

    def _personalize_one_step(
        self,
        recipient: Dict[str, Any],
        rid: str,
        step: int,
        template_subject: str,
        template_content: str,
        template_id: str,
        recipient_summary: str,
        company_brief: str,
        merge_dict: Dict[str, str],
    ) -> bool:
        """Write + judge + refine + validate + resolve + upsert one step.

        Returns True if the personalized version was successfully written to Mongo.
        """
        sequence_name = self.sequence_doc.get("name", "") if self.sequence_doc else ""

        # Fix 3 (gate web_fetch): only enable when a real company website was
        # successfully pre-fetched. Without this, the writer burns turns guessing
        # URLs like pattonstaff.com -> pattonstaffing.com -> pattonstaff.co.
        has_website = bool(company_brief)

        # Fix 5 (tiered thresholds): when no company data is available, the
        # writer cannot ground personalization_depth claims. Refining just
        # rephrases the same generic text. Relax the threshold in that case.
        effective_threshold = QUALITY_THRESHOLD if has_website else QUALITY_THRESHOLD_NO_RESEARCH
        effective_floor = QUALITY_HARD_FLOOR if has_website else QUALITY_HARD_FLOOR_NO_RESEARCH

        # Fix 2 (merge-field gaps): tell the writer which merge keys are
        # actually populated for this recipient so it doesn't emit {{company}}
        # when business_name is blank.
        available_merge_keys = set(merge_dict.keys())

        # Bug A2: in rewrite mode, fetch the existing personalized doc BEFORE
        # the writer runs so we can (a) show it to the writer and (b) regression-
        # guard the upsert (A1). Snapshot capture for rewrite_history[] happens
        # later but reuses this same record.
        previous_version = None
        if self.is_rewrite:
            previous_version = self._snapshot_existing_for_history(rid, step)

        prev_subject = previous_version.get("subject") if previous_version else None
        prev_content = previous_version.get("content") if previous_version else None
        prev_score = previous_version.get("quality_score") if previous_version else None

        # T1.4: detect vertical from the template text so the prompt can
        # inject matching vocabulary. Falls back to None if no keyword hits.
        vertical = detect_vertical(template_content or "")
        # T1.2: load the gold-standard calibration example (cached).
        quality_example = _load_quality_example()

        user_prompt = _build_writer_user_prompt(
            template_subject=template_subject,
            template_content=template_content,
            step=step,
            recipient_summary=recipient_summary,
            company_brief=company_brief,
            sequence_name=sequence_name,
            available_merge_keys=available_merge_keys,
            enable_web_fetch=has_website,
            previous_subject=prev_subject,
            previous_content=prev_content,
            previous_score=prev_score if isinstance(prev_score, (int, float)) else None,
            vertical=vertical,
            quality_example=quality_example,
        )

        # Targeted-rewrite: inject user feedback into the writer prompt so the
        # model knows what to change vs the prior draft.
        if self.feedback:
            user_prompt += (
                f"\n\n---\n\n## User feedback on the previous version\n\n"
                f"{self.feedback}\n\n"
                f"The prior draft was sent back with this feedback. Address it "
                f"directly in your rewrite. This feedback is a hard constraint "
                f"and takes precedence over maintaining the prior score."
            )

        # Bug A3: in rewrite mode, lower writer temperature so the rewrite
        # iterates carefully instead of randomly resampling. Leave None for
        # full-batch runs where higher variance helps quality.
        writer_temperature = 0.3 if self.is_rewrite else None

        # Phase 1: Writer
        writer_result, writer_tokens = write_personalized_email(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            enable_web_fetch=has_website,
            temperature=writer_temperature,
        )
        self.cost.record("writer", WRITER_MODEL, writer_tokens["input_tokens"], writer_tokens["output_tokens"])
        if not writer_result:
            logger.warning(
                "step=%d recipient=%s reason=writer_timeout has_website=%s",
                step, rid, has_website,
            )
            return False

        subject = writer_result.get("subject", "")
        content = writer_result.get("content", "")
        company_insight = writer_result.get("company_insight", "")
        data_grounding = writer_result.get("data_grounding", [])
        advisor_used = writer_result.get("advisor_used", False)

        # Phase 2/3: Judge + Refine loop with best-so-far tracking (Bug A4).
        # The old loop kept the LAST iteration's output even if it regressed
        # from an earlier iteration. Now we snapshot whichever iteration
        # scored highest and use that as the final answer.
        # Fix 4 (judge model routing): use the cheap Sonnet judge for
        # intermediate iterations; only use the Opus judge on the final call.
        final_judgment = None
        best_subject = subject
        best_content = content
        best_score = 0.0
        best_judgment: Optional[Dict[str, Any]] = None
        iters_done = 0
        for refine_iteration in range(MAX_REFINE_LOOPS + 1):
            iters_done = refine_iteration + 1
            is_final_iter = refine_iteration >= MAX_REFINE_LOOPS
            judge_model = JUDGE_MODEL_FINAL if is_final_iter else JUDGE_MODEL
            # T3.1: judge verifies writer's self-reported grounding against
            # the copy. Lets the judge catch ungrounded claims the writer
            # hallucinated.
            judgment, judge_tokens = judge_email(
                subject=subject,
                content=content,
                company_brief=company_brief,
                recipient_summary=recipient_summary,
                model=judge_model,
                data_grounding=data_grounding or None,
                company_insight=company_insight or None,
            )
            self.cost.record("judge", judge_model, judge_tokens["input_tokens"], judge_tokens["output_tokens"])
            final_judgment = judgment
            score = judgment.get("overall_score", 0.0)
            should_refine = judgment.get("should_refine", False)

            # T2.1: programmatic slop gate. Run validator on the current
            # subject/content BEFORE accepting the judge's verdict. Merge any
            # violations into the issues list and force should_refine=True
            # regardless of score. Catches em-dashes, banned phrases, and
            # structural bounds that the LLM judge misses.
            slop_violations = validate_email(subject=subject, content=content, position=step)
            if slop_violations:
                logger.info(
                    "Recipient=%s step=%d: slop_gate iter=%d violations=%d types=%s",
                    rid, step, refine_iteration, len(slop_violations),
                    sorted({v.pattern_type for v in slop_violations})[:5],
                )
                slop_issues = [v.to_judge_issue() for v in slop_violations]
                judgment_issues = list(judgment.get("issues", []) or [])
                judgment["issues"] = judgment_issues + slop_issues
                should_refine = True
                # Track best score BEFORE we consider the slop-flagged version.
                # A slop-contaminated draft should not be selected as best even
                # if its LLM score is high; penalize it out of the running.
                score_for_best = max(0.0, score - 0.1 * len(slop_violations))
            else:
                score_for_best = score

            # A4: track best-so-far across iterations (slop-penalized).
            if score_for_best > best_score:
                best_subject = subject
                best_content = content
                best_score = score_for_best
                best_judgment = judgment

            # Accept only if the score clears the threshold (or judge says stop)
            # AND slop is clean. slop violations always force another refine.
            if (score >= effective_threshold or not should_refine) and not slop_violations:
                logger.info(
                    "Recipient=%s step=%d: score=%.2f passes (iter=%d, threshold=%.2f, judge=%s)",
                    rid, step, score, refine_iteration, effective_threshold, judge_model,
                )
                break

            # Fix 5 (skip futile refine): if the only failing dimension is
            # personalization_depth and we have no data, refining cannot help.
            if not has_website:
                dim_scores = judgment.get("dimension_scores", {}) or {}
                failing_dims = [k for k, v in dim_scores.items() if isinstance(v, (int, float)) and v < 0.6]
                if failing_dims == ["personalization_depth"]:
                    logger.info(
                        "Recipient=%s step=%d: skipping refine - no data for personalization_depth",
                        rid, step,
                    )
                    break

            if is_final_iter:
                logger.info(
                    "Recipient=%s step=%d: score=%.2f after %d refines, accepting (threshold=%.2f)",
                    rid, step, score, refine_iteration, effective_threshold,
                )
                break

            # A5: thread user_feedback into refine so a "don't mention
            # company" constraint isn't silently undone by refine chasing the
            # judge's tone critique.
            # T3.2: also pass recipient + step + vertical so the refiner can
            # re-ground claims and enforce CTA tier rules per email position.
            refined, refine_tokens = refine_email(
                current_subject=subject,
                current_content=content,
                issues=judgment.get("issues", []),
                company_brief=company_brief,
                user_feedback=self.feedback,
                recipient_summary=recipient_summary,
                step=step,
                step_role=STEP_ROLES.get(step),
                vertical=vertical,
            )
            self.cost.record("refine", REFINE_MODEL, refine_tokens["input_tokens"], refine_tokens["output_tokens"])
            if refined:
                subject = refined.get("subject", subject)
                content = refined.get("content", content)
            else:
                logger.warning(
                    "Refine failed for recipient=%s step=%d, accepting current draft",
                    rid, step,
                )
                break

        # A4: use the best-seen iteration, not whatever the last one produced.
        final_judgment = best_judgment or final_judgment
        subject = best_subject
        content = best_content
        final_score = best_score if best_judgment else (
            final_judgment.get("overall_score", 0.0) if final_judgment else 0.0
        )

        # Hard floor: if final score is unsalvageable, skip this recipient/step.
        # message_scheduler will fall back to the template version.
        if final_score < effective_floor:
            logger.warning(
                "step=%d recipient=%s decision=below_hard_floor score=%.2f "
                "floor=%.2f has_website=%s iters=%d had_feedback=%s",
                step, rid, final_score, effective_floor, has_website,
                iters_done, bool(self.feedback),
            )
            return False

        # A1: regression guard. When a rewrite would replace an existing
        # doc with a LOWER-scoring version, keep the old one unless the user
        # provided explicit feedback (which means they asked for a change
        # that might trade score for the constraint they want honored).
        if (
            self.is_rewrite
            and previous_version is not None
            and not self.feedback
            and isinstance(prev_score, (int, float))
            and (prev_score - final_score) > 0.05
        ):
            logger.warning(
                "step=%d recipient=%s decision=rejected_regression "
                "old_score=%.2f new_score=%.2f delta=%.2f iters=%d",
                step, rid, prev_score, final_score, prev_score - final_score,
                iters_done,
            )
            # Return True because we "succeeded" in the sense that the user's
            # existing copy is preserved. No upsert needed.
            return True

        # Phase 4: Programmatic slop validation
        slop_violations = validate_email(subject=subject, content=content, position=step)
        slop_warnings = [v.to_dict() for v in slop_violations]

        # Phase 5: Resolve merge fields (so message_scheduler can skip content_updater)
        resolved_subject = resolve_merge_fields(subject, merge_dict)
        resolved_content = resolve_merge_fields(content, merge_dict)

        # Targeted-rewrite: snapshot the existing doc for rewrite_history
        # before the upsert replaces it. Runs only in targeted mode so the
        # full-batch path stays unchanged.
        # previous_version was already captured earlier in this step (A2).
        # Reuse it here for rewrite_history[] push instead of re-fetching.

        # A6: structured decision log so post-run audits can filter by outcome.
        logger.info(
            "step=%d recipient=%s decision=accepted score=%.2f "
            "old_score=%s iters=%d had_feedback=%s has_website=%s "
            "best_of_n_iterations=True",
            step, rid, final_score,
            f"{prev_score:.2f}" if isinstance(prev_score, (int, float)) else "none",
            iters_done, bool(self.feedback), has_website,
        )

        # Phase 6: Upsert
        ok = upsert_personalized_email(
            email_sequence_id=self.sequence_id,
            recipient_id=rid,
            step=step,
            subject=resolved_subject,
            content=resolved_content,
            personalization_run_id=self.personalization_run_id,
            quality_score=final_score,
            company_insight=company_insight,
            data_grounding=data_grounding,
            slop_warnings=slop_warnings,
            advisor_used=advisor_used,
            original_template_id=template_id,
            dimension_scores=final_judgment.get("dimension_scores") if final_judgment else None,
            previous_version=previous_version,
            last_rewrite_feedback=self.feedback,
        )
        return ok

    # ────────────────────────────────────────────────────────────
    # Targeted-rewrite helpers
    # ────────────────────────────────────────────────────────────

    def _resolve_targets_from_scope(self) -> List[Dict[str, Any]]:
        """Build the target list for a bulk-scope rewrite by querying the
        existing personalized_sequence_emails docs for this sequence.

        - scope='recipient': filters by recipient_id
        - scope='all': no recipient filter

        Projects only (recipient_id, step) fields. Returns a plain list of
        dicts compatible with the rest of the targeted-rewrite pipeline.
        """
        from utils.mongo import _get_primary_db
        db = _get_primary_db()
        query: Dict[str, Any] = {"email_sequence_id": str(self.sequence_id)}
        if self.rewrite_scope == "recipient":
            # Bug C1: guard against missing/empty recipient_id silently
            # expanding to an all-sequence rewrite.
            if not self.rewrite_recipient_id:
                raise ValueError(
                    "rewrite_scope='recipient' requires a non-empty "
                    f"rewrite_recipient_id but got {self.rewrite_recipient_id!r}"
                )
            query["recipient_id"] = str(self.rewrite_recipient_id)
        cursor = db.personalized_sequence_emails.find(
            query, {"recipient_id": 1, "step": 1, "_id": 0},
        )
        return [
            {"recipient_id": doc["recipient_id"], "step": int(doc["step"])}
            for doc in cursor
        ]

    def _fetch_recipients_by_ids(
        self, recipient_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch specific recipient docs from the READ cluster by _id list.

        Mirrors the cluster choice in query_recipients (recipients live on the
        READ cluster, not PRIMARY, for this deployment).
        """
        from utils.mongo import _get_read_db
        oids: List[ObjectId] = []
        for rid in recipient_ids:
            if not rid or len(rid) != 24:
                continue
            try:
                oids.append(ObjectId(rid))
            except Exception:
                continue
        if not oids:
            return []
        db = _get_read_db()
        return list(db.recipients.find({"_id": {"$in": oids}}))

    def _snapshot_existing_for_history(
        self, rid: str, step: int,
    ) -> Optional[Dict[str, Any]]:
        """Snapshot the current personalized_sequence_emails doc so we can
        push it into rewrite_history[] before the new upsert replaces it.

        Returns None if no existing doc (first write for this pair).
        """
        from utils.mongo import _get_primary_db
        db = _get_primary_db()
        existing = db.personalized_sequence_emails.find_one({
            "email_sequence_id": str(self.sequence_id),
            "recipient_id": str(rid),
            "step": step,
        })
        if not existing:
            return None
        return {
            "personalization_run_id": existing.get("personalization_run_id"),
            "quality_score": existing.get("quality_score"),
            "subject": existing.get("subject"),
            "content": existing.get("content"),
            "created_at": existing.get("created_at"),
            "feedback": self.feedback,
        }
