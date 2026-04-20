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


def _load_system_prompt() -> str:
    """Load the rewriter harness system prompt from S3."""
    return load_knowledge(HARNESS_BUCKET, f"{PROMPT_PREFIX}/system.md")


def _build_writer_user_prompt(
    template_subject: str,
    template_content: str,
    step: int,
    recipient_summary: str,
    company_brief: str,
    sequence_name: str,
    available_merge_keys: Optional[Set[str]] = None,
    enable_web_fetch: bool = True,
) -> str:
    """Build the per-recipient user prompt for the writer LLM."""
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
    return (
        f"You are personalizing email {step} of a sequence ({sequence_name!r}).\n\n"
        f"## Template (the baseline written for the segment)\n\n"
        f"### Subject\n{template_subject}\n\n"
        f"### Body\n{template_content}\n\n"
        f"---\n\n"
        f"## Recipient context\n\n{recipient_summary}\n\n"
        f"## Company research\n\n"
        f"{brief_block}\n\n"
        f"---\n\n"
        f"## Task\n\n"
        f"Rewrite this email FOR THIS RECIPIENT. Keep the strategic intent of the "
        f"template (this is email {step}; preserve the role it plays in the sequence) "
        f"but ground every claim in specifics about the recipient's company. "
        f"Reference real data points ('673 placements', 'San Antonio DMA', 'plant HR directors'). "
        f"Use the swap test: if your sentence still works for any other company in the same "
        f"segment, rewrite it.\n\n"
        f"{fetch_hint}Submit via submit_personalized_email when done. "
        f"Keep merge field placeholders like {{{{first_name}}}} UNRESOLVED."
        f"{merge_constraint}"
    )


class PersonalizerPipeline:
    """Per-recipient personalizer for one email_sequence."""

    def __init__(
        self,
        org_id: str,
        sequence_id: str,
        personalization_run_id: str,
        concurrency: int = 5,
        max_recipients: int = 0,
        resume: bool = False,
    ):
        self.org_id = org_id
        self.sequence_id = sequence_id
        self.personalization_run_id = personalization_run_id
        self.concurrency = concurrency
        self.max_recipients = max_recipients
        self.resume = resume

        self.cost = CostAccumulator()
        self.system_prompt = _load_system_prompt()

        # Will be loaded in run()
        self.sequence_doc: Optional[Dict[str, Any]] = None
        self.template_emails: List[Dict[str, Any]] = []
        self.recipients: List[Dict[str, Any]] = []
        self.completed_ids: Set[str] = set()
        self.failed_ids: Set[str] = set()

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

        # 2. Query recipients matching the sequence's targeting
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
        init_personalization_run(
            sequence_id=self.sequence_id,
            personalization_run_id=self.personalization_run_id,
            total=len(self.recipients) + len(self.completed_ids),
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

        for template in self.template_emails:
            step = template.get("step", 1)
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

        user_prompt = _build_writer_user_prompt(
            template_subject=template_subject,
            template_content=template_content,
            step=step,
            recipient_summary=recipient_summary,
            company_brief=company_brief,
            sequence_name=sequence_name,
            available_merge_keys=available_merge_keys,
            enable_web_fetch=has_website,
        )

        # Phase 1: Writer
        writer_result, writer_tokens = write_personalized_email(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            enable_web_fetch=has_website,
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

        # Phase 2: Judge + (Phase 3) Refine loop
        # Fix 4 (judge model routing): use the cheap Sonnet judge for
        # intermediate iterations; only use the Opus judge on the final call
        # (after max refines, or as the accept/reject decision).
        final_judgment = None
        for refine_iteration in range(MAX_REFINE_LOOPS + 1):
            is_final_iter = refine_iteration >= MAX_REFINE_LOOPS
            judge_model = JUDGE_MODEL_FINAL if is_final_iter else JUDGE_MODEL
            judgment, judge_tokens = judge_email(
                subject=subject,
                content=content,
                company_brief=company_brief,
                recipient_summary=recipient_summary,
                model=judge_model,
            )
            self.cost.record("judge", judge_model, judge_tokens["input_tokens"], judge_tokens["output_tokens"])
            final_judgment = judgment
            score = judgment.get("overall_score", 0.0)
            should_refine = judgment.get("should_refine", False)

            if score >= effective_threshold or not should_refine:
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
                        "Recipient=%s step=%d: skipping refine — no data for personalization_depth",
                        rid, step,
                    )
                    break

            if is_final_iter:
                logger.info(
                    "Recipient=%s step=%d: score=%.2f after %d refines, accepting (threshold=%.2f)",
                    rid, step, score, refine_iteration, effective_threshold,
                )
                break

            # Refine
            refined, refine_tokens = refine_email(
                current_subject=subject,
                current_content=content,
                issues=judgment.get("issues", []),
                company_brief=company_brief,
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

        # Hard floor: if final score is unsalvageable, skip this recipient/step.
        # message_scheduler will fall back to the template version.
        final_score = final_judgment.get("overall_score", 0.0) if final_judgment else 0.0
        if final_score < effective_floor:
            logger.warning(
                "step=%d recipient=%s reason=below_hard_floor score=%.2f floor=%.2f has_website=%s",
                step, rid, final_score, effective_floor, has_website,
            )
            return False

        # Phase 4: Programmatic slop validation
        slop_violations = validate_email(subject=subject, content=content, position=step)
        slop_warnings = [v.to_dict() for v in slop_violations]

        # Phase 5: Resolve merge fields (so message_scheduler can skip content_updater)
        resolved_subject = resolve_merge_fields(subject, merge_dict)
        resolved_content = resolve_merge_fields(content, merge_dict)

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
        )
        return ok
