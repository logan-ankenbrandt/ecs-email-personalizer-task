#!/usr/bin/env python3
"""Email Personalizer entrypoint.

Per-recipient rewrite of a template sequence using the proven write-judge-refine
pattern from agentic-email-rewriter-v2.py. Productionized for ECS Fargate.

Usage:
    python main.py --org_id <id> --sequence_id <id> --personalization_run_id <id>
    python main.py ... --concurrency 5 --max_recipients 50 --resume
"""

import argparse
import json
import logging
import os
import sys

from config import (
    CONCURRENCY_DEFAULT,
    get_api_key,
)
from pipeline import PersonalizerPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-recipient personalization for sequence-architect templates",
    )
    parser.add_argument("--org_id", required=True)
    parser.add_argument(
        "--sequence_id", required=True,
        help="The email_sequences _id (PRIMARY cluster) to personalize",
    )
    parser.add_argument(
        "--personalization_run_id", required=True,
        help="Idempotency key + S3 checkpoint key + audit reference",
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY_DEFAULT,
        help=f"Recipients in flight at once (default {CONCURRENCY_DEFAULT})",
    )
    parser.add_argument(
        "--max_recipients", type=int, default=0,
        help="Cap recipient count (0 = no cap; for budget-controlled testing)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip recipients already in the S3 checkpoint",
    )
    parser.add_argument(
        "--targets", type=str, default=None,
        help='Targeted rewrite mode. JSON list of {"recipient_id": "...", '
             '"step": N} objects. When present, the full-batch query is '
             'skipped and only these (recipient, step) pairs are processed.',
    )
    parser.add_argument(
        "--feedback", type=str, default=None,
        help="Optional user feedback string injected into the writer prompt "
             "when running a targeted rewrite. Tells the model what to fix.",
    )
    parser.add_argument(
        "--rewrite_scope", type=str, default=None,
        choices=["custom", "recipient", "all"],
        help="Bulk rewrite scope. When 'recipient' or 'all', the pipeline "
             "queries personalized_sequence_emails at runtime to build the "
             "target list, avoiding the ECS command-length limit for large "
             "target sets. 'custom' (or absent) uses --targets JSON as before.",
    )
    parser.add_argument(
        "--rewrite_recipient_id", type=str, default=None,
        help="Recipient ID for --rewrite_scope=recipient. Ignored otherwise.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    parsed_targets = None
    if args.targets:
        try:
            parsed_targets = json.loads(args.targets)
        except json.JSONDecodeError as e:
            print(f"Invalid --targets JSON: {e}", file=sys.stderr)
            sys.exit(2)
        # Shape validation (Bug H2). Each entry must be {recipient_id: str, step: int}.
        if not isinstance(parsed_targets, list):
            print("--targets must be a JSON array", file=sys.stderr)
            sys.exit(2)
        for i, item in enumerate(parsed_targets):
            if not isinstance(item, dict):
                print(f"--targets[{i}]: must be an object", file=sys.stderr)
                sys.exit(2)
            if "recipient_id" not in item or not isinstance(item["recipient_id"], str):
                print(f"--targets[{i}]: missing or non-string 'recipient_id'", file=sys.stderr)
                sys.exit(2)
            if "step" not in item:
                print(f"--targets[{i}]: missing 'step'", file=sys.stderr)
                sys.exit(2)
            try:
                int(item["step"])
            except (TypeError, ValueError):
                print(
                    f"--targets[{i}]: 'step' must be int-coercible, got {item['step']!r}",
                    file=sys.stderr,
                )
                sys.exit(2)

    # Bug C1: scope=recipient requires a recipient id; otherwise the pipeline
    # silently rewrites every personalized doc for the sequence.
    if args.rewrite_scope == "recipient" and not args.rewrite_recipient_id:
        print(
            "Error: --rewrite_scope=recipient requires --rewrite_recipient_id",
            file=sys.stderr,
        )
        sys.exit(2)

    # Bulk scopes ignore --targets (the pipeline resolves them from Mongo).
    if args.rewrite_scope in ("recipient", "all") and parsed_targets:
        print(
            "Warning: --targets ignored when --rewrite_scope is 'recipient' or 'all'",
            file=sys.stderr,
        )
        parsed_targets = None

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate API key early
    try:
        api_key = get_api_key()
        os.environ["ANTHROPIC_API_KEY"] = api_key
    except ValueError as e:
        logger.error("API key error: %s", e)
        sys.exit(1)

    logger.info(
        "Starting email-personalizer: org=%s sequence=%s run=%s concurrency=%d max=%s",
        args.org_id, args.sequence_id, args.personalization_run_id,
        args.concurrency, args.max_recipients or "unbounded",
    )

    pipeline = PersonalizerPipeline(
        org_id=args.org_id,
        sequence_id=args.sequence_id,
        personalization_run_id=args.personalization_run_id,
        concurrency=args.concurrency,
        max_recipients=args.max_recipients,
        resume=args.resume,
        targets=parsed_targets,
        feedback=args.feedback,
        rewrite_scope=args.rewrite_scope,
        rewrite_recipient_id=args.rewrite_recipient_id,
    )

    try:
        metadata = pipeline.run()
    except Exception as e:
        logger.error("Personalization run failed: %s", e, exc_info=True)
        # Try to mark the run as failed in Mongo so the UI sees status="failed"
        try:
            from utils.mongo import finalize_personalization_run
            # Early crash before pipeline.run(): infer is_rewrite from CLI args.
            is_rewrite_top = bool(parsed_targets) or args.rewrite_scope in ("recipient", "all")
            finalize_personalization_run(
                sequence_id=args.sequence_id,
                completed=0, failed=0,
                metadata={"error_type": type(e).__name__},
                error=str(e),
                personalization_run_id=args.personalization_run_id,
                is_rewrite=is_rewrite_top,
            )
        except Exception:
            pass
        sys.exit(1)

    logger.info("Personalization complete: %s", metadata)


if __name__ == "__main__":
    main()
