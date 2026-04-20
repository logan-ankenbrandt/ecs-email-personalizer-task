#!/usr/bin/env python3
"""Email Personalizer entrypoint.

Per-recipient rewrite of a template sequence using the proven write-judge-refine
pattern from agentic-email-rewriter-v2.py. Productionized for ECS Fargate.

Usage:
    python main.py --org_id <id> --sequence_id <id> --personalization_run_id <id>
    python main.py ... --concurrency 5 --max_recipients 50 --resume
"""

import argparse
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

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
    )

    try:
        metadata = pipeline.run()
    except Exception as e:
        logger.error("Personalization run failed: %s", e, exc_info=True)
        # Try to mark the run as failed in Mongo so the UI sees status="failed"
        try:
            from utils.mongo import finalize_personalization_run
            finalize_personalization_run(
                sequence_id=args.sequence_id,
                completed=0, failed=0,
                metadata={"error_type": type(e).__name__},
                error=str(e),
            )
        except Exception:
            pass
        sys.exit(1)

    logger.info("Personalization complete: %s", metadata)


if __name__ == "__main__":
    main()
