"""Configuration for the email personalizer pipeline."""

import os
from pathlib import Path

# S3 buckets
S3_BUCKET = "copy-generation"
HARNESS_BUCKET = "agent-harnesses"
KNOWLEDGE_PREFIX = "email-personalizer/knowledge"
PROMPT_PREFIX = "email-personalizer/prompt"

# Anthropic
# WRITER_MODEL: drafts the per-recipient copy (Sonnet 4.6 — fast, cheap, good).
# JUDGE_MODEL: scores intermediate drafts (Sonnet 4.6 by default — cheaper for
# the 1st-pass accept/reject call). JUDGE_MODEL_FINAL is used on the final
# iteration (Opus 4.6). Previously Opus ran every iter (2-3x per step) which
# made the ~$0.20/recipient estimate explode to $1.07/recipient in live run.
# REFINE_MODEL: rewrites flagged emails (Sonnet — refinement is mechanical).
# RESEARCH_MODEL: summarizes the per-recipient web fetch into a company brief.
WRITER_MODEL = os.environ.get("WRITER_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL_FINAL = os.environ.get("JUDGE_MODEL_FINAL", "claude-opus-4-6")
REFINE_MODEL = os.environ.get("REFINE_MODEL", "claude-sonnet-4-6")
RESEARCH_MODEL = os.environ.get("RESEARCH_MODEL", "claude-haiku-4-5-20251001")

# Writer loop bound. 8 was the rewriter default, but the personalizer writer
# tends to burn turns on URL-hallucination doom-loops when no company website
# is on file. 4 caps the damage; Fix 3 (enable_web_fetch gating + circuit
# breaker) also prevents the hallucination loop from starting.
WRITER_MAX_TURNS = int(os.environ.get("WRITER_MAX_TURNS", "4"))

API_KEY_PATH = Path.home() / ".config" / "cold" / "auth" / "anthropic_api_key"

# MongoDB — same dual-cluster split as sequence-architect-task.
# READ cluster (cluster0.725a4j4) hosts copy_generator_runs.
# PRIMARY cluster (34.29.235.153) hosts email_sequences, sequence_emails,
# personalized_sequence_emails, recipients, and is what cold-api UI reads.
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://root_v2:2v_toor@cluster0.725a4j4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
)
PRIMARY_MONGO_URI = os.environ.get(
    "PRIMARY_MONGO_URI",
    "mongodb://root_v2:123PasswordSecureVery@34.29.235.153:27017/administrator?authSource=admin",
)
MONGO_DB = "administrator"

# Quality threshold — the rewriter rubric returns 0-1 scores. <0.75 triggers
# refinement. Hard floor: if a recipient's final score is below this, log
# the failure and skip the recipient (template fallback). Don't ship low-quality.
QUALITY_THRESHOLD = float(os.environ.get("QUALITY_THRESHOLD", "0.75"))
QUALITY_HARD_FLOOR = float(os.environ.get("QUALITY_HARD_FLOOR", "0.50"))
MAX_REFINE_LOOPS = int(os.environ.get("MAX_REFINE_LOOPS", "2"))

# Tiered thresholds: when no company website is available, the writer cannot
# ground personalization_depth claims (no data to inject) and refines become
# rephrasing churn that burns cost without raising score. Apply a relaxed
# threshold + floor in that case. Set these env vars equal to the non-suffixed
# values to disable the tiering.
QUALITY_THRESHOLD_NO_RESEARCH = float(os.environ.get("QUALITY_THRESHOLD_NO_RESEARCH", "0.60"))
QUALITY_HARD_FLOOR_NO_RESEARCH = float(os.environ.get("QUALITY_HARD_FLOOR_NO_RESEARCH", "0.40"))

# Concurrency + batching
CONCURRENCY_DEFAULT = int(os.environ.get("CONCURRENCY_DEFAULT", "5"))
CHECKPOINT_EVERY_N = int(os.environ.get("CHECKPOINT_EVERY_N", "50"))

# Web research
WEB_FETCH_TIMEOUT_SECONDS = int(os.environ.get("WEB_FETCH_TIMEOUT_SECONDS", "30"))
WEB_FETCH_CACHE_DAYS = int(os.environ.get("WEB_FETCH_CACHE_DAYS", "7"))

# Cost telemetry — Anthropic pricing per 1M tokens (approx, 2026-04)
COST_PER_M_INPUT = {
    "claude-sonnet-4-6": 3.0,
    "claude-opus-4-6": 15.0,
    "claude-haiku-4-5-20251001": 0.80,
}
COST_PER_M_OUTPUT = {
    "claude-sonnet-4-6": 15.0,
    "claude-opus-4-6": 75.0,
    "claude-haiku-4-5-20251001": 4.0,
}


def get_api_key() -> str:
    """Resolve Anthropic API key from env or file."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    if API_KEY_PATH.exists():
        key = API_KEY_PATH.read_text().strip()
        if key:
            return key
    raise ValueError(
        f"No ANTHROPIC_API_KEY in env and no key file at {API_KEY_PATH}"
    )
