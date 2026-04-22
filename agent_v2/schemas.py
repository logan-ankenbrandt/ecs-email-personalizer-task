"""Anthropic tool schemas for every tool the orchestrator and sub-agents expose.

Centralizing schemas here means every sub-agent imports its tool list from
one place, there's no schema drift between caller and handler, and the
schemas are trivially testable. Schema shape matches Anthropic's tools API:
{name, description, input_schema} with input_schema being a JSON Schema
dict.

Tool name conventions:
- list_recipient_gaps, get_recipient_brief, dispatch_*, read_draft,
  submit_step, skip_step: orchestrator-level tools (routed in
  orchestrator.py's tool handler).
- web_fetch, submit_brief: researcher sub-agent tools.
- validate_draft, submit_draft: writer sub-agent tools.
- submit_output: critic is single-shot via generate_structured; uses that
  helper's built-in submit_output tool, not a schema here.
"""

# ============================================================
# Orchestrator-level tools
# ============================================================

LIST_RECIPIENT_GAPS_TOOL = {
    "name": "list_recipient_gaps",
    "description": (
        "Return which steps in the sequence are done, missing (no personalized "
        "doc), or need rewrite (doc exists but below threshold). Call this FIRST "
        "in every orchestrator session to see what needs work."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "recipient_id": {"type": "string"},
            "sequence_id": {"type": "string"},
        },
        "required": ["recipient_id", "sequence_id"],
    },
}

GET_RECIPIENT_BRIEF_TOOL = {
    "name": "get_recipient_brief",
    "description": (
        "Return the cached company brief if one exists for this session, or "
        "trigger the researcher to create one. Returns {brief_id, "
        "brief_summary, vertical, cached, sources}. You need a brief_id "
        "before dispatching any writer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "recipient_id": {"type": "string"},
        },
        "required": ["recipient_id"],
    },
}

DISPATCH_RESEARCHER_TOOL = {
    "name": "dispatch_researcher",
    "description": (
        "Force a fresh research pass (bypasses the cache). Use only when the "
        "cached brief is missing key data or you need research on a specific "
        "angle via the focus parameter. Cost ~$0.01."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "recipient_id": {"type": "string"},
            "focus": {
                "type": "string",
                "description": "Optional focus string (e.g. 'recent hiring news')",
            },
        },
        "required": ["recipient_id"],
    },
}

DISPATCH_WRITER_TOOL = {
    "name": "dispatch_writer",
    "description": (
        "Dispatch the writer sub-agent to produce one email. The writer "
        "self-validates before returning. Returns {draft_id, word_count, "
        "validation_passed, subject_preview}. Cost ~$0.03."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "step": {"type": "integer", "description": "Email position (1-N)"},
            "brief_id": {"type": "string", "description": "From get_recipient_brief; required"},
            "prior_summary": {
                "type": "string",
                "description": (
                    "What earlier accepted steps said (opener + proof points). "
                    "The writer avoids recycling these."
                ),
            },
            "constraints": {
                "type": "string",
                "description": "Optional specific constraints for this rewrite",
            },
        },
        "required": ["step", "brief_id"],
    },
}

DISPATCH_CRITIC_TOOL = {
    "name": "dispatch_critic",
    "description": (
        "Dispatch the Opus critic to assess a draft on 5 dimensions. Use ONLY "
        "when the writer returned validation_passed=false, when the draft is "
        "near word-count limits, or as a final-pass confidence check. Do NOT "
        "call on every draft. Cost ~$0.02."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "draft_id": {"type": "string"},
            "step": {"type": "integer"},
        },
        "required": ["draft_id", "step"],
    },
}

READ_DRAFT_TOOL = {
    "name": "read_draft",
    "description": (
        "Surgically reveal specific fields of a draft. Use sparingly; prefer "
        "the summary returned by dispatch_writer. Fields: subject, "
        "first_sentence, last_sentence, word_count, content, company_insight, "
        "data_grounding."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "draft_id": {"type": "string"},
            "fields": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["draft_id", "fields"],
    },
}

SUBMIT_STEP_TOOL = {
    "name": "submit_step",
    "description": (
        "Accept this draft as final. Upserts to Mongo. Only submit drafts "
        "with validation_passed=true OR critic overall_score >= 0.70."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
            "draft_id": {"type": "string"},
            "quality_score": {
                "type": "number",
                "description": (
                    "The score to store for this draft. Use critic's "
                    "overall_score if critic was called, else 0.80 "
                    "(writer-self-validated baseline)."
                ),
            },
        },
        "required": ["step", "draft_id", "quality_score"],
    },
}

SKIP_STEP_TOOL = {
    "name": "skip_step",
    "description": (
        "Give up on this step. The template fallback will send. Valid "
        "reasons: budget_exhausted, persistent_quality_failure, "
        "no_research_data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["step", "reason"],
    },
}

ORCHESTRATOR_TOOLS = [
    LIST_RECIPIENT_GAPS_TOOL,
    GET_RECIPIENT_BRIEF_TOOL,
    DISPATCH_RESEARCHER_TOOL,
    DISPATCH_WRITER_TOOL,
    DISPATCH_CRITIC_TOOL,
    READ_DRAFT_TOOL,
    SUBMIT_STEP_TOOL,
    SKIP_STEP_TOOL,
]


# ============================================================
# Researcher sub-agent tools
# ============================================================

WEB_FETCH_TOOL = {
    "name": "web_fetch",
    "description": (
        "Fetch a URL and return its plain text (HTML stripped). You may call "
        "this multiple times in a single response to fetch in parallel (e.g. "
        "/about + /services + /team)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
        },
        "required": ["url"],
    },
}

SUBMIT_BRIEF_TOOL = {
    "name": "submit_brief",
    "description": (
        "Submit the final company brief. Call exactly once when research is "
        "complete."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "vertical": {
                "type": "string",
                "enum": [
                    "healthcare", "it_tech", "exec_search", "hr_peo",
                    "va_offshore", "light_industrial", "construction_trades",
                    "general",
                ],
            },
            "team_size": {"type": ["integer", "null"]},
            "differentiator": {"type": "string"},
            "markets": {"type": "array", "items": {"type": "string"}},
            "notable_metrics": {"type": "array", "items": {"type": "string"}},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "vertical", "differentiator", "markets", "notable_metrics", "sources",
        ],
    },
}

RESEARCHER_TOOLS = [WEB_FETCH_TOOL, SUBMIT_BRIEF_TOOL]


# ============================================================
# Writer sub-agent tools
# ============================================================

VALIDATE_DRAFT_TOOL = {
    "name": "validate_draft",
    "description": (
        "Run structural + slop validation on your draft. Returns a list of "
        "issues, each with field/rule/offending_snippet/suggestion. Call "
        "BEFORE submit_draft. If issues are returned, fix them and call "
        "validate_draft again. Only call submit_draft when the list is empty."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "content": {"type": "string"},
            "step": {"type": "integer"},
        },
        "required": ["subject", "content", "step"],
    },
}

SUBMIT_DRAFT_TOOL = {
    "name": "submit_draft",
    "description": (
        "Submit the final draft when validate_draft returns no issues. Call "
        "exactly once when done."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "content": {
                "type": "string",
                "description": (
                    "HTML body. Use <p>, <br>, <strong>, <em>, <a> only. "
                    "Keep merge fields like {{first_name}} UNRESOLVED."
                ),
            },
            "company_insight": {
                "type": "string",
                "description": "One sentence on what you found and how you used it.",
            },
            "data_grounding": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of specific facts/numbers/names from research that "
                    "appear in the copy."
                ),
            },
            "word_count": {"type": "integer"},
        },
        "required": [
            "subject", "content", "company_insight", "data_grounding", "word_count",
        ],
    },
}

WRITER_TOOLS = [VALIDATE_DRAFT_TOOL, SUBMIT_DRAFT_TOOL]


# ============================================================
# Critic schema (used as `schema` arg to generate_structured)
# ============================================================

CRITIC_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "number"},
        "dimension_scores": {
            "type": "object",
            "properties": {
                "personalization_depth": {"type": "number"},
                "slop_absence": {"type": "number"},
                "tone_authenticity": {"type": "number"},
                "structural_compliance": {"type": "number"},
                "segment_specificity": {"type": "number"},
            },
            "required": [
                "personalization_depth", "slop_absence", "tone_authenticity",
                "structural_compliance", "segment_specificity",
            ],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "excerpt": {"type": "string"},
                    "slop_type": {"type": ["string", "null"]},
                    "issue": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["excerpt", "issue", "suggestion"],
            },
        },
        "should_refine": {"type": "boolean"},
        "swap_test_result": {
            "type": "object",
            "properties": {
                "sentences_tested": {"type": "integer"},
                "sentences_passing": {"type": "integer"},
                "sentences_failing": {"type": "integer"},
            },
        },
    },
    "required": [
        "overall_score", "dimension_scores", "issues", "should_refine",
    ],
}
