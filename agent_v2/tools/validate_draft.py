"""validate_draft tool: structured slop validation for the writer's inner loop.

Wraps utils.slop_validation.validate_email into the issue-list shape the
writer's system prompt expects. Each issue has:
  - field: "subject" or "content"
  - rule: the pattern_type (e.g. "em_dash", "sentence_too_long")
  - offending_snippet: the matched text (<=200 chars)
  - suggestion: concrete rewrite direction

The writer calls validate_draft repeatedly until the list is empty (or 4
iterations are exhausted), then calls submit_draft. No LLM involved.
"""

from typing import Dict, List


def validate_draft(subject: str, content: str, step: int) -> List[Dict[str, str]]:
    """Return a structured issue list the writer can iterate on.

    Empty list means the draft passes all structural + slop checks. The
    writer should submit at that point.
    """
    # Import inside function to avoid circular import risk and to keep the
    # tool module lightweight when merely introspected.
    from utils.slop_validation import validate_email

    violations = validate_email(subject or "", content or "", int(step))
    issues: List[Dict[str, str]] = []
    for v in violations:
        issues.append({
            "field": v.field,
            "rule": v.pattern_type,
            "offending_snippet": (v.excerpt or "")[:200],
            # _suggestion is intentionally part of the public contract
            # even though the leading underscore suggests otherwise — it's
            # a Violation method, kept private only because the dataclass
            # is a NamedTuple.
            "suggestion": v._suggestion(),
        })
    return issues
