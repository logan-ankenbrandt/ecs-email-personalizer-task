"""Researcher sub-agent: fetch the recipient's company website in parallel,
extract structured intelligence, submit a brief via the submit_brief tool.

Uses call_with_tools_loop. The web_fetch handler wraps utils.web_fetch;
the submit_brief handler signals is_terminal and the brief payload is
returned to the caller via LoopResult.terminal_payload.

Anti-URL-guessing: reuses writer.py's circuit-breaker idea — after 2
consecutive fetch failures we inject a "STOP" message so the model stops
burning turns hallucinating URLs. Fewer turns, better quality when the
real site is down.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from config import RESEARCH_MODEL
from utils.web_fetch import fetch_url_cached, normalize_url

from agent_v2.loop import LoopResult, ToolResult, TokenUsage, call_with_tools_loop
from agent_v2.schemas import RESEARCHER_TOOLS

logger = logging.getLogger(__name__)

# After this many consecutive failed fetches, tell the model to stop guessing
# URLs and submit with whatever context it has. Same threshold as writer.py.
_MAX_CONSECUTIVE_FETCH_FAILURES = 2


def _build_researcher_task(recipient: Dict[str, Any], focus: Optional[str]) -> str:
    """User prompt for the researcher, seeded with the recipient info we have."""
    name = recipient.get("first_name", "") or ""
    if recipient.get("last_name"):
        name = f"{name} {recipient['last_name']}".strip()
    title = recipient.get("title", "") or "(unknown title)"
    company = recipient.get("business_name") or recipient.get("company") or "(unknown company)"
    location = recipient.get("location") or recipient.get("city") or "(unknown location)"

    # Prefer explicit website fields; fall back to email domain.
    website = (
        recipient.get("company_website")
        or recipient.get("website")
        or (recipient.get("custom_fields") or {}).get("company_website")
        or ""
    )
    if not website and recipient.get("email"):
        domain = recipient["email"].split("@")[-1].strip().lower() if "@" in recipient["email"] else ""
        if domain and "." in domain and domain not in {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}:
            website = f"https://{domain}"

    lines: List[str] = [
        f"## Recipient",
        f"- Name: {name or '(unknown)'}",
        f"- Title: {title}",
        f"- Company: {company}",
        f"- Location: {location}",
    ]
    if website:
        lines.append(f"- Website: {website}")
    else:
        lines.append(
            "- Website: (none on file). Submit a brief based only on "
            "the recipient context above — do NOT guess URLs."
        )
    if focus:
        lines.append(f"\n## Focus for this research pass\n\n{focus}")

    lines.append(
        "\n## Instructions\n\n"
        "Fetch the relevant pages (homepage + /about + /services + /team + "
        "/news as applicable) in parallel where possible, then call "
        "submit_brief with the structured fields. If the site is down or "
        "missing data, submit a thin brief using the recipient context only. "
        "Do not fabricate."
    )
    return "\n".join(lines)


def run_researcher(
    system_prompt: str,
    recipient: Dict[str, Any],
    focus: Optional[str] = None,
    max_turns: int = 5,
) -> Tuple[Optional[Dict[str, Any]], TokenUsage]:
    """Execute a researcher session.

    Returns (brief_dict, usage). brief_dict keys: vertical, team_size,
    differentiator, markets, notable_metrics, sources. None if the model
    failed to submit (exhausted max_turns).
    """
    consecutive_failures = [0]  # list so closure can mutate
    fetched_urls: List[str] = []

    def handle_tool(name: str, tool_input: Dict[str, Any], tool_use_id: str) -> ToolResult:
        if name == "web_fetch":
            url = normalize_url((tool_input.get("url") or "").strip())
            if not url:
                return ToolResult(content="[fetch failed] empty url", is_error=True)
            text = fetch_url_cached(url, max_chars=5000)
            if text:
                consecutive_failures[0] = 0
                fetched_urls.append(url)
                return ToolResult(content=text)
            consecutive_failures[0] += 1
            if consecutive_failures[0] >= _MAX_CONSECUTIVE_FETCH_FAILURES:
                # Hard STOP to prevent URL-guessing doom-loop.
                return ToolResult(
                    content=(
                        f"[fetch failed] STOP: you have failed to fetch "
                        f"{consecutive_failures[0]} URLs in a row. Do NOT "
                        f"guess more URLs. Submit the brief with the "
                        f"recipient context you already have."
                    ),
                    is_error=True,
                )
            return ToolResult(content="[fetch failed or empty]", is_error=True)

        if name == "submit_brief":
            # Ensure sources is populated even if the model omitted it.
            if not tool_input.get("sources"):
                tool_input["sources"] = fetched_urls
            return ToolResult(
                content="brief accepted",
                is_terminal=True,
                payload=tool_input,
            )

        return ToolResult(content=f"[unknown tool: {name}]", is_error=True)

    messages = [{"role": "user", "content": _build_researcher_task(recipient, focus)}]
    result: LoopResult = call_with_tools_loop(
        system_prompt=system_prompt,
        messages=messages,
        tools=RESEARCHER_TOOLS,
        model=RESEARCH_MODEL,
        max_turns=max_turns,
        tool_handler=handle_tool,
        max_tokens_per_turn=2048,
        temperature=0.0,
    )

    if result.stop_reason == "tool_terminal" and isinstance(result.terminal_payload, dict):
        brief = result.terminal_payload
        # Ensure sources is a list (defensive).
        if not isinstance(brief.get("sources"), list):
            brief["sources"] = fetched_urls
        logger.info(
            "researcher submitted: vertical=%s sources=%d metrics=%d",
            brief.get("vertical"), len(brief.get("sources", [])),
            len(brief.get("notable_metrics", [])),
        )
        return brief, result.usage

    logger.warning(
        "researcher did not submit: stop_reason=%s turns=%d",
        result.stop_reason, result.usage.turns_used,
    )
    return None, result.usage
