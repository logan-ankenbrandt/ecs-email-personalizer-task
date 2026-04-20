"""Sonnet writer with optional web_fetch tool.

Multi-turn loop using raw anthropic.messages.create() because the writer
needs tool-use (the structured `generate_structured` helper used elsewhere
forces a single tool call with no follow-ups).

Pattern ported from ~/.scripts/agentic-email-rewriter-v2.py. The Opus
advisor tool was removed (Fix 4) — it doubled the writer-phase cost and
models don't self-restrain on "use sparingly" guidance.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic

from config import WRITER_MAX_TURNS, WRITER_MODEL, get_api_key
from utils.web_fetch import fetch_url_cached, normalize_url

logger = logging.getLogger(__name__)


# Tool definitions exposed to the writer LLM
_WEB_FETCH_TOOL = {
    "name": "web_fetch",
    "description": (
        "Fetch a URL and return its plain text content (HTML stripped). "
        "Use this to research the recipient's company website before writing. "
        "Returns up to 5000 chars."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch"},
        },
        "required": ["url"],
    },
}

# JSON schema for the writer's final output
_WRITE_RESULT_TOOL = {
    "name": "submit_personalized_email",
    "description": "Submit the final personalized email. Call exactly once at the end.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Personalized subject line"},
            "content": {
                "type": "string",
                "description": (
                    "Personalized body as HTML (use <p>, <br>, <strong>, "
                    "<em>, <a> tags only — no markdown). All merge fields "
                    "({{first_name}}, etc.) should remain UNRESOLVED — the "
                    "downstream processor handles substitution."
                ),
            },
            "company_insight": {
                "type": "string",
                "description": (
                    "1-2 sentence summary of what you learned about the "
                    "recipient's company that grounded the rewrite."
                ),
            },
            "data_grounding": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of specific facts/numbers/names from research that "
                    "appear in the copy. Used for downstream audit."
                ),
            },
        },
        "required": ["subject", "content", "company_insight", "data_grounding"],
    },
}


# Circuit-breaker threshold for consecutive web_fetch failures. Two in a row
# is the signal to stop guessing URLs (the writer was burning turns on doom
# loops like pattonstaff.com -> pattonstaffing.com -> pattonstaff.co -> ...).
MAX_CONSECUTIVE_FETCH_FAILURES = 2


def write_personalized_email(
    system_prompt: str,
    user_prompt: str,
    max_turns: Optional[int] = None,
    enable_web_fetch: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    """Run the writer loop. Returns (result_dict, token_counts).

    result_dict has keys: subject, content, company_insight, data_grounding,
    advisor_used. None if the writer failed to call submit_personalized_email
    within max_turns.

    When enable_web_fetch is False (no known website on file for the recipient),
    web_fetch is omitted from the tool list. Forces the writer to ground copy
    in the provided recipient_summary rather than guessing URLs.

    token_counts: {input_tokens, output_tokens} accumulated across all turns.
    """
    if max_turns is None:
        max_turns = WRITER_MAX_TURNS
    client = Anthropic(api_key=get_api_key())
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    tools: List[Dict[str, Any]] = [_WRITE_RESULT_TOOL]
    if enable_web_fetch:
        tools.insert(0, _WEB_FETCH_TOOL)
    # advisor_used kept in result shape for backward compat; always False now.
    advisor_used = False
    consecutive_fetch_failures = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for turn in range(max_turns):
        try:
            response = client.messages.create(
                model=WRITER_MODEL,
                max_tokens=4000,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
        except Exception as e:
            logger.error("Writer LLM call failed at turn %d: %s", turn, e)
            return None, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}

        # Accumulate tokens
        if hasattr(response, "usage"):
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        # Append the assistant's response to history
        messages.append({"role": "assistant", "content": response.content})

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Writer finished without calling submit_personalized_email. Bad.
            logger.warning("Writer ended turn without submitting. Turn=%d", turn)
            return None, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}

        if response.stop_reason != "tool_use":
            logger.warning("Unexpected stop_reason=%s at turn %d", response.stop_reason, turn)
            return None, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}

        # Process tool uses
        tool_results: List[Dict[str, Any]] = []
        for block in response.content:
            if not (hasattr(block, "type") and block.type == "tool_use"):
                continue

            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            if tool_name == "submit_personalized_email":
                # Final submission
                tool_input["advisor_used"] = advisor_used
                logger.info(
                    "Writer submitted on turn %d: subject_len=%d, content_len=%d, advisor=%s",
                    turn, len(tool_input.get("subject", "")),
                    len(tool_input.get("content", "")), advisor_used,
                )
                return tool_input, {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                }

            if tool_name == "web_fetch":
                url = normalize_url(tool_input.get("url", ""))
                fetched = fetch_url_cached(url, max_chars=5000)
                if fetched:
                    consecutive_fetch_failures = 0
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": fetched,
                    })
                else:
                    consecutive_fetch_failures += 1
                    if consecutive_fetch_failures >= MAX_CONSECUTIVE_FETCH_FAILURES:
                        # Circuit breaker: stop the URL-guessing doom loop.
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": (
                                f"[fetch failed] STOP: You have failed to fetch "
                                f"{consecutive_fetch_failures} URLs in a row. Do NOT guess "
                                f"more URLs. Use the provided recipient context and submit "
                                f"via submit_personalized_email now."
                            ),
                        })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "[fetch failed or empty]",
                        })
                continue

            # Unknown tool
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"[Unknown tool: {tool_name}]",
                "is_error": True,
            })

        if not tool_results:
            # Stop reason was tool_use but no tool blocks? Shouldn't happen.
            logger.warning("tool_use stop with no tool blocks at turn %d", turn)
            return None, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}

        messages.append({"role": "user", "content": tool_results})

    # Exhausted max_turns
    logger.warning("Writer exhausted max_turns=%d without submitting", max_turns)
    return None, {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}
