"""Generic multi-turn tool-use loop, modeled on Claude Code's query loop.

Extracted from writer.py's per-personalizer loop and decoupled from any
specific tool set. The caller provides tool schemas and a tool_handler
callback that dispatches each tool_use block to the appropriate backend.

Termination conditions (see LoopResult.stop_reason):
  - "end_turn":        model returned no tool_use blocks.
  - "tool_terminal":   a tool_handler return ToolResult(is_terminal=True).
  - "max_turns":       loop exhausted the turn budget.
  - "max_tokens":      the API returned stop_reason=max_tokens.
  - (other API stops): passed through as-is.

The caller accumulates usage via the returned TokenUsage and records cost
via CostAccumulator. Tool_handler errors become is_error tool_result blocks
sent back to the model (so the model can adapt), not exceptions propagated
to the caller.

Retry policy on API errors: jittered exponential backoff on RateLimitError
and 5xx APIStatusError, copied from utils.llm:90-112. Non-retryable errors
propagate.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Accumulated token counts across all turns of a single loop run."""
    input_tokens: int = 0
    output_tokens: int = 0
    turns_used: int = 0

    def add(self, input_t: int, output_t: int) -> None:
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.turns_used += 1


@dataclass
class ToolResult:
    """Result returned by the tool_handler callback for each tool_use block.

    content: what the model will see as the tool_result content.
    is_error: set True to mark tool_result as is_error (model sees failure).
    is_terminal: set True to end the loop. The loop returns immediately
      after appending this tool_result, with stop_reason="tool_terminal"
      and terminal_payload set to the handler's returned payload (or
      content if payload is None).
    payload: structured object to surface back to the caller when
      is_terminal; if None, content is used.
    """
    content: str
    is_error: bool = False
    is_terminal: bool = False
    payload: Optional[Any] = None


@dataclass
class LoopResult:
    """Outcome of call_with_tools_loop.

    final_content: the last assistant message's content blocks (list of
      ContentBlock objects from the SDK). For end_turn this is usually the
      final text; for tool_terminal this is the assistant message that
      issued the terminal tool call.
    messages: the full messages[] list at loop end (caller can inspect
      for debugging or chain further calls).
    usage: accumulated token counts.
    stop_reason: see module docstring.
    terminal_payload: payload from the tool_handler that signaled terminal.
      None when stop_reason != "tool_terminal".
    """
    final_content: List[Any]
    messages: List[Dict[str, Any]]
    usage: TokenUsage
    stop_reason: str
    terminal_payload: Optional[Any] = None


def _call_with_retry(
    client: anthropic.Anthropic,
    create_kwargs: Dict[str, Any],
    max_retries: int,
) -> Any:
    """messages.create() wrapper with jittered exponential backoff.

    Pattern copied from utils.llm:90-112 so retry behavior is consistent
    across every LLM call site in the pipeline.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**create_kwargs)
        except anthropic.RateLimitError as e:
            last_error = e
            if attempt < max_retries:
                base_wait = 2 ** (attempt + 1)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                logger.warning(
                    "Rate limited, retrying in %.1fs (attempt %d)",
                    wait, attempt + 1,
                )
                time.sleep(wait)
            else:
                raise
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries:
                last_error = e
                base_wait = 2 ** (attempt + 1)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                logger.warning(
                    "Server error %d, retrying in %.1fs (attempt %d)",
                    e.status_code, wait, attempt + 1,
                )
                time.sleep(wait)
            else:
                raise
    # Unreachable in practice; defensive.
    raise last_error  # type: ignore[misc]


def call_with_tools_loop(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: str,
    max_turns: int,
    tool_handler: Callable[[str, Dict[str, Any], str], ToolResult],
    max_tokens_per_turn: int = 4096,
    temperature: Optional[float] = None,
    max_retries: int = 3,
    client: Optional[anthropic.Anthropic] = None,
) -> LoopResult:
    """Run a multi-turn tool-use loop.

    Args:
        system_prompt: passed as the `system` kwarg to messages.create.
        messages: mutable list of messages. Modified in-place (assistant +
          tool_result messages are appended per turn). Caller can pre-seed
          with the initial user message.
        tools: list of tool schemas (see schemas.py for examples).
        model: Anthropic model ID.
        max_turns: hard cap on iterations.
        tool_handler: callback invoked per tool_use block with
          (tool_name, tool_input, tool_use_id). Must return a ToolResult.
          Exceptions from the handler are caught and converted to is_error
          tool_result blocks.
        max_tokens_per_turn: Anthropic max_tokens per API call.
        temperature: optional sampling temperature.
        max_retries: retry count for rate limits / 5xx.
        client: optional pre-built Anthropic client (for testing). Defaults
          to a fresh one.

    Returns:
        LoopResult capturing final state. See dataclass docstring.
    """
    if client is None:
        client = anthropic.Anthropic()
    usage = TokenUsage()

    for turn in range(max_turns):
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens_per_turn,
            "system": system_prompt,
            "tools": tools,
            "messages": messages,
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature

        response = _call_with_retry(client, create_kwargs, max_retries)

        # Accumulate tokens (SDK response.usage is always present for
        # successful calls; defensive getattr anyway).
        if hasattr(response, "usage"):
            usage.add(response.usage.input_tokens, response.usage.output_tokens)

        # Append the assistant turn so the next messages.create sees it.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return LoopResult(
                final_content=response.content,
                messages=messages,
                usage=usage,
                stop_reason="end_turn",
            )

        if response.stop_reason != "tool_use":
            # max_tokens, pause_turn, refusal, or an unexpected value.
            # Pass through so the caller can branch on it.
            logger.warning(
                "Loop: unexpected stop_reason=%s at turn %d",
                response.stop_reason, turn,
            )
            return LoopResult(
                final_content=response.content,
                messages=messages,
                usage=usage,
                stop_reason=response.stop_reason or "unknown",
            )

        # stop_reason == "tool_use": dispatch every tool_use block in this
        # assistant message. Model may emit multiple tool_uses in one turn
        # (parallel tool calls) — we handle all of them sequentially. For
        # genuinely parallel backends, the tool_handler can be wrapped in
        # asyncio.gather externally; the loop itself is synchronous.
        tool_results: List[Dict[str, Any]] = []
        terminal_payload: Optional[Any] = None
        for block in response.content:
            if not (hasattr(block, "type") and block.type == "tool_use"):
                continue
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", {}) or {}
            tool_use_id = getattr(block, "id", None)

            if not tool_name or not tool_use_id:
                logger.warning(
                    "Loop: malformed tool_use block at turn %d: %s",
                    turn, block,
                )
                continue

            try:
                result = tool_handler(tool_name, tool_input, tool_use_id)
            except Exception as e:  # noqa: BLE001 - convert any handler error
                logger.error(
                    "Loop: tool_handler raised for %s: %s", tool_name, e,
                    exc_info=True,
                )
                result = ToolResult(
                    content=f"[tool error] {type(e).__name__}: {e}",
                    is_error=True,
                )

            tr_block: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result.content,
            }
            if result.is_error:
                tr_block["is_error"] = True
            tool_results.append(tr_block)

            if result.is_terminal:
                # Capture the payload and stop dispatching further tool_use
                # blocks from this same assistant response. This matches
                # Claude Code's behavior: once a terminal tool fires, we
                # don't honor sibling tool calls in the same turn.
                terminal_payload = (
                    result.payload if result.payload is not None else result.content
                )
                break

        # Must append tool_result messages even on terminal to keep the
        # conversation well-formed (Anthropic requires tool_result for
        # every tool_use from the prior assistant turn).
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if terminal_payload is not None:
            return LoopResult(
                final_content=response.content,
                messages=messages,
                usage=usage,
                stop_reason="tool_terminal",
                terminal_payload=terminal_payload,
            )

        if not tool_results:
            # stop_reason said tool_use but no tool blocks found. Shouldn't
            # happen with a well-behaved model; treat as end_turn.
            logger.warning(
                "Loop: tool_use stop_reason but no tool blocks at turn %d",
                turn,
            )
            return LoopResult(
                final_content=response.content,
                messages=messages,
                usage=usage,
                stop_reason="end_turn",
            )

    # Exhausted max_turns without end_turn or terminal.
    return LoopResult(
        final_content=messages[-1].get("content", []) if messages else [],
        messages=messages,
        usage=usage,
        stop_reason="max_turns",
    )
