"""Anthropic API caller with structured output via tool-use."""

import json
import logging
import random
import time

import anthropic

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> anthropic.Anthropic:
    """Lazy-init Anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def generate_structured(
    prompt: str,
    schema: dict,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 8192,
    max_retries: int = 5,
    temperature: float = 0.0,
    system: str = "",
    return_usage: bool = False,
):
    """Call Anthropic API with tool-use to get structured JSON output.

    Uses the forced tool_choice pattern: define a "submit_output" tool
    with the desired JSON schema, force the model to call it, and
    extract the tool input as the structured result.

    Args:
        prompt: The user message content.
        schema: JSON Schema for the expected output structure.
        model: Anthropic model ID.
        max_tokens: Max output tokens.
        max_retries: Number of retries on transient errors.
        temperature: Sampling temperature. Defaults to 0.0 so that judges and
            other structured-output callers score deterministically. Set
            higher (e.g. 0.2) for refinement where light variation helps.
        system: Optional system prompt. When empty, no system is sent.
            Added in Round 3 Phase 2 so critic sub-agent can pass a full
            rubric system prompt.
        return_usage: When True, return (dict, {"input_tokens": int,
            "output_tokens": int}) instead of just dict. Default False
            preserves the existing V1 callers.

    Returns:
        Parsed dict from the tool call input, OR (dict, usage_counts) if
        return_usage=True.

    Raises:
        ValueError: If the response contains no tool_use block.
        anthropic.APIError: On non-retryable API errors.
    """
    client = _get_client()

    tool = {
        "name": "submit_output",
        "description": "Submit the generated output.",
        "input_schema": schema,
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            create_kwargs: dict = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": "submit_output"},
            }
            if system:
                create_kwargs["system"] = system
            response = client.messages.create(**create_kwargs)

            # Extract the tool_use block
            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_output":
                    logger.info(
                        "LLM call: model=%s, input_tokens=%d, output_tokens=%d",
                        model,
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                    )
                    if return_usage:
                        return block.input, {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        }
                    return block.input

            raise ValueError(
                f"No submit_output tool_use block in response. "
                f"Content types: {[b.type for b in response.content]}"
            )

        except anthropic.RateLimitError as e:
            last_error = e
            if attempt < max_retries:
                # Jittered exponential backoff to avoid 17 concurrent ECS
                # tasks all retrying at the same instant.
                base_wait = 2 ** (attempt + 1)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                logger.warning("Rate limited, retrying in %.1fs (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
            else:
                raise

        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries:
                last_error = e
                base_wait = 2 ** (attempt + 1)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                logger.warning("Server error %d, retrying in %.1fs", e.status_code, wait)
                time.sleep(wait)
            else:
                raise

    raise last_error  # type: ignore[misc]
