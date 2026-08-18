"""Anthropic SDK adapter for `LLMPlanner` - the ONLY file that imports `anthropic`.

Everything about the planner's contract lives in `planner.py`; this file just
turns "prompt + JSON schema" into "JSON string" using the Claude API, with the
failure modes a robot planner has to care about:

  * no SDK / no credentials  -> `make_anthropic_completer()` returns (None, why)
    and the planner stays rule-based. The reason is surfaced, never swallowed.
  * network / rate limit / 5xx -> raises; `LLMPlanner` catches and falls back.
  * safety refusal (`stop_reason == "refusal"`) -> raises with the category;
    the planner falls back to its deterministic plan. Deliberately no server-
    side model fallback here: for a task planner the correct fallback is the
    rule-based planner, not another LLM.
  * slow response -> 30 s timeout, one retry. A planner must never stall the
    control loop indefinitely.

Structured outputs (`output_config.format`) make the reply valid JSON by
construction, so the planner never has to scrape braces out of prose.

Usage:
    complete, why = make_anthropic_completer()
    planner = LLMPlanner(complete=complete)   # complete may be None -> rule-based

Env:
    ANTHROPIC_API_KEY   (or an `ant auth login` profile - the SDK resolves both)
    RVC_LLM_MODEL       default "claude-opus-5"
    RVC_LLM_EFFORT      default "medium"  (low|medium|high|xhigh|max)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"
TIMEOUT_S = 30.0
MAX_TOKENS = 8000  # cap on thinking + answer; the JSON itself is tiny


class PlannerRefused(RuntimeError):
    """The model declined the request (stop_reason == 'refusal')."""


def make_anthropic_completer(
    model: str | None = None,
    effort: str | None = None,
) -> tuple[Callable[..., str] | None, str]:
    """Return (completer, reason). completer is None when the LLM path is unavailable.

    The completer signature is `complete(prompt: str, schema: dict | None = None) -> str`.
    """
    try:
        import anthropic
    except ImportError:
        return None, "anthropic SDK not installed (uv pip install -e '.[llm]')"

    model = model or os.environ.get("RVC_LLM_MODEL", DEFAULT_MODEL)
    effort = effort or os.environ.get("RVC_LLM_EFFORT", DEFAULT_EFFORT)

    # Zero-arg client: resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an
    # `ant auth login` profile. Construction does not hit the network - and it
    # does NOT fail without credentials (auth errors surface on the first
    # request), so check explicitly: reporting "llm active" and then silently
    # falling back on every call would be exactly the dishonesty this project
    # is built to avoid.
    try:
        client = anthropic.Anthropic()
    except Exception as exc:
        return None, f"anthropic client unavailable: {type(exc).__name__}: {exc}"
    if not (client.api_key or client.auth_token or getattr(client, "credentials", None)):
        return None, (
            "no Anthropic credentials found (set ANTHROPIC_API_KEY, or run "
            "`ant auth login`) -> LLM planner off"
        )
    client = client.with_options(timeout=TIMEOUT_S, max_retries=1)

    def complete(prompt: str, schema: dict[str, Any] | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": (
                "You are the high-level task planner for a robot manipulator. "
                "Answer with the requested JSON and nothing else."
            ),
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": effort},
        }
        if schema is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": schema}

        response = client.messages.create(**kwargs)

        # A refusal is a normal HTTP 200 - check before touching content.
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise PlannerRefused(f"model refused (category={category})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("planner reply truncated at max_tokens")

        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            raise RuntimeError("planner reply contained no text block")
        return text

    complete.__name__ = f"anthropic:{model}"
    return complete, f"anthropic SDK {anthropic.__version__}, model={model}, effort={effort}"
