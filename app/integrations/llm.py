"""OpenAI (GPT-4o) wrapper used by Modules 4, 5 and 8.

``LLMClient.available`` tells callers whether a key is configured; every module
must keep working (with a documented heuristic fallback) when it is ``False``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError, FeatureDisabledError
from app.core.metrics import metrics

logger = structlog.get_logger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None

    def as_assistant_message(self) -> dict[str, Any]:
        """Serialise back into an OpenAI ``assistant`` message (for tool-call loops)."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in self.tool_calls
            ]
        return message


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.openai_model
        self.stats = metrics.provider("openai")
        self._client: Any = None
        self.total_tokens = 0
        if self.settings.openai_configured:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.settings.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
                base_url=self.settings.openai_base_url,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=2,
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    def _require(self) -> Any:
        if self._client is None:
            raise FeatureDisabledError("OPENAI_API_KEY is not configured")
        return self._client

    async def _create(self, **kwargs: Any) -> Any:
        client = self._require()
        started = time.perf_counter()
        try:
            response = await client.chat.completions.create(model=self.model, **kwargs)
        except Exception as exc:  # the SDK raises many error types; map them all to one
            self.stats.record((time.perf_counter() - started) * 1000, error=f"{type(exc).__name__}: {exc}")
            raise ExternalServiceError(f"OpenAI request failed: {type(exc).__name__}") from exc
        self.stats.record((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        return response

    # ----------------------------------------------------------- helpers
    async def complete_text(self, *, system: str, user: str, temperature: float = 0.4, max_tokens: int = 1200) -> str:
        response = await self._create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        schema_name: str = "result",
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        """Return a JSON object; uses structured outputs when a schema is given."""
        response_format: dict[str, Any]
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": False},
            }
        else:
            response_format = {"type": "json_object"}
        response = await self._create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format=response_format,
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except ValueError as exc:
            raise ExternalServiceError("OpenAI returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ExternalServiceError("OpenAI returned a non-object JSON payload")
        return data

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {"messages": messages, "temperature": temperature, "max_completion_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        response = await self._create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except ValueError:
                arguments = {"_raw": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        usage = getattr(response, "usage", None)
        return ChatResult(
            content=message.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
            if usage
            else {},
            model=getattr(response, "model", None),
        )


__all__ = ["ChatResult", "LLMClient", "ToolCall"]
