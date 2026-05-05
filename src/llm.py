"""OpenRouter (OpenAI-compatible) Chat Completions — responses include usage.cost when using OpenRouter."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

from config import Settings
from trace import TraceRecorder
from usage_cost import extract_usage_fields


class ChatLLM:
    def __init__(self, settings: Settings, trace: TraceRecorder | None = None) -> None:
        if not settings.api_key:
            raise RuntimeError(
                "No API key: set OPENROUTER_API_KEY (or OPENAI_API_KEY for compatible endpoints)."
            )
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": settings.api_key,
            "base_url": settings.base_url,
        }
        headers: dict[str, str] = {}
        if headers:
            kwargs["default_headers"] = headers

        self._client: OpenAI = OpenAI(**kwargs)
        self._model = settings.model_name
        self._trace = trace

    @property
    def client(self) -> "OpenAI":
        return self._client

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        trace_phase: str,
        trace_extra: dict[str, Any] | None = None,
    ) -> str:
        if self._trace:
            self._trace.add(
                trace_phase,
                direction="request",
                messages=messages,
                temperature=temperature,
                **(trace_extra or {}),
            )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        if self._trace:
            uc = extract_usage_fields(resp, model_fallback=self._model)
            self._trace.add(
                trace_phase,
                direction="response",
                content=text,
                completion_id=getattr(resp, "id", None),
                model=getattr(resp, "model", None),
                finish_reason=(
                    resp.choices[0].finish_reason if getattr(resp, "choices", None) else None
                ),
                **uc,
            )
        return text
