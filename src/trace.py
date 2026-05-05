"""Structured trace of model reasoning and execution steps (no LangChain)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """One observable step in the pipeline."""

    phase: str
    detail: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self.total_llm_cost_usd: float = 0.0

    def add(self, phase: str, **detail: Any) -> None:
        self.events.append(TraceEvent(phase=phase, detail=detail))
        c = detail.get("cost_usd")
        if c is not None and isinstance(c, (int, float)):
            self.total_llm_cost_usd += float(c)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [{"phase": e.phase, **e.detail} for e in self.events]
