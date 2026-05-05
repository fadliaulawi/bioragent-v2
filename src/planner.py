"""Parse planner JSON and build a QueryPlan (mirrors agent_planner structures)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from llm import ChatLLM
from prompts import PLANNER_SYSTEM_PROMPT
from trace import TraceRecorder


@dataclass
class PlanStep:
    step: int
    tool: str
    input: str
    purpose: str
    extract: Optional[str] = None
    depends_on: Optional[int] = None


@dataclass
class QueryPlan:
    query_type: str
    entities: list[str]
    steps: list[PlanStep]

    def __str__(self) -> str:
        lines = [f"[{self.query_type.upper()}]  entities: {', '.join(self.entities)}"]
        for s in self.steps:
            dep = f"  ← depends on step {s.depends_on}" if s.depends_on else ""
            lines.append(f"  Step {s.step}: [{s.tool}]  input='{s.input}'{dep}")
            lines.append(f"           purpose: {s.purpose}")
        return "\n".join(lines)


def _parse_step(raw: dict[str, Any]) -> PlanStep:
    return PlanStep(
        step=int(raw["step"]),
        tool=raw["tool"],
        input=raw["input"],
        purpose=raw["purpose"],
        extract=raw.get("extract"),
        depends_on=raw.get("depends_on"),
    )


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    return text


def create_plan(llm: ChatLLM, query: str, trace: TraceRecorder | None = None) -> QueryPlan:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Create a retrieval plan for: {query}"},
    ]
    raw_text = llm.chat(
        messages=messages,
        temperature=0.0,
        trace_phase="planner",
        trace_extra={"user_query": query},
    )
    cleaned = _strip_json_fences(raw_text)
    if trace:
        trace.add("planner_parse", raw_model_text=raw_text, cleaned_json_text=cleaned)
    data = json.loads(cleaned)
    steps = [_parse_step(s) for s in data["steps"]]
    return QueryPlan(
        query_type=data.get("query_type", "single_hop"),
        entities=data.get("entities", []),
        steps=steps,
    )
