"""OpenAI Chat Completions tool-calling loop for refinement (replaces LangChain agent_data)."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from prompts import TOOL_CATALOG
from tools import format_tool_output, run_tool
from trace import TraceRecorder
from usage_cost import extract_usage_fields


def _tool_slug(display_name: str) -> str:
    s = display_name.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    return s[:64] or "tool"


SLUG_TO_DISPLAY: dict[str, str] = {_tool_slug(k): k for k in TOOL_CATALOG}
DISPLAY_TO_SLUG: dict[str, str] = {v: k for k, v in SLUG_TO_DISPLAY.items()}


REFINE_SYSTEM_PROMPT = """You are a biomedical retrieval assistant with access to database tools only.
You must call the appropriate tools to gather facts; do not fabricate database content.
When tools return enough information, answer in clear prose summarizing what was retrieved.
If tools return "Not Found" or empty results, say so honestly.
"""


def _openai_tool_schemas() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for display_name, desc in TOOL_CATALOG.items():
        slug = DISPLAY_TO_SLUG[display_name]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": slug,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Primary search term (gene, disease, SNP rs id, phenotype name, etc.)",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        )
    return tools


def refine_with_openai_tools(
    *,
    client: OpenAI,
    model: str,
    user_instruction: str,
    trace: TraceRecorder | None = None,
    max_rounds: int = 8,
    temperature: float = 0.0,
) -> str:
    """
    Multi-turn tool loop: model may call any catalog tool by slug; results are fed back until
    the model returns a normal assistant message without tool_calls.
    """
    tools = _openai_tool_schemas()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction},
    ]

    for round_i in range(max_rounds):
        if trace:
            trace.add(
                "refine_openai_round",
                round=round_i + 1,
                message_count=len(messages),
            )

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )
        choice = resp.choices[0]
        msg = choice.message

        if trace:
            uc = extract_usage_fields(resp, model_fallback=model)
            trace.add(
                "refine_openai_completion",
                round=round_i + 1,
                finish_reason=choice.finish_reason,
                has_tool_calls=bool(msg.tool_calls),
                content_preview=(msg.content or "")[:400],
                **uc,
            )

        if not msg.tool_calls:
            return (msg.content or "").strip()

        # Append assistant turn with tool_calls (API expects serialized tool_calls)
        assistant_payload: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        tool_calls_list: list[dict[str, Any]] = []
        for tc in msg.tool_calls:
            tool_calls_list.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
            )
        assistant_payload["tool_calls"] = tool_calls_list
        messages.append(assistant_payload)

        for tc in msg.tool_calls:
            slug = tc.function.name
            display = SLUG_TO_DISPLAY.get(slug)
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            q = args.get("query") or args.get("input") or ""
            if display:
                try:
                    raw_out = run_tool(display, str(q))
                except Exception as e:
                    raw_out = f"Tool error: {e}"
            else:
                raw_out = f"Unknown tool slug: {slug}"

            if trace:
                trace.add(
                    "refine_tool_call",
                    round=round_i + 1,
                    slug=slug,
                    display_name=display,
                    query=str(q)[:500],
                    output_preview=str(raw_out)[:1500],
                )

            content = format_tool_output(raw_out)
            if len(content) > 12000:
                content = content[:12000] + "\n...[truncated]"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )

    return (
        "Sorry — the refinement step stopped after the maximum number of tool rounds "
        "without a final answer."
    )
