"""End-to-end biomedical query pipeline without LangChain orchestration."""

from __future__ import annotations

import re
from typing import Any

from config import Settings, get_settings
from llm import ChatLLM
from planner import QueryPlan, create_plan
from prompts import (
    EVALUATION_SYSTEM_PROMPT,
    RESPONSE_NON_MEDICAL_SYSTEM_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
)
from refine_loop import refine_with_openai_tools
from tools import run_tool
from trace import TraceRecorder


def _is_medical(llm: ChatLLM, query: str) -> bool:
    text = (
        f"Please determine whether the following query is related to medicine: "
        f"'{query}', only answer 'Yes' or 'No'"
    )
    user_only = [{"role": "user", "content": text}]
    out = llm.chat(
        messages=user_only,
        temperature=0.0,
        trace_phase="medical_gate",
        trace_extra={"query": query},
    )
    return "yes" in out.strip().lower()


def _eval_verdict(llm: ChatLLM, query: str, response: str) -> str:
    eval_prompt = (
        f"Please evaluate: Retrieved answer to the query '{query}' "
        f"is '{response}', only answer 'Yes' or 'No'"
    )
    messages = [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": eval_prompt},
    ]
    out = llm.chat(
        messages=messages,
        temperature=0.0,
        trace_phase="evaluation",
        trace_extra={"query": query},
    )
    ans = re.findall(r"yes|no", out.lower())
    return ans[0] if ans else "yes"


def _extract_with_llm(llm: ChatLLM, *, raw: str, extract: str, tool: str, step: int) -> str:
    # Keep extraction prompt bounded; tool outputs can be very large.
    raw_preview = raw[:8000]
    messages = [
        {
            "role": "system",
            "content": (
                "You extract specific information from tool outputs.\n"
                "Return ONLY the extracted value(s) as plain text.\n"
                "If multiple values are needed, return a comma-separated list.\n"
                "If nothing relevant is found, return exactly: NOT_FOUND"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Tool: {tool}\n"
                f"Step: {step}\n"
                f"Extraction target: {extract}\n\n"
                f"Tool output:\n{raw_preview}"
            ),
        },
    ]
    out = llm.chat(
        messages=messages,
        temperature=0.0,
        trace_phase="extractor",
        trace_extra={"step": step, "tool": tool, "extract": extract},
    ).strip()
    if out.upper() == "NOT_FOUND":
        return ""
    return out


def _execute_plan(plan: QueryPlan, original_query: str, trace: TraceRecorder, llm: ChatLLM) -> str:
    step_results: dict[int, str] = {}

    for step in plan.steps:
        step_input = step.input
        if step.depends_on is not None:
            placeholder = f"output_of_step_{step.depends_on}"
            if placeholder in step_input:
                prior = step_results.get(step.depends_on, "")
                step_input = step_input.replace(placeholder, str(prior))

        parts = [
            f"Use the '{step.tool}' tool.",
            f"Input: {step_input}.",
            f"Goal: {step.purpose}.",
        ]
        if step.extract:
            parts.append(f"From the result, extract: {step.extract}.")
        instruction = " ".join(parts)

        trace.add(
            "plan_step_start",
            step=step.step,
            total=len(plan.steps),
            tool=step.tool,
            resolved_input=step_input,
            instruction_preview=instruction[:500],
        )

        raw = run_tool(step.tool, step_input)
        forwarded = str(raw or "")
        if step.extract:
            extracted = _extract_with_llm(
                llm,
                raw=forwarded,
                extract=step.extract,
                tool=step.tool,
                step=step.step,
            )
            if extracted:
                forwarded = extracted
        trace.add(
            "plan_step_tool",
            step=step.step,
            tool=step.tool,
            input=step_input,
            output_preview=str(raw)[:2000],
            output_chars=len(str(raw)),
            extract=step.extract,
            forwarded_preview=forwarded[:500],
        )
        step_results[step.step] = forwarded

    return step_results.get(len(plan.steps), "")


def _validate(llm: ChatLLM, wrapped: str) -> str:
    messages = [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": wrapped},
    ]
    return llm.chat(
        messages=messages,
        temperature=1.0,
        trace_phase="validation",
    )


def run_pipeline(
    user_question: str,
    *,
    settings: Settings | None = None,
    trace: TraceRecorder | None = None,
) -> tuple[str, TraceRecorder]:
    settings = settings or get_settings()
    trace = trace or TraceRecorder()
    llm = ChatLLM(settings, trace=trace)

    if not _is_medical(llm, user_question):
        messages = [
            {"role": "system", "content": RESPONSE_NON_MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]
        out = llm.chat(
            messages=messages,
            temperature=0.7,
            trace_phase="non_medical_response",
        )
        trace.add("done", route="non_medical", answer_preview=out[:500])
        return out, trace

    plan = create_plan(llm, user_question, trace=trace)
    trace.add("plan_ready", plan_text=str(plan))

    response = _execute_plan(plan, user_question, trace, llm)

    max_iterations = 2
    num_try = 0
    hasno_flag = True

    while num_try < max_iterations and hasno_flag:
        num_try += 1
        hasno_flag = False
        try:
            verdict_raw = _eval_verdict(llm, user_question, response)
        except Exception:
            verdict_raw = "yes"
        verdict = "no" if verdict_raw == "no" else "yes"

        trace.add("evaluation_summary", iteration=num_try, verdict=verdict)

        if verdict == "no":
            refine_instruction = (
                f"The previous answer to the question '{user_question}' was: "
                f"'{response}'. This answer is incomplete or insufficiently "
                f"detailed. Please use the appropriate tools to retrieve "
                f"more information and provide a complete, accurate answer "
                f"to: '{user_question}'."
            )
            trace.add(
                "refine_attempt",
                iteration=num_try,
                instruction_preview=refine_instruction[:400],
                mechanism="openai_tool_loop",
            )
            response = refine_with_openai_tools(
                client=llm.client,
                model=llm.model,
                user_instruction=refine_instruction,
                trace=trace,
            )
            trace.add(
                "refine_result",
                iteration=num_try,
                output_preview=str(response)[:2000],
            )
            hasno_flag = True

    final_wrapped = f'The answer to "{user_question}" is {response}.'
    final_answer = _validate(llm, final_wrapped)
    trace.add("done", route="medical", answer_preview=final_answer[:500])
    return final_answer, trace

def trace_as_json_serializable(trace: TraceRecorder) -> list[dict[str, Any]]:
    """Flatten trace for logging; drop huge raw_completion if present."""
    out: list[dict[str, Any]] = [
        {
            "phase": "_summary",
            "total_llm_cost_usd": round(trace.total_llm_cost_usd, 8),
            "note": "total_llm_cost_usd sums cost_usd on LLM completions (OpenRouter usage.cost when present).",
        }
    ]
    for row in trace.as_dicts():
        if row.get("raw_completion"):
            row = {**row, "raw_completion": "<omitted>"}
        out.append(row)
    return out
