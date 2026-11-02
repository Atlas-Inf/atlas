#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Reference prompt renders for Qwen3.8-Flash-Next (nvidia FP4-Local checkpoint).

WHY THIS EXISTS
---------------
The reference server for this checkpoint is vLLM (`--reasoning-parser qwen3
--tool-call-parser qwen3_xml --enable-auto-tool-choice`, eugr/spark-vllm-b12x).
It renders the checkpoint's own chat_template.jinja through transformers with
the kwargs vLLM derives from the request, and nothing else touches the prompt.
This script reproduces that render for a set of OpenAI chat requests and writes
them, request + expected prompt text, to a JSON fixture that
`crates/spark-server/src/api/chat/reference_prompt_tests.rs` replays through
Atlas's own request path. Any byte Atlas adds, drops or reorders fails there.

vLLM's request -> template mapping (vllm 8244d1c4, renderers/ and
entrypoints/openai/chat_completion/protocol.py `build_chat_params`):
  * kwargs = chat_template_kwargs | {add_generation_prompt: True,
    reasoning_effort: <request field>} with None values dropped;
  * a request `reasoning_effort` also sets enable_thinking = (effort != "none")
    unless chat_template_kwargs carries enable_thinking;
  * tools are `[tool.model_dump()]` ({type, function: {name, description,
    parameters}}), passed even under tool_choice "none";
  * assistant `reasoning`/`reasoning_content` reaches the template as
    reasoning_content; tool-call arguments are parsed to dicts;
  * response_format and thinking_token_budget do not change the prompt.

Regenerate (needs transformers >= 4.45 and jinja2):
    python3 scripts/qwen38_flash_next_template_golden.py
"""
import copy
import json
import pathlib

from transformers.utils.chat_template_utils import render_jinja_template

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "crates/spark-server/test_data/chat_templates"
TEMPLATE = DATA / "qwen3.8-flash-next-fp4-local.jinja"
OUT = DATA / "qwen3.8-flash-next-fp4-local.cases.json"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns <temp> & sky, e.g. \"18C\" in Zürich.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}
SHELL = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "timeout": {"type": "integer"},
                "env": {"type": "object"},
                "dry": {"type": "boolean"},
            },
            "required": ["cmd"],
        },
    },
}
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
USER = {"role": "user", "content": "What is 12 x 8?"}


def call(cid, name, args):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


TOOL_TURNS = [
    {"role": "user", "content": "Weather in Paris and Rome, then summarize."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Two cities, two calls.",
        "tool_calls": [
            call("c1", "get_weather", {"city": "Paris"}),
            call("c2", "get_weather", {"city": "Rome", "unit": "c"}),
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "18C, sunny"},
    {"role": "tool", "tool_call_id": "c2", "content": "24C, cloudy"},
    {
        "role": "assistant",
        "content": "Paris 18C sunny; Rome 24C cloudy.",
        "reasoning": "Both results are in; summarize.",
    },
    {"role": "user", "content": "Thanks - which is warmer?"},
]

CASES = [
    ("plain_default", {"messages": [SYSTEM, USER]}),
    ("no_system_default", {"messages": [USER]}),
    ("effort_low", {"messages": [SYSTEM, USER], "reasoning_effort": "low"}),
    ("effort_medium", {"messages": [SYSTEM, USER], "reasoning_effort": "medium"}),
    ("effort_xhigh", {"messages": [USER], "reasoning_effort": "xhigh"}),
    ("effort_none", {"messages": [SYSTEM, USER], "reasoning_effort": "none"}),
    (
        "kwargs_effort_low",
        {"messages": [USER], "chat_template_kwargs": {"reasoning_effort": "low"}},
    ),
    (
        "kwargs_thinking_off",
        {"messages": [SYSTEM, USER], "chat_template_kwargs": {"enable_thinking": False}},
    ),
    (
        "kwargs_thinking_on",
        {"messages": [USER], "chat_template_kwargs": {"enable_thinking": True}},
    ),
    ("thinking_token_budget", {"messages": [USER], "thinking_token_budget": 64}),
    ("tools_auto", {"messages": [SYSTEM, USER], "tools": [WEATHER, SHELL]}),
    ("tools_no_system", {"messages": [USER], "tools": [WEATHER]}),
    (
        "tools_effort_low",
        {"messages": [SYSTEM, USER], "tools": [WEATHER], "reasoning_effort": "low"},
    ),
    (
        "tools_thinking_off",
        {
            "messages": [USER],
            "tools": [WEATHER],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ),
    (
        "tools_required",
        {"messages": [USER], "tools": [WEATHER], "tool_choice": "required"},
    ),
    (
        "tools_named",
        {
            "messages": [USER],
            "tools": [WEATHER, SHELL],
            "tool_choice": {"type": "function", "function": {"name": "run"}},
        },
    ),
    ("tool_multiturn", {"messages": [SYSTEM] + TOOL_TURNS, "tools": [WEATHER]}),
    (
        "tool_multiturn_preserve_off",
        {
            "messages": [SYSTEM] + TOOL_TURNS,
            "tools": [WEATHER],
            "chat_template_kwargs": {"preserve_thinking": False},
        },
    ),
    (
        "tool_multiturn_thinking_off",
        {
            "messages": [SYSTEM] + TOOL_TURNS,
            "tools": [WEATHER],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ),
    (
        "tool_result_pending",
        {"messages": [SYSTEM] + TOOL_TURNS[:4], "tools": [WEATHER]},
    ),
    (
        "assistant_text_and_typed_args",
        {
            "messages": [
                USER,
                {
                    "role": "assistant",
                    "content": "Let me run that.",
                    "tool_calls": [
                        call(
                            "r1",
                            "run",
                            {"cmd": "ls -la", "timeout": 30, "env": {"A": "1"}, "dry": True},
                        )
                    ],
                },
                {"role": "tool", "tool_call_id": "r1", "content": "total 0"},
            ],
            "tools": [SHELL],
        },
    ),
    (
        "tool_error_text",
        {
            "messages": [
                {"role": "system", "content": "You are a coding agent.\nWorking directory: /work/app"},
                {"role": "user", "content": "List the files."},
                {"role": "assistant", "content": "", "tool_calls": [call("e1", "run", {"cmd": "lss"})]},
                {"role": "tool", "tool_call_id": "e1", "content": "Error: bash: lss: command not found (exit code 127)"},
                {"role": "assistant", "content": "", "tool_calls": [call("e2", "run", {"cmd": "lss"})]},
                {"role": "tool", "tool_call_id": "e2", "content": "Error: bash: lss: command not found (exit code 127)"},
            ],
            "tools": [SHELL],
        },
    ),
    (
        "repeated_tool_rounds",
        {
            "messages": [
                {"role": "user", "content": "Check the weather in Oslo."},
                {"role": "assistant", "content": "", "tool_calls": [call("o1", "get_weather", {"city": "Oslo"})]},
                {"role": "tool", "tool_call_id": "o1", "content": "3C, rain"},
                {"role": "assistant", "content": "", "tool_calls": [call("o2", "get_weather", {"city": "Oslo"})]},
                {"role": "tool", "tool_call_id": "o2", "content": "3C, rain"},
                {"role": "assistant", "content": "", "tool_calls": [call("o3", "get_weather", {"city": "Oslo"})]},
                {"role": "tool", "tool_call_id": "o3", "content": "3C, rain"},
            ],
            "tools": [WEATHER],
        },
    ),
    (
        "content_free_system",
        {"messages": [{"role": "system", "content": "User Context:\n\n"}, USER]},
    ),
    (
        "response_format_json_schema",
        {
            "messages": [USER],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                },
            },
        },
    ),
    (
        "tools_and_response_format",
        {
            "messages": [USER],
            "tools": [WEATHER],
            "response_format": {"type": "json_object"},
        },
    ),
    (
        "tool_choice_none",
        {"messages": [USER], "tools": [WEATHER], "tool_choice": "none"},
    ),
]


def template_kwargs(req):
    """vLLM ChatCompletionRequest.build_chat_params + merge_kwargs."""
    user = dict(req.get("chat_template_kwargs") or {})
    extra = {
        "add_generation_prompt": True,
        "reasoning_effort": req.get("reasoning_effort"),
    }
    if req.get("reasoning_effort") is not None and "enable_thinking" not in user:
        extra["enable_thinking"] = req["reasoning_effort"] != "none"
    return user | {k: v for k, v in extra.items() if v not in (None, "auto")}


def template_messages(req):
    """vLLM chat_utils parse + _postprocess_messages."""
    out = []
    for m in copy.deepcopy(req["messages"]):
        reasoning = m.pop("reasoning", None)
        if reasoning is None:
            reasoning = m.pop("reasoning_content", None)
        m.pop("reasoning_content", None)
        if reasoning is not None:
            m["reasoning"] = reasoning
            m["reasoning_content"] = reasoning
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        out.append(m)
    return out


def main():
    template = TEMPLATE.read_text()
    cases = []
    for name, body in CASES:
        req = {"model": "qwen3.8-flash-next", "max_tokens": 4096, **body}
        tools = req.get("tools")
        expected = render_jinja_template(
            conversations=[template_messages(req)],
            tools=tools,
            chat_template=template,
            **template_kwargs(req),
        )[0][0]
        cases.append({"name": name, "request": req, "expected": expected})
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
