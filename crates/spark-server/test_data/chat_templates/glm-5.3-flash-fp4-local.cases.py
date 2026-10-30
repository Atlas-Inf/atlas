#!/usr/bin/env python3
"""Regenerate glm-5.3-flash-fp4-local.cases.json.

Renders each OpenAI chat request the way the reference engine (vLLM, eugr
spark-vllm-b12x) does for GLM-5.3-Flash-FP4-Local: the checkpoint's own
chat_template.jinja under HF's jinja2 settings, with vLLM's kwargs rules:
  * the request's `reasoning_effort` is passed verbatim (None when absent),
  * `enable_thinking = reasoning_effort != "none"` is added when an effort is
    set and chat_template_kwargs does not carry enable_thinking,
  * chat_template_kwargs are merged underneath (None never overrides),
  * assistant `reasoning_content` is passed as both `reasoning` and
    `reasoning_content`; tool-call arguments are parsed to dicts; tool
    messages keep `tool_call_id`.

Usage: python3 glm-5.3-flash-fp4-local.cases.py  (writes the .json beside it)
"""
import copy
import json
import os

import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "glm-5.3-flash-fp4-local.jinja")


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def env():
    e = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                      extensions=[jinja2.ext.loopcontrols])
    e.filters["tojson"] = tojson
    return e


def vllm_messages(messages):
    out = []
    for m in copy.deepcopy(messages):
        if m.get("content") is None:
            m["content"] = ""
        if m["role"] == "assistant":
            reasoning = m.pop("reasoning_content", None)
            if reasoning is not None:
                m["reasoning"] = reasoning
                m["reasoning_content"] = reasoning
            for tc in m.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    tc["function"]["arguments"] = json.loads(args)
        out.append(m)
    return out


def vllm_render(template, request):
    effort = request.get("reasoning_effort")
    kwargs = dict(request.get("chat_template_kwargs") or {})
    extra = {"add_generation_prompt": True, "reasoning_effort": effort}
    if effort is not None and "enable_thinking" not in kwargs:
        extra["enable_thinking"] = effort != "none"
    kwargs |= {k: v for k, v in extra.items() if v not in (None, "auto")}
    tools = request.get("tools")
    return template.render(messages=vllm_messages(request["messages"]),
                           tools=tools, **kwargs)


WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the user's current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "City <name>"},
                                  "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get_time",
    "description": "Local time in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}

SINGLE = [{"role": "system", "content": "You are terse."},
          {"role": "user", "content": "What's the weather in Paris and the time in Oslo?"}]
MULTI = SINGLE + [
    {"role": "assistant", "content": "",
     "reasoning_content": "Two lookups; call both tools.",
     "tool_calls": [
         {"id": "call_a", "type": "function",
          "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"unit\": \"c\"}"}},
         {"id": "call_b", "type": "function",
          "function": {"name": "get_time", "arguments": "{\"city\": \"Oslo\"}"}}]},
    # Results arrive out of call order; the template sorts them by id.
    {"role": "tool", "tool_call_id": "call_b", "content": "14:05"},
    {"role": "tool", "tool_call_id": "call_a", "content": "{\"temp\": 18}"},
    {"role": "assistant", "content": "Paris is 18 C; Oslo is 14:05.",
     "reasoning_content": "Both results are in."},
    {"role": "user", "content": "Thanks. And tomorrow?"},
]

FORMS = [
    ("no_flag", {}),
    ("enable_thinking_false", {"enable_thinking": False}),
    ("kwargs_enable_thinking_false", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("kwargs_enable_thinking_true", {"chat_template_kwargs": {"enable_thinking": True}}),
    ("kwargs_effort_high", {"chat_template_kwargs": {"reasoning_effort": "high"}}),
    ("effort_low_kwargs_enable_thinking_false",
     {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}}),
] + [(f"effort_{e}", {"reasoning_effort": e})
     for e in ["none", "minimal", "low", "medium", "high", "max", "xhigh"]] + [
    ("response_format", {"response_format": {"type": "json_schema", "json_schema": {
        "name": "answer", "schema": {"type": "object", "properties": {
            "answer": {"type": "string"}}, "required": ["answer"]}}}}),
]


def cases():
    out = []
    for conv_name, messages in [("single", SINGLE), ("multi", MULTI)]:
        for tools_name, tools in [("plain", None), ("tools", [WEATHER, TIME])]:
            if conv_name == "multi" and tools is None:
                continue  # a tool round-trip needs its tools
            for form_name, form in FORMS:
                request = {"model": "glm", "messages": messages, "max_tokens": 64, **form}
                if tools:
                    request["tools"] = tools
                out.append({"name": f"{conv_name}/{tools_name}/{form_name}",
                            "request": request})
    return out


def main():
    template = env().from_string(open(TEMPLATE).read())
    rendered = []
    for case in cases():
        case["prompt"] = vllm_render(template, case["request"])
        rendered.append(case)
    path = os.path.join(HERE, "glm-5.3-flash-fp4-local.cases.json")
    with open(path, "w") as f:
        json.dump(rendered, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(rendered)} cases -> {path}")


if __name__ == "__main__":
    main()
