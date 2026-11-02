#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from Jason McCartney / Enntity, commit ece58a9e2a0f5bff45a58b60f58f0264302f238d.
"""tools + response_format probe (ToolEvalBench TC-65..69 shape).

Two turns against a live OpenAI-compatible server, both carrying `tools` AND a
strict `response_format` json_schema with tool_choice unset (auto):

  1. the user asks for the weather -> the model should call get_weather
  2. the tool result is fed back  -> the final content must be the bare JSON
     object: json.loads() succeeds (no ```json fence, no prose) and it
     validates against the schema (jsonschema when installed, else key /
     type / enum checks).

Usage: tools_rfmt_probe.py [--url http://127.0.0.1:8081] [--model M]
                           [--runs N] [--thinking]
Exit status: 0 when every run's final answer is schema-valid JSON.
"""
import argparse
import json
import sys
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

# Nested object + enum + integer: TC-66 / TC-67 constraints in one schema.
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "city": {"type": "string"},
        "temperature_c": {"type": "integer"},
        "conditions": {"type": "string", "enum": ["sunny", "cloudy", "rain", "snow"]},
        "wind": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "speed_kmh": {"type": "number"},
                "direction": {"type": "string"},
            },
            "required": ["speed_kmh", "direction"],
        },
    },
    "required": ["city", "temperature_c", "conditions", "wind"],
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "weather_report", "strict": True, "schema": SCHEMA},
}

TOOL_RESULT = {"city": "Paris", "temp_c": 18, "sky": "cloudy",
               "wind_kmh": 12.5, "wind_dir": "NW"}


def post(url, body):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def first_model(url):
    with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def check_schema(obj, schema, path="$"):
    """Minimal fallback validator: types, required, enum, no extra keys."""
    types = {"object": dict, "string": str, "array": list, "boolean": bool}
    t = schema.get("type")
    if t == "integer":
        if not isinstance(obj, int) or isinstance(obj, bool):
            return [f"{path}: expected integer, got {obj!r}"]
    elif t == "number":
        if not isinstance(obj, (int, float)) or isinstance(obj, bool):
            return [f"{path}: expected number, got {obj!r}"]
    elif t in types and not isinstance(obj, types[t]):
        return [f"{path}: expected {t}, got {type(obj).__name__}"]
    errs = []
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if t == "object":
        props = schema.get("properties", {})
        errs += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in obj]
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected {k}" for k in obj if k not in props]
        for k, v in obj.items():
            if k in props:
                errs += check_schema(v, props[k], f"{path}.{k}")
    return errs


def validate(obj):
    try:
        import jsonschema
    except ImportError:
        return check_schema(obj, SCHEMA)
    v = jsonschema.Draft202012Validator(SCHEMA)
    return [f"{'/'.join(map(str, e.path)) or '$'}: {e.message}" for e in v.iter_errors(obj)]


def run_once(args, model):
    base = {"model": model, "tools": TOOLS, "response_format": RESPONSE_FORMAT,
            "temperature": 0, "max_tokens": 4096}
    if args.thinking is not None:
        base["chat_template_kwargs"] = {"enable_thinking": args.thinking}
    messages = [{"role": "user", "content":
                 "What's the weather in Paris right now? Use the tool, then report it."}]
    r1 = post(args.url, {**base, "messages": messages})["choices"][0]
    calls = r1["message"].get("tool_calls") or []
    if calls:
        print(f"  turn 1: tool call {calls[0]['function']['name']}"
              f"({calls[0]['function']['arguments']})")
        messages.append({"role": "assistant", "content": r1["message"].get("content") or "",
                         "tool_calls": calls})
        for c in calls:
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(TOOL_RESULT)})
        final = post(args.url, {**base, "messages": messages})["choices"][0]
    else:
        print("  turn 1: answered without a tool call; checking that answer")
        final = r1
    content = final["message"].get("content") or ""
    print(f"  final ({final.get('finish_reason')}): {content[:300]!r}")
    if final["message"].get("tool_calls"):
        return ["final turn called a tool again"]
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        return [f"not valid JSON: {e}"]
    return validate(obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                    help="send chat_template_kwargs.enable_thinking (default: server's)")
    args = ap.parse_args()
    args.url = args.url.rstrip("/")
    model = args.model or first_model(args.url)
    failed = 0
    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs} ({model})")
        errs = run_once(args, model)
        print("  PASS" if not errs else "  FAIL: " + "; ".join(errs))
        failed += bool(errs)
    print(f"{args.runs - failed}/{args.runs} schema-valid final answers")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
