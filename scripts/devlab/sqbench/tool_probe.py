#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""tool_probe.py TAG [RUNS] -- OpenAI tool calling against the engine, end to end.

Seven cases, each once greedy and RUNS-1 more times with the model's default sampling:
  single     one call with a string and an enum argument
  typed      integer arguments arrive as JSON numbers
  parallel   two independent calls in one turn
  followup   a tool result in, a final answer out (no further call)
  no_tool    tools offered, none needed: a plain answer
  choice_none  tool_choice "none": no call even when one fits
  stream     the single case streamed: the deltas assemble into the same call
Writes ./tools-TAG.json. SQ_URL (default http://127.0.0.1:8893) and SQ_MODEL select the server;
SQ_API_KEY, when set, is sent as a bearer token."""
import json, os, sys, urllib.request

URL = os.environ.get("SQ_URL", "http://127.0.0.1:8893") + "/v1/chat/completions"
MODEL = os.environ.get("SQ_MODEL", "qwen3.8-flash-next-atlas")
TAG = sys.argv[1]
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}}}
ADD = {"type": "function", "function": {
    "name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}


def post(body, stream=False):
    body = {"model": MODEL, "max_tokens": 1024, **body}
    if stream:
        body["stream"] = True
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SQ_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SQ_API_KEY"]
    req = urllib.request.Request(URL, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=600) as resp:
        if not stream:
            return json.load(resp)["choices"][0]
        calls, content, finish = {}, "", None
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:]).get("choices") or []
            if not ch:
                continue
            delta = ch[0].get("delta", {})
            finish = ch[0].get("finish_reason") or finish
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                c["arguments"] += fn.get("arguments") or ""
        return {"finish_reason": finish, "message": {"content": content, "tool_calls": [
            {"function": calls[i]} for i in sorted(calls)]}}


def calls(choice):
    out = []
    for tc in choice["message"].get("tool_calls") or []:
        fn = tc["function"]
        try:
            args = json.loads(fn["arguments"] or "{}")
        except json.JSONDecodeError:
            args = None
        out.append((fn["name"], args))
    return out


def sampling(greedy):
    return {"temperature": 0} if greedy else {}


def single(greedy, stream=False):
    ch = post({"tools": [WEATHER], "messages": [{"role": "user", "content":
              "What's the weather in Oslo right now? Use celsius."}], **sampling(greedy)}, stream)
    c = calls(ch)
    ok = (len(c) == 1 and c[0][0] == "get_weather" and isinstance(c[0][1], dict)
          and "oslo" in str(c[0][1].get("city", "")).lower() and c[0][1].get("unit") == "celsius"
          and ch["finish_reason"] == "tool_calls")
    return ok, c, ch["finish_reason"]


def typed(greedy):
    ch = post({"tools": [ADD], "messages": [{"role": "user", "content":
              "Use the add tool to add 17 and 25."}], **sampling(greedy)})
    c = calls(ch)
    ok = (len(c) == 1 and c[0][0] == "add" and isinstance(c[0][1], dict)
          and sorted([c[0][1].get("a"), c[0][1].get("b")], key=str) == [17, 25]
          and all(type(c[0][1].get(k)) is int for k in "ab"))
    return ok, c, ch["finish_reason"]


def parallel(greedy):
    ch = post({"tools": [WEATHER], "messages": [{"role": "user", "content":
              "Get the weather in Oslo and in Lima, both in celsius, with one tool call for each city."}],
               **sampling(greedy)})
    c = calls(ch)
    cities = sorted(str(a.get("city", "")).lower() for n, a in c if n == "get_weather" and isinstance(a, dict))
    ok = len(c) == 2 and any("oslo" in x for x in cities) and any("lima" in x for x in cities)
    return ok, c, ch["finish_reason"]


def followup(greedy):
    msgs = [{"role": "user", "content": "What's the weather in Oslo right now? Use celsius."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo", "unit": "celsius"})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"temperature_c": 7, "sky": "light rain"})}]
    ch = post({"tools": [WEATHER], "messages": msgs, **sampling(greedy)})
    text = ch["message"].get("content") or ""
    ok = not calls(ch) and "7" in text
    return ok, text[-200:], ch["finish_reason"]


def no_tool(greedy):
    ch = post({"tools": [WEATHER], "messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number."}],
               **sampling(greedy)})
    text = ch["message"].get("content") or ""
    return not calls(ch) and "4" in text, text[-120:], ch["finish_reason"]


def choice_none(greedy):
    ch = post({"tools": [WEATHER], "tool_choice": "none", "messages": [{"role": "user", "content":
              "What's the weather in Oslo right now? Use celsius."}], **sampling(greedy)})
    return not calls(ch), (ch["message"].get("content") or "")[-120:], ch["finish_reason"]


CASES = {"single": single, "typed": typed, "parallel": parallel, "followup": followup,
         "no_tool": no_tool, "choice_none": choice_none, "stream": lambda g: single(g, stream=True)}
rows, summary = [], {}
for name, fn in CASES.items():
    passed = 0
    for run in range(RUNS):
        greedy = run == 0
        try:
            ok, detail, finish = fn(greedy)
        except Exception as e:  # noqa: BLE001 -- a probe records every failure
            ok, detail, finish = False, f"error: {e}", None
        passed += ok
        rows.append({"case": name, "run": run, "greedy": greedy, "ok": ok, "finish": finish, "detail": detail})
    summary[name] = f"{passed}/{RUNS}"
json.dump({"tag": TAG, "model": MODEL, "summary": summary, "rows": rows}, open(f"tools-{TAG}.json", "w"), indent=1)
print(TAG, " ".join(f"{k} {v}" for k, v in summary.items()))
