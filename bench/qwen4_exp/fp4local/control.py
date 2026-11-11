"""Fixed 4-request output control. Part of the FP4-Local measurement record (docs/checkpoints/qwen3.8-flash-next-fp4-local.md).
usage: control.py <port> <outdir>   -> <outdir>/{math,explain,code,tool}.json + summary line
temperature 0, reasoning_effort "none", max_tokens 192, distinct prompts."""
import json, os, sys, urllib.request

port, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city"]}}}]
REQS = {
    "math": {"messages": [{"role": "user", "content": "What is 37*43? Show the multiplication briefly, then the answer."}]},
    "explain": {"messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]},
    "code": {"messages": [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation."}]},
    "tool": {"messages": [{"role": "user", "content": "What's the weather in Lisbon right now, in celsius?"}], "tools": TOOLS},
}
ok = 0
for name, extra in REQS.items():
    body = {"model": "x", "temperature": 0, "max_tokens": 192, "reasoning_effort": "none", **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        m = r["choices"][0]["message"]
        rec = {"content": m.get("content"), "tool_calls": m.get("tool_calls"), "finish": r["choices"][0].get("finish_reason")}
        ok += 1
    except Exception as e:
        rec = {"error": str(e)[:300]}
    json.dump(rec, open(f"{outdir}/{name}.json", "w"), indent=1, sort_keys=True)
print(json.dumps({"outdir": outdir, "ok": ok, "of": len(REQS)}))
