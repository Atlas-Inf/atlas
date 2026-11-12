#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Replay BFCL live_irrelevance items against an OpenAI-compatible server and record, per item, whether the reply
carries tool_calls (a FAIL on this subset: the right answer is to call nothing), plus the raw content.

    irrelevance_probe.py <port> <model-id> <BFCL_v4_live_irrelevance.json> <out.jsonl> [n]

Tools are passed the way BFCL's FC mode passes them (OpenAI `tools`), after its type mapping (dict->object,
float->number, tuple->array, any->string). Greedy, thinking off, max_tokens 512.
"""
import json
import sys
import urllib.request

port, model, data_path, out_path = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 60
TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "list": "array", "int": "integer",
         "str": "string", "bool": "boolean"}


def fix(schema):
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = TYPES.get(v, v)
            else:
                out[k] = fix(v)
        return out
    if isinstance(schema, list):
        return [fix(v) for v in schema]
    return schema


items = [json.loads(line) for line in open(data_path) if line.strip()][:n]
called = 0
with open(out_path, "w") as out:
    for it in items:
        tools = [{"type": "function", "function": {"name": f["name"].replace(".", "_"), "description": f.get("description", ""),
                                                   "parameters": fix(f.get("parameters", {"type": "object", "properties": {}}))}}
                 for f in it["function"]]
        messages = it["question"][0] if it["question"] and isinstance(it["question"][0], list) else it["question"]
        req = {"model": model, "messages": messages, "tools": tools, "temperature": 0, "max_tokens": 512,
               "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(req).encode(),
                headers={"Content-Type": "application/json"}), timeout=600).read())
            msg = r["choices"][0]["message"]
            tc = msg.get("tool_calls") or []
            rec = {"id": it["id"], "tool_calls": [c["function"]["name"] for c in tc], "content": (msg.get("content") or "")[:600],
                   "finish": r["choices"][0].get("finish_reason")}
        except Exception as e:  # a server error is a result too
            rec = {"id": it["id"], "error": str(e)[:300], "tool_calls": []}
        called += bool(rec["tool_calls"])
        out.write(json.dumps(rec) + "\n")
print(f"  {model}: {called}/{len(items)} items answered with a tool call (each one a live_irrelevance FAIL)")
