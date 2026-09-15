#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""One BFCL relevant-control probe for the Strix bisection.

Posts a single dataset row (default live_multiple_4-2-1) to a running Atlas
serve at temperature 0 and reports wall-to-first-token, total wall, the
assembled text head, and any parsed tool calls, so a serve-path pathology
(prefill stall, degenerate first token, decode loop) is visible per row.
"""
import json
import sys
import time
import urllib.request

url, model, root, sample_id, max_tokens = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4] if len(sys.argv) > 4 else "live_multiple_4-2-1",
    int(sys.argv[5]) if len(sys.argv) > 5 else 256,
)

row = None
with open(f"{root}/dataset.jsonl", encoding="utf-8") as handle:
    for line in handle:
        r = json.loads(line)
        if r["sample_id"] == sample_id:
            row = r
            break
if row is None:
    sys.exit(f"sample {sample_id} not found in {root}/dataset.jsonl")

body = {
    "model": model,
    "stream": True,
    "temperature": 0.0,
    "max_tokens": max_tokens,
    "messages": row["messages"],
    "tools": row["tools"],
    "tool_choice": row["tool_choice"],
}
req = urllib.request.Request(
    url.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)

started = time.time()
first_token_s = None
text_parts = []
tool_calls = []
finish = None
try:
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data:"):])
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                if first_token_s is None:
                    first_token_s = time.time() - started
                text_parts.append(content)
            for tc in delta.get("tool_calls") or []:
                if first_token_s is None:
                    first_token_s = time.time() - started
                tool_calls.append(tc)
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
except Exception as exc:  # noqa: BLE001 - diagnostic probe, print and continue
    print(f"PROBE ERROR: {exc!r}")

total = time.time() - started
text = "".join(text_parts)
print(f"sample={sample_id} total_wall={total:.1f}s first_token={first_token_s}s finish={finish}")
print(f"tool_calls={json.dumps(tool_calls)[:400]}")
print(f"text_head={text[:300]!r}")
