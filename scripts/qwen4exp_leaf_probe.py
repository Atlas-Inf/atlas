#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""qwen4exp_leaf_probe.py [URL] [CTX_KEYS] -- exercise the qwen4_exp decode leaf
(ATLAS_QWEN4EXP_FINISH_LEAF=1 with its preconditions).

The leaf only serves a next turn whose prompt reproduces the previous output
token for token past the leaf's block, so this probe sends the assistant's full
content back verbatim, with thinking off (a template that drops reasoning from
history would not reproduce it). Turn 1: a long registry context and a request
for ~400 tokens of output. Turn 2: the same messages, the reply verbatim, and a
short question. Prints turn-2 TTFT and cached tokens; the server log shows where
it restored ("Marconi ... restored from checkpoint at token N" / "cache hit").
CTX_KEYS 2000 is ~20K tokens (one chunk); 7000 is ~70K (several)."""
import json, random, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8893/v1/chat/completions"
KEYS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
KW = {"enable_thinking": False, "reasoning_effort": "low"}


def chat(msgs, mt):
    body = {"model": "qwen3.8-flash-next-atlas", "messages": msgs, "max_tokens": mt, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": KW}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, out, usage = time.time(), None, [], {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content"):
                    first = first or time.time()
                out.append(dl.get("content") or "")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"prompt": usage.get("prompt_tokens"), "cached": cached, "n": usage.get("completion_tokens"),
            "ttft": round((first or time.time()) - t0, 2)}, "".join(out)


r = random.Random(41)
reg = "\n".join(f"k{r.randrange(10**6):06d} => {r.randrange(10**8):08d}" for _ in range(KEYS))
msgs = [{"role": "user", "content": reg + "\n\nWrite about 350 words on how a team should maintain a registry "
                                          "like this one. Plain prose, no lists."}]
s1, reply = chat(msgs, 450)
print("turn1", s1)
msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": "Summarize that in one sentence."}]
s2, _ = chat(msgs, 60)
print("turn2", s2)
