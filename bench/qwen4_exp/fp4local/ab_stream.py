"""Engine-neutral streaming A/B client (Atlas vs vLLM). Part of the FP4-Local measurement record (docs/checkpoints/qwen3.8-flash-next-fp4-local.md).

usage: ab_stream.py <port> <model> <outdir>

Identical requests to any OpenAI-compatible server, timed on the CLIENT clock
from the SSE stream, so both engines are measured the same way:
  TTFT    = first delta carrying content OR reasoning text
  decode  = (completion_tokens - 1) / (t_last_delta - t_first_delta)
completion_tokens comes from the server's final usage chunk (stream_options
include_usage), and includes reasoning tokens on both engines.
Workloads (each 3 runs, distinct prompt per run so no prefix-cache reuse):
  short  — short chat questions, max_tokens 512
  agent  — ~14k-token agent first turn: long system prompt of tool schemas +
           project docs + a user task, with tools attached, max_tokens 512
Both send chat_template_kwargs {"reasoning_effort": "low"} (Atlas's recipe
default) and temperature 0. Writes <outdir>/ab.jsonl and prints a summary.
"""
import glob, json, os, statistics, sys, time, urllib.request

port, model, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
corpus = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read()
                     for p in sorted(glob.glob(os.path.join(os.environ.get("FP4L_CORPUS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs")), "**", "*.md"), recursive=True)))

SHORT = ["What is the capital of Australia, and why was it chosen?",
         "Explain the difference between TCP and UDP in a short paragraph.",
         "Give me three tips for writing clear commit messages."]
TOOLS = [{"type": "function", "function": {
    "name": f"tool_{i}", "description": f"Utility {i}: performs operation {i} on the workspace and returns JSON.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file path"},
                   "query": {"type": "string", "description": "search query"},
                   "limit": {"type": "integer", "description": "max results"}}, "required": ["path"]}}}
    for i in range(24)]


def agent_msgs(run):
    # ~14k prompt tokens: ~13k of docs (offset per run so prompts differ) + tool schemas.
    start = 50_000 * run
    docs = corpus[start:start + 46_000]
    sys_p = ("You are a coding agent working in a repository. Use the tools when needed. "
             "Project documentation follows.\n\n" + docs)
    return [{"role": "system", "content": sys_p},
            {"role": "user", "content": "Summarize the build and serve workflow described above in five bullet "
                                        "points, then say which tool you would call first and why."}]


def stream(messages, max_tokens, tools=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None; n_delta = 0; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece or d.get("tool_calls"):
                    now = time.time(); n_delta += 1
                    t_first = t_first or now; t_last = now
                    text.append(piece)
    ct = (usage or {}).get("completion_tokens") or 0
    ct_from = "usage"
    if not ct:  # server sent no usage chunk: fall back to delta count (a lower bound under MTP)
        ct, ct_from = n_delta, "deltas"
    pt = (usage or {}).get("prompt_tokens") or 0
    dec = (ct - 1) / (t_last - t_first) if (t_first and t_last and t_last > t_first and ct > 1) else None
    return {"prompt_tokens": pt, "completion_tokens": ct, "ct_from": ct_from,
            "ttft_s": round(t_first - t0, 3) if t_first else None,
            "decode_tok_s": round(dec, 2) if dec else None, "wall_s": round(time.time() - t0, 2),
            "deltas": n_delta, "text_head": "".join(text)[:120]}


rows = []
with open(f"{outdir}/ab.jsonl", "w") as f:
    stream([{"role": "user", "content": "Say hi."}], 16)  # warm-up, unmeasured
    for i, q in enumerate(SHORT):
        r = {"workload": "short", "run": i, **stream([{"role": "user", "content": q}], 512)}
        rows.append(r); f.write(json.dumps(r) + "\n"); f.flush(); print(json.dumps(r))
    for i in range(3):
        r = {"workload": "agent", "run": i, **stream(agent_msgs(i), 512, TOOLS)}
        rows.append(r); f.write(json.dumps(r) + "\n"); f.flush(); print(json.dumps(r))
for w in ("short", "agent"):
    rs = [r for r in rows if r["workload"] == w and r["decode_tok_s"]]
    if rs:
        print(json.dumps({"summary": w, "n": len(rs),
                          "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in rs),
                          "ttft_s_median": statistics.median(r["ttft_s"] for r in rs),
                          "prompt_tokens": [r["prompt_tokens"] for r in rs],
                          "completion_tokens": [r["completion_tokens"] for r in rs]}))
