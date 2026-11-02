"""Long-context passcode recall for the FP4-Local server. Part of the FP4-Local measurement record (docs/checkpoints/qwen3.8-flash-next-fp4-local.md).

usage: needle_long.py <port> <target_tokens> [<target_tokens> ...]

Filler is real English prose/markdown: every docs/**/*.md file of the Atlas
checkout (or $FP4L_CORPUS), concatenated in sorted order and repeated as needed (deterministic,
no synthetic small-vocabulary filler, which inflates n-gram cache hit rates).
For each target length the passcode is placed at depths 10%, 50% and 90%.
The request uses the recipe's thinking default (reasoning_effort "low"),
temperature 0, max_tokens 800. PASS = the exact passcode appears in content.
Prints one JSON line per trial, using the SERVER's usage.prompt_tokens.
"""
import glob, json, os, sys, time, urllib.request

port = sys.argv[1]
targets = [int(x) for x in sys.argv[2:]]
root = os.environ.get("FP4L_CORPUS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs"))  # the Atlas docs/ tree (override: FP4L_CORPUS)
text = "\n\n".join(open(p, encoding="utf-8", errors="ignore").read()
                   for p in sorted(glob.glob(f"{root}/**/*.md", recursive=True)))
# ~3.6 chars/token for this markdown on Qwen tokenizers; refined per trial below.
CPT = 3.6
CODES = ["AMBER-FALCON-731", "COBALT-RIVER-284", "VIOLET-ANCHOR-956"]


def build(n_tok, depth, code, cpt):
    need = int(n_tok * cpt)
    body = (text * (need // len(text) + 1))[:need]
    cut = body.rfind("\n", 0, int(len(body) * depth)) + 1
    needle = f"\n\nIMPORTANT NOTE: the secret passcode is {code}. Remember it.\n\n"
    return body[:cut] + needle + body[cut:]


def ask(prompt):
    body = {"model": "x", "temperature": 0, "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt +
                          "\n\nWhat is the secret passcode mentioned in the text above? "
                          "Reply with only the passcode."}],
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r, time.time() - t


cpt = CPT
for n in targets:
    for depth, code in zip([0.1, 0.5, 0.9], CODES):
        try:
            r, dt = ask(build(n, depth, code, cpt))
            u = r.get("usage", {})
            msg = r["choices"][0]["message"]
            content = msg.get("content") or ""
            pt = u.get("prompt_tokens", 0)
            if pt:
                cpt = cpt * n / pt if depth == 0.1 else cpt  # calibrate once per length
            print(json.dumps({"target": n, "prompt_tokens": pt, "depth": depth,
                              "pass": code in content, "content": content[:80],
                              "completion_tokens": u.get("completion_tokens"),
                              "ttft_ms": u.get("time_to_first_token_ms"),
                              "wall_s": round(dt, 1)}), flush=True)
        except Exception as e:  # a failed request is a FAIL, recorded, not skipped
            print(json.dumps({"target": n, "depth": depth, "pass": False,
                              "error": str(e)[:300]}), flush=True)
