#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""calib.py ATLAS_REPO CAPTURE_DIR [URL] -- the calibration prefills for `requant.py capture-gptq`.

Serve the FP8 checkpoint from a build with capture_hook.diff applied and
ATLAS_GDN_CAPTURE=CAPTURE_DIR set, start `requant.py capture-gptq`, then run
this. 40 prefill-only requests of 12K characters, about 125K tokens on Ubuntu
24.04: 20 Python files from /usr/lib/python3/dist-packages, 6 Rust files and 4
CUDA files from the engine source, 10 Markdown documents from its docs/ (4 of
them as User:/Assistant: turns). None of it is in the held-out NLL
set (the python3.12 stdlib, /usr/share docs and licenses, pydoc topics, GSM8K
questions). After each request it waits for the consumer to drain the capture
files, so the rows never pile up on disk; at the end it writes CAPTURE_DIR/DONE."""
import glob, json, os, random, sys, time, urllib.request

repo, cap = sys.argv[1], sys.argv[2]
url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8897"
r = random.Random(17)


def pick(pattern, n, min_size=16000):
    files = sorted(f for f in glob.glob(pattern, recursive=True) if os.path.getsize(f) > min_size)
    return r.sample(files, n)


texts = []
sources = [(pick("/usr/lib/python3/dist-packages/**/*.py", 20), False),
           (pick(f"{repo}/crates/**/*.rs", 6), False),
           (pick(f"{repo}/kernels/**/*.cu", 4), False),
           (pick(f"{repo}/docs/**/*.md", 10, 14000), True)]
for files, prose in sources:
    for j, f in enumerate(files):
        t = open(f, errors="ignore").read()
        i = r.randrange(0, max(1, len(t) - 12000))
        t = t[i:i + 12000]
        if prose and j % 3 == 0:
            t = f"User: Can you walk me through this part of the documentation?\n\nAssistant: {t}\n"
        texts.append(t)
tokens = 0
for t in texts:
    body = {"model": "qwen3.8-27b-atlas", "prompt": t, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    tokens += json.load(urllib.request.urlopen(req, timeout=900))["usage"]["prompt_tokens"]
    while any(f.endswith(".bin") for f in os.listdir(cap)):
        time.sleep(0.5)
open(os.path.join(cap, "DONE"), "w").write(str(tokens))
print("calibration prompt tokens", tokens, "chunks", len(texts))
