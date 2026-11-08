#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""calib.py ATLAS_REPO [URL] -- the calibration prefills for `requant.py --mode gptq`.

Serve the FP8 checkpoint from a build with capture_hook.diff applied and
ATLAS_GDN_CAPTURE=<dir> set, then run this: seven prefill-only requests of
~9K characters (three Python files from /usr/lib/python3/dist-packages, one
Rust file and three Markdown documents from the engine source, the last one as
a User:/Assistant: turn), about 18.7K tokens on Ubuntu 24.04. None of it is in
the held-out NLL set (the python3.12 stdlib, /usr/share docs and licenses,
pydoc topics, GSM8K questions)."""
import glob, json, os, random, sys, urllib.request

repo = sys.argv[1]
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8897"
r = random.Random(5)
py = sorted(f for f in glob.glob("/usr/lib/python3/dist-packages/**/*.py", recursive=True) if os.path.getsize(f) > 12000)
rs = sorted(f for f in glob.glob(f"{repo}/crates/**/*.rs", recursive=True) if os.path.getsize(f) > 12000)
md = sorted(f for f in glob.glob(f"{repo}/docs/**/*.md", recursive=True) if os.path.getsize(f) > 12000)
texts = []
for files, n in ((py, 3), (rs, 1), (md, 3)):
    for f in r.sample(files, n):
        t = open(f, errors="ignore").read()
        i = r.randrange(0, max(1, len(t) - 9000))
        texts.append(t[i:i + 9000])
texts[-1] = f"User: Can you explain this part of the documentation?\n\nAssistant: {texts[-1]}\n"
tokens = 0
for t in texts:
    body = {"model": "qwen3.8-27b-atlas", "prompt": t, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    tokens += json.load(urllib.request.urlopen(req, timeout=900))["usage"]["prompt_tokens"]
print("calibration prompt tokens", tokens, "chunks", len(texts))
