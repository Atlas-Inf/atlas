#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""ppl.py — perplexity + per-token logprob-delta gate for kernel/env A/Bs.

Uses the legacy /v1/completions loglikelihood call (echo=true, logprobs=1,
max_tokens=0): the response's token_logprobs array covers the prompt tokens
(first entry null), so one request scores a whole window with no generation.

Corpus sources (first match wins):
    --tokens-file F   JSON list of token ids (bypasses tokenization entirely)
    --text-file F     raw UTF-8 text
    --corpus-dir D    concatenated .md/.txt under D (recursive, sorted)

Windows are non-overlapping chunks of --ctx characters on the text path, or
--ctx tokens on the token-id path. Two runs of this script against the same
checkpoint produce byte-identical token arrays, so --compare can diff the
token_logprobs arrays position-by-position.

Run a gate:
    ppl.py --url http://127.0.0.1:8095 --corpus-dir docs --out ppl-ref.json
    ppl.py --url http://127.0.0.1:8095 --corpus-dir docs --out ppl-new.json
    ppl.py --compare ppl-ref.json ppl-new.json

Stdlib only — no requests/transformers/datasets.
"""
import argparse, json, math, os, sys, time, urllib.error, urllib.request

def corpus_text(args):
    if args.tokens_file:
        return None, json.load(open(args.tokens_file))
    if args.text_file:
        return open(args.text_file, encoding="utf-8", errors="replace").read(), None
    if args.corpus_dir:
        parts = []
        for root, _, files in os.walk(args.corpus_dir):
            for f in sorted(files):
                if f.endswith((".md", ".txt")):
                    try:
                        parts.append(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
                    except OSError:
                        pass
        text = "\n\n".join(parts)
        if len(text) < args.ctx * 2:
            sys.exit(f"corpus-dir {args.corpus_dir} too small ({len(text)} chars)")
        return text, None
    sys.exit("need --tokens-file, --text-file, or --corpus-dir")

def score(url, model, prompt, tokids):
    body = {"model": model, "prompt": prompt,
            "echo": True, "logprobs": 1, "max_tokens": 0,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    except urllib.error.HTTPError as e:
        print(f"  window skipped: HTTP {e.code} {e.read()[:160]}", file=sys.stderr)
        return None
    ch = r["choices"][0]
    lp = ch["logprobs"]
    out = {"tokens": lp["tokens"], "lps": lp["token_logprobs"],
           "wall_s": round(time.time() - t0, 2), "usage": r.get("usage", {})}
    if tokids is not None and len(lp["tokens"]) != len(tokids):
        print(f"  warn: echoed {len(lp['tokens'])} tokens != sent {len(tokids)}", file=sys.stderr)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=False)
    ap.add_argument("--tokens-file"); ap.add_argument("--text-file"); ap.add_argument("--corpus-dir")
    ap.add_argument("--ctx", type=int, default=8192, help="chars per window (text) or tokens per window (ids)")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "NEW"))
    args = ap.parse_args()

    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        wa, wb = a["windows"], b["windows"]
        assert len(wa) == len(wb), "window count differs"
        dmax, dsum, ntot = 0.0, 0.0, 0
        worst = None
        for i, (x, y) in enumerate(zip(wa, wb)):
            if x["tokens"] != y["tokens"]:
                sys.exit(f"window {i}: token arrays differ — runs are not comparable")
            for j, (lx, ly) in enumerate(zip(x["lps"], y["lps"])):
                if lx is None or ly is None:
                    continue
                d = abs(ly - lx); ntot += 1
                if d > dmax:
                    dmax, worst = d, (i, j, y["tokens"][j], lx, ly)
                dsum += d
        pa, pb = a["ppl"], b["ppl"]
        print(f"compare: ppl {pa:.4f} -> {pb:.4f} (ratio {pb/pa:.4f})")
        print(f"         per-token |dlp| max={dmax:.4f} mean={dsum/max(ntot,1):.5f} n={ntot} worst={worst}")
        print(f"         walls: ref={sum(w['wall_s'] for w in wa):.1f}s new={sum(w['wall_s'] for w in wb):.1f}s")
        ok = (pb / pa < 1.02) and dmax < 0.5
        print("VERDICT:", "PASS" if ok else "FAIL (ppl ratio>1.02 or a token moved >0.5 nats)")
        return

    text, tokids = corpus_text(args)
    model = json.loads(urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=15).read())["data"][0]["id"]
    print(f"model={model} url={args.url}")
    windows, nll_sum, n_tok = [], 0.0, 0
    if tokids is not None:
        for i in range(0, min(len(tokids), args.ctx * args.windows), args.ctx):
            windows.append(score(args.url, model, tokids[i:i + args.ctx], tokids[i:i + args.ctx]))
    else:
        for i in range(args.windows):
            chunk = text[i * args.ctx:(i + 1) * args.ctx]
            if len(chunk) < args.ctx // 2:
                break
            w = score(args.url, model, chunk, None)
            if w is None:
                continue
            windows.append(w)
            print(f"  window {i}: {len(w['tokens'])} tokens wall={w['wall_s']}s")
    for w in windows:
        for lp in w["lps"]:
            if lp is not None:
                nll_sum -= lp; n_tok += 1
    ppl = math.exp(nll_sum / max(n_tok, 1))
    rec = {"model": model, "ctx": args.ctx, "windows": windows, "n_tokens": n_tok, "ppl": round(ppl, 5)}
    if args.out:
        json.dump(rec, open(args.out, "w"))
    print(f"ppl={ppl:.4f} over {n_tok} tokens in {len(windows)} windows")

if __name__ == "__main__":
    main()
