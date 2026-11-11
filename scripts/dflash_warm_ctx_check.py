#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Check the DFlash drafter window of warm turns against cold prefills.

Run against a server started with

    ATLAS_DFLASH_CTX_DUMP_DIR=<dir>   (rank 0 writes ctx-<pid>-<n>.bin/.json
                                       at every request's first propose)
    ATLAS_DFLASH_WARM_CTX=<mode>      (legacy|exact|seed|restore)

and the production prefix-cache flags. Each case sends its turns one at a
time and reads the dump each turn left:

  repeat  the same prompt twice: the second prefill restores at its own tail
          cut and recomputes only the rows above it.
  chain   turn 1, then turn 1 + a short suffix (a warm agentic turn).

Under `restore` the second window's rows under turn 1's tail cut are the ones
turn 1 stored, so they must equal turn 1's bit for bit; rows above the cut
are recomputed (in the repeat case by a pass of the same shape) and only
reported.

Every window is also checked for zero rows and for labels that run
contiguously up to the propose position. Exit status 1 on a failed check.

    scripts/dflash_warm_ctx_check.py --url http://127.0.0.1:8081 --dump-dir <dir> \
        --model <served name> [--prompt-tokens 6000] [--suffix-tokens 300]
"""

import argparse
import json
import random
import struct
import sys
import time
import urllib.request
from pathlib import Path

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango").split()


def text(n_words, rng):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def complete(url, model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 8,
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"{url}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()


def dumps(d):
    return sorted(d.glob("ctx-*.json"))


def send(args, prompt):
    """Send one turn; return the dump its first propose wrote."""
    before = set(dumps(args.dump_dir))
    complete(args.url, args.model, prompt)
    for _ in range(50):
        new = [p for p in dumps(args.dump_dir) if p not in before]
        if new:
            meta = json.loads(new[-1].read_text())
            raw = new[-1].with_suffix(".bin").read_bytes()
            step = meta["ctx_slot_bytes"]
            meta["rows"] = [raw[i:i + step] for i in range(0, len(raw), step)]
            return meta
        time.sleep(0.1)
    raise SystemExit("no dump appeared: is ATLAS_DFLASH_CTX_DUMP_DIR set on rank 0?")


def bf16(row):
    return [struct.unpack("<f", b"\0\0" + row[i:i + 2])[0]
            for i in range(0, len(row), 2)]


def window_ok(name, w):
    pos = w["positions"]
    zeros = sum(1 for r in w["rows"] if not any(r))
    contiguous = all(b == a + 1 for a, b in zip(pos, pos[1:]))
    top = pos[-1] if pos else None
    print(f"  {name}: {w['warm_ctx']} ctx_len={w['ctx_len']} window=[{pos[0] if pos else '-'},"
          f" {top + 1 if pos else '-'}) floor={w['ctx_floor']} zero_rows={zeros}"
          f" contiguous={contiguous}")
    return zeros == 0 and contiguous


def tail_cut(prompt_len, bs):
    """`pc_policy::tail_cut`: one block under the last boundary below the end."""
    return max((prompt_len - 1) // bs * bs - bs, 0)


def overlap(name, a, b, must_equal, cut):
    """Compare the rows two windows hold at the same positions; with
    `must_equal`, those under `cut` must be bit-equal."""
    ia = {p: r for p, r in zip(a["positions"], a["rows"])}
    shared = [(p, ia[p], r) for p, r in zip(b["positions"], b["rows"]) if p in ia]
    equal = sum(1 for _, x, y in shared if x == y)
    under = [(x, y) for p, x, y in shared if p < cut]
    under_equal = sum(1 for x, y in under if x == y)
    worst = 0.0
    for _, x, y in shared:
        if x != y:
            fx, fy = bf16(x), bf16(y)
            num = sum((u - v) ** 2 for u, v in zip(fx, fy)) ** 0.5
            den = sum(u * u for u in fx) ** 0.5 or 1.0
            worst = max(worst, num / den)
    print(f"  {name}: {len(shared)} shared positions, {equal} bit-equal"
          f" ({under_equal}/{len(under)} under the cut {cut}),"
          f" worst relative L2 {worst:.3g}")
    return not must_equal or under_equal == len(under)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--prompt-tokens", type=int, default=6000,
                    help="approximate prompt length (words ~ tokens)")
    ap.add_argument("--suffix-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()
    rng = random.Random(args.seed or time.time_ns())
    ok = True

    print("repeat:")
    prompt = f"[{rng.random()}] " + text(args.prompt_tokens, rng)
    cold, warm = send(args, prompt), send(args, prompt)
    for name, w in (("cold", cold), ("warm", warm)):
        ok &= window_ok(name, w)
    restore = warm["warm_ctx"] == "Restore"
    cut = tail_cut(cold["position"], args.block_size)
    ok &= overlap("warm vs cold", cold, warm, restore, cut)

    print("chain:")
    turn1 = f"[{rng.random()}] " + text(args.prompt_tokens, rng)
    t1 = send(args, turn1)
    t2 = send(args, turn1 + " " + text(args.suffix_tokens, rng))
    for name, w in (("turn 1", t1), ("turn 2", t2)):
        ok &= window_ok(name, w)
    cut = tail_cut(t1["position"], args.block_size)
    ok &= overlap("turn 2 vs turn 1", t1, t2, restore, cut)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
