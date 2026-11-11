#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Wave-level breakdown of an `ATLAS_MTP_STEP_TRACE=1` serve log.

Reads the `MTP STEP ...` lines (`crates/spark-server/src/scheduler/
mtp_step_trace.rs` documents the format) from a rank-0 log and reports where a
wave's delivered tokens per step go:

* per wave: wall, tokens, steps, tokens/step/nmax (the "3200 / 124 / 8"
  quantity) next to tokens per sequence-step (what vLLM's spec-decode metrics
  report), and the time and steps spent below full width (ramp, tail);
* by active width n: step time, tokens per step and per sequence-step;
* acceptance by step index since each sequence's first traced step;
* how often rows are draftless (decode rows, bootstraps) and the paths taken;
* target forwards per step (more than one = a step paid extra weight passes).

Usage: step_trace_summary.py LOG [LOG...] [--gap-ms 1000] [--waves] [--steps N]
       [--min-seqs 2]
"""
import argparse
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"MTP STEP (.*)$")
INDEX_BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 10**9)]
PATH_NAMES = {
    "v": "batched verify",
    "d": "decode row (in verify)",
    "s": "serial verify",
    "B": "batched bootstrap",
    "b": "per-seq bootstrap",
    "n": "plain decode",
    "r": "rode prefill chunk",
    "m": "mixed decode",
    "?": "unrecorded",
}
DRAFTLESS = set("dbBnm")


def parse(paths):
    steps = []
    for p in paths:
        fh = sys.stdin if p == "-" else open(p, errors="replace")
        for raw in fh:
            m = LINE.search(ANSI.sub("", raw))
            if not m or " n=0 " in m.group(0):
                continue
            kv = {}
            for tok in m.group(1).split():
                k, _, v = tok.partition("=")
                kv[k] = v
            seqs = []
            for e in filter(None, kv.get("s", "").split(",")):
                slot, gen, pd, emitted = e.split(":")
                seqs.append((int(slot), int(gen), pd[0], int(pd[1:]), int(emitted)))
            # `dur` covers the whole tick; `pf` (newer builds) is its prefill part
            # before the decode dispatch, which counts as a gap here.
            pf = int(kv.get("pf", 0))
            steps.append({
                "t": int(kv["t"]) + pf, "dur": int(kv["dur"]) - pf, "pf": pf, "kind": kv["kind"], "n": int(kv["n"]),
                "rows": int(kv["rows"]), "fwd": int(kv["fwd"]), "nd": int(kv["nd"]),
                "deep": kv["deep"] == "1", "dl": int(kv["dl"]), "seqs": seqs,
            })
        if fh is not sys.stdin:
            fh.close()
    steps.sort(key=lambda s: s["t"])
    return steps


def assign_ids(steps):
    """Give each sequence a stable id (slot + epoch) and its step index."""
    last_gen = {}
    epoch = defaultdict(int)
    seen_steps = defaultdict(int)
    for st in steps:
        ids = []
        for slot, gen, path, drafts, emitted in st["seqs"]:
            if slot in last_gen and gen < last_gen[slot]:
                epoch[slot] += 1
            last_gen[slot] = gen
            sid = (slot, epoch[slot])
            ids.append((sid, seen_steps[sid]))
            seen_steps[sid] += 1
        st["ids"] = ids


def split_waves(steps, gap_us):
    """A new wave starts when a step shares no sequence with the previous one
    or follows a gap longer than `gap_us`."""
    waves, cur, prev_ids, prev_end = [], [], set(), None
    for st in steps:
        ids = {sid for sid, _ in st["ids"]}
        if cur and (not (ids & prev_ids) or st["t"] - prev_end > gap_us):
            waves.append(cur)
            cur = []
        cur.append(st)
        prev_ids, prev_end = ids, st["t"] + st["dur"]
    if cur:
        waves.append(cur)
    return waves


def tokens(st):
    return sum(e[4] for e in st["seqs"])


def bucket_of(i):
    for lo, hi in INDEX_BUCKETS:
        if lo <= i <= hi:
            return (lo, hi)
    return INDEX_BUCKETS[-1]


def label(b):
    lo, hi = b
    return f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")


def ms(us):
    return us / 1000.0


def wave_report(w, idx):
    nseq = len({sid for st in w for sid, _ in st["ids"]})
    nmax = max(st["n"] for st in w)
    tok = sum(tokens(st) for st in w)
    seq_steps = sum(st["n"] for st in w)
    busy = sum(st["dur"] for st in w)
    wall = w[-1]["t"] + w[-1]["dur"] - w[0]["t"]
    full = [i for i, st in enumerate(w) if st["n"] == nmax]
    first_full, last_full = (full[0], full[-1]) if full else (len(w), -1)
    ramp, tail = w[:first_full], w[last_full + 1:]
    mid = w[first_full:last_full + 1] if full else []
    mid_part = [st for st in mid if st["n"] < nmax]
    full_steps = [st for st in mid if st["n"] == nmax]
    full_tok = sum(tokens(st) for st in full_steps)
    full_tps = full_tok / max(1, nmax * len(full_steps))
    full_dur = sum(st["dur"] for st in full_steps) / max(1, len(full_steps))

    def at_full_rate(t):
        # Time the full-width steps would take to deliver `t` tokens.
        return t / max(1e-9, nmax * full_tps) * full_dur

    def seg(name, ss):
        if not ss:
            return 0
        t = sum(tokens(s) for s in ss)
        d = sum(s["dur"] for s in ss)
        lost = sum(nmax - s["n"] for s in ss)
        excess = d - at_full_rate(t) if full_steps else 0
        print(f"    {name:<14} steps={len(ss):>4} time={ms(d):8.1f} ms tokens={t:>5} "
              f"tok/step={t / len(ss):6.2f} empty_slots={lost:>4} excess={ms(excess):7.1f} ms")
        return excess

    print(f"wave {idx}: seqs={nseq} nmax={nmax} steps={len(w)} tokens={tok} wall={ms(wall):.1f} ms "
          f"busy={ms(busy):.1f} ms gaps={ms(wall - busy):.1f} ms")
    print(f"    tokens/step/nmax={tok / max(1, len(w) * nmax):.3f}  "
          f"tokens/seq-step={tok / max(1, seq_steps):.3f}  "
          f"full-width tokens/seq-step={full_tps:.3f} @ {ms(full_dur):.1f} ms/step  "
          f"agg={tok / max(1e-9, wall / 1e6):.1f} tok/s")
    ex = [seg("ramp (n<max)", ramp), seg("full width", full_steps),
          seg("mid dips", mid_part), seg("tail (n<max)", tail)]
    if full_steps:
        ideal = at_full_rate(tok)
        print(f"    at the full-width rate the wave needs {tok / (nmax * full_tps):.1f} steps = "
              f"{ms(ideal):.1f} ms; wall {ms(wall):.1f} ms = ideal + ramp {ms(ex[0]):.1f} "
              f"+ dips {ms(ex[2]):.1f} + tail {ms(ex[3]):.1f} + gaps {ms(wall - busy):.1f} ms")
    extra_fwd = [st for st in w if st["fwd"] > 1]
    dl_rows = sum(st["dl"] for st in w)
    print(f"    draftless rows={dl_rows} ({dl_rows / max(1, seq_steps):.1%} of seq-steps); "
          f"steps with >1 target forward={len(extra_fwd)} "
          f"(time {ms(sum(s['dur'] for s in extra_fwd)):.1f} ms); "
          f"kinds={dict(sorted(count_by(w, lambda s: s['kind']).items()))}")


def count_by(steps, key):
    c = defaultdict(int)
    for st in steps:
        c[key(st)] += 1
    return c


def by_width(steps):
    print("\nby active width n:")
    print(f"  {'n':>3} {'steps':>6} {'ms/step':>8} {'tok/step':>9} {'tok/seq-step':>13} "
          f"{'rows':>6} {'fwd':>5} {'draftless':>10} {'nd':>4}")
    agg = defaultdict(list)
    for st in steps:
        agg[st["n"]].append(st)
    for n in sorted(agg):
        ss = agg[n]
        t = sum(tokens(s) for s in ss)
        rows = sum(s["rows"] for s in ss) / len(ss)
        fwd = sum(s["fwd"] for s in ss) / len(ss)
        dl = sum(s["dl"] for s in ss) / max(1, n * len(ss))
        nds = count_by(ss, lambda s: s["nd"])
        nd = max(nds, key=nds.get)
        print(f"  {n:>3} {len(ss):>6} {ms(sum(s['dur'] for s in ss)) / len(ss):8.1f} "
              f"{t / len(ss):9.2f} {t / (n * len(ss)):13.3f} {rows:6.1f} {fwd:5.2f} {dl:10.1%} {nd:>4}")


def by_index(steps):
    print("\nacceptance by step index since the sequence's first traced step:")
    print(f"  {'index':>7} {'seq-steps':>10} {'tok/seq-step':>13} {'drafted':>8} "
          f"{'mean drafts':>12} {'accept rate':>12} {'p1':>6} {'draftless':>10}")
    agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    for st in steps:
        for (sid, i), (_, _, path, drafts, emitted) in zip(st["ids"], st["seqs"]):
            a = agg[bucket_of(i)]
            a[0] += 1
            a[1] += emitted
            if drafts > 0:
                a[2] += 1
                a[3] += drafts
                a[4] += min(drafts, max(0, emitted - 1))
                a[5] += emitted >= 2
            if path in DRAFTLESS:
                a[6] += 1
    for b in INDEX_BUCKETS:
        if b not in agg:
            continue
        n, em, dr, dsum, acc, p1, dl = agg[b]
        print(f"  {label(b):>7} {n:>10} {em / n:13.3f} {dr:>8} {dsum / max(1, dr):12.2f} "
              f"{acc / max(1, dsum):12.3f} {p1 / max(1, dr):6.3f} {dl / n:10.1%}")


def paths(steps):
    print("\npaths (sequence-steps):")
    agg = defaultdict(lambda: [0, 0, 0])
    for st in steps:
        for _, _, path, drafts, emitted in st["seqs"]:
            a = agg[path]
            a[0] += 1
            a[1] += emitted
            a[2] += drafts
    total = sum(a[0] for a in agg.values())
    for p, (n, em, dr) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {p} {PATH_NAMES.get(p, p):<24} {n:>7} ({n / total:6.1%})  "
              f"tok/seq-step={em / n:.3f} mean drafts={dr / n:.2f}")
    fw = count_by(steps, lambda s: s["fwd"])
    print("target forwards per step: " + ", ".join(f"{k}:{fw[k]}" for k in sorted(fw)))


def dump_steps(w, limit):
    print(f"\nfirst/last {limit} steps of the wave (t ms from wave start):")
    t0 = w[0]["t"]
    rows = w if len(w) <= 2 * limit else w[:limit] + [None] + w[-limit:]
    for st in rows:
        if st is None:
            print("  ...")
            continue
        shape = " ".join(f"{p}{d}:{e}" for _, _, p, d, e in st["seqs"])
        print(f"  {ms(st['t'] - t0):8.1f} {ms(st['dur']):6.1f}ms {st['kind']:<5} n={st['n']} "
              f"rows={st['rows']:>3} fwd={st['fwd']} nd={st['nd']}{'D' if st['deep'] else ''} | {shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--gap-ms", type=float, default=1000.0,
                    help="a gap longer than this between steps starts a new wave")
    ap.add_argument("--min-seqs", type=int, default=2,
                    help="report waves with at least this many sequences in the tables")
    ap.add_argument("--waves", action="store_true", help="print every wave, not only multi-sequence ones")
    ap.add_argument("--steps", type=int, default=0, help="dump the first/last N steps of each reported wave")
    args = ap.parse_args()
    steps = parse(args.logs)
    if not steps:
        sys.exit("no 'MTP STEP' lines (was ATLAS_MTP_STEP_TRACE=1 set?)")
    assign_ids(steps)
    waves = split_waves(steps, args.gap_ms * 1000)
    chosen = []
    for i, w in enumerate(waves):
        nseq = len({sid for st in w for sid, _ in st["ids"]})
        if args.waves or nseq >= args.min_seqs:
            chosen.append((i, w))
    print(f"{len(steps)} steps, {len(waves)} waves, {len(chosen)} reported "
          f"(>= {args.min_seqs} sequences)\n")
    for i, w in chosen:
        wave_report(w, i)
        if args.steps:
            dump_steps(w, args.steps)
        print()
    sel = [st for _, w in chosen for st in w] or steps
    by_width(sel)
    by_index(sel)
    paths(sel)


if __name__ == "__main__":
    main()
