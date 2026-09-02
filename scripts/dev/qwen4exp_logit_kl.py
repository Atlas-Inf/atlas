#!/usr/bin/env python3
"""KL drift between the GPU's logits and the CPU reference's, same prompt.

Why KL and not cosine. The highway tap diff measures how similar two hidden
states are, and by layer 35 the GPU sits at cosine ~0.96 against an f32 CPU
reference. That number cannot say whether the port is fine: a 0.96 hidden
state can land on the vocabulary harmlessly or fatally. KL over the softmax
says how differently the two models would SAMPLE, which is what reaches a
user.

Reported per position:

  KL(ref||gpu)   nats. The reference is the truth, so this is the one to read:
                 it is large when the GPU puts little mass where the reference
                 puts a lot.
  KL(gpu||ref)   the reverse, which catches the GPU inventing mass the
                 reference does not have.
  top1           whether the argmax agrees at all. The single most
                 consequential bit, because greedy decode is exactly this.
  rank_of_ref1   where the reference's top token sits in the GPU's ordering.
                 1 is agreement; a rank inside top_k is recoverable under
                 sampling; far outside it is not.
  top20_overlap  |intersection| of the two top-20 sets. 20 because the model
                 card samples top_k=20, so this is the window that actually
                 gates what can be emitted.
  temp-scaled KL at the card's sampling temperature, since that is the
                 distribution really drawn from.

Inputs are both flat FP32, row-major `[rows, vocab]`:
    GPU  <gpu_dir>/logits_fetch.bin   (ATLAS_DUMP_LOGITS_PATH, appended)
    REF  <ref_dir>/ref_logits.bin     (ATLAS_QWEN4EXP_REF_DUMP)

Usage:
    qwen4exp_logit_kl.py <gpu_dir> <ref_dir> --vocab 248077 [--temp 1.0]
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys


def rows(path: str, vocab: int) -> list[list[float]]:
    with open(path, "rb") as fh:
        raw = fh.read()
    n = len(raw) // 4
    flat = struct.unpack(f"<{n}f", raw[: n * 4])
    if vocab <= 0 or n < vocab:
        raise SystemExit(f"{path}: {n} floats is fewer than one row of {vocab}")
    return [list(flat[i * vocab:(i + 1) * vocab]) for i in range(n // vocab)]


def softmax(v: list[float], temp: float) -> list[float]:
    m = max(v)
    ex = [math.exp((x - m) / temp) for x in v]
    s = sum(ex)
    return [e / s for e in ex]


def kl(p: list[float], q: list[float]) -> float:
    """KL(p||q) in nats.

    `q` is floored rather than skipped: a vocabulary entry where the reference
    holds mass and the GPU holds none is exactly the disagreement worth
    measuring, and dropping it would hide it.
    """
    floor = 1e-12
    return sum(pi * math.log(pi / max(qi, floor))
               for pi, qi in zip(p, q) if pi > 0.0)


def topk(v: list[float], k: int) -> list[int]:
    return sorted(range(len(v)), key=lambda i: v[i], reverse=True)[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu_dir")
    ap.add_argument("ref_dir")
    ap.add_argument("--vocab", type=int, required=True,
                    help="GPU row width; the server CAPS vocab to the "
                         "tokenizer's size (248077 here)")
    ap.add_argument("--ref-vocab", type=int, default=None,
                    help="reference row width if it differs — the CPU forward "
                         "uses config.vocab_size (248320), which the server "
                         "caps. Compared over the common prefix.")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="card sampling temperature (Qwen3.8-Flash-Next: 1.0)")
    ap.add_argument("--kl-warn", type=float, default=0.05,
                    help="KL(ref||gpu) in nats above which to flag a position")
    args = ap.parse_args()

    # Two sites append under ATLAS_DUMP_LOGITS_PATH and which one fires
    # depends on the sampling path taken: `spark_runtime::sampler` writes
    # `logits_fetch.bin`, `scheduler::sample_step` writes `logits_stok.bin`.
    # Accept either rather than silently reporting "missing" for the one that
    # actually ran.
    gpu_path = next(
        (p for p in (os.path.join(args.gpu_dir, n)
                     for n in ("logits_stok.bin", "logits_fetch.bin"))
         if os.path.exists(p)),
        os.path.join(args.gpu_dir, "logits_stok.bin"),
    )
    ref_path = os.path.join(args.ref_dir, "ref_logits.bin")
    for p in (gpu_path, ref_path):
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            return 2

    ref_vocab = args.ref_vocab or args.vocab
    gpu = rows(gpu_path, args.vocab)
    ref = rows(ref_path, ref_vocab)
    print(f"gpu file: {os.path.basename(gpu_path)}")
    print(f"gpu rows: {len(gpu)} x {args.vocab}   "
          f"ref rows: {len(ref)} x {ref_vocab}")
    # The tail the server trimmed is unreachable ids; comparing the common
    # prefix is the only apples-to-apples window. Truncating rather than
    # padding matters: a zero-padded tail would look like agreed-upon mass.
    common = min(args.vocab, ref_vocab)
    if args.vocab != ref_vocab:
        print(f"widths differ — comparing the common prefix of {common} "
              f"(the server caps vocab to the tokenizer's size)")
        gpu = [r[:common] for r in gpu]
        ref = [r[:common] for r in ref]

    # The GPU dump holds the sampled step(s) only; the reference holds every
    # prompt position. Compare the LAST row of each, which is the position both
    # actually predicted from.
    pairs = [(len(gpu) - 1, len(ref) - 1)]
    print(f"comparing gpu row {pairs[0][0]} against ref row {pairs[0][1]} "
          f"(the position both predicted from)\n")

    # ── CONTROL 1: the GPU against itself. ──
    # Identical prompts should give identical logits; whatever KL this shows
    # is the measurement's own noise floor, and the GPU-vs-reference number
    # means nothing without it.
    if len(gpu) > 1:
        selfk = max(kl(softmax(gpu[0], 1.0), softmax(gpu[i], 1.0))
                    for i in range(1, len(gpu)))
        ident = all(gpu[0] == gpu[i] for i in range(1, len(gpu)))
        print(f"  control: gpu vs gpu over {len(gpu)} identical prompts — "
              f"KL {selfk:.9f}, bitwise identical: {ident}")
        print()

    worst = 0.0
    for gi, ri in pairs:
        g, r = gpu[gi], ref[ri]
        for temp in (1.0, args.temp) if args.temp != 1.0 else (1.0,):
            pg, pr = softmax(g, temp), softmax(r, temp)
            fwd, rev = kl(pr, pg), kl(pg, pr)
            worst = max(worst, fwd)
            g20, r20 = topk(g, 20), topk(r, 20)
            overlap = len(set(g20) & set(r20))
            order = sorted(range(len(g)), key=lambda i: g[i], reverse=True)
            rank = order.index(r20[0]) + 1
            print(f"  temp {temp:>4}:  KL(ref||gpu) {fwd:9.6f}   "
                  f"KL(gpu||ref) {rev:9.6f}")
            print(f"             top1 {'AGREE' if g20[0] == r20[0] else 'DIFFER'}"
                  f"  (gpu {g20[0]}, ref {r20[0]})   "
                  f"rank_of_ref1 {rank}   top20_overlap {overlap}/20")
        print(f"             logit range: gpu [{min(g):.3f}, {max(g):.3f}]  "
              f"ref [{min(r):.3f}, {max(r):.3f}]")

        # ── CONTROL 2: KL over what sampling can actually REACH. ──
        # The card samples top_k=20, so every token outside the top 20 is
        # truncated before a draw happens. Full-vocab KL therefore charges the
        # port for disagreement over ~248k tokens that can never be emitted.
        # Restrict to the union of the two top-20 sets and renormalize: this
        # is the distribution a user is actually exposed to.
        for k in (20, 100):
            keep = sorted(set(topk(g, k)) | set(topk(r, k)))
            gk = softmax([g[i] for i in keep], temp)
            rk = softmax([r[i] for i in keep], temp)
            print(f"             top-{k} union ({len(keep)} ids), renormalized: "
                  f"KL(ref||gpu) {kl(rk, gk):.6f}")

    print()
    if worst <= args.kl_warn:
        print(f"KL(ref||gpu) <= {args.kl_warn} nats: the two distributions "
              f"agree to within quantization noise.")
        return 0
    print(f"KL(ref||gpu) = {worst:.6f} nats, above {args.kl_warn}. The port "
          f"and the reference would sample differently; this is a real gap, "
          f"not rounding.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
