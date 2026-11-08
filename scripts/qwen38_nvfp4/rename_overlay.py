#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""rename_overlay.py IN.safetensors OUT.safetensors [--bf16 DIR] [--report R.json]

Re-key an external NVFP4 overlay whose tensors are named `layers.N.<module>.<proj>.
{weight, weight_scale, weight_scale_2}` (ModelOpt layout: E2M1 low nibble first,
E4M3 scale per 16 inputs in a plain [N, K/16] grid, F32 scale_2 per tensor) to the
checkpoint's `model.language_model.layers.N...` names, for `requant.py overlay`.
Bytes are copied unchanged, with one exception the loader forces: a GDN layer's
in_proj_qkv and in_proj_z must share one weight_scale_2, because they are
concatenated into qkvz. Where an input carries two, one of them is kept exact
and the other's E4M3 block scales are re-expressed against the kept scale_2
(scale * s2_old / s2_new, rounded to nearest E4M3, saturating at 448; the codes
are untouched). Each layer keeps whichever side makes the smaller squared error
of the re-expressed tensor against its own original dequantization. With --bf16
(the BF16 cuts of stream_bf16.sh) the report also gives every GDN projection's
relative error against the BF16 weights, before and after, which checks the
layout and nibble order too.
"""
import argparse
import glob
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GRID = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
PRE = "model.language_model.layers"


def deq(w, s, s2):
    """[N, K/2] U8 + [N, K/16] E4M3 + F32 -> [N, K] FP32."""
    lo, hi = (w & 0xF).long(), (w >> 4).long()
    code = torch.stack([lo, hi], -1).reshape(w.shape[0], -1)
    val = GRID[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    n, k = val.shape
    return (val.view(n, k // 16, 16) * (s.float() * s2).unsqueeze(-1)).view(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--bf16")
    ap.add_argument("--report")
    a = ap.parse_args()
    f = safe_open(a.inp, "pt")
    bf = {}
    for p in sorted(glob.glob(f"{a.bf16}/*.safetensors")) if a.bf16 else []:
        g = safe_open(p, "pt")
        bf.update({k: (g, k) for k in g.keys()})
    out, report = {}, {}
    keys = sorted(f.keys())
    for k in keys:
        out[f"{PRE}.{k[len('layers.'):]}"] = f.get_tensor(k)

    def rel(name, d):
        if f"{name}.weight" not in bf:
            return None
        g, kk = bf[f"{name}.weight"]
        w = g.get_tensor(kk).float()
        return float(((w - d).norm() / w.norm()).item())

    layers = sorted({int(k.split(".")[1]) for k in keys if ".linear_attn." in k})
    for li in layers:
        base = f"{PRE}.{li}.linear_attn"
        q, z = f"{base}.in_proj_qkv", f"{base}.in_proj_z"
        if q + ".weight" not in out:
            continue
        s2 = {n: float(out[n + ".weight_scale_2"].item()) for n in (q, z)}
        orig = {n: deq(out[n + ".weight"], out[n + ".weight_scale"], s2[n]) for n in (q, z)}
        r = {"s2_qkv": s2[q], "s2_z": s2[z]}
        if s2[q] != s2[z]:
            best = None
            for keep, other in ((q, z), (z, q)):
                sc = (out[other + ".weight_scale"].float() * (s2[other] / s2[keep])).clamp(max=448)
                sc = sc.to(torch.float8_e4m3fn)
                d = deq(out[other + ".weight"], sc, s2[keep])
                err = float(((d - orig[other]) ** 2).sum())
                if best is None or err < best[0]:
                    changed = float((sc.float() * s2[keep] != out[other + ".weight_scale"].float() * s2[other]).float().mean())
                    best = (err, keep, other, sc, d, changed)
            err, keep, other, sc, d, changed = best
            out[other + ".weight_scale"] = sc
            out[other + ".weight_scale_2"] = torch.tensor(s2[keep], dtype=torch.float32)
            tot = float((orig[q] ** 2).sum() + (orig[z] ** 2).sum())
            r.update({"kept": keep.rsplit(".", 1)[1], "blocks_changed": changed,
                      "reexpress_rel_err_qkvz": (err / tot) ** 0.5})
            new = dict(orig)
            new[other] = d
        else:
            new = orig
        for n in (q, z):
            e0, e1 = rel(n, orig[n]), rel(n, new[n])
            if e0 is not None:
                r[n.rsplit(".", 1)[1]] = {"rel_vs_bf16": e0, "after": e1}
        o = f"{base}.out_proj"
        if o + ".weight" in out:
            e = rel(o, deq(out[o + ".weight"], out[o + ".weight_scale"], float(out[o + ".weight_scale_2"].item())))
            if e is not None:
                r["out_proj"] = {"rel_vs_bf16": e}
        report[li] = r
        print(li, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}, flush=True)
    save_file(out, a.out, metadata={"renamed_from": a.inp.rsplit("/", 1)[-1], **(f.metadata() or {})})
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    print("wrote", a.out, len(out), "tensors")


if __name__ == "__main__":
    main()
