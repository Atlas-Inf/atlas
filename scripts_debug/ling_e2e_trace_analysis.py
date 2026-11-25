#!/usr/bin/env python3
"""Offline layout forensics for the layer-0 KDA trace (f341.

Compares atlas lane dumps vs oracle trace stages under candidate layouts and
per-row/per-plane breakdowns to name the first wrong op precisely.
"""
import numpy as np

D = "/home/spark/.cache/atlas-e2e2"
P, T, H = 2, 4096, 2560
np.set_printoptions(precision=4, suppress=True)

def rd(name):
    return np.fromfile(f"{D}/{name}", dtype=np.float32)

def stats(a, b):
    d = np.abs(a - b)
    rel = d / (np.abs(b) + 1e-6)
    cos = 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    return d.max(), np.median(d), rel.max(), cos

def show(name, a, b):
    mx, med, rel, cos = stats(a, b)
    print(f"{name:34s} max_abs {mx:9.5f} med {med:9.6f} max_rel {rel:9.4f} 1-cos {cos:9.6f}")

atlas = {s: rd(f"atlas_lane_0_{s}.f32") for s in
         ["normed", "preconv", "qkv", "f", "g", "b_raw", "kda_out", "kda_o",
          "attn_out", "ffn_in", "shared_h", "ffn_out"]}
oracle = {s: rd(f"trace_0_{s}.f32") for s in
          ["normed", "preconv", "qkv", "f", "g", "b_raw", "kda_out", "kda_o",
           "attn_out", "ffn_in", "shared_h", "ffn_out"]}

print("== stage-wise (oracle layouts) ==")
for s in oracle:
    n = min(len(atlas[s]), len(oracle[s]))
    show(f"{s} (n={n})", atlas[s][:n], oracle[s][:n])

print("\n== preconv layout forensics ==")
ap = atlas["preconv"]; op = oracle["preconv"]
print(f"atlas len {len(ap)}, oracle len {len(op)} (3*P*T = {3*P*T})")
for rows_cap in [2, 8, 16, 32, 64, 512]:
    # hypothesis: plane p written at rows_cap * p * T from base
    got = np.concatenate([ap[rows_cap * p * T: rows_cap * p * T + P * T] for p in range(3)])
    show(f"plane stride rows_cap={rows_cap}", got, op[:3*P*T])
# per-plane, per-row at plane stride 2 (as coded)
for p in range(3):
    for t in range(P):
        a = ap[p * P * T + t * T: p * P * T + (t + 1) * T]
        o = op[p * P * T + t * T: p * P * T + (t + 1) * T]
        show(f"preconv plane {p} row {t}", a, o)

print("\n== qkv per-plane-row ==")
aq, oq = atlas["qkv"], oracle["qkv"]
for t in range(P):
    for p in range(3):
        a = aq[t * 3 * T + p * T: t * 3 * T + (p + 1) * T]
        o = oq[t * 3 * T + p * T: t * 3 * T + (p + 1) * T]
        show(f"qkv row {t} plane {p}", a, o)

print("\n== normed detail ==")
an, on = atlas["normed"], oracle["normed"]
for t in range(P):
    show(f"normed row {t}", an[t * H:(t + 1) * H], on[t * H:(t + 1) * H])

print("\n== first 8 elements, preconv plane0 row0 ==")
print("atlas:", ap[:8])
print("oracle:", op[:8])
print("atlas row1:", ap[T:T + 8])
print("oracle row1:", op[T:T + 8])
