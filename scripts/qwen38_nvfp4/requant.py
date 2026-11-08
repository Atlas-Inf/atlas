#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Requantize the FP8 projections of nvidia/Qwen3.8-27B-NVFP4 to NVFP4 and build
a checkpoint overlay the dense qwen3_5 loader serves natively.

The checkpoint ships the 48 Gated DeltaNet (GDN) layers' in_proj_qkv / in_proj_z
/ out_proj and the 16 attention layers' q/k/v/o as FP8 E4M3 with one scale per
tensor. Atlas serves the GDN ones as FP8 (`w8a16`) and requantizes attention to
NVFP4 at load (round to nearest). This tool writes them in the ModelOpt NVFP4
layout instead: E2M1 codes two to a byte, low nibble first (`.weight`, U8
[N, K/2]); one E4M3 scale per 16 inputs (`.weight_scale`, [N, K/16]); one F32
`.weight_scale_2` per tensor. in_proj_qkv and in_proj_z share one
weight_scale_2, because the loader concatenates them into qkvz.

  requant.py quant SRC OUT.safetensors --which gdn|attn --mode rtn|mse|gptq
             [--cap DIR] [--report R.json] [--layers 0,1,...] [--device cuda]
  requant.py overlay SRC OUT_DIR FILE[:LAYERS[:PROJS]] ...

Modes (each starts from the FP8 weight times its scale, in FP32):
  rtn   block scale = E4M3(amax / 6 / s2), s2 = tensor amax / (6 * 448); round to nearest.
  mse   the same s2; per 16-block the E4M3 scale code with the least squared error
        among the round-to-nearest code and its neighbours (c-4 .. c+1).
  gptq  GPTQ (Frantar et al., 2022) on the NVFP4 grid, GDN only: with H = X^T X of
        the projection's calibration inputs, columns are rounded in blocks of 128 and
        each column's error is spread over the later columns through the Cholesky
        factor of H^-1 (damping 1% of mean diag H). Each 16-group's scale is chosen as
        in mse on the error-updated weights when the group's first column is reached.
        --cap holds L{ssm:02}_qkvz.bin / L{ssm:02}_out.bin: raw BF16 rows of each GDN
        layer's in_proj_qkvz input [*, 5120] and out_proj input [*, 6144], in SSM-layer
        order, captured from calibration prefills (capture_hook.diff + calib.py).
The report gives each projection's relative output error tr(D H D^T) / tr(W H W^T)
for rtn, mse and gptq when calibration inputs are given, else the weight error.

overlay: a directory that is SRC with the projections found in each FILE replaced
(LAYERS: comma list or a-b ranges; PROJS: e.g. in_proj_qkv,in_proj_z to keep
out_proj FP8). Large files are relative symlinks, so SRC and the overlay must sit
side by side under the directory the server mounts.
"""
import argparse
import json
import os
import struct

GRID = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
PRE = "model.language_model.layers"


def quant_main(a):
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    dev = a.device
    grid = torch.tensor(GRID, device=dev)
    mid = torch.tensor(MID, device=dev)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    types = cfg.get("text_config", cfg)["layer_types"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(a.src, f), "pt", device="cpu")
        return handles[f].get_tensor(name)

    def fp8_weight(prefix):
        s = get(prefix + ".weight_scale")
        assert s.numel() == 1, f"{prefix}: expected one FP8 scale per tensor"
        return get(prefix + ".weight").to(dev).float() * s.to(dev).float()

    def e2m1(x):
        i = torch.bucketize(x.abs(), mid)
        neg = ((x < 0) & (i > 0)).to(i.dtype)
        return torch.where(x < 0, -grid[i], grid[i]), (i | (neg << 3)).to(torch.uint8)

    def code_val(c):
        return c.to(torch.uint8).view(torch.float8_e4m3fn).float()

    def rtn_code(b, s2):
        amax = b.abs().amax(-1)
        return (amax / 6 / s2).clamp(max=448).to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)

    def mse_code(b, s2):
        c0 = rtn_code(b, s2)
        best_err = best = None
        for d in (-4, -3, -2, -1, 0, 1):
            c = (c0 + d).clamp(0, 0x7E)
            sc = code_val(c) * s2
            v, _ = e2m1(b / torch.where(sc > 0, sc, torch.ones_like(sc)).unsqueeze(-1))
            err = ((v * sc.unsqueeze(-1) - b) ** 2).sum(-1)
            if best is None:
                best_err, best = err, c
            else:
                m = err < best_err
                best_err, best = torch.where(m, err, best_err), torch.where(m, c, best)
        return best

    def blockwise(w, s2, mse):
        n, k = w.shape
        b = w.view(n, k // 16, 16)
        c = mse_code(b, s2) if mse else rtn_code(b, s2)
        sc = code_val(c) * s2
        v, cd = e2m1(b / torch.where(sc > 0, sc, torch.ones_like(sc)).unsqueeze(-1))
        return (v * sc.unsqueeze(-1)).view(n, k), cd.view(n, k), c

    def gptq(w, h, s2, blk=128):
        n, k = w.shape
        w = w.clone()
        h = h.clone()
        dead = torch.diag(h) == 0
        h[dead, dead] = 1
        w[:, dead] = 0
        h += 0.01 * torch.mean(torch.diag(h)) * torch.eye(k, device=dev)
        hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
        q = torch.zeros_like(w)
        codes = torch.zeros(n, k, dtype=torch.uint8, device=dev)
        scodes = torch.zeros(n, k // 16, dtype=torch.int16, device=dev)
        for i1 in range(0, k, blk):
            i2 = min(i1 + blk, k)
            w1 = w[:, i1:i2].clone()
            e1 = torch.zeros_like(w1)
            hi = hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                col = i1 + i
                if col % 16 == 0:
                    c = mse_code(w1[:, i:i + 16].unsqueeze(1), s2)[:, 0]
                    scodes[:, col // 16] = c
                    sc = code_val(c) * s2
                    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
                v, cd = e2m1(w1[:, i] / safe)
                q[:, col] = v * sc
                codes[:, col] = cd
                err = (w1[:, i] - q[:, col]) / hi[i, i]
                w1[:, i:] -= err.unsqueeze(1) * hi[i, i:].unsqueeze(0)
                e1[:, i] = err
            w[:, i2:] -= e1 @ hinv[i1:i2, i2:]
        return q, codes, scodes

    def load_h(path, k):
        x = np.fromfile(path, dtype=np.uint16).reshape(-1, k)
        t = torch.from_numpy(x.astype(np.int32) << 16).view(torch.float32).to(dev)
        return t.T @ t / t.shape[0], t.shape[0]

    def rel_out(w, q, h):
        d = w - q
        if h is None:
            return float((d.norm() / w.norm()).item())
        return float((torch.einsum("nk,kj,nj->", d, h, d) / torch.einsum("nk,kj,nj->", w, h, w)).item())

    if a.mode == "gptq":
        assert a.which == "gdn" and a.cap, "gptq needs --which gdn and --cap"
    want = {int(x) for x in a.layers.split(",") if x}
    out, report, ssm = {}, {}, -1
    for li, t in enumerate(types):
        ssm += t == "linear_attention"
        if want and li not in want:
            continue
        if a.which == "gdn" and t == "linear_attention":
            groups = [("qkvz", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]), ("out", ["linear_attn.out_proj"])]
        elif a.which == "attn" and t == "full_attention":
            groups = [(p, [f"self_attn.{p}"]) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        else:
            continue
        for kind, projs in groups:
            ws = [fp8_weight(f"{PRE}.{li}.{p}") for p in projs]
            w = torch.cat(ws, 0)
            s2 = float(w.abs().max().item()) / (6 * 448)
            h = None
            if a.cap:
                h, _ = load_h(os.path.join(a.cap, f"L{ssm:02d}_{kind}.bin"), w.shape[1])
            if a.mode == "gptq":
                q, codes, sc = gptq(w, h, s2)
            else:
                q, codes, sc = blockwise(w, s2, a.mode == "mse")
            r = {a.mode: rel_out(w, q, h)}
            if h is not None:
                for m in ("rtn", "mse"):
                    if m != a.mode:
                        r[m] = rel_out(w, blockwise(w, s2, m == "mse")[0], h)
            report[f"{li}.{kind}"] = r
            print(li, kind, {m: round(v, 5) for m, v in r.items()}, flush=True)
            r0 = 0
            for p, wp in zip(projs, ws):
                r1 = r0 + wp.shape[0]
                cd = codes[r0:r1]
                name = f"{PRE}.{li}.{p}"
                out[name + ".weight"] = (cd[:, 0::2] | (cd[:, 1::2] << 4)).contiguous().cpu()
                out[name + ".weight_scale"] = sc[r0:r1].to(torch.uint8).view(torch.float8_e4m3fn).contiguous().cpu()
                out[name + ".weight_scale_2"] = torch.tensor(s2, dtype=torch.float32)
                r0 = r1
            del ws, w, h, q, codes, sc
    save_file(out, a.out, metadata={"format": "pt", "quantizer": f"requant.py {a.which} {a.mode}"})
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    print("wrote", a.out, len(out), "tensors")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def ranges(spec):
    s = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            s |= set(range(int(lo), int(hi) + 1))
        elif part:
            s.add(int(part))
    return s


def overlay_main(a):
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(src):
        dst = os.path.join(out, name)
        if name not in ("model.safetensors.index.json", "hf_quant_config.json") and not os.path.lexists(dst):
            os.symlink(os.path.relpath(os.path.join(src, name), out), dst)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = index["weight_map"]
    hq = json.load(open(os.path.join(src, "hf_quant_config.json")))
    ql = hq["quantization"]["quantized_layers"]
    n = 0
    for arg in a.files:
        parts = arg.split(":")
        path = os.path.abspath(parts[0])
        layers = ranges(parts[1]) if len(parts) > 1 and parts[1] else None
        projs = set(parts[2].split(",")) if len(parts) > 2 and parts[2] else None
        link = os.path.basename(path)
        if not os.path.lexists(os.path.join(out, link)):
            os.symlink(os.path.relpath(path, out), os.path.join(out, link))
        mods = {}
        for t in header(path):
            if t == "__metadata__":
                continue
            mod = t.rsplit(".", 1)[0]
            li = int(mod.split(".layers.")[1].split(".")[0])
            if (layers is None or li in layers) and (projs is None or mod.rsplit(".", 1)[1] in projs):
                mods.setdefault(mod, []).append(t)
        for mod, ts in mods.items():
            for k in [k for k in wm if k.startswith(mod + ".")]:
                del wm[k]
            for t in ts:
                wm[t] = link
            ql[mod] = {"quant_algo": "NVFP4", "group_size": 16}
            n += 1
    json.dump(index, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    json.dump(hq, open(os.path.join(out, "hf_quant_config.json"), "w"), indent=4)
    print(out, "replaced", n, "modules")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quant")
    q.add_argument("src")
    q.add_argument("out")
    q.add_argument("--which", choices=["gdn", "attn"], required=True)
    q.add_argument("--mode", choices=["rtn", "mse", "gptq"], required=True)
    q.add_argument("--cap")
    q.add_argument("--report")
    q.add_argument("--layers", default="")
    q.add_argument("--device", default="cuda")
    o = sub.add_parser("overlay")
    o.add_argument("src")
    o.add_argument("out")
    o.add_argument("files", nargs="+")
    args = ap.parse_args()
    (quant_main if args.cmd == "quant" else overlay_main)(args)
