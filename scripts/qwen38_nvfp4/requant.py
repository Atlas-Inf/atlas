#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Requantize Qwen3.8-27B's GDN (and attention) projections to NVFP4 and build a
checkpoint overlay of nvidia/Qwen3.8-27B-NVFP4 that the dense qwen3_5 loader
serves natively.

nvidia/Qwen3.8-27B-NVFP4 ships the 48 Gated DeltaNet (GDN) layers' in_proj_qkv /
in_proj_z / out_proj and the 16 attention layers' q/k/v/o as FP8 E4M3 with one
scale per tensor. Atlas serves the GDN ones as FP8 (`w8a16`) and requantizes
attention to NVFP4 at load. This tool writes them in the ModelOpt NVFP4 layout:
E2M1 codes two to a byte, low nibble first (`.weight`, U8 [N, K/2]); one E4M3
scale per 16 inputs (`.weight_scale`, [N, K/16]); one F32 `.weight_scale_2` per
tensor. in_proj_qkv and in_proj_z share one weight_scale_2, because the loader
concatenates them into qkvz.

  requant.py quant SRC OUT.safetensors --which gdn|attn --mode rtn|mse
             [--bf16 DIR] [--report R.json] [--layers 0,1,...] [--device cuda]
  requant.py capture-gptq SRC --cap DIR --bf16 DIR --gptq OUT --mse OUT --report R.json
  requant.py overlay SRC OUT_DIR FILE[:LAYERS[:PROJS]] ...

Source weights: SRC's FP8 tensors times their scale, or, with --bf16, the BF16
tensors of the same names found in DIR's *.safetensors (for example the GDN
projections cut from Qwen/Qwen3.8-27B one shard at a time, stream_bf16.sh).

  rtn   block scale = E4M3(amax / 6 / s2), s2 = tensor amax / (6 * 448); round to nearest.
  mse   the same s2; per 16-block the E4M3 scale code with the least squared error
        among the round-to-nearest code and its neighbours (c-4 .. c+1).
  gptq  (capture-gptq; GDN) GPTQ (Frantar et al., 2022) on the NVFP4 grid: with H =
        X^T X of the projection's calibration inputs, columns are rounded in blocks of
        128 and each column's error is spread over the later columns through the
        Cholesky factor of H^-1 (damping 1% of mean diag H); each 16-group's scale is
        chosen as in mse on the error-updated weights when its first column is reached.

capture-gptq folds X^T X on the GPU from the files a server built with
capture_hook.diff writes under ATLAS_GDN_CAPTURE (L{ssm:02}_{qkvz,out}_{seq}.bin,
raw BF16 rows; each is deleted once read) while calib.py sends the calibration
prefills, until DIR/DONE appears; then it writes the GPTQ and the mse files.
H never touches the disk. The report gives each GDN projection's relative output
error tr(D H D^T) / tr(W H W^T) for rtn, mse and gptq.

overlay: a directory that is SRC with the projections found in each FILE replaced
(LAYERS: comma list or a-b ranges; PROJS: e.g. in_proj_qkv,in_proj_z to keep
out_proj FP8). Large files are relative symlinks, so SRC and the overlay must sit
side by side under the directory the server mounts.
"""
import argparse
import glob
import json
import os
import re
import struct
import time

GRID = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
PRE = "model.language_model.layers"


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


class Quant:
    """The NVFP4 grid, scale choices and GPTQ, on one torch device."""

    def __init__(self, dev):
        import torch

        self.t, self.dev = torch, dev
        self.grid = torch.tensor(GRID, device=dev)
        self.mid = torch.tensor(MID, device=dev)

    def e2m1(self, x):
        t = self.t
        i = t.bucketize(x.abs(), self.mid)
        neg = ((x < 0) & (i > 0)).to(i.dtype)
        return t.where(x < 0, -self.grid[i], self.grid[i]), (i | (neg << 3)).to(t.uint8)

    def code_val(self, c):
        return c.to(self.t.uint8).view(self.t.float8_e4m3fn).float()

    def rtn_code(self, b, s2):
        amax = b.abs().amax(-1)
        e4 = self.t.float8_e4m3fn
        return (amax / 6 / s2).clamp(max=448).to(e4).view(self.t.uint8).to(self.t.int16)

    def mse_code(self, b, s2):
        t = self.t
        c0 = self.rtn_code(b, s2)
        best_err = best = None
        for d in (-4, -3, -2, -1, 0, 1):
            c = (c0 + d).clamp(0, 0x7E)
            sc = self.code_val(c) * s2
            v, _ = self.e2m1(b / t.where(sc > 0, sc, t.ones_like(sc)).unsqueeze(-1))
            err = ((v * sc.unsqueeze(-1) - b) ** 2).sum(-1)
            if best is None:
                best_err, best = err, c
            else:
                m = err < best_err
                best_err, best = t.where(m, err, best_err), t.where(m, c, best)
        return best

    def blockwise(self, w, s2, mse):
        t = self.t
        n, k = w.shape
        b = w.view(n, k // 16, 16)
        c = self.mse_code(b, s2) if mse else self.rtn_code(b, s2)
        sc = self.code_val(c) * s2
        v, cd = self.e2m1(b / t.where(sc > 0, sc, t.ones_like(sc)).unsqueeze(-1))
        return (v * sc.unsqueeze(-1)).view(n, k), cd.view(n, k), c

    def gptq(self, w, h, s2, blk=128):
        t = self.t
        n, k = w.shape
        w = w.clone()
        h = h.clone()
        dead = t.diag(h) == 0
        h[dead, dead] = 1
        w[:, dead] = 0
        h += 0.01 * t.mean(t.diag(h)) * t.eye(k, device=self.dev)
        hinv = t.linalg.cholesky(t.cholesky_inverse(t.linalg.cholesky(h)), upper=True)
        q = t.zeros_like(w)
        codes = t.zeros(n, k, dtype=t.uint8, device=self.dev)
        scodes = t.zeros(n, k // 16, dtype=t.int16, device=self.dev)
        for i1 in range(0, k, blk):
            i2 = min(i1 + blk, k)
            w1 = w[:, i1:i2].clone()
            e1 = t.zeros_like(w1)
            hi = hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                col = i1 + i
                if col % 16 == 0:
                    c = self.mse_code(w1[:, i:i + 16].unsqueeze(1), s2)[:, 0]
                    scodes[:, col // 16] = c
                    sc = self.code_val(c) * s2
                    safe = t.where(sc > 0, sc, t.ones_like(sc))
                v, cd = self.e2m1(w1[:, i] / safe)
                q[:, col] = v * sc
                codes[:, col] = cd
                err = (w1[:, i] - q[:, col]) / hi[i, i]
                w1[:, i:] -= err.unsqueeze(1) * hi[i, i:].unsqueeze(0)
                e1[:, i] = err
            w[:, i2:] -= e1 @ hinv[i1:i2, i2:]
        return q, codes, scodes

    def rel_out(self, w, q, h):
        d = w - q
        if h is None:
            return float((d.norm() / w.norm()).item())
        es = self.t.einsum
        return float((es("nk,kj,nj->", d, h, d) / es("nk,kj,nj->", w, h, w)).item())


class Source:
    """Projection weights in FP32: SRC's FP8 times scale, or BF16 from --bf16."""

    def __init__(self, src, bf16, dev):
        from safetensors import safe_open

        self.open, self.dev, self.handles = safe_open, dev, {}
        self.idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
        self.idx = {k: os.path.join(src, f) for k, f in self.idx.items()}
        self.bf16 = {}
        for p in sorted(glob.glob(os.path.join(bf16, "*.safetensors"))) if bf16 else []:
            self.bf16.update({k: p for k in header(p) if k != "__metadata__"})
        cfg = json.load(open(os.path.join(src, "config.json")))
        self.types = cfg.get("text_config", cfg)["layer_types"]

    def get(self, path, name):
        if path not in self.handles:
            self.handles[path] = self.open(path, "pt", device="cpu")
        return self.handles[path].get_tensor(name)

    def weight(self, prefix):
        if self.bf16:
            return self.get(self.bf16[prefix + ".weight"], prefix + ".weight").to(self.dev).float()
        s = self.get(self.idx[prefix + ".weight_scale"], prefix + ".weight_scale")
        assert s.numel() == 1, f"{prefix}: expected one FP8 scale per tensor"
        w = self.get(self.idx[prefix + ".weight"], prefix + ".weight")
        return w.to(self.dev).float() * s.to(self.dev).float()

    def groups(self, which, layers):
        """(layer, ssm index, kind, projection names) in model order."""
        ssm = -1
        for li, t in enumerate(self.types):
            ssm += t == "linear_attention"
            if layers and li not in layers:
                continue
            if which == "gdn" and t == "linear_attention":
                yield li, ssm, "qkvz", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
                yield li, ssm, "out", ["linear_attn.out_proj"]
            elif which == "attn" and t == "full_attention":
                for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    yield li, ssm, p, [f"self_attn.{p}"]


def pack(out, li, projs, ws, codes, sc, s2):
    import torch

    r0 = 0
    for p, wp in zip(projs, ws):
        r1 = r0 + wp.shape[0]
        cd = codes[r0:r1]
        name = f"{PRE}.{li}.{p}"
        out[name + ".weight"] = (cd[:, 0::2] | (cd[:, 1::2] << 4)).contiguous().cpu()
        e4 = torch.float8_e4m3fn
        out[name + ".weight_scale"] = sc[r0:r1].to(torch.uint8).view(e4).contiguous().cpu()
        out[name + ".weight_scale_2"] = torch.tensor(s2, dtype=torch.float32)
        r0 = r1


def quant_main(a):
    import torch
    from safetensors.torch import save_file

    q, src = Quant(a.device), Source(a.src, a.bf16, a.device)
    want = {int(x) for x in a.layers.split(",") if x}
    out, report = {}, {}
    for li, _, kind, projs in src.groups(a.which, want):
        ws = [src.weight(f"{PRE}.{li}.{p}") for p in projs]
        w = torch.cat(ws, 0)
        s2 = float(w.abs().max().item()) / (6 * 448)
        dq, codes, sc = q.blockwise(w, s2, a.mode == "mse")
        report[f"{li}.{kind}"] = {a.mode: q.rel_out(w, dq, None)}
        print(li, kind, report[f"{li}.{kind}"], flush=True)
        pack(out, li, projs, ws, codes, sc, s2)
    tag = "bf16" if a.bf16 else "fp8"
    save_file(out, a.out, metadata={"format": "pt", "quantizer": f"requant.py {a.which} {a.mode} from {tag}"})
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    print("wrote", a.out, len(out), "tensors")


def capture_gptq_main(a):
    import numpy as np
    import torch
    from safetensors.torch import save_file

    q, src = Quant("cuda"), Source(a.src, a.bf16, "cuda")
    dims = {"qkvz": 5120, "out": 6144}
    hs, ns = {}, {}
    pat = re.compile(r"L(\d\d)_(qkvz|out)_(\d+)\.bin$")
    t0 = time.time()
    while True:
        files = sorted((f for f in os.listdir(a.cap) if pat.search(f)), key=lambda f: int(pat.search(f)[3]))
        if not files:
            if os.path.exists(os.path.join(a.cap, "DONE")):
                break
            time.sleep(0.5)
            continue
        for f in files:
            m = pat.search(f)
            key = (int(m[1]), m[2])
            path = os.path.join(a.cap, f)
            x = np.fromfile(path, dtype=np.uint16).reshape(-1, dims[m[2]])
            os.unlink(path)
            xt = torch.from_numpy(x.astype(np.int32) << 16).view(torch.float32).cuda()
            if key not in hs:
                hs[key] = torch.zeros(xt.shape[1], xt.shape[1], device="cuda")
                ns[key] = 0
            hs[key].addmm_(xt.T, xt)
            ns[key] += xt.shape[0]
        print(f"{time.time() - t0:7.1f}s folded, tokens per layer {sorted(set(ns.values()))}", flush=True)
    assert len(set(ns.values())) == 1, f"uneven capture: {sorted(set(ns.values()))}"
    gout, mout, report = {}, {}, {}
    for li, ssm, kind, projs in src.groups("gdn", set()):
        h = hs.pop((ssm, kind)) / ns[(ssm, kind)]
        ws = [src.weight(f"{PRE}.{li}.{p}") for p in projs]
        w = torch.cat(ws, 0)
        s2 = float(w.abs().max().item()) / (6 * 448)
        dq, codes, sc = q.gptq(w, h, s2)
        pack(gout, li, projs, ws, codes, sc, s2)
        r = {"tokens": ns[(ssm, kind)], "gptq": q.rel_out(w, dq, h)}
        dq, codes, sc = q.blockwise(w, s2, True)
        pack(mout, li, projs, ws, codes, sc, s2)
        r["mse"] = q.rel_out(w, dq, h)
        r["rtn"] = q.rel_out(w, q.blockwise(w, s2, False)[0], h)
        report[f"{li}.{kind}"] = r
        print(li, kind, {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        del h, ws, w, dq, codes, sc
    tag = "bf16" if a.bf16 else "fp8"
    save_file(gout, a.gptq, metadata={"format": "pt", "quantizer": f"requant.py gdn gptq from {tag}"})
    save_file(mout, a.mse, metadata={"format": "pt", "quantizer": f"requant.py gdn mse from {tag}"})
    json.dump(report, open(a.report, "w"), indent=1)
    print("wrote", a.gptq, a.mse)


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
    q.add_argument("--mode", choices=["rtn", "mse"], required=True)
    q.add_argument("--bf16")
    q.add_argument("--report")
    q.add_argument("--layers", default="")
    q.add_argument("--device", default="cuda")
    g = sub.add_parser("capture-gptq")
    g.add_argument("src")
    for k in ("--cap", "--gptq", "--mse", "--report"):
        g.add_argument(k, required=True)
    g.add_argument("--bf16")
    o = sub.add_parser("overlay")
    o.add_argument("src")
    o.add_argument("out")
    o.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"quant": quant_main, "capture-gptq": capture_gptq_main, "overlay": overlay_main}[args.cmd](args)
