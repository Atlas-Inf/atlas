#!/usr/bin/env python3
# Ling-3.2-air-NVFP4 end-to-end sequential oracle (campaign slice
# parity-mla-head, endgame localization). CPU torch, naive ops, real weights.
#
# Conventions (all validated per-path on this branch):
#   - KDA (fused_recurrent_kda semantics, PR #118): conv kernel 4 fused SiLU,
#     q l2-norm * 128^-0.5, k l2-norm, safe-gate decay exp(LB/(1+exp(-a*(g+dt)))),
#     sigmoid beta, H[K,V] state, FusedRMSNormGated(eps 1e-6, sigmoid) o_norm.
#   - gated MLA (PR #119): eager per-head attention, partial pair-interleaved
#     rope theta 6e6, scale 192^-0.5, head-wise sigmoid output gate.
#   - MoE (PR #120): reciprocal NVFP4 convention w = e2m1*sf/wgs; activation
#     quant dequant q*sf/ig (a_scale2 = 1/ig of expert-0 gate); re-quant with
#     per-expert ig(down); sigmoid+bias group routing (8 groups x top4),
#     norm-topk * 2.5, bias selection-only; routed SwiGLU limit 4 @35-41;
#     shared BF16 expert limit 5 @34-40, 7 @41; final = rb(shared + rb(sum w*y)).
#   - Dense MLP (layers 0-1): full SwiGLU, BF16, no clamps.
#   - Embeddings: model.word_embeddings rows, bf16.
#   - Head: model.norm (RMS eps 1e-6) + lm_head, vocab 157184.
#
# Output (OUT dir): oracle_pre.f32 [43, PROMPT, 2560] (index 0 = embed,
# L+1 = after layer L), oracle_dec.f32 [DECODE, 43, 2560], oracle_meta.json.

import json
import math
import os
import struct
import sys

import torch

torch.set_grad_enabled(False)
torch.set_num_threads(20)

CKPT = os.environ.get("ATLAS_LING_CKPT", "/home/spark/ling/Ling-3.2-air-NVFP4")
OUT = os.environ.get("ATLAS_LING_E2E_OUT", "e2e_oracle")

_env_tokens = os.environ.get("ATLAS_LING_E2E_TOKENS")
TOKENS = ([int(t) for t in _env_tokens.split(",") if t.strip()]
          if _env_tokens else
          [25951, 16278, 113926, 21053, 94766, 1061, 97529, 15953,
           128054, 109169, 137868, 60077, 69809, 26636, 133761, 41880,
           40020, 45362, 38145, 33557])
PROMPT = int(os.environ.get("ATLAS_LING_E2E_PROMPT_LEN", "16"))
DECODE = 4
NTOK = PROMPT + DECODE

H = 2560
LAYERS = 42
HEADS = 32
KDA_T = HEADS * 128
CONV = 4
LORA = 512
NOPE = 128
ROPE = 64
VD = 128
EXPERTS = 512
TOPK = 8
INTER = 768
N_GROUP = 8
TOPK_GROUP = 4
SCALING = 2.5
EPS = 1e-6
LOWER = -5.0
THETA = 6_000_000.0
VOCAB = 157_184
MLA_LAYERS = {5, 11, 17, 23, 29, 35, 41}
ROUTED_LIMIT = [0] * 35 + [4] * 7          # layers 35..41 -> 4
SHARED_LIMIT = [0] * 34 + [5] * 7 + [7]     # 34..40 -> 5, 41 -> 7

from safetensors import safe_open  # noqa: E402

_index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
_handles = {}


def raw(name):
    shard = _index[name]
    h = _handles.get(shard)
    if h is None:
        h = safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu")
        _handles[shard] = h
    return h.get_tensor(name)


def w32(name):
    t = raw(name)
    assert t.dtype in (torch.bfloat16, torch.float32), (name, t.dtype)
    return t.to(torch.float32)


def rb(x):
    return x.to(torch.bfloat16).to(torch.float32)


def rms(x, w):
    """BailingMoeV3RMSNorm: fp32 stats, round to bf16, weight product."""
    inv = torch.rsqrt(x.pow(2).mean(-1) + EPS)
    return rb(w * rb(x * inv))


def gemv(x, w):
    """fp32 dot, one bf16 round (the house dense_gemv contract)."""
    return rb(x.double() @ w.double().T if x.numel() <= H else x @ w.T)


# ── NVFP4 mirrors of ling_moe_quant.cuh (bit-identical by construction) ────
def _e2m1_quantize_scalar(v):
    absv = abs(v)
    sign = 8 if v < 0 else 0
    if absv <= 0.25:
        idx = 0
    elif absv <= 0.75:
        idx = 1
    elif absv <= 1.25:
        idx = 2
    elif absv <= 1.75:
        idx = 3
    elif absv <= 2.5:
        idx = 4
    elif absv <= 3.5:
        idx = 5
    elif absv <= 5.0:
        idx = 6
    else:
        idx = 7
    return sign | idx


_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_LUT = torch.tensor([_E2M1_MAG[c & 7] * (-1.0 if c & 8 else 1.0) for c in range(16)])


def _f32_to_e4m3(v):
    bits = struct.unpack(">I", struct.pack(">f", v))[0]
    sign = (bits >> 31) & 1
    if bits & 0x7FFFFFFF == 0:
        return sign << 7
    a = abs(v)
    if a > 448.0:
        a = 448.0
    abits = struct.unpack(">I", struct.pack(">f", a))[0]
    f32_exp = ((abits >> 23) & 0xFF) - 127
    f32_man = abits & 0x7FFFFF
    if f32_exp < -9:
        return sign << 7
    if f32_exp < -6:
        man = min(max(int(a * 512.0 + 0.5), 0), 7)
        return (sign << 7) | man
    e = f32_exp + 7
    if e < 1:
        e = 1
    if e > 15:
        e = 15
        man = 6
    else:
        m = (f32_man + (1 << 19)) >> 20
        if m > 7:
            m = 0
            e += 1
            if e > 15:
                e = 15
                m = 6
        man = m
    return (sign << 7) | (e << 3) | man


def _e4m3_decode(b7):
    exp = (b7 >> 3) & 0xF
    man = b7 & 7
    if exp == 0:
        return man * 0.001953125
    if exp == 15 and man == 7:
        return 0.0
    return struct.unpack(">f", struct.pack(
        ">I", ((exp + 120) << 23) | (man << 20)))[0]


E4M3_LUT = torch.tensor([_e4m3_decode(b & 0x7F) for b in range(128)])


def quantize_group_act(x16, scale2):
    """ling_quantize_group mirror -> dequantized 16 values (q*sf*scale2)."""
    amax = float(x16.abs().max())
    fp8_float = amax / scale2 / 6.0 if amax > 0.0 else 0.0
    decoded = float(E4M3_LUT[_f32_to_e4m3(fp8_float)])
    effective = decoded * scale2
    inv_eff = 1.0 / effective if effective > 0.0 else 0.0
    nibs = torch.tensor([_e2m1_quantize_scalar(float(v) * inv_eff) for v in x16])
    return E2M1_LUT[nibs] * decoded * scale2


def requant_group_mid(h16, ig_down):
    """ling_group_scale mirror for the gate/up -> down intermediate."""
    amax = float(h16.abs().max())
    byte = _f32_to_e4m3(amax * ig_down / 6.0)
    decoded = float(E4M3_LUT[byte])
    inv = ig_down / decoded if decoded > 0.0 else 0.0
    nibs = torch.tensor([_e2m1_quantize_scalar(float(v) * inv) for v in h16])
    return E2M1_LUT[nibs] * decoded / ig_down


def dequant_nvfp4(packed, scale, n, k, wgs):
    """w = e2m1(nibble) * e4m3_group / weight_global_scale, [n, k] f32."""
    p = packed.view(torch.uint8).reshape(n, k // 2)
    lo = (p & 0xF).long()
    hi = (p >> 4).long()
    vals = torch.empty(n, k)
    vals[:, 0::2] = E2M1_LUT[lo]
    vals[:, 1::2] = E2M1_LUT[hi]
    s = scale.view(torch.uint8).reshape(n, k // 16).long()
    sf = E4M3_LUT[s]
    return vals * sf.repeat_interleave(16, dim=1) / wgs


def silu_clamped(g, limit):
    s = rb(g * torch.sigmoid(g))
    if limit > 0.0:
        s = torch.minimum(s, torch.tensor(limit))
    return s


def gate_up_mul(gs, u, limit):
    if limit > 0.0:
        u = u.clamp(-limit, limit)
    return rb(gs * u)


def rope_pairs(x64, pos):
    """Pair-interleaved partial rope, theta 6e6."""
    out = x64.clone()
    for p in range(ROPE // 2):
        ang = (THETA ** (-(2 * p) / ROPE)) * pos
        c, s = math.cos(ang), math.sin(ang)
        e, o = float(x64[2 * p]), float(x64[2 * p + 1])
        out[2 * p] = e * c - o * s
        out[2 * p + 1] = o * c + e * s
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    TRACE_LAYER = int(os.environ.get("ATLAS_LING_E2E_TRACE_LAYER", "-1"))
    trace = {}
    final_norm = w32("model.norm.weight")
    lm_head = w32("lm_head.weight")
    embed = w32("model.word_embeddings.weight")

    h = torch.stack([embed[t] for t in TOKENS])  # [NTOK, H]

    pre_h = torch.zeros(LAYERS + 1, PROMPT, H)
    pre_h[0] = h[:PROMPT]
    dec_h = torch.zeros(DECODE, LAYERS + 1, H)
    dec_h[:, 0] = h[PROMPT:]

    # Layer-major pass: KDA/MLA states belong to one layer each, so each
    # layer advances token-by-token in causal order.
    conv_state = None
    kda_h = None
    mla_cache = []
    expert_cache = {}

    routes = {}
    for t in range(NTOK):
        key = f"pre_{t}" if t < PROMPT else f"dec_{t - PROMPT}"
        routes[key] = {}

    def moe_ffn(L, n, token_key):
        lp = f"model.layers.{L}.mlp"
        gate_w = w32(f"{lp}.gate.weight")
        bias = w32(f"{lp}.gate.expert_bias")
        logits = (n.double() @ gate_w.double().T).float()
        scores = torch.sigmoid(logits)
        sel = scores + bias
        gsize = EXPERTS // N_GROUP
        gsel = sel.view(N_GROUP, gsize)
        top2 = gsel.sort(dim=1, descending=True).values[:, :2].sum(1)
        picked = torch.topk(top2, TOPK_GROUP).indices.tolist()
        work = sel.view(N_GROUP, gsize).clone()
        for g in range(N_GROUP):
            if g not in picked:
                work[g, :] = -float("inf")
        flat = work.reshape(-1)
        chosen = []
        for _ in range(TOPK):
            at = int(torch.argmax(flat).item())
            chosen.append(at)
            flat[at] = -float("inf")
        vals = scores[torch.tensor(chosen)]
        wts = vals / vals.sum() * SCALING
        routes[token_key][L] = {
            "ids": chosen,
            "w": [round(float(v), 6) for v in wts],
        }

        ig_a = float(raw(f"{lp}.experts.0.gate_proj.input_global_scale"))
        a_scale2 = 1.0 / ig_a
        a_deq = torch.cat([
            quantize_group_act(n[g * 16:(g + 1) * 16], a_scale2)
            for g in range(H // 16)
        ])

        rl = ROUTED_LIMIT[L]
        slot_y = []
        for e in chosen:
            if (L, e) not in expert_cache:
                def dq(proj, n_, k_):
                    wgs = float(raw(f"{lp}.experts.{e}.{proj}.weight_global_scale"))
                    return dequant_nvfp4(
                        raw(f"{lp}.experts.{e}.{proj}.weight_packed"),
                        raw(f"{lp}.experts.{e}.{proj}.weight_scale"),
                        n_, k_, wgs)
                expert_cache[(L, e)] = (
                    dq("gate_proj", INTER, H),
                    dq("up_proj", INTER, H),
                    dq("down_proj", H, INTER),
                )
            wg, wu, wd = expert_cache[(L, e)]
            g_row = rb(a_deq.double() @ wg.double().T)
            u_row = rb(a_deq.double() @ wu.double().T)
            hmid = gate_up_mul(silu_clamped(g_row, rl), u_row, rl)
            ig_dn = float(raw(f"{lp}.experts.{e}.down_proj.input_global_scale"))
            h_deq = torch.cat([
                requant_group_mid(hmid[g * 16:(g + 1) * 16], ig_dn)
                for g in range(INTER // 16)
            ])
            slot_y.append(rb(h_deq.double() @ wd.double().T))
        acc = torch.zeros(H)
        for t_, y in enumerate(slot_y):
            acc += wts[t_] * y
        routed = rb(acc)

        sl = SHARED_LIMIT[L]
        sg = w32(f"{lp}.shared_experts.gate_proj.weight")
        su = w32(f"{lp}.shared_experts.up_proj.weight")
        sd = w32(f"{lp}.shared_experts.down_proj.weight")
        g_row = rb(n.double() @ sg.double().T)
        u_row = rb(n.double() @ su.double().T)
        hsh = gate_up_mul(silu_clamped(g_row, sl), u_row, sl)
        ysh = rb(hsh.double() @ sd.double().T)
        return rb(routed + ysh)

    # ── the layer-major forward ─────────────────────────────────────────────
    for L in range(LAYERS):
        lp = f"model.layers.{L}"
        in_norm = w32(f"{lp}.input_layernorm.weight")
        post_norm = w32(f"{lp}.post_attention_layernorm.weight")
        is_mla = L in MLA_LAYERS
        if is_mla:
            q_w = w32(f"{lp}.attention.q_proj.weight")
            kva_w = w32(f"{lp}.attention.kv_a_proj_with_mqa.weight")
            kv_a_ln = w32(f"{lp}.attention.kv_a_layernorm.weight")
            kv_b = w32(f"{lp}.attention.kv_b_proj.weight")
            g_w = w32(f"{lp}.attention.g_proj.weight")
            o_w = w32(f"{lp}.attention.dense.weight")
            hf = {}  # per-head nope/roped weight rows
            for hh in range(HEADS):
                hf[hh] = (
                    kv_b[hh * (NOPE + VD):hh * (NOPE + VD) + NOPE],
                    kv_b[hh * (NOPE + VD) + NOPE:(hh + 1) * (NOPE + VD)],
                )
            mla_cache = []
        else:
            a_proj = {
                p: w32(f"{lp}.attention.{p}_proj.weight") for p in ("q", "k", "v")
            }
            convw = torch.cat([
                w32(f"{lp}.attention.q_conv1d.weight").reshape(KDA_T, CONV),
                w32(f"{lp}.attention.k_conv1d.weight").reshape(KDA_T, CONV),
                w32(f"{lp}.attention.v_conv1d.weight").reshape(KDA_T, CONV),
            ])
            f_w = w32(f"{lp}.attention.f_proj.weight")
            g_w = w32(f"{lp}.attention.g_proj.weight")
            b_w = w32(f"{lp}.attention.b_proj.weight")
            a_log = w32(f"{lp}.attention.A_log")
            dt_bias = w32(f"{lp}.attention.dt_bias")
            o_norm_w = w32(f"{lp}.attention.o_norm.weight")
            o_w = w32(f"{lp}.attention.o_proj.weight")
            conv_state = torch.zeros(3 * KDA_T, CONV)
            kda_h = torch.zeros(HEADS, 128, 128)
        dense_mlp = L < 2
        if dense_mlp:
            dg = w32(f"{lp}.mlp.gate_proj.weight")
            du = w32(f"{lp}.mlp.up_proj.weight")
            dd = w32(f"{lp}.mlp.down_proj.weight")

        for t in range(NTOK):
            token_key = f"pre_{t}" if t < PROMPT else f"dec_{t - PROMPT}"
            pos = t
            x = h[t]
            n = rms(x, in_norm)
            if is_mla:
                q = rb(n.double() @ q_w.double().T)
                kva = rb(n.double() @ kva_w.double().T)
                latent = rms(kva[:LORA], kv_a_ln)
                k_rot = rope_pairs(kva[LORA:], pos)
                kn_all = torch.stack([hf[hh][0] @ latent for hh in range(HEADS)])
                v_all = torch.stack([hf[hh][1] @ latent for hh in range(HEADS)])
                mla_cache.append({"k_rot": k_rot, "kn": kn_all, "v": v_all})
                g_logits = rb(n.double() @ g_w.double().T)
                out = torch.zeros(HEADS, VD)
                k_rope_all_cache = torch.stack([e["k_rot"] for e in mla_cache])
                for hh in range(HEADS):
                    qn_ = q[hh * (NOPE + ROPE):hh * (NOPE + ROPE) + NOPE]
                    qr = rope_pairs(
                        q[hh * (NOPE + ROPE) + NOPE:(hh + 1) * (NOPE + ROPE)], pos)
                    kn_ctx = torch.stack([e["kn"][hh] for e in mla_cache])
                    scores = ((kn_ctx @ qn_) + (k_rope_all_cache @ qr)) / math.sqrt(192.0)
                    mx = scores.max()
                    p = torch.exp(scores - mx)
                    p = p / p.sum()
                    v_all_ctx = torch.stack([e["v"][hh] for e in mla_cache])
                    out[hh] = (p[:, None] * v_all_ctx).sum(0)
                gate = torch.sigmoid(g_logits)
                gated = (out * gate[:, None]).reshape(HEADS * VD)
                attn = rb(gated.double() @ o_w.double().T)
            else:
                qkv_raw = torch.cat([
                    rb(n.double() @ a_proj["q"].double().T),
                    rb(n.double() @ a_proj["k"].double().T),
                    rb(n.double() @ a_proj["v"].double().T),
                ])
                conv_state[:, :-1] = conv_state[:, 1:].clone()
                conv_state[:, -1] = qkv_raw
                acc = (conv_state * convw).sum(-1)
                qkv = rb(acc * torch.sigmoid(acc))
                f = rb(n.double() @ f_w.double().T)
                g = rb(n.double() @ g_w.double().T)
                b_raw = rb(n.double() @ b_w.double().T)
                if L == TRACE_LAYER and t < PROMPT:
                    tk = f"t{t}"
                    trace.setdefault("normed", {})[tk] = n.clone()
                    trace.setdefault("preconv", {})[tk] = torch.stack(
                        [qkv_raw[0:KDA_T], qkv_raw[KDA_T:2 * KDA_T],
                         qkv_raw[2 * KDA_T:3 * KDA_T]])
                    trace.setdefault("qkv", {})[tk] = qkv.clone()
                    trace.setdefault("f", {})[tk] = f.clone()
                    trace.setdefault("g", {})[tk] = g.clone()
                    trace.setdefault("b_raw", {})[tk] = b_raw.clone()
                gg = f.view(HEADS, 128)
                og = g.view(HEADS, 128)
                decay = torch.exp(LOWER / (1.0 + torch.exp(
                    -(a_log[:, None] * (gg + dt_bias.view(HEADS, 128))))))
                qq = qkv[0:KDA_T].view(HEADS, 128)
                kk = qkv[KDA_T:2 * KDA_T].view(HEADS, 128)
                vv = qkv[2 * KDA_T:3 * KDA_T].view(HEADS, 128)
                sq = (qq * qq).sum(-1, keepdim=True)
                sk = (kk * kk).sum(-1, keepdim=True)
                qn = qq * torch.rsqrt(sq + 1e-6) / math.sqrt(128.0)
                kn = kk * torch.rsqrt(sk + 1e-6)
                beta = torch.sigmoid(b_raw)
                dot = (kda_h * decay[:, :, None] * kn[:, :, None]).sum(1)
                delta = (vv - dot) * beta[:, None]
                h_new = (kda_h * decay[:, :, None]
                         + delta[:, None, :] * kn[:, :, None])
                o = (h_new * qn[:, :, None]).sum(1)
                kda_h = h_new
                inv = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
                onrm = rb(o * inv * o_norm_w[None, :] * torch.sigmoid(og))
                attn = rb(onrm.reshape(KDA_T).double() @ o_w.double().T)
                if L == TRACE_LAYER and t < PROMPT:
                    tk = f"t{t}"
                    trace.setdefault("kda_out", {})[tk] = o.reshape(KDA_T).clone()
                    trace.setdefault("kda_o", {})[tk] = onrm.reshape(KDA_T).clone()
                    trace.setdefault("attn_out", {})[tk] = attn.clone()

            h1 = rb(x + attn)
            if L == TRACE_LAYER and (not is_mla) and t < PROMPT:
                trace.setdefault("h1", {})[f"t{t}"] = h1.clone()
            n2 = rms(h1, post_norm)
            if L == TRACE_LAYER and (not is_mla) and t < PROMPT:
                trace.setdefault("ffn_in", {})[f"t{t}"] = n2.clone()
            if dense_mlp:
                g_row = rb(n2.double() @ dg.double().T)
                u_row = rb(n2.double() @ du.double().T)
                hmid = rb(rb(g_row * torch.sigmoid(g_row)) * u_row)
                ffn = rb(hmid.double() @ dd.double().T)
                if L == TRACE_LAYER and t < PROMPT:
                    trace.setdefault("shared_h", {})[f"t{t}"] = hmid.clone()
                    trace.setdefault("ffn_out", {})[f"t{t}"] = ffn.clone()
            else:
                ffn = moe_ffn(L, n2, token_key)
            out_h = rb(h1 + ffn)
            h[t] = out_h
            if t < PROMPT:
                pre_h[L + 1, t] = out_h
            else:
                dec_h[t - PROMPT, L + 1] = out_h
        expert_cache.clear()
        print(f"layer {L} done", file=sys.stderr)

    last = dec_h[DECODE - 1, LAYERS]
    logits = rms(last, final_norm) @ lm_head.T
    top = torch.topk(logits, 5)
    print("oracle last-token top-5:",
          list(zip(top.indices.tolist(), [round(float(v), 3) for v in top.values])))

    pre_h.numpy().tofile(os.path.join(OUT, "oracle_pre.f32"))
    dec_h.numpy().tofile(os.path.join(OUT, "oracle_dec.f32"))
    meta = {
        "tokens": TOKENS,
        "prompt": PROMPT,
        "decode": DECODE,
        "hidden": H,
        "layers": LAYERS,
        "routed_limit": ROUTED_LIMIT,
        "shared_limit": SHARED_LIMIT,
        "routes": routes,
    }
    with open(os.path.join(OUT, "oracle_meta.json"), "w") as f:
        json.dump(meta, f)
    # Trace dumps: one f32 file per traced stage, layouts matching the atlas
    # scratch lanes: `preconv` is plane-major [3, P, T]; `qkv`/`f`/`g`/
    # `kda_out`/`kda_o` are [P, width]; row stages are [P, width].
    if TRACE_LAYER >= 0:
        order = [f"t{t}" for t in range(PROMPT)]
        for stage, vals in trace.items():
            stacked = torch.stack([vals[k] for k in order]).float()
            if stage == "preconv":
                # oracle stacks per token [P, 3, T]; the atlas lane is
                # plane-major [3, P, T].
                stacked = stacked.permute(1, 0, 2).contiguous()
            np_fn = os.path.join(OUT, f"trace_{TRACE_LAYER}_{stage}.f32")
            stacked.numpy().tofile(np_fn)
        print(f"wrote {len(trace)} trace_{TRACE_LAYER}_*.f32 stage files",
              file=sys.stderr)
    print(f"wrote {OUT}/oracle_pre.f32 {tuple(pre_h.shape)} "
          f"oracle_dec.f32 {tuple(dec_h.shape)} oracle_meta.json")


if __name__ == "__main__":
    main()
