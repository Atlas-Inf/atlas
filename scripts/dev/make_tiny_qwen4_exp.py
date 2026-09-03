#!/usr/bin/env python3
"""Emit a tiny but structurally faithful `qwen4_exp` checkpoint.

Why this exists: the smallest published Qwen3.8-Flash-Next repack is 135.3 GB
(RadixArk NVFP4), which does not fit on a GB10 and does not fit on its disk
either. A weight loader cannot be developed against a checkpoint nobody can
hold, so this generates one that is a few MB and wrong about nothing that
matters structurally.

What is faithful:
  * every tensor NAME the real checkpoint carries, at the same nesting
  * the hybrid layer schedule, and PLE on a linear-attention layer, one-indexed
  * the n-gram head geometry -- prime table sizes, cumulative offsets, the
    concatenation padded to `make_ngram_vocab_size_divisible_by`
  * `mrope_section * 2 == head_dim * partial_rotary_factor`
  * `norm_topk_prob` and `seed` OMITTED, exactly as the published config omits
    them, so a loader that mishandles those defaults fails here too

What is not: dimensions, and the weights are random. This checks plumbing, not
numerics.

    python scripts/dev/make_tiny_qwen4_exp.py /tmp/tiny-qwen4-exp
"""
import json, math, pathlib, sys

import torch
from safetensors.torch import save_file

H = 128            # hidden_size
LAYERS = 8         # 4-stride hybrid -> 6 linear, 2 full
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 1, 32
LIN_K_HEADS, LIN_K_DIM = 2, 16
LIN_V_HEADS, LIN_V_DIM = 4, 16
CONV_K = 4
EXPERTS, TOPK, MOE_INT = 8, 2, 32
VOCAB, EOS = 512, 511
NGRAM_SIZE, HEADS_PER_NGRAM = 3, 4
NGRAM_BASE, NGRAM_DIVISOR, NGRAM_SHARDS = 1024, 128, 4
PLE_LAYER_IDS = [2]                       # ONE-indexed -> decoder layer 1
HC_COUNT, HC_LOWRANK = 4, 16
MTP_LAYERS = 1

NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
NGRAM_HEAD_DIM = H // NGRAM_HEADS


MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
SEED_STRIDE = 10007
SEED = 1234


def splitmix64(value):
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def layer_multipliers(unigram_vocab, ngram_size, ple_layer_index, seed):
    """The odd multipliers the reference draws in __init__."""
    multiplier_max = ((1 << 63) - 1) // max(unigram_vocab, 1)
    half = max(1, multiplier_max // 2)
    base = (seed + SEED_STRIDE * ple_layer_index) & MASK64
    return [2 * (splitmix64((base + SPLITMIX_GAMMA * (i + 1)) & MASK64) % half) + 1
            for i in range(ngram_size)]


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


def nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


def layer_types():
    # Same rule as the real config: every 4th layer is full attention.
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(LAYERS)]


def config():
    types = layer_types()
    for one_indexed in PLE_LAYER_IDS:
        assert types[one_indexed - 1] == "linear_attention", \
            "HF refuses PLE on a non-linear layer"
    rotary = int(HEAD_DIM * 0.25)
    section = [rotary // 4, rotary // 8, rotary // 8]
    assert sum(section) * 2 == rotary, (section, rotary)
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_size": H, "num_hidden_layers": LAYERS,
        "num_attention_heads": Q_HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM, "attention_bias": False,
        "layer_types": types, "full_attention_interval": 4,
        "linear_num_key_heads": LIN_K_HEADS, "linear_key_head_dim": LIN_K_DIM,
        "linear_num_value_heads": LIN_V_HEADS, "linear_value_head_dim": LIN_V_DIM,
        "linear_conv_kernel_dim": CONV_K, "output_gate_type": "sigmoid",
        "mamba_ssm_dtype": "float32",
        "num_experts": EXPERTS, "num_experts_per_tok": TOPK,
        "moe_intermediate_size": MOE_INT,
        "shared_expert_intermediate_size": MOE_INT,
        "hc_count": HC_COUNT, "hc_lowrank": HC_LOWRANK,
        "indexer_n_heads": 2, "indexer_head_dim": 16, "indexer_kv_heads": 1,
        "indexer_budget": 64, "indexer_compress_ratio": 4,
        "ple_layer_ids": list(PLE_LAYER_IDS), "ple_embed_dim": H,
        "ple_conv_kernel_size": CONV_K,
        "ngram_size": NGRAM_SIZE, "heads_per_ngram": HEADS_PER_NGRAM,
        "ngram_vocab_size_base": NGRAM_BASE,
        "make_ngram_vocab_size_divisible_by": NGRAM_DIVISOR,
        "split_ngram_parts": NGRAM_SHARDS,
        # `norm_topk_prob` and `seed` are DELIBERATELY absent -- see docstring.
        # (seed defaults to 1234, which is what layer_multipliers() uses.)
        "vocab_size": VOCAB, "bos_token_id": EOS, "eos_token_id": EOS,
        "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False, "partial_rotary_factor": 0.25,
        "mtp_num_hidden_layers": MTP_LAYERS,
        "mtp_use_dedicated_embeddings": False,
        "mtp": {"hybrid": True, "layer_types": ["full_attention"],
                "num_hidden_layers": MTP_LAYERS, "rope_theta": 10000000,
                "mtp_use_hidden_state_from_layer": None},
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                            "partial_rotary_factor": 0.25,
                            "mrope_interleaved": True, "mrope_section": section},
    }
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp", "text_config": text,
        "tie_word_embeddings": False,
        "image_token_id": VOCAB - 8, "video_token_id": VOCAB - 7,
        "vision_start_token_id": VOCAB - 6, "vision_end_token_id": VOCAB - 5,
        "vision_config": {
            "model_type": "qwen4_exp", "depth": 2, "hidden_size": 64,
            "num_heads": 4, "intermediate_size": 128, "in_channels": 3,
            "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": H, "num_position_embeddings": 64,
            "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [],
        },
    }


def hyper_connection(prefix, tensors, inject=True):
    # These operate on the CONCATENATED residual: hc_count streams of `hidden`,
    # so every width here is hc_count * H, not H. The published checkpoint's
    # hc_norm is [10240] = 4 x 2560, not [2560].
    wide = HC_COUNT * H
    tensors[f"{prefix}.hc_norm.weight"] = torch.ones(wide)
    tensors[f"{prefix}.input_mix_weight_down.weight"] = torch.randn(HC_LOWRANK, wide) * 0.02
    tensors[f"{prefix}.input_mix_weight_up.weight"] = torch.randn(wide, HC_LOWRANK) * 0.02
    if inject:
        tensors[f"{prefix}.block_inject_weight.weight"] = torch.randn(HC_COUNT, wide) * 0.02


def moe(prefix, tensors):
    tensors[f"{prefix}.gate.weight"] = torch.randn(EXPERTS, H) * 0.02
    for e in range(EXPERTS):
        for proj, shape in (("gate_proj", (MOE_INT, H)), ("up_proj", (MOE_INT, H)),
                            ("down_proj", (H, MOE_INT))):
            tensors[f"{prefix}.experts.{e}.{proj}.weight"] = torch.randn(*shape) * 0.02
    for proj, shape in (("gate_proj", (MOE_INT, H)), ("up_proj", (MOE_INT, H)),
                        ("down_proj", (H, MOE_INT))):
        tensors[f"{prefix}.shared_expert.{proj}.weight"] = torch.randn(*shape) * 0.02
    tensors[f"{prefix}.shared_expert_gate.weight"] = torch.randn(1, H) * 0.02


def full_attention(prefix, tensors):
    # 2x: Q and its gate, interleaved per head. The real q_proj is
    # [12288, 2560] against 24 heads x head_dim 256 = 6144.
    tensors[f"{prefix}.q_proj.weight"] = torch.randn(Q_HEADS * HEAD_DIM * 2, H) * 0.02
    tensors[f"{prefix}.k_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, H) * 0.02
    tensors[f"{prefix}.v_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, H) * 0.02
    tensors[f"{prefix}.o_proj.weight"] = torch.randn(H, Q_HEADS * HEAD_DIM) * 0.02
    tensors[f"{prefix}.q_norm.weight"] = torch.ones(HEAD_DIM)
    tensors[f"{prefix}.k_norm.weight"] = torch.ones(HEAD_DIM)
    # Fused indexer q and k: (n_heads + kv_heads) * head_dim. Real is
    # (4 + 1) * 128 = 640.
    tensors[f"{prefix}.indexer.index_qk_proj.weight"] = torch.randn((2 + 1) * 16, H) * 0.02
    tensors[f"{prefix}.indexer.q_layernorm.weight"] = torch.ones(16)
    tensors[f"{prefix}.indexer.k_layernorm.weight"] = torch.ones(16)


def linear_attention(prefix, tensors):
    qkv = LIN_K_HEADS * LIN_K_DIM * 2 + LIN_V_HEADS * LIN_V_DIM
    tensors[f"{prefix}.in_proj_qkv.weight"] = torch.randn(qkv, H) * 0.02
    tensors[f"{prefix}.in_proj_a.weight"] = torch.randn(LIN_V_HEADS, H) * 0.02
    tensors[f"{prefix}.in_proj_b.weight"] = torch.randn(LIN_V_HEADS, H) * 0.02
    tensors[f"{prefix}.in_proj_z.weight"] = torch.randn(LIN_V_HEADS * LIN_V_DIM, H) * 0.02
    tensors[f"{prefix}.conv1d.weight"] = torch.randn(qkv, 1, CONV_K) * 0.02
    tensors[f"{prefix}.A_log"] = torch.randn(LIN_V_HEADS)
    tensors[f"{prefix}.dt_bias"] = torch.randn(LIN_V_HEADS)
    tensors[f"{prefix}.norm.weight"] = torch.ones(LIN_V_DIM)
    tensors[f"{prefix}.out_proj.weight"] = torch.randn(H, LIN_V_HEADS * LIN_V_DIM) * 0.02


def ple(prefix, tensors, ple_layer_index):
    sizes, offsets, total = [], [], 0
    first = ple_layer_index * NGRAM_HEADS
    for head in range(NGRAM_HEADS):
        size = nth_prime_after(NGRAM_BASE - 1, first + head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    padded = math.ceil(total / NGRAM_DIVISOR) * NGRAM_DIVISOR
    assert padded % NGRAM_SHARDS == 0, "shards must divide the padded table"
    rows = padded // NGRAM_SHARDS
    emb = f"{prefix}.ple_embedding"
    for shard in range(NGRAM_SHARDS):
        tensors[f"{emb}.ngram_embedding.shard_{shard}.weight"] = (
            torch.randn(rows, NGRAM_HEAD_DIM) * 0.02)
    tensors[f"{emb}.ngram_embedding.weight_scale"] = torch.ones(1)
    tensors[f"{emb}.ngram_heads_vocab_sizes"] = torch.tensor(sizes, dtype=torch.long)
    tensors[f"{emb}.ngram_heads_offsets"] = torch.tensor(offsets, dtype=torch.long)
    # DERIVED, not zeroed. These are the SplitMix64 draws the reference builds
    # in __init__; a checkpoint carrying zeros overwrites them on load, every
    # mixed id collapses to 0, and every token hashes to its head's first row.
    # That looks like a working model right up until you check the output.
    tensors[f"{emb}.layer_multipliers"] = torch.tensor(
        layer_multipliers(VOCAB, NGRAM_SIZE, ple_layer_index, SEED), dtype=torch.long
    )
    tensors[f"{prefix}.conv1d.weight"] = torch.randn(HC_COUNT * H, 1, CONV_K) * 0.02
    tensors[f"{prefix}.key_proj.weight"] = torch.randn(HC_COUNT * H, H) * 0.02
    tensors[f"{prefix}.value_proj.weight"] = torch.randn(H, H) * 0.02
    for n in ("norm_conv", "norm_key", "norm_query"):
        tensors[f"{prefix}.{n}.weight"] = torch.ones(HC_COUNT * H)
    return padded


def build():
    tensors, types = {}, layer_types()
    lm = "model.language_model"
    tensors[f"{lm}.embed_tokens.weight"] = torch.randn(VOCAB, H) * 0.02
    tensors["lm_head.weight"] = torch.randn(VOCAB, H) * 0.02
    hyper_connection(f"{lm}.hyper_connection_mixer", tensors, inject=False)

    for i, kind in enumerate(types):
        base = f"{lm}.layers.{i}"
        hyper_connection(f"{base}.attn_hyper_connection", tensors)
        hyper_connection(f"{base}.mlp_hyper_connection", tensors)
        if kind == "full_attention":
            full_attention(f"{base}.self_attn", tensors)
        else:
            linear_attention(f"{base}.linear_attn", tensors)
        moe(f"{base}.mlp", tensors)
        if (i + 1) in PLE_LAYER_IDS:
            ple(f"{base}.ple", tensors, PLE_LAYER_IDS.index(i + 1))

    for i in range(MTP_LAYERS):
        base = f"mtp.layers.{i}"
        hyper_connection(f"{base}.attn_hyper_connection", tensors)
        hyper_connection(f"{base}.mlp_hyper_connection", tensors)
        full_attention(f"{base}.self_attn", tensors)
        moe(f"{base}.mlp", tensors)
    hyper_connection("mtp.hyper_connection_mixer", tensors, inject=False)
    tensors["mtp.fc_embedding.weight"] = torch.randn(H, H) * 0.02
    tensors["mtp.fc_hidden.weight"] = torch.randn(H, H) * 0.02
    # Different widths: the embedding side normalises a token embedding (H),
    # the hidden side normalises the trunk's hc_count-wide hyper-connection
    # state. The published checkpoints are [2560] and [10240].
    tensors["mtp.pre_fc_norm_embedding.weight"] = torch.ones(H)
    tensors["mtp.pre_fc_norm_hidden.weight"] = torch.ones(HC_COUNT * H)

    vis = "model.visual"
    tensors[f"{vis}.patch_embed.proj.weight"] = torch.randn(64, 3, 2, 16, 16) * 0.02
    tensors[f"{vis}.patch_embed.proj.bias"] = torch.zeros(64)
    tensors[f"{vis}.pos_embed.weight"] = torch.randn(64, 64) * 0.02
    for b in range(2):
        p = f"{vis}.blocks.{b}"
        tensors[f"{p}.attn.qkv.weight"] = torch.randn(192, 64) * 0.02
        tensors[f"{p}.attn.qkv.bias"] = torch.zeros(192)
        tensors[f"{p}.attn.proj.weight"] = torch.randn(64, 64) * 0.02
        tensors[f"{p}.attn.proj.bias"] = torch.zeros(64)
        tensors[f"{p}.mlp.linear_fc1.weight"] = torch.randn(128, 64) * 0.02
        tensors[f"{p}.mlp.linear_fc1.bias"] = torch.zeros(128)
        tensors[f"{p}.mlp.linear_fc2.weight"] = torch.randn(64, 128) * 0.02
        tensors[f"{p}.mlp.linear_fc2.bias"] = torch.zeros(64)
        for n in ("norm1", "norm2"):
            tensors[f"{p}.{n}.weight"] = torch.ones(64)
            tensors[f"{p}.{n}.bias"] = torch.zeros(64)
    for n in ("linear_fc1", "linear_fc2"):
        o, i = (H, 64 * 4) if n == "linear_fc1" else (H, H)
        tensors[f"{vis}.merger.{n}.weight"] = torch.randn(o, i) * 0.02
        tensors[f"{vis}.merger.{n}.bias"] = torch.zeros(o)
    tensors[f"{vis}.merger.norm.weight"] = torch.ones(64 * 4)
    tensors[f"{vis}.merger.norm.bias"] = torch.zeros(64 * 4)
    return tensors


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-qwen4-exp")
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.contiguous() for k, v in build().items()}
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(config(), indent=2) + "\n")
    n_params = sum(v.numel() for v in tensors.values())
    print(f"{len(tensors)} tensors, {n_params/1e6:.2f}M params -> {out}")
    print(f"  ngram heads {NGRAM_HEADS} x {NGRAM_HEAD_DIM} wide, {NGRAM_SHARDS} shards")


if __name__ == "__main__":
    main()
