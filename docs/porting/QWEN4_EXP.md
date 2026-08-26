# `qwen4_exp` — Qwen3.8-Flash-Next: what is in Atlas, and what is not

Status as of 2026-08-26. Read against the published
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
`config.json` and `model.safetensors.index.json`, plus the reference
implementation in HuggingFace Transformers (Apache-2.0),
`src/transformers/models/qwen4_exp/`.

**Atlas cannot serve this checkpoint today.** One piece of it — the n-gram id
core — is implemented and pinned against the real weights. Everything else in
the list below is absent. This file is the gap list, so that "drop it in" is a
plan rather than a surprise.

## The headline constraint: it does not fit on a GB10

`model.safetensors.index.json` declares **185.5 GB** of FP8 weights. A DGX Spark
GB10 has ~119 GB of usable unified memory. Single-box serving is not a matter of
tuning `--gpu-memory-utilization`; the weights alone are ~1.6x the box.

Where the bytes are, from the index:

| component | bytes | note |
|---|---|---|
| routed experts, 48 layers x 512 x 3 x (640 x 2560) | ~121 GB | the bulk |
| n-gram embedding table, `320_001_536 x 160` FP8 | ~51 GB | ONE layer's PLE tower |
| MTP block's own 512 experts | ~2.5 GB | |
| backbone, vision tower, norms, scales | remainder | |

The n-gram table is the cheapest 51 GB to move off-device: it is gathered by
row, and the reference already excludes it from placement
(`_no_placement_params = ["ple.ple_embedding.ngram_embedding.weight"]`). Host
residency plus a row gather is the intended deployment, not a workaround. Doing
that leaves ~134 GB device-resident — still over a GB10, so routed-expert
offload or expert parallelism across boxes is required on top.

## What is implemented

`crates/atlas-core/src/config/ngram_qwen4exp.rs` — the n-gram hashed-embedding
id core and its table geometry, derived from `config.json` alone and asserted
bit-exact against the checkpoint's own `layer_multipliers`,
`ngram_heads_vocab_sizes` and `ngram_heads_offsets` buffers and its
`[2_500_012, 160] x 128` shard shapes.

It is **not** the LongCat mechanism in `config/ngram.rs`. Both hash token ids
into `K*(N-1)` heads of width `embed_dim/(K*(N-1))` and both reset shifts at
document boundaries, so both need only the last `N-1` tokens to decode. They
agree on nothing else:

|                | LongCat (`ngram.rs`)                | `qwen4_exp` (`ngram_qwen4exp.rs`)     |
|----------------|-------------------------------------|---------------------------------------|
| mixing         | polynomial `Σ shift_d · (V^d mod T)` | XOR of `shift_d · m_d`               |
| multipliers    | powers of the vocab size            | seeded SplitMix64 draws, odd          |
| table rows     | `ratio·V + 2i + 1` (odd)            | consecutive PRIMES above a base       |
| storage        | one tensor per table                | one concatenated table + head offsets |
| shift fill     | `0`                                 | `eos_token_id`                        |

Two traps are encoded as tests rather than left as comments:

* **`ple_layer_ids` is ONE-indexed.** The config says `[2]`; the weights are at
  `model.language_model.layers.1.ple.*`. Off by one here loads a real tensor
  from the wrong layer.
* **`seed` is absent from the published `config.json`.** It must default to
  1234. A zero default draws different multipliers and hashes every token to an
  unrelated row — degraded output, no error.

## What is missing

### Dispatch
`qwen4_exp` is in neither `config/dispatch.rs::parse_config` nor
`spark-model/src/factory.rs::loader_for_config`. Both currently reject it by
name. The config is nested under `text_config` like the Qwen3.5/VL families, but
does not otherwise share their shape.

### Hyper-connections (`hc_count = 4`, `hc_lowrank = 320`)
Four residual streams, with `attn_hyper_connection` and `mlp_hyper_connection`
per block plus a top-level `hyper_connection_mixer`. The mix is a low-rank
sigmoid gate — `silu(down(x)/hc_count)` then `sigmoid(up(·))`, unflattened to
`(hc_count, hidden)` and mean-reduced — with a separate
`2·sigmoid(block_inject_weight(x)/hc_count)` injection.

Atlas already carries `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` for
DeepSeek-V4's mHC. **That is a different formulation** — Sinkhorn-normalised
mixing, not a low-rank gate — so those fields describe the concept but not this
model, and reusing them without a distinct path would be wrong.

### Sparse-attention indexer
`indexer_n_heads = 4`, `indexer_head_dim = 128`, `indexer_kv_heads = 1`,
`indexer_budget = 2048`, `indexer_compress_ratio = 4`; tensors
`self_attn.indexer.{index_qk_proj, q_layernorm, k_layernorm}` on the 12
full-attention layers only. Atlas's `index_n_heads` / `index_head_dim` /
`index_topk` (DeepSeek-V4 CSA) are the nearest relatives; the mapping between
`indexer_budget`/`indexer_compress_ratio` and `index_topk` has not been
established.

### PLE tower
`conv1d [10240,1,4]`, `key_proj [10240,2560]`, `value_proj [2560,2560]`,
`norm_conv` / `norm_key` / `norm_query [10240]`. `10240 = hc_count x hidden` —
one key per residual stream. The n-gram context (the last `N-1` token ids) rides
in **conv-state slot 2** of the cache; the reference sets
`number_of_conv_states = 3` whenever `ple_layer_ids` is non-empty. Atlas's SSM
conv-state plumbing would have to carry a third slot holding token ids rather
than activations.

### Linear attention
36 of 48 layers. `in_proj_qkv` / `in_proj_a` / `in_proj_b` / `in_proj_z`,
`conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj`; 16 key heads x 128, 48 value
heads x 128, conv kernel 4, `output_gate_type = "sigmoid"`, SSM state in fp32.
Related to the Qwen3-Next GDN path but not tensor-name compatible.

### MoE
512 experts, top-10, `moe_intermediate_size = 640`, a shared expert of the same
width plus `shared_expert_gate`. Expert count and per-expert width are both well
outside what the existing Qwen3.5 MoE kernels were sized for.

### MTP
One hybrid block with `fc_embedding` / `fc_hidden` /
`pre_fc_norm_embedding` / `pre_fc_norm_hidden`, its own full-attention layer,
its own indexer and its own 512 experts. `mtp_use_dedicated_embeddings = false`.
Not the Qwen3.5/Nemotron MTP head shape.

### Vision
27 blocks, hidden 1152, 16 heads, intermediate 4304, patch 16, spatial merge 2,
temporal patch 2, out_hidden 2560, `num_position_embeddings = 2304`, and
`deepstack_visual_indexes = []` (empty — no deepstack fusion). Tensor layout
resembles the Qwen3-VL tower.

### Quantization
`quant_method = "fp8"`, dynamic activation scaling, **`weight_block_size =
[128, 128]`** — block-scaled, not per-tensor or per-row, with `weight_scale_inv`
siblings. Atlas's `"fp8"` path handles `weight_scale_inv`; whether it handles a
2-D 128x128 block grid at this scale is unverified.

### Position embeddings
MRoPE interleaved, `mrope_section = [11, 11, 10]`, `partial_rotary_factor =
0.25`, `rope_theta = 1e7`, `max_position_embeddings = 262144`. This part maps
cleanly onto the existing Qwen3.6 MRoPE support.

## Provenance

`crates/atlas-core/src/config/ngram_qwen4exp.rs` is an independent derivation of
the algorithm implemented in the Apache-2.0 licensed

    https://github.com/huggingface/transformers
    src/transformers/models/qwen4_exp/modeling_qwen4_exp.py

(`Qwen4ExpTextNGramEmbedding`, `_build_layer_multipliers`,
`_find_nth_prime_after`), checked against the published
`Qwen/Qwen3.8-Flash-Next-FP8` weights. No upstream file is vendored, and the
Rust is not a transliteration — but the constants, the mixing order and the
prime-selection rule are the reference's, and the Apache-2.0 attribution travels
with them. The rest of Atlas remains AGPL-3.0-only.

This mirrors how `bench/ngram_embed/LICENSE` carries Meituan's MIT notice for
the LongCat derivation.
