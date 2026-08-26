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

### There is no host/device split to offload across

**Corrected 2026-08-26.** An earlier version of this file said host-resident PLE
would leave ~83 GB "on device", which would fit. That is wrong on this hardware,
and it pointed the plan at the wrong first feature.

The GB10 is fully coherent unified memory — `nvidia-smi -q` reports
`Addressing Mode: ATS` and `memory.total [N/A]`, because there is no separate
GPU pool. `MemTotal` is 125,418,660 kB (~119.6 GiB) and that single pool is what
both the CPU and the GPU allocate from; a serving process showing 100 GB of
"Used GPU Memory" is holding 100 GB of the same RAM the OS sees. Moving a tensor
"to host" therefore frees nothing.

What works is not residency but **demand paging**. The n-gram table is a pure
row gather: one token touches at most `K*(N-1)` = 16 rows of 160 bytes, about
2.5 KB. `mmap` the `model-plefp8-*` shards and let the page cache hold the hot
rows; cold rows come off NVMe. The reference points the same way — it excludes
that tensor from placement entirely
(`_no_placement_params = ["ple.ple_embedding.ngram_embedding.weight"]`).

That gives a workable budget on one GB10:

| | |
|---|---|
| resident weights (everything but the n-gram table) | ~83 GB |
| n-gram table, mmap'd, only hot rows in page cache | ~52 GB on disk |
| left for KV cache, activations, page cache | ~36 GB |

So the first loader feature is **mmap-backed n-gram gather**, not offload. Full
residency of all 135.3 GB is not reachable on this box by any placement
strategy.

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

### Weight manifest — done
`atlas_core::weight_manifest::qwen4_exp_manifest` enumerates every
language-model and MTP tensor with its shape, and matches the published
checkpoint's index exactly (76,492 names, zero missing, zero unexpected; 1,653
shapes cross-checked against real headers). Use
`scripts/dev/verify_qwen4_exp_manifest.py <dir>` against the tiny checkpoint
while building the loader — it catches a wrong width the moment it appears,
which it already did three times.

Quantization-aware as of the FP8 release: `quantization_siblings` derives the
`weight_scale_inv` set from the checkpoint's declared `weight_block_size` and
`modules_to_not_convert`. Manifest plus siblings is **151,756 tensors — the
entire published checkpoint except its 333 vision tensors — with zero missing
and zero unexpected**, and 3,189 shapes cross-checked against real headers.

Two rules that are not guessable from the tensor names:

* Only 2-D linear weights take a block scale. Norms, biases, integer buffers
  and the 3-D conv kernels do not, and none of them appear in
  `modules_to_not_convert` either — so the rank check cannot be left to the
  ignore list.
* A group carrying its own `weight_scale` is quantized PER TENSOR and takes no
  `weight_scale_inv`. The 128 n-gram shards are the case: FP8, sharing one BF16
  scale, and absent from `modules_to_not_convert` because they *are* converted,
  just by another scheme. Treating them as block-quantized over-generates by
  exactly 128.

**Verified against the real weights.** `cargo run -p atlas-core --example
qwen4exp_preflight -- <dir>` against the downloaded 126 GiB RadixArk checkpoint:
296,142 tensors, **0 missing, 0 unexpected, 0 mismatched**, in 2.81 s and 372 MB
RSS — it reads headers, not weights, so it is cheap enough to run at load time.
It auto-detected the stacked-MTP layout and the NVFP4 config from the checkpoint
itself.

That run is also what caught the last width bug (`mtp.pre_fc_norm_hidden` is
`hc_count * hidden`, not `hidden`) — an index-only check could never have seen
it, having no shapes.

**Both published releases are described exactly**, weights and scales:

| release | tensors (ex-vision) | manifest | discrepancies |
|---|---|---|---|
| `Qwen/…-FP8` (block FP8) | 151,756 | 151,756 | **0** |
| `RadixArk/…-NVFP4` (ModelOpt) | 296,142 | 296,142 | **0** |

NVFP4 does more than add siblings — it **repacks the weight**: a `[2560, 640]`
projection is stored U8 `[2560, 320]`, two FP4 values per byte, with
`weight_scale [2560, 40]` (one per group of 16 along the input dim) and scalar
`weight_scale_2` / `input_scale`.

Two packaging differences that are NOT architectural, and that a loader must
handle rather than assume:

* **Expert layout.** HF's native `Qwen4ExpTextExperts` stores `gate_up_proj` as
  one `[experts, 2*moe_intermediate, hidden]` tensor and chunks it at use.
  ModelOpt works per `nn.Linear`, so quantizing splits the stack into
  `experts.{i}.{gate,up,down}_proj`. The FP8 release is split throughout;
  RadixArk is split for the quantized routed experts and **stacked for the MTP
  block**, which it excludes from quantization entirely (`mtp.*`). Both are
  expressible via `Qwen4ExpLayout`.
* **Ignore-list syntax.** ModelOpt globs (`*.self_attn.*`, `mtp.*`) and `*`
  spans dots; HF's native FP8 list carries no globs at all and spells out all
  943 modules. Both forms are matched.

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

## Cross-checked against the reference, not just against ourselves

`bench/ngram_embed/qwen4exp_xcheck.py` runs the real
`Qwen4ExpTextNGramEmbedding.forward` from Transformers and diffs the row ids it
would gather with against `cargo run -p atlas-core --example
qwen4exp_ngram_ids`. This matters because the Rust was written by reading HF's
algorithm — a transcription error in the XOR mix or the shift semantics would
not be caught by any test Atlas writes about itself.

It needs no GPU and no weights: the embedding tensor is `320_001_536 x 160`
(~51 GB), so `nn.Embedding` is stubbed during construction. Every buffer that
actually feeds the ids is built by the real `__init__`, untouched.

As of 2026-08-26, over 32 token streams — the EOS edge cases plus 25 seeded
random ones — **5408 / 5408 ids match**.

## The offload boundary is already packaged for us

`RadixArk/Qwen3.8-Flash-Next-NVFP4` (135.3 GB, the smallest repack published)
quantizes only the routed experts. Its `hf_quant_config.json` excludes
`*.ple.*`, so the n-gram tower stays FP8 **in its own `model-plefp8-*`
shards** — 10 files, ~52 GB.

That is the whole offload boundary, pre-separated:

| block | precision | size | placement |
|---|---|---|---|
| routed experts | NVFP4 | ~60 GB | device |
| PLE / n-gram table | FP8 | ~52 GB | **host** (HF does this already) |
| backbone, vision, MTP, norms | BF16 | ~23 GB | device |

~83 GB of that has to be resident. The remaining ~52 GB is the n-gram table,
which must be **mmap'd and demand-paged rather than placed anywhere** — see
"There is no host/device split to offload across" above. Full residency of
135.3 GB is not reachable on a 119 GB box by any placement strategy.

## How to work on this without the checkpoint

`scripts/dev/make_tiny_qwen4_exp.py` emits an 8.7 MB / 2.17M-param checkpoint
carrying every tensor name at the same nesting, the same hybrid schedule, PLE
on a one-indexed linear-attention layer, and real n-gram head geometry. Both
Atlas and HF's own `Qwen4ExpTextConfig` accept it.

```
python scripts/dev/make_tiny_qwen4_exp.py /tmp/tiny-qwen4-exp
python bench/ngram_embed/qwen4exp_xcheck.py /tmp/tiny-qwen4-exp/config.json
```

Develop the loader against that. It checks plumbing, not numerics — dimensions
and values are not faithful — but it is the only way to iterate at all, given
the arithmetic below.

## Why the real checkpoint is not an option yet

On the GB10 available (`reiner`, 119 GB unified):

| | |
|---|---|
| smallest published repack (RadixArk NVFP4) | 135.3 GB |
| free disk | ~72 GB |
| free disk after deleting everything of ours | ~95 GB |
| total system memory | 119 GB |

It fits neither the disk nor the memory. Host-resident PLE would bring the
device-resident set to ~83 GB, which *would* fit — but the download still needs
~40 GB more disk than exists. Serving this on that box needs either more disk,
a different box, or a repack that does not exist today.

## Suggested order

1. **Weight loader** (`Qwen4ExpWeightLoader` + a `factory.rs` arm). Comparable
   loaders in this tree run 1000–1900 lines. Develop against the tiny
   checkpoint.
2. **mmap-backed n-gram gather.** Out of order on purpose: it is what decides
   whether the model runs on a GB10 at all. `RadixArk`'s `model-plefp8-*`
   shards mean the boundary needs no repacking — the table is already its own
   set of files.
3. **Layers, cheapest first** — the 512-expert MoE and the linear-attention
   pathway are adaptations of the Qwen3.5 / Qwen3-Next paths; the low-rank
   hyper-connections, the PLE tower (including the third conv-state slot that
   carries token ids) and the sparse-attention indexer are new work.
4. **MTP head**, then **vision**, both of which the model runs without.
5. **SM121 kernels** at these dimensions.

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
