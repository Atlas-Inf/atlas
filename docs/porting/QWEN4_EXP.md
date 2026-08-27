# `qwen4_exp` — Qwen3.8-Flash-Next: what is in Atlas, and what is not

Status as of 2026-08-27. Read against the published
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
`config.json` and `model.safetensors.index.json`, plus the reference
implementation in HuggingFace Transformers (Apache-2.0),
`src/transformers/models/qwen4_exp/`.

**Atlas serves this checkpoint.** As of the #754 merge
(`feat/qwen4exp-serve`), the mHC highway runs on all 48 layers, the PLE tower
runs at model layer 1 off an NVMe row cache, the QSA indexer selects on both
decode and prefill, the vision tower loads, CUDA graphs capture, prefix caching
carries the PLE and QSA state, and concurrency is honoured. What is still
refused — by name, at pre-flight — is MTP and the stacked expert layout.

Read the rest of this file as two things layered: the ARCHITECTURAL notes are
still current and are the reason each piece is shaped the way it is; the
STATUS lines have been updated in place, and every section that used to say
"missing" now says what it is checked against instead. `QWEN4_EXP_PORT_LOG.md`
carries the merge decisions and the traps that came across with the port.

## The headline constraint: the WEIGHTS do not fit on a GB10, the model does

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

So the first loader feature is **demand-paged n-gram gather**, not offload. Full
residency of all 135.3 GB is not reachable on this box by any placement
strategy.

**Measured, and it is comfortably viable.** `NgramTable` (`pread`, not `mmap` —
same page cache, no SIGBUS and no invisible page-fault stall in a CUDA-adjacent
thread) against the real 51.2 GB table on a GB10:

| | per token (16 rows) | ceiling |
|---|---|---|
| cold, every row off NVMe | 820 µs | 1,219 tok/s |
| warm, page-cached | 8.85 µs | 112,994 tok/s |

Opening the table — validating that 128 shards tile it — takes 443 ms. Even the
fully-cold figure is an order of magnitude above the decode rate this class of
model runs at, so the gather is not the bottleneck; at ~100 tok/s it costs under
a tenth of a decode step, and that is the worst case where no row is resident.
Resident memory did not move during the run.

That settles the memory plan: ~83 GB of resident weights, the 52 GB table read
on demand, and ~36 GB left for KV cache and activations.

### Confirmed on the box, end to end

`spark serve` against the real RadixArk checkpoint on a GB10, 2026-08-26:

```
Selected kernel target: (sm_121, qwen3.8-flash-next, nvfp4) (167 modules)
Model config: 48 layers, 12 attention, 36 SSM, 512 experts, head_dim=256
Quantization config: method="modelopt", algo="NVFP4"
Demand-paged (never resident): [".ple.ple_embedding.ngram_embedding.shard_"]
Fast-load pre-flight: 78.19 GB on-disk, 1.3x = 101.65 GB peak, 114.25 GB free
Loaded 296347 weight tensors
qwen4_exp weights load (48 layers, 1 PLE towers), but the forward pass is not
implemented ...
```

Every tensor the loader asks for is present and loads. 69 s, peak ~88 GB used
with ~26 GB spare. Without the demand-paged exclusion the same run refuses at
163.64 GB — so that one change is the difference between "does not fit" and
"loads with headroom".

What remains is the forward pass, and only the forward pass.

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

## Component notes — what each piece is, and what it is checked against

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

### Dispatch — done
`config/parsers/qwen4_exp.rs` is the single parser, reached from
`dispatch.rs::parse_config` under both `qwen4_exp` and `qwen3_8_flash_next`,
and `factory.rs::loader_for_config` binds `Qwen4ExpWeightLoader`. The config is
nested under `text_config` like the Qwen3.5/VL families but shares nothing else
with them.

Two things that parser does which are easy to omit: it sets
`weight_prefix = "model.language_model"`, so `config.layer_prefix(i)` yields
real checkpoint keys and the shared loader helpers need no qwen4_exp-specific
naming; and it calls `finalize_config`, which is where `quantization_config` is
read off the TOP level — ModelOpt writes it beside `text_config`, so serde on
`text_config` alone never sees it, and without that call an NVFP4 checkpoint
parses as unquantized.

It populates BOTH field spellings on purpose: `hc_mult` / `index_*` /
`emb_*` because that is what the engine dispatches on, and `hc_count` /
`indexer_*` / `ngram_size` / `heads_per_ngram` because the weight manifest, the
CPU reference and `Qwen4ExpNgram` read the checkpoint in its own vocabulary.
`Qwen4ExpNgram` DERIVES the multipliers, per-head primes and offsets and
asserts they equal the buffers the checkpoint ships, so the derivation and the
shipped values check each other.

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

### Sparse-attention indexer — NOT on the critical path for a first deployment

`block_topk = indexer_budget / indexer_compress_ratio = 2048 / 4 = 512`, and the
selection takes `min(block_topk, num_complete_blocks)`. A sequence with no more
complete blocks than that has **every** block selected, so the mask cannot mask
anything: the indexer is exactly a no-op and dense attention is numerically
identical.

Measured against HF's own module (`scripts/dev/probe_indexer_threshold.py`),
counting causally-visible tokens the indexer removes:

| kv length | tokens masked |
|---|---|
| 16 | 0 — no-op |
| 512 | 0 — no-op |
| 2048 | 0 — no-op |
| 2052 | 4 — restricts |
| 2064 | 112 — restricts |

So a first bring-up capped at **2048 context needs no indexer at all** and is
exact, not approximate. That takes one of the three new layers off the critical
path and leaves two new (PLE tower, hyper-connections — both with oracles) plus
three adaptations.

### Sparse-attention indexer — what it does past the budget
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

### Linear attention — mostly REUSE, not adaptation
36 of 48 layers. `in_proj_qkv` / `in_proj_a` / `in_proj_b` / `in_proj_z`,
`conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj`; 16 key heads x 128, 48 value
heads x 128, conv kernel 4, SSM state in fp32.

**Atlas's `SsmWeightsQwen35` is already a structural match** — the same nine
tensors under the same `{layer}.linear_attn` prefix, with the separate
`in_proj_qkv`/`in_proj_z` split rather than the older fused `in_proj_qkvz`. The
dimensions differ but are config-driven.

The real delta is one flag. `qwen4_exp` declares **`output_gate_type =
"sigmoid"`**; Atlas's GDN hardcodes SiLU on the gated output norm, which is
correct for Qwen3.5/3.6 and wrong here. That field is now parsed
(`ModelConfig::output_gate_type`) and pinned by a test; making the gated norm
select on it is the adaptation.

Worth noting the two norm conventions this model uses, because they are not the
same and sit a few lines apart in the reference:

* `Qwen4ExpTextRMSNorm` — weight is an **offset from 1** (`x * (1 + w)`), and
  optionally **grouped**. Used by the PLE tower and hyper-connections.
* `Qwen4ExpTextRMSNormGated` — weight is a **plain multiplier** (`w * x`), then
  multiplied by `activation(gate)`. Used by the GDN output.

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

## The real checkpoint — loaded, measured, twice

Superseding an earlier section that said the checkpoint fit neither the disk
nor the memory of `reiner`. It fits both, once the n-gram table is
demand-paged, and it has been loaded independently by two people on two GB10s:

| | |
|---|---|
| pre-flight | 78.19 GB on-disk, 1.3x = 101.65 GB projected peak |
| measured peak | ~90.4 GB, ~27.8 GB free, 8 GiB guard held throughout |
| load | 206/206 shards, 296,347 tensors, ~139 s |
| without the demand-paged exclusion | refuses at 163.64 GB |

That last line is the whole point: one exclusion is the difference between
"does not fit" and "loads with headroom".

The FP8 release still does not fit — its routed experts alone are 123.3 GB
resident — so NVFP4 is not the smaller option here, it is the only workable
one.

## Suggested order — spent

This section used to sequence the port. Every step of it has landed; the
history is in `QWEN4_EXP_PORT_LOG.md` and in #754's comment thread, which
records each defect in the order it was found. Kept as a heading so links to it
do not dangle.

## The forward pass, end to end

Read off `modeling_qwen4_exp.py` and confirmed by running the full model on the
tiny checkpoint (zero missing keys). This is the wiring the GPU layers have to
reproduce; the two novel blocks in it already have oracles in
`atlas_core::qwen4exp_reference`.

```
embeds  = embed_tokens[ids]                      # [S, hidden]
hidden  = tile(embeds, hc_count)                 # [S, hc*hidden]  <- literally repeated
for layer in 0..48:
    if layer hosts PLE:                          # layer 1 only (ple_layer_ids is 1-indexed)
        hidden += ple(hidden, ids)               # [S, hc*hidden], see PLE oracle

    mixed, inject = attn_hyper_connection(hidden) # [S, hidden], [S, hc]
    hyper = hidden                                # kept UN-normalised for the residual
    mixed = linear_attn(mixed)  or  self_attn(mixed)
    hidden = hyper + broadcast(mixed, inject)     # mixed[:,None,:] * inject[:,:,None], flattened

    mixed, inject = mlp_hyper_connection(hidden)
    hyper = hidden
    mixed = moe(mixed)
    hidden = hyper + broadcast(mixed, inject)

hidden = hyper_connection_mixer(hidden)          # [S, hidden]; no injection on this one
logits = lm_head(hidden)
```

Three things worth stating because they are not obvious from the shapes:

* The residual stream is `hc_count * hidden` wide for the **whole** trunk. Each
  block collapses it to `hidden`, computes, and broadcasts back with per-stream
  injection gains. There is no point at which the model is `hidden`-wide except
  inside a block.
* The initial state is the token embedding **tiled** `hc_count` times, not
  projected or zero-padded.
* There is **no final norm** — `hyper_connection_mixer` is what normalises
  before the LM head, and it is the `use_combine=False` variant, so it returns
  only the mixed value and no injection.

### What is left to write

| piece | state |
|---|---|
| PLE tower | **SERVES** at model layer 1. CPU oracle 5.1e-7; GPU vs the real reference module cos 0.999998, max_rel 0.40–1.24%; GPU vs the CPU oracle in `ops/qwen4exp_oracle_tests.rs` |
| hyper-connections | **SERVES** on all 48 layers. Oracle 1.6e-7; GPU vs the reference golden cos 0.999998 with `hc_post` bit-exact; also gated against the CPU oracle at T=1/64/96, which is what selects the split / GEMM collapse |
| trunk entry (`hc_expand`) | **SERVES**. Exact, sentinel-checked |
| linear attention (36 layers) | **SERVES**. Atlas's GDN + the sigmoid output gate selected from `output_gate_type` |
| gated-Q full attention (12 layers) | **SERVES** on the production PAGED path. Our flat-buffer `q4e_attn_decode` stays as the microtest oracle |
| 512-expert MoE | **SERVES** — reuse, with `norm_topk_prob` pinned true |
| QSA indexer | **SERVES**: decode-side and per-query prefill selection, parity-gated at T=2200 where selection actively prunes |
| layer wiring | **SERVES**, as hc arms on `qwen3_ssm` / `qwen3_attention` rather than a standalone layer — which is what carries paging, graphs and prefix caching |
| vision tower | **LOADS AND RUNS** via the qwen35 ViT loader |
| CUDA graphs | capture + replay clean; correct but roughly neutral on speed at C=1 |
| prefix caching | correct AND faster — PLE/QSA aux rides the snapshots warm turns hit (3.9x warm TTFT) |
| concurrency | C>1 honoured; C=2 measured at 20.5 tok/s aggregate |
| MTP | **not implemented**. `load_mtp_weights` returns `None`, so `--speculative` is refused at pre-flight rather than half-wired |
| stacked expert layout | **unreached** — both published releases split their trunk experts. The manifest describes it, so pre-flight would name the missing tensors rather than failing inside a kernel |

Measured on a GB10 (rsafier, #754): decode 16.5 tok/s, prefill 747 tok/s at a
2191-token prompt.

### The 512-expert MoE is REUSE, like the linear attention was

This was listed as "adapt Qwen3.5's path". It needs no adaptation at all.

`load_moe_inner` (`weight_map/loaders_moe.rs`) expects exactly the names this
model uses:

    {layer}.mlp.gate.weight
    {layer}.mlp.shared_expert_gate.weight
    {layer}.mlp.shared_expert.{gate,up,down}_proj.weight
    {layer}.mlp.experts.{e}.{gate,up,down}_proj.weight

and the target checkpoint's trunk experts are `PerExpert`, which is that
function's default arm. Passing the bare layer prefix is enough; `Layer.moe` is
a plain `MoeWeights`.

Two caveats worth keeping visible:

* The `Stacked` layout (HuggingFace-native — `experts.gate_up_proj` as one
  `[experts, 2*inter, hidden]` tensor, chunked at use) is NOT handled by that
  path. Both published releases split their trunk experts, so it is unreached
  today. `weight_manifest::qwen4_exp` still describes it, so preflight would
  name the missing tensors rather than failing inside a kernel.
* Reuse of the LOADER is not reuse of the dispatch. `MoeLayer`'s routing has to
  agree with this model's rule -- softmax over ALL experts first, then top-K,
  with `norm_topk_prob` forced true because the published config.json omits it
  and HF's default is true. Top-K then softmax silently renormalises away the
  router's confidence.

## Where the layer implementation actually went

This section planned a standalone `Qwen4ExpLayer` implementing
`TransformerLayer::{decode, alloc_state}` and inheriting
`default_loops::prefill_default`, which loops `decode` per token — correct, and
enough to serve. The #754 merge chose the other road, and it is the better one:
mHC, PLE and QSA hang off the EXISTING `qwen3_ssm` and `qwen3_attention` layers
as hc arms (`trait_decode_hc.rs`, `trait_prefill_hc.rs`,
`trait_prefill_block.rs`). A per-token prefill loop cannot scale, and hanging
off the existing layers is what carries the paged attention path, CUDA graphs,
prefix caching and C>1 for free.

The extraction is worth knowing about because it is the delicate part: the SSM
prefill fuses its residual adds into its norms (steps 1/11/13), so under a
highway they double-count. Steps 2–10 are residual-free, so they moved verbatim
into `prefill_block` — verified byte-identical against the pre-move source —
with `prefill_inner_hc` / `decode_inner_hc` as a second entry path.

The scaffold layer is retired. What survives from it, and is still load-bearing:

| piece | where it went |
|---|---|
| grouped RMS norm | `common/rms_norm.cu`, verified vs oracle |
| the three sigmoid gated-norm twins | `common/rms_norm.cu`, selected at layer init from `output_gate_type` |
| `q4e_hc_*` / `q4e_ple_*` / `q4e_attn_decode` | microtest oracles, independently reproduced on a second GB10 |
| `atlas_core::ngram_table` | the CPU forward and the `qwen4exp_*_check` examples |
| `ForwardContext::token_ids` | what the PLE tower reads — a device `[num_tokens]` u32 buffer added for DeepSeek-V4's hash-MoE routing, already there |

The serving n-gram rows come from `spark_storage::NgramRowCache` instead, whose
`open_segmented` handles the fact that the 128 PLE shards are NOT contiguous
(they span 26.4 GB of a 102.4 GB table), so a single base offset would read
wrong-but-valid rows silently.

### The PLE gather is a host round-trip, deliberately

The n-gram table is on NVMe, so layer 1 has to: read the ids, hash them
(`Qwen4ExpNgram::ngram_ids`), `pread` the rows, upload ~2.5 KB, then run the two
PLE kernels. That serialises with the GPU stream, which sounds bad and is not:
measured at 820 µs/token fully cold and 8.85 µs warm, against a decode step two
orders of magnitude slower. Making it a device-side gather would require the
51.2 GB table to be resident, which is the thing that does not fit.

### Remaining work

1. **MTP.** Its shape is not the `MtpWeights` any other family uses — its own
   512-expert stack, its own indexer, two input norms of *different* widths.
   When revisited, note the trap #746 documents for LongCat: the `verify_*`
   paths call bare `embed()`, which an n-gram guard refuses.
2. **The stacked expert layout**, if a release ever ships one.
3. **Thinking-body quality at the card's temperature 1.0.** With
   `norm_topk_prob` fixed the control flow is clean and greedy is coherent;
   whether the residual weakness is distribution quality is the open thread on
   #754, and `bench/qwen4_exp/{forward_ref,logit_quality_aligned}.py` plus
   `ATLAS_DUMP_LOGITS_PATH` are the harness for it.
4. **QSA prefix-cache re-ingest** needs the raw indexer keys in the radix
   payload; the guard errors loudly rather than serving stale keys.
5. **Chunk-1+ QSA prefill above 8193 tokens** is wired and correct by
   construction; the >8K envelope was validated end to end at 10,749 tokens
   through the chat template.

## GPU kernels: what exists and what it is checked against

Each block is checked against a CPU oracle that is itself checked against
HuggingFace at real weights, so agreement chains to the reference rather than
to another guess. Every check carries a **control that must fail** — otherwise
a kernel that ignored the thing being tested would pass.

| block | layers | vs oracle | control |
|---|---|---|---|
| `rms_norm_grouped` | all | 2.0e-3 | ungrouped norm, 300x larger |
| hyper-connection collapse | all (x2) | 3.9e-3 | injection must not saturate |
| PLE tower | 1 | 5.2e-3 | zeroed conv taps must change the answer |
| `gated_delta_rule_decode` | 36 | 3.3e-3 | shared-head mapping exercised |
| `q4e_attn_decode` | 12 | 3.4e-3 | ungated recompute must move the answer |
| `q4e_hc_expand` | trunk entry | exact | sentinel must be fully overwritten |

All five block types, at BF16-ulp level. Run them with
`cargo run --release -p spark-model --example qwen4exp_grouped_norm_microtest`.

### The attention oracle is extracted, not re-written

`attention_decode_step` was split OUT of `attention_forward`, which now calls
it. A second transcription of the same equations would have been free to be
wrong in the same direction as the kernel; this way agreement chains to the
code that matches HuggingFace at 8.0e-7.

Its gate control matters more than it looks. `q_proj` emits `[query | gate]`
per head and the gate is applied ELEMENTWISE before `o_proj` — so a kernel that
ignored `gate` entirely would still produce well-formed attention output. The
check recomputes with every gate forced to +8 (sigmoid ~ 1) and requires the
answer to move; it moves by 300x the error.

### The trunk entry: `hc_expand`, from this target's own shadow

Corrected after the #754 merge. This tree briefly carried
`q4e_hc_expand` in `common/` because DeepSeek-V4's `hc_expand` lives in THAT
model's shadow and so does not resolve here. The merge brought a
`hyper_connection.cu` shadow for `qwen3.8-flash-next` itself, which supplies
`hc_expand` / `hc_pre` / `hc_post` / `hc_head` under the names Atlas's mHC
dispatch already looks up — so the engine uses those, and `q4e_hc_expand`
remains only as a microtest oracle.

Both forms have the same load-bearing property, below.

The streams must start IDENTICAL — a copy of the embedding into all `hc_count`
slots. Zero-initialising them instead makes the first hyper-connection collapse
read a zero mean, and the model does not recover, while still emitting tokens.

### hc_streams: FP32 for both families, and why the per-family branch went

`hc_streams` was once gated on `hc_mult > 0`, DeepSeek-V4's spelling.
`qwen4_exp` says `hc_count`, so it silently received the 256-byte placeholder
and the first write past it would have been another buffer. Fixing that
introduced a second, subtler version of the same hazard: the element size was
branched per family, f32 for `hc_mult` and bf16 for `hc_count`.

That was right while this tree's own `q4e_hc_stream_mix` /
`q4e_hc_scatter_add` read the buffer — they stride by `__nv_bfloat16`. It is
wrong for the kernels that now serve: every entry point in
`hyper_connection.cu` declares this buffer `float*`. So it is unconditionally
f32, and the branch was REMOVED rather than left in place, because reading the
code as written would lead the next person to restore it.

| family | field | element | why |
|---|---|---|---|
| DeepSeek-V4 | `hc_mult` | f32 | manifold mixing is norm-preserving; bf16 swamps the per-layer signal and collapses generation |
| qwen4_exp | `hc_count` | f32 | `hyper_connection.cu` declares `const float* streams` / `float* out` on every entry point |

Getting it wrong does not fail: it reads the wrong half of every value, on all
48 layers. `buffers/sizes.rs` pins all three fillings — `hc_mult` only,
`hc_count` only, and both, which is what the parser actually produces — to the
same size.

**CLOSED** (was open): qwen4_exp's mixer is contractive, dividing by
`hc_count`, rather than norm-preserving, so V4's drift argument does not
transfer. The port keeps the FP32 highway regardless and generates coherently
across all 48 layers, so there is nothing to trade and nothing left to measure.

### The GDN kernel is reusable, with one interface trap

Atlas's `gated_delta_rule_decode` computes exactly this model's recurrence —
`hk_dot` on the undecayed state, then `(v - g*hk_dot)*beta`, which is
algebraically the decay-first form the reference uses.

But **it scales the OUTPUT by `1/sqrt(key_head_dim)`, where HuggingFace scales
the QUERY.** Same result — the output is linear in `q` and neither placement
touches the state — so a layer must pass `q` **unscaled** and let the kernel do
it. Passing the HF-shaped pre-scaled query applies the factor twice and shrinks
every linear-attention layer's contribution by 11x, across 36 of 48 layers,
while still producing text.

### Module names collide across model shadows

The qwen4_exp kernels are `qwen4exp_hc` and `qwen4exp_ple`, not
`hyper_connection` and `ple`. `hyper_connection` already belongs to
DeepSeek-V4's Sinkhorn mHC, and its model shadow beats `common/` — the lookup
failed with "named symbol not found" while the PTX held someone else's kernels.
That failure was loud. The reverse collision would have been silent, running one
model's weights through the other's mixing.

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
