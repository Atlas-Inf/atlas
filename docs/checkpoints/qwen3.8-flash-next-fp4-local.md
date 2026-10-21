# Qwen3.8-Flash-Next FP4-Local on one GB10 — native formats, measurements, open gap

`nvidia/Qwen3.8-Flash-Next-FP4-Local` (snapshot `b9ce18a6887e18c089fdea8509a33e63429a3300`, ~96 GiB,
ModelOpt QAD) served by Atlas on a single DGX Spark (GB10, 121.6 GB unified memory), with every
tensor family run on the format the checkpoint ships. This is the record of what was built, how it
was verified, what it measures, and where it still trails NVIDIA's own vLLM recipe on the same box.

**Status in one paragraph.** It loads, serves and is accurate: BFCL golden ST-995 87.44 overall /
87.32 normalized (n=995), passcode recall 12/12 from 32K to 245K tokens, multi-turn state and
cross-request isolation gates pass. Weights are checkpoint-native (no widening copies, no 8→4-bit
requantization). It is **not yet as fast as NVIDIA's vLLM recipe**: on the same box with the same
client, vLLM decodes 1.6–1.9x faster and prefills ~1.7x faster (table in §6). Closing that is the
next piece of work (§8).

All numbers below were measured on `sparky` (GB10, driver 580.178.04, CUDA 13.0) with Atlas built
at `a78d2fb54` (single-target `qwen3.8-flash-next`, binary sha256 `8d566e6895f6f544…`). Each cites the
`spark benchmark` run id (`~/.atlas/runs/<id>`), or names the streaming client used. Single-run
figures are labelled as such; nothing here is cross-validated beyond what is stated. After the
branch was rebased onto main `e29809db9`, the rebased build answered the 4-request output control
byte-identically to the measured `a78d2fb54` build, so the numbers carry over to the PR head.

## 1. What the checkpoint ships

| Tensor family | Count | On disk |
|---|---|---|
| Routed experts (48 layers × 512, gate/up/down) | 73,728 | NVFP4 (E2M1 + per-16 E4M3 scale + F32 scale_2), **scales CSF-compressed** (`weight_scale.nvfp4_csf_fixed` + `nvfp4_csf_exceptions`, U32) |
| GDN `in_proj_qkv`, `in_proj_z`, `out_proj` | 108 | MXFP8 (E4M3 + E8M0 per 32 inputs) |
| GDN `in_proj_a`, `in_proj_b` | 72 | MXFP8 |
| Full attention q/k/v/o (12 layers) | 48 | MXFP8 |
| QSA indexer `index_qk_proj` | 12 | MXFP8 |
| Shared expert gate/up/down | 144 | MXFP8 |
| Vision `attn.qkv`, `attn.proj` (27 blocks) | 54 | MXFP8 |
| Vision MLP `linear_fc1/fc2` + merger | 56 | NVFP4 (W4A16) |
| hyper-connection `hc_norm` + PLE `norm_conv/key/query` | 100 (+3 MTP) | **F32** (every other release ships BF16) |
| PLE n-gram table | 128 shards × 2,500,012 rows | NVFP4 rows U8 `[R,80]` + per-shard E4M3 `[R,10]` + one F32 `weight_scale_2` (28.8 GB) |
| MTP experts | 512 | per-expert NVFP4 |
| MTP shared expert, mHC mixing weights, LM head, embeddings | — | BF16 |

Shards live under `tensors/` in the index.

## 2. What Atlas does with each (branch `feat/fp4-local-flash-next`)

| Family | Before this branch | Now |
|---|---|---|
| CSF expert scales | not loadable | decoded losslessly to plain E4M3 at load (`atlas-core::modelopt_codec`), byte-identical to vLLM's reference decoder (vllm#60646 spec) on the real weights; compressed originals never resident. Same as vLLM and TensorFold do. |
| GDN qkv/z/out, attention q/k/v/o | expanded to BF16 (attention then requantized to NVFP4 — lossy) | **native MXFP8**: `mxfp8_proj.cu` GEMV (1 row), batched GEMV (2–8 rows, every row bit-identical to the 1-row kernel), transient exact dequant + BF16 GEMM for prefill |
| Shared expert | MXFP8 → BF16 → **NVFP4 requant (8→4-bit loss on the one expert every token uses)** | **native MXFP8** through the separate-shared-expert seam (`SeparateSharedExpert::{Bf16,Mxfp8}`); gate‖up one `[1280,2560]` projection; fused routed kernels run with a NULL shared expert in every MoE arm |
| MTP shared expert | BF16 → NVFP4 requant | BF16 as shipped (same seam) |
| F32 norms | read as BF16 → NaN, then rounded to BF16 (lossy on all 100 tensors) | **F32 end to end**: hc/PLE norm kernel arguments are `const float*`; BF16 packs are widened exactly (`bits << 16`) |
| Vision MXFP8 attn / NVFP4 MLP | host-expanded to BF16 | **native**: uploaded as shipped, expanded per GEMM into a 10 MB encoder scratch (bit-identical to the old expansion; −0.52 GB) |
| PLE n-gram table | host decode of every row + stream sync per gather | **disk-backed, hot rows in pinned RAM, decoded on the GPU** (`batched_embed_nvfp4`, bit-identical to the host decoder); one cache holds rows and their block scales slot-parallel; `ATLAS_PLE_CACHE_GB` sizes it |
| GDN `in_proj_a/b`, indexer `index_qk_proj` | — | still expanded to BF16 (exact for MXFP8, ~28 MB). Not yet native. |

Commits (rebased onto main for the PR; measured at the pre-rebase SHAs):
codec · spark-runtime loader (`tensors/` shards, U32, deferred host-decoded modules) · serve FP4-Local
· MTP per-expert NVFP4 · GDN native MXFP8 · attention native MXFP8 · F32 norms native · shared expert
native MXFP8 · gate‖up single buffer · PLE GPU decode + scalable hot cache · vision native.

## 3. How exactness was verified

* **Codecs:** CSF decode byte-identical to vLLM's reference decoder on the real tensors; MXFP8 dequant
  exact vs host codec; NVFP4 PLE GPU decode identical to the host decoder over all 256 scale codes ×
  all packed bytes and over 4,096 real rows read through the cache; vision dequant identical for all
  54 MXFP8 + 56 NVFP4 projections (`examples/*_microtest.rs`).
* **NVFP4 pack never moved:** after every commit the NVFP4 pack (`nvidia/Qwen3.8-Flash-Next-NVFP4`)
  answered a fixed 4-request control (math, explanation, code, tool call; temp 0) byte-identically
  on the old and new binary.
* **FP4-Local output identity across the non-numeric commits** (GPU PLE decode, gate‖up buffer,
  vision): the same 4-request control byte-identical old vs new (one tool request varies run to run
  on the *same* server through an optional preamble sentence — prefix-cache path, not a decode
  delta).
* **Speculation and batching are exact on FP4-Local's native paths** (live `*_CHECK` diagnostics,
  every checked step re-run as serial decode on the live state):
  `ATLAS_QWEN4EXP_EXACT_VERIFY` + `MTP_DEPTH=3` + `HC_FAST`: 64 verify steps / 256 rows, 0 differ;
  `BATCH_FAST`+`BATCH_SMALL` at C=4: 128 steps / 512 rows, 0 differ; with MTP K=1 at C>1: 64 steps /
  444 rows, 0 differ; with `HC_FAST`+`MOE_UNITS`: 128 steps / 1,024 rows, 0 differ.

## 4. Memory: why the n-gram table stays on disk

The PLE table is 28.8 GB (25.6 GB packed rows + 3.2 GB scales). Making it resident on one Spark
needs ~131 GB against 121.6 GB (Atlas pre-KV ~93 GB + table + 4.2 GB working reserve + 5.5 GB OS).
It does not compress: measured on shards 0/57/127, the FP4 payload's nibble entropy is
**3.966 of 4 bits** and zstd-19 ratio **1.000**; only the scales compress (ratio 0.455, ~1.5 GB at
best). Getting ~17 GB elsewhere would have meant dropping MTP, a MoE-kernel rework for compressed
scales and still ~1 user at ~128K context. NVIDIA's own Spark recipe (`VLLM_PLE_TABLE_MEMORY=disk`)
and TensorFold also read the table from disk. Decision: **disk-backed, hot rows pinned in RAM,
GPU decode.**

Hot-row cache (FP4-Local, same realistic workload per size — BFCL small draw, 16K-token prose
needles, code generation, the BFCL draw again):

| `ATLAS_PLE_CACHE_GB` | slots | tracked weights | KV | misses | hit rate | evictions |
|---|---|---|---|---|---|---|
| unset (84 MB) | 524,288 | 91.32 GB | 11.2 GB (473K tok) | 433,477 | 94.6% | 0 |
| 2 | 22.2M | 93.26 GB | 9.3 GB (393K tok) | first-touch only | — | 0 |
| 8 | 88.9M | 99.19 GB | 3.1 GB (130K tok) | 433,477 | 94.8% | 0 |

Every miss in this workload is a first touch; repeating the BFCL draw added zero misses even at
84 MB. A larger cache only pays once a session's unique rows exceed it, so the recipe uses **2 GB**,
which keeps a 262K-token context. "Tracked" is the driver-accounted figure; the boot log's
free-memory pre-KV estimate swings by up to ±7 GB between boots of the same binary and must not be
used for A/Bs.

## 5. Serve configuration that was benchmarked

Existing default-off qwen4_exp levers were swept on FP4-Local (fresh boot per config; decode-floor =
3 × 1,500-token MinHeap code generation, reasoning off, server-side rate; quick = 512-in/512-out):

| C=1 config | decode-floor | quick | accept len |
|---|---|---|---|
| recipe default, MTP K=1 | 28.3 (run-1792317977888315647) | 27.4 | 1.91 |
| + exact verify + exact batching lane + fused decode, MTP depth 3 | 31.9 (run-1792342260484689522) | — | 3.32 |
| + `ATLAS_QWEN4EXP_HC_FAST` (adopted) | **33.0** (run-1792342559125109723) | **34.5** (run-1792342600593663107) | 3.32 |

Drafts deeper than 3 did not engage: the scheduler capped single-sequence MTP at `k_drafts=3` even
with `MTP_DEPTH=5/7` and matching `--num-drafts`. Exact verify also lets MTP run inside `<think>`.

Chosen serve line (single user):

```
ATLAS_QWEN4EXP_EXACT_VERIFY=1 ATLAS_QWEN4EXP_DECODE_FUSE=1 ATLAS_QWEN4EXP_BATCH_FAST=1 \
ATLAS_QWEN4EXP_BATCH_SMALL=1 ATLAS_QWEN4EXP_MTP_DEPTH=3 ATLAS_QWEN4EXP_HC_FAST=1 ATLAS_PLE_CACHE_GB=2 \
spark serve <FP4-Local snapshot> --kernel-target qwen3.8-flash-next --gpu-memory-utilization 0.9 \
  --max-seq-len 32768 --max-prefill-tokens 8192 --kv-cache-dtype bf16 --max-num-seqs 1 --max-batch-size 1 \
  --speculative --num-drafts 3 --enable-prefix-caching --fast-load-prefetch-shards \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}'
```

Multi-user: same levers minus `MTP_DEPTH`/`HC_FAST`, plus `ATLAS_MTP_MAX_SEQS=8`, `--max-num-seqs 8
--max-batch-size 8 --speculative --num-drafts 1`.

## 6. Results

### Quality and robustness (FP4-Local)

| Gate | Result | Run |
|---|---|---|
| BFCL golden ST-995 (no overrides, reasoning low) | **87.44 overall / 87.32 normalized**, n=995 | run-1792340347555552942 |
| Passcode recall, real-prose filler, 3 depths each | **12/12** at 32K, 64K, 128K, 245K prompt tokens | needle_long.py |
| SSM state poisoning (12 replayed rounds) | PASS, 12/12 invariant | run-1792328829391992739 |
| Cross-request contamination | PASS, 8/8 identical | run-1792329447184876759 |
| Video fidelity | PASS, 7/7 legs | run-1792328221225737347 |
| Vision fidelity | FAIL on 1 of 10 integrity probes (`multi-choice-image`); geometry 14/14, probes 3/3. Fails identically on the NVFP4 pack and on the pre-vision-change binary — pre-existing, not FP4-Local | run-1792328210080553327 |

For reference, the NVFP4 recipe documents ST-995 84.92 on an older binary; a same-binary NVFP4 BFCL
run is still queued, so no FP4-Local-vs-NVFP4 accuracy claim is made here.

### Speed, Atlas-internal harness (single run each; v1 = before HC_FAST)

| | FP4-Local | NVFP4 pack (same flags) |
|---|---|---|
| decode-floor, C=1 | 32.5 (v1, run-1792327978365873482) / 33.0 (v2) | 31.9 (v2, run-1792343818643478918) |
| quick 512/512 decode / TTFT | 34.1 / 224 ms (v1) | 32.6 / 214 ms (v2) |
| cold TTFT 1K / 4K / 16K | 1.05 s / 2.94 s / 12.1 s (run-1792328125909646904) | 3.17 s median (run-1792343954260808403) |
| warm TTFT 4K / 16K | 213 ms / 259 ms | — |
| 262K serve: KV capacity, decode-floor | 257,056 tokens, 31.8 tok/s | — |
| 240K-token prompt TTFT | ~233 s (~1,050 tok/s prefill) | — |
| aggregate tok/s, isl 512, C1/C2/C4/C5/C8 | 24.1 / 28.7 / 36.9 / 37.4 / 42.3 (run-1792329307611412276) | 22.7 / 26.5 / 35.2 / 40.3 / 42.9 (run-1792345177245084155) |
| aggregate tok/s, isl 2048, C1/C4/C8 | 22.6 / 33.2 / 35.2 | 20.6 / 30.9 / 35.0 |

### Same box, same client: Atlas vs NVIDIA's vLLM recipe

NVIDIA's DGX Spark recipe from the model card, flag for flag (`eugr/spark-vllm-b12x:latest`, MTP 4
drafts, MXFP8 LM head, b12x kernels, FP8 KV, PLE table on disk, util 0.8) against Atlas's chosen
single-user config, both driven by `ab_stream.py` (client clock from the SSE stream, server usage
for token counts, `reasoning_effort: low`, temperature 0, 3 distinct prompts each, medians):

| | Atlas | vLLM (NVIDIA recipe) |
|---|---|---|
| short chat: decode / first token | 25.6 tok/s / 0.38 s | **41.5 tok/s / 0.26 s** |
| ~17–19K-token agent first turn (24 tools + docs): decode / first token | 22.9 tok/s / 15.4 s | **43.6 tok/s / 9.2 s** |

This reproduces the community-reported numbers for that recipe (44 tok/s short, 0.27 s first token,
41.9 tok/s at a ~14K agent turn). Atlas also renders ~2K more prompt tokens than vLLM for the same
agent request (tool-schema templating) — worth its own look. TensorFold's README claims (~50 tok/s
single stream, ~130 aggregate at 5 users) were not reproduced here.

## 7. Where the time goes (nsys, FP4-Local, before the lever sweep)

Plain C=1 decode, 46.9 ms GPU per token: native MXFP8 projections 13.6 ms (at the ~216 GB/s ceiling
measured for this box), routed MoE gate/up + silu/down 11.6 ms (~53% of ceiling: 1.33 GB of expert
bytes a token), mHC down/finish 10.4 ms (~56%), BF16 LM head 5.0 ms. With MTP K=1 a verify cycle
read the BF16 LM head ~2.4×. At C=4 every row runs its own MoE (each row routes to its own 10 of 512
experts) and the LM head ran on `dense_gemm_bf16` (13.8 ms a step; the exact batching lane replaces
it with a batched GEMV).

## 8. What is left (ranked by expected gain)

1. **Routed-MoE decode kernels at ~53% of bandwidth** — the largest single lever at every
   concurrency and for MTP verify rows, which each read their own experts.
2. **MTP depth 4** like the vLLM recipe: lift the scheduler's single-sequence 3-draft cap.
3. **LM head**: vLLM ships it MXFP8 (`VLLM_MXFP8_LM_HEAD=1`); an opt-in MXFP8 head (quality-gated)
   would halve its 1.27 GB read.
4. **Prefill**: ~1,050–1,400 tok/s vs vLLM's ~1.7x; MoE prefill and the PLE first-touch faults.
5. Remaining non-native tensors: GDN `in_proj_a/b`, indexer `index_qk_proj` (exact today, 28 MB).
6. Nits: boot log prints the PLE cache arena size wrong (tracked memory is right); strix-hip mirror
   of the hc kernels updated but not compiled on HIP here; Atlas's template renders tool schemas
   longer than vLLM's.

## 9. Reproduction

Scripts in `bench/qwen4_exp/fp4local/`: `control.py` (4-request output control), `needle_long.py`
(long-context recall on real prose), `ab_stream.py` (engine-neutral streaming A/B), `prof_req.py`
(nsys decode load). Everything else is the in-tree `spark benchmark` suite (`decode-floor`,
`quick-speed-bench`, `concurrency-sweep`, `ttft-cold-gate`, `ttft-warm-gate`, `bfcl-subset`,
`vision-fidelity`, `video-fidelity`, `cross-contamination`, `ssm-state-poisoning-gate`).

Environment note: the BFCL scorer's venv (`~/.atlas/artifacts/bfcl/venv`) needed `soundfile`
(`qwen_agent` imports it) plus `libsndfile1`; without it every `bfcl-subset` run fails at scoring
after generating all responses.

## 10. Update — overnight results and format-driven loading

**Same-binary quality A/B** (only the checkpoint differs; BFCL golden ST-995, no overrides; single run
each): FP4-Local **87.44 / 87.32** (run-1792340347555552942) vs NVFP4 pack 86.23 / 85.64
(run-1792378038850732185). An observation, not a significance claim.

**v2 profile (adds `ATLAS_QWEN4EXP_HC_FAST`):** decode-floor 33.0 tok/s (run-1792378753368339758),
quick 512/512 34.7 tok/s, SSM-poisoning 12/12 invariant, long-context recall 12/12 at 32K–245K.

**Image:** `azeezish/atlas-gb10:a78d2fb` built from the measured commit (all targets, 3.0 GB), served by
sparkrun from the rendered recipe: coherent, needle PASS. Its temperature-0 answers are correct but
not byte-identical to the bare-metal single-target binary — not yet root-caused (multi-target kernel
selection or the in-container template path). Not pushed to any registry.

**Format-driven loading:** `weight_map/modelopt_plan.rs` reads the checkpoint's declared per-module
formats (`quantization_config.quantized_layers`, or `hf_quant_config.json`) — NVFP4 / W4A16_NVFP4 /
MXFP8 / FP8 block, group size in either spelling, `weight_scale_encoding`. CSF scales decode only for
encoding `"csf"`; any other encoding fails at load naming module and encoding. A per-model
native-consumer registry says which roles run natively (qwen4_exp declares the set above); every other
declared module is expanded exactly to BF16, so a new FP4-Local checkpoint loads correctly before it
is optimized. FP4-Local and NVFP4 outputs are byte-identical to the previous build.
