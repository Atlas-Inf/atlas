# Qwen3.8-Flash-Next FP4-Local on two GB10s (TP=EP=2): resident n-grams, native formats

The two-Spark continuation of [qwen3.8-flash-next-fp4-local.md](qwen3.8-flash-next-fp4-local.md)
(PR #195). `nvidia/Qwen3.8-Flash-Next-FP4-Local` (snapshot `b9ce18a6887e18c089fdea8509a33e63429a3300`)
served by Atlas on two DGX Sparks joined by one ConnectX-7 cable, tensor and expert parallel
(`--tp-size 2 --ep-size 2`), with the 28.8 GB NVFP4 PLE n-gram table held **resident in device
memory on both ranks**. That is the thing one Spark cannot do: there the table stays on NVMe behind a
RAM row cache (§4 of the single-Spark doc).

**Status.** It serves and is exact where exactness is claimed (§3a lists each contract and what
proves it). Greedy output with speculation equals serial greedy output (8/8 at 1 and 8 concurrent
streams) after every change below. The resident table is bit-identical to the NVMe-cached path. BFCL
golden ST-995 is 87.44 / 87.45 (n=995) on the current numerics. All numbers are from one pair (two
GB10s, driver 580, CUDA 13.0, cuBLAS 13.0.2.14 pinned on both ranks) with the rc16 TP2 profile: util
0.88, 8 sequences x 32K, MTP 7 drafts, exact verify, prefix cache, `reasoning_effort` low. Single runs
unless repetitions are listed.

## 1. What changed (on top of PR #195)

| Change | Why |
|---|---|
| MXFP8 tensor parallelism by byte slicing | GDN `in_proj_qkv`/`in_proj_z` (by q/k/v segment and value head) and attention q/k/v (by head) are sliced by rows. `o_proj` and GDN `out_proj` are sliced by columns on whole 32-input E8M0 groups (`(k/tp) % 32 == 0` is enforced). No dequant or requant; a shard is byte-identical to the shipped rows/columns. |
| Shared expert added once under EP | The MXFP8 shared expert stays replicated. Under EP it is blended as a zero row before the all-reduce and added once after it. Two batched decode paths blended it before the reduce, counting it once per rank. **Cross-model:** the generic `forward_batched` EP tail is shared by every EP model with a shared expert; it now follows the contract the single-row `forward` already had (`moe/forward.rs`, "EP fix"), which assumes the shared expert is replicated on every EP rank (FP4-Local's loader, `weight_loader/qwen4_exp/ffn.rs`, loads the full module on each rank with no shard call). Any EP model that reached this tail with a shared expert had it counted once per rank before. |
| Resident PLE table, `ATLAS_PLE_RESIDENT=1` | The 128 NVFP4 shards are streamed through pinned staging into one rows buffer and one block-scale buffer per rank, then the files are fadvise(DONTNEED)'d. A boot self-check compares sampled rows (first, last, each head's first and last row, 64 pseudo-random) against the host oracle. The gather uses the existing GPU decode with slot = row id. No row cache, keepalive or fault pool. Startup parity refuses a rank mismatch. |
| BF16 router (dtype fix) | `mlp.gate` ships BF16 (not in `quantized_layers`), but Atlas requantized it to NVFP4 at load and routed on that. FP4-Local now routes on the shipped weight: decode/verify on row-invariant GEMV tiers (scoped to these layers by a load-time flag), prefill on the pipelined BF16 GEMM. `ATLAS_QWEN4EXP_ROUTER_NVFP4=1` restores the old path for A/B. |
| TC v3 MoE units without a fused shared expert | The plan kernel takes `has_shared`. FP4-Local runs the tensor-core MoE units with its MXFP8 shared expert as a separate chain (previously it fell back to the slower rows path). |
| MXFP8 multi-row projections | 2..=32 rows are served in one weight pass (bucketed GEMV, plus a register-tiled 17..=32 tier), every row bit-identical to the 1-row kernel. Before, verify windows re-streamed each weight once per 8 rows. |
| Decode/verify MXFP8 rows exact at any batch | `mxfp8_proj_exact_rows` serves any row count as consecutive <=32-row GEMV launches. Decode and verify callers (GDN `exact_rows_proj`, attention `try_mx8(exact = true)`, the shared expert's decode/verify entries) use it; prefill keeps `mxfp8_proj` (dequant + GEMM past 32 rows). Before this, a batched verify passes every sequence's rows in one call (`qwen3_ssm/trait_decode_batched.rs`, `trait_decode_multi_seq/ssm_batched/impl.rs`): at 8 sequences x 8 verify rows that is 64 rows, which took the dequant + GEMM path, so C8 verify rows were not the serial GEMV's bytes. The C8 speculative-equals-serial gates matched on text, but bit-identity was not guaranteed there until this change. |

A tensor-core MXFP8 tier was also built and tested (row-invariant, within 1 BF16 ULP of an FP64
reference) but measured ~32% slower than the GEMV tiers at decode, so it is not shipped; it is kept
on the `wip/mxfp8-tc` branch.

## 2. Dtype contract at TP2

| Tensor family | Ships | Served |
|---|---|---|
| Routed experts | NVFP4, CSF-compressed scales | CSF decoded once at load; W4A16; 256 experts per rank |
| Attention q/k/v/o, GDN qkv/z/out | MXFP8 | native bytes, sliced per rank |
| Shared expert | MXFP8 | native, replicated |
| GDN `in_proj_a/b`, indexer `index_qk_proj` | MXFP8 | exact widening to BF16 (~28 MB), as on one Spark |
| Router, `shared_expert_gate`, mHC, PLE projections, embed, lm_head | BF16 | BF16 |
| `hc_norm` and PLE norms | F32 | F32 |
| PLE n-gram table | NVFP4 (U8 + E4M3 + F32 global) | resident per rank, GPU-decoded |
| Vision attention / MLP | MXFP8 / W4A16 NVFP4 | native bytes (attention dequantized on GPU into scratch) |
| KV cache | unspecified | BF16 |
| MTP attention, MTP shared expert, draft head | BF16 | draft-only paths; may approximate, which changes acceptance, never tokens |

## 3. Verification

- Exactness on the pair after each change: serial vs rc16 greedy 8/8 at C1 and C8; cached-prefix C4
  8/8 vs no cache; cross-contamination 8/8; tool probe 21/21; needles at 32K / 128K / 240K (4x262k profile).
- Resident vs NVMe-cached PLE: outputs bit-identical, arm for arm. Under a full benchmark set,
  `/proc/<pid>/io` read_bytes moved by 0 on both ranks, with no checkpoint files open.
- Kernel tests: MXFP8 shard byte-exactness (column, segmented, row-parallel) with an off-by-one-scale
  negative control, and row-parallel reassembly `dequant(shard0) || dequant(shard1) == dequant(full)`
  compared bit for bit on hostile E4M3/E8M0 bytes; resident vs cached gather parity on interleaved
  shards incl. the last row; the multi-row MXFP8 tiers byte-identical to the 1-row kernel at
  M = 1..=32 on the real shapes with hostile bytes, and the exact-rows path at M = 64 / 65.
- Launch-recording tests: `m = 0` is refused with no launch; exact rows at M = 33 / 64 / 65 launch
  only GEMV tiers (ceil(M/32) per projection, never the dequant); the shared expert's exact chain runs
  the fused gate_up -> SiLU -> down sequence once per <=32-row chunk with per-chunk offsets; the EP
  tails (qwen4exp fast path and the generic `forward_batched`) consume the shared expert exactly once
  (zero row before the single all-reduce, one post-reduce add; without EP, blended once in place).
- BFCL ST-995 golden draw: 87.64 / 87.71 (`run-1792365985129633322`, first resident build, NVFP4
  router) and 87.44 / 87.45 (`run-1792406664787105023`, BF16 router + TC MoE). The difference is
  within run-to-run noise.

## 3a. Exactness and determinism contracts

| Contract | Holds by | Proven by |
|---|---|---|
| A speculative verify row equals the serial decode row | Every decode/verify projection takes a row-invariant kernel: MXFP8 through `mxfp8_proj_exact_rows` (any M), the BF16 router through the `dense_rows` GEMV tiers, TC MoE units row-invariant by construction | Kernel byte-identity tests (§3); speculative == serial greedy on the pair |
| Prefill may use other numerics than decode | Prefill takes dequant + BF16 GEMM for MXFP8 past 32 rows and the pipelined BF16 GEMM for the router. Each logit/output is an independent in-order chain, so prefill is chunk-invariant, which the cached-prefix gate requires | Cached-prefix C4 8/8 vs no cache |
| Both ranks route identically | The router input is the replicated hidden state after the TP all-reduce. With two ranks every element is one BF16 add of the two partials, which is commutative, so both ranks hold the same bits (the same argument `forward_ep_reduce` documents for the EP reduce) | Both ranks produce the same tokens; any divergence would desynchronize EP dispatch and break the 8/8 gates |
| Router rows and ties are deterministic | `dense_rows` (all tiers: 8-row narrow, 16/32-row wide, 32-row tile) gives each row the single-row `dense_gemv_bf16` bytes; the softmax top-k kernels break exact ties by the lower expert id | `moe/router_determinism_gpu_tests.rs` (GB10): row identity at M = 1..=32, 33, 64; prefill GEMM vs decode GEMV max logit difference 1.56e-2 (2^-6; not bit-identical, as the prefill contract allows), top-10 sets equal on every row whose 10th/11th gap exceeds twice that (35 of 64 random rows; 29 were near ties); single-row and multi-row top-k identical on every tie incl. 512 equal logits, prefill's batched top-k identical on boundary ties ({300, 7} -> 7, {450, 20, 333} -> 20, {511, 256} -> 256) but not on a row of 512 equal logits |
| The shared expert counts exactly once | Replicated weights; under EP the pre-reduce blend takes a zero row and `forward_ep_reduce` adds it once after the all-reduce | Launch-recording tests on both EP tails (§3) |
| TP shards are the shipped bytes | Row slices by head and column slices on whole 32-input scale groups; startup refuses head counts not divisible by `tp_size` (`serve_phases/topology.rs`), segments not divisible by `tp_size` (`tp_shard/gdn.rs`) and `(k/tp) % 32 != 0` (`tp_shard/mxfp8.rs`) | Shard byte-exactness and reassembly tests (§3) |

## 4. Results (same pair; tok/s; every row on its own fresh boot)

| | FP4-Local, first resident build | FP4-Local, V5 | NVFP4 pack, same flags |
|---|---|---|---|
| decode-floor (1500 tok) x3 (accept) | 89.0 / 88.9 / 88.7 (4.37) | 84.6 / 84.6 / 84.3 (4.27) | 82.1 / 82.3 / 82.1 (4.13) |
| short chat decode, first token | 67.98, 0.165 s | 63.02, 0.157 s | 60.86, 0.144 s |
| ~18K agent turn decode, first token | 46.27, 5.73 s | 43.05, 5.83 s | 42.07, 6.15 s |
| cold TTFT median (1K / 4K / 16K) | 1493 ms | 1503 ms | 1408 ms |
| total at 1 / 2 / 4 / 5 / 8 users (isl 512, osl 256) | 63.6 / 104.1 / 137.3 / 148.2 / 165.9 | 64.5 / 108.1 / 146.0 / 161.0 / 205.7 | 61.3 / 102.5 / 156.6 / 176.8 / 232.1 |

- The NVFP4-pack column predates the TC-MoE and BF16-router changes.
- Single-stream speed moves with acceptance, which moves with each numerics baseline: the TC MoE
  family, the BF16 router and the prefill router each changed the exact outputs.
- For reference, measured on one Spark with the same streaming client (another box, not an A/B):
  - Atlas FP4-Local: short 25.6 tok/s; agent turn 22.9 tok/s with a 15.4 s first token.
  - NVIDIA's vLLM recipe: short 41.5 tok/s; agent turn 43.6 tok/s with a 9.2 s first token.

## 5. Where the time goes (nsys with CUDA-graph node tracing, rank 0, per decode step)

| | 1 stream (V4, 45.1 ms) | 8 streams (V4, 102.2 ms) | 8 streams, NVFP4 pack (93 ms) |
|---|---|---|---|
| routed MoE (TC v3 units) | 10.9 | 42.6 | 40.9 (shared expert fused) |
| MXFP8 projections | 11.1 | 21.6 | n/a (BF16/NVFP4 tiers ~16.5) |
| mHC (replicated) | 7.4 | 8.8 | 8.6 |
| RDMA all-reduce | 3.0 | 6.3 | 7.1 |

Profile captures must trace graph nodes (`--cuda-graph-trace=node`). At graph level the kernel tables
show only the ~25% of GPU time spent outside graphs.

## 6. What is left (ranked)

1. MXFP8 projections are still ~11 ms/step (1 stream) and ~22 ms/step (8 streams), against a ~7.5 ms
   bandwidth floor. The GEMV tiers are load/latency-bound. Next: cp.async-pipelined staging, plus more
   CTAs per small-n projection.
2. MTP depth / draft policy sweep on the current numerics (measurement only).
3. Head-sharded resident table (each rank holds 8 of the 16 n-gram heads, then an all-gather of
   `[T, 8x160]` before the replicated projections; bit-exact): frees 14.4 GB per rank, so the 0.70
   sidecar setting fits.
4. MXFP8 verify LM head as an opt-in (NVIDIA ships one; lossy, so off by default); shared-expert TP
   split; mHC split.

## 7. Running it

rc16 TP2 profile plus `ATLAS_PLE_RESIDENT=1` on both ranks, the same binary on both, cuBLAS 13.0.2.14
pinned on both. Start rank 1 first:
`spark serve --model-from-path=<snapshot> --kernel-target=qwen3.8-flash-next --world-size=2 --tp-size=2
--ep-size=2 --gpu-memory-utilization=0.88 --kv-cache-dtype=bf16 --enable-prefix-caching
--max-prefill-tokens=16384 --speculative --num-drafts=7 --max-num-seqs=8 --max-seq-len=32768
--rank={0,1} --master-addr=<rank-0 cable IP>`.
Per rank, the boot log shows pre-KV ≈ 85–87 GB, a ~1.1–1.3M-token KV pool and the line
`PLE resident: 320001536 rows, 28.8 GB on device ... self-check PASS`.

### 7a. Switches this PR adds, and rollback

| Variable | Default | Effect | Rollback |
|---|---|---|---|
| `ATLAS_PLE_RESIDENT=1` | off | Holds the whole NVFP4 n-gram table on device on each rank (28.8 GB). Refuses tables that are not the NVFP4 U8 pack. Must match on both ranks (startup parity). `ATLAS_PLE_CACHE_GB`, `ATLAS_PLE_CACHE_SLOTS` and `ATLAS_PLE_NVME_KEEPALIVE_MS` are ignored with a warning while it is on. | Unset it: the NVMe + RAM row cache of PR #195, bit-identical output. |
| `ATLAS_QWEN4EXP_ROUTER_NVFP4=1` | off | Routes on the old load-time NVFP4 requantization of the BF16 router. | This is the rollback for the BF16 router; it changes routing numerics (lossy). |

Everything else on the served path (TC MoE units, multi-row MXFP8 tiers, exact rows, shared expert
once under EP, MXFP8 TP slicing) is unconditional for FP4-Local or keyed by the checkpoint's own
formats, and has no switch. The remaining variables come from the rc16 TP2 profile this PR does not
change.

### 7b. Memory admission with the resident table

The resident table is allocated during model load, before the KV cache is sized, so it is charged
like any weight:

1. `factory/build/kv_budget.rs` sizes the pool from the larger of two own-footprint measures (the
   drop in free memory since context init, and what the process itself allocated by driver
   accounting) and leaves the reserves plus a headroom floor free. The resident load fadvises its
   source files out of the page cache, so free memory falls by much less than Atlas allocated; the
   tracked measure catches that. Boot log, rank 0: `KV own-footprint: Atlas allocated 85.10 GB but
   free memory fell by only 43.86 GB ... sizing from the tracked figure`, then `119.6 GB total x 88%
   util = 105.3 GB budget; 85.1 GB pre-KV + 4.7 GB reserve -> 15.5 GB for KV ... 1271712 max KV
   tokens`.
2. Both ranks agree on the smaller block count before allocating (`agree_kv_blocks`).
3. The scheduler rejects a request larger than the pool instead of admitting it
   (`scheduler/admission.rs`, `reject_oversized`).

At util 0.88 that leaves ~1.27M KV tokens per rank, enough for the 8 x 32K profile several times
over. At util 0.70 (co-tenant setting) the replicated table does not fit; that needs the
head-sharded table (§6). Residency stays opt-in until then.

### 7c. Two-GB10 checklist

- One ConnectX-7 cable, a private IPv4 subnet on the cable ports (`ping` both ways), RoCE v2 GID index
  known per node (`show_gids`); it can move after a power cycle. The measured pair runs the cable
  ports at the default MTU 1500 on both ends; keep the two ends equal.
- Per rank: `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` / `ATLAS_RDMA_RAILS` = that node's cable interface
  and RDMA device, `NCCL_IB_GID_INDEX` = its RoCE v2 IPv4 GID. The rc16 profile sets
  `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_NET=IB NCCL_IB_ROCE_VERSION_NUM=2 NCCL_NVLS_ENABLE=0
  NCCL_CUMEM_ENABLE=0`.
- The same `spark` binary, the same profile and the same opt-ins on both ranks; startup parity refuses
  a mismatch (including `ATLAS_PLE_RESIDENT`). cuBLAS 13.0.2.14 pinned on both.
- `--gpu-memory-utilization 0.88`, `--kv-cache-dtype bf16`, `--max-seq-len 32768`,
  `--max-num-seqs 8`; check the KV pool line on both ranks after boot.
- Nothing else large on either GPU while it loads (unified memory: exhaustion stalls the node rather
  than failing an allocation).
- Start rank 1, then rank 0; ready when `/v1/models` answers and a 2-token completion returns.
