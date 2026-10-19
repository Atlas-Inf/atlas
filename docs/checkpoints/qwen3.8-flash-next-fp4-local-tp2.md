# Qwen3.8-Flash-Next FP4-Local on two GB10s (TP=EP=2): resident n-grams, native formats

The two-Spark continuation of [qwen3.8-flash-next-fp4-local.md](qwen3.8-flash-next-fp4-local.md)
(PR #195). `nvidia/Qwen3.8-Flash-Next-FP4-Local` (snapshot `b9ce18a6887e18c089fdea8509a33e63429a3300`)
served by Atlas on two DGX Sparks joined by one ConnectX-7 cable, tensor and expert parallel
(`--tp-size 2 --ep-size 2`), with the 28.8 GB NVFP4 PLE n-gram table held **resident in device
memory on both ranks**. That is the thing one Spark cannot do: there the table stays on NVMe behind a
RAM row cache (§4 of the single-Spark doc).

**Status.** It serves and is exact where exactness is claimed. Greedy output with speculation equals
serial greedy output (8/8 at 1 and 8 concurrent streams) after every change below. The resident table
is bit-identical to the NVMe-cached path. BFCL golden ST-995 is 87.44 / 87.45 (n=995) on the current
numerics. All numbers are from one pair (two GB10s, driver 580, CUDA 13.0, cuBLAS 13.0.2.14 pinned on
both ranks) with the rc16 TP2 profile: util 0.88, 8 sequences x 32K, MTP 7 drafts, exact verify,
prefix cache, `reasoning_effort` low. Single runs unless repetitions are listed.

## 1. What changed (on top of PR #195)

| Change | Why |
|---|---|
| MXFP8 tensor parallelism by byte slicing | GDN `in_proj_qkv`/`in_proj_z` (by q/k/v segment and value head) and attention q/k/v (by head) are sliced by rows. `o_proj` and GDN `out_proj` are sliced by columns on whole 32-input E8M0 groups (`(k/tp) % 32 == 0` is enforced). No dequant or requant; a shard is byte-identical to the shipped rows/columns. |
| Shared expert added once under EP | The MXFP8 shared expert stays replicated. Under EP it is blended as a zero row before the all-reduce and added once after it. Two batched decode paths blended it before the reduce, counting it once per rank. |
| Resident PLE table, `ATLAS_PLE_RESIDENT=1` | The 128 NVFP4 shards are streamed through pinned staging into one rows buffer and one block-scale buffer per rank, then the files are fadvise(DONTNEED)'d. A boot self-check compares sampled rows (first, last, each head's first and last row, 64 pseudo-random) against the host oracle. The gather uses the existing GPU decode with slot = row id. No row cache, keepalive or fault pool. Startup parity refuses a rank mismatch. |
| BF16 router (dtype fix) | `mlp.gate` ships BF16 (not in `quantized_layers`), but Atlas requantized it to NVFP4 at load and routed on that. FP4-Local now routes on the shipped weight: decode/verify on row-invariant GEMV tiers (scoped to these layers by a load-time flag), prefill on the pipelined BF16 GEMM. `ATLAS_QWEN4EXP_ROUTER_NVFP4=1` restores the old path for A/B. |
| TC v3 MoE units without a fused shared expert | The plan kernel takes `has_shared`. FP4-Local runs the tensor-core MoE units with its MXFP8 shared expert as a separate chain (previously it fell back to the slower rows path). |
| MXFP8 multi-row projections | 2..=32 rows are served in one weight pass (bucketed GEMV, plus a register-tiled 17..=32 tier), every row bit-identical to the 1-row kernel. Before, verify windows re-streamed each weight once per 8 rows. |
| Opt-in tensor-core MXFP8 tier, `ATLAS_QWEN4EXP_MXFP8_TC=1` | Row-invariant and within 1 BF16 ULP of an FP64 reference, but slower than the GEMV tiers today (§4). Off by default; not recommended yet. |

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
  negative control; resident vs cached gather parity on interleaved shards incl. the last row; the
  multi-row MXFP8 tiers byte-identical to the 1-row kernel at M = 1..=32 on the real shapes with
  hostile bytes; TC tier row invariance + one-hot leak map + strict FP64 reference.
- BFCL ST-995 golden draw: 87.64 / 87.71 (`run-1792365985129633322`, first resident build, NVFP4
  router) and 87.44 / 87.45 (`run-1792406664787105023`, BF16 router + TC MoE). The difference is
  within run-to-run noise.

## 4. Results (same pair; tok/s; every row on its own fresh boot)

| | FP4-Local, first resident build | FP4-Local, current (V5) | V5 + `MXFP8_TC=1` | NVFP4 pack, same flags |
|---|---|---|---|---|
| decode-floor (1500 tok) x3 (accept) | 89.0 / 88.9 / 88.7 (4.37) | 84.6 / 84.6 / 84.3 (4.27) | 57.8 / 57.7 / 57.7 (4.19) | 82.1 / 82.3 / 82.1 (4.13) |
| short chat decode, first token | 67.98, 0.165 s | 63.02, 0.157 s | 43.54, 0.161 s | 60.86, 0.144 s |
| ~18K agent turn decode, first token | 46.27, 5.73 s | 43.05, 5.83 s | 27.68, 5.86 s | 42.07, 6.15 s |
| cold TTFT median (1K / 4K / 16K) | 1493 ms | 1503 ms | 1506 ms | 1408 ms |
| total at 1 / 2 / 4 / 5 / 8 users (isl 512, osl 256) | 63.6 / 104.1 / 137.3 / 148.2 / 165.9 | 64.5 / 108.1 / 146.0 / 161.0 / 205.7 | 40.5 / 73.3 / 118.9 / 138.3 / 182.4 | 61.3 / 102.5 / 156.6 / 176.8 / 232.1 |

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
   bandwidth floor. The GEMV tiers are load/latency-bound and the first TC kernel is under-occupied on
   small-n shapes. Next: cp.async-pipelined staging, plus more CTAs per small-n projection.
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
Per rank, the boot log shows pre-KV ≈ 85–87 GB, a ~1.1M-token KV pool and the line
`PLE resident: 320001536 rows, 28.8 GB on device ... self-check PASS`.
