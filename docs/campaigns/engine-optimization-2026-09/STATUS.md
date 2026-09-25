# Engine optimization campaign — status (2026-09-25)

Qwen3.8-Flash-Next (`qwen4_exp`) and Qwen3.8-27B on the DGX Spark GB10 and Strix Halo
(gfx1151). Every number here is **measured** unless marked *projection*, and names the
reiner jobqueue job that produced it. All GB10 numbers: `nvidia/*-NVFP4` packs, one box,
consecutive arms, same binary where the table says so.

**Where the work lives:** the job board (label `job`) and the PRs below. Follow a
workstream by watching its job.

| workstream | job | PRs |
|---|---|---|
| Flash-Next long-context cold prefill | #74 | #68 (draft) |
| Flash-Next multi-turn TTFT (prefix caching) | #69 | — |
| Flash-Next decode (MTP, draft vocab, graphs) | #70 | — |
| GB10 tensor-core GEMM ceiling (shared) | #71 | — |
| 27B DFlash2 concurrency | #58 | design: [`dflash2-batched-propose.md`](../../design/dflash2-batched-propose.md) |
| 27B on Strix Halo: prefill + DP4A verify | #72 | #75 |
| Per-request host-memory growth | #73 | — |
| Client streaming latency | — | #67 (draft) |
| cuBLASLt per-call setup (shared runtime) | #71 | #77 (draft) |
| MTP memory on the nvidia Flash-Next pack | #70 | #78 (draft) |
| This status page | — | #76 (draft) |

---

## Done (measured)

### Flash-Next cold TTFT: 92.2 s → 24.3 s at 31.5k — PR #68

Jobs 239/240/243. `main66` (`5b07b953`) vs the campaign branch (`c8bbd90f8`); serve util
0.88, max-seq 34000, prefill chunk 16384, prefix caching off.

| prompt | main66 | #68 defaults | #68 + QSA TC flags | main66 (repeat) |
|---:|---:|---:|---:|---:|
| ~160 tok | 0.866 s | **0.332 s** | 0.330 s | 0.862 s |
| ~2.6k | 3.98 s | **1.78 s** | 1.68 s | 3.97 s |
| ~10.2k | 19.40 s | **7.39 s** | 6.34 s | 19.29 s |
| 31.5k | 92.24 s | **24.33 s** | 19.92 s | 93.45 s |

- Greedy output main == #68 defaults on the probe prompts; main == main (control).
- Logprobs are **near-identical, not bit-identical**: dense-region ppl +0.036 %, above
  the QSA bound −0.014 % (8 × 12k windows). Kernel attribution queued (job 251).
- What is in it: QSA scorer that replays the reference reduction DAG in one thread
  (bit-identical), head-grouped QSA attention, GPU prefill top-k, skipping the dense
  pass QSA overwrites, mHC skinny GEMMs routed to split-K by machine fill (prefill
  fixed cost 143 → 22 ms), MoE prefill m-tiles strided instead of ~1 M no-op CTAs, the
  MoE transpose orchestrator extended to qwen4_exp, and the `--gdn-fused-norm`
  SiLU-vs-sigmoid correctness fix.
- `ATLAS_GDN_PIPE` made the default on NVIDIA (job 243: ppl 0.000 % above and below the
  bound, needles 12/12, kl_drift top-1 100 %). 27B check queued (job 252).
- `ATLAS_QSA_ATTN_TC2` + `ATLAS_QSA_SCORE_TC` stay **opt-in**: ppl neutral (−0.030 %) but
  needles 11/12 and kl_drift top-1 91.7 % (job 243).

### Client streaming latency: 0.206 s → 0.002 s — PR #67

Job 241. With a leak-marker tool parser (`qwen3_coder`), the stream sanitizer held back
a flat ~18 bytes of every delta. It now holds only a suffix that could still become a
marker: identical bytes, sent sooner. Client-minus-server first-token lag, 15 requests
per arm: main66 0.206 s / #67 0.002 s (repeat 0.205 / 0.002). `spark-server` suite 2368
passed.

### 27B DFlash2 concurrency: root cause found

Job 235: batched verify scales (122 ms at n=1 → 285 ms at n=12) but propose is exactly
41 ms × n, because generic DFlash2 proposes one sequence at a time
(`propose_batch_max() = 1`). Job 242: `ATLAS_DFLASH_PROPOSE_LANES=4` makes it **worse**
(C4/C8/C16 = 25.6 / 31.4 / 29.2 tok/s vs 36.8 / 46.6 / 47.1): each lane re-reads ~4.5 GB
of drafter + head weights. Only a B×γ batched propose fixes this — see the design doc.

### PR hygiene

#46, #48, #53 closed: their content is already on main (patch-equivalent / byte-identical);
the two still-unmerged commits live in #75.

---

## In flight (reiner jobqueue)

| job | what it answers |
|---|---|
| 250 `fnext-mtp-vocab-ab2` | Flash-Next decode: MTP K=2 (per-row QSA verify) with and without draft-head vocab slicing (`--mtp-vocab` 65536 / 131072); serial baseline 18.0-18.9 tok/s |
| 251 `campaign-drift-pin` | pin the +0.036 % dense-ppl drift to the mHC GEMM path (`ATLAS_QWEN4EXP_NO_HC_GEMM=1` arm) |
| 252 `gdn-pipe-27b` | GDN pipe spine on the 27B: ppl + needles + kl_drift vs vfused |
| 253 `dflash-gamma-resweep` | γ 8/12/16 under Option B. The old "γ=8 optimal" sweep and the γ=16 CUDA 700 both predate #33, which fixed the drafter attention reading the neighbouring head at head_dim 128 |
| 255 `cublaslt-plan-cache` | #77: build + clippy + CUTLASS-vs-cuBLASLt GPU tests + 27B greedy byte-identity and TTFT vs main66 |
| 256 `mtp-fp8-reclaim` | #78: pre-KV with `--speculative` (main66 104.6 GB, refuses util 0.88) vs releasing the 512 MTP experts' FP8 sources (~2.5 GB est.); greedy identity at 0.93 |
| 254 `mem-slope-ab2` | per-request host-memory growth, main66 vs #68 (the #68 BFCL run lost ~7-27 MB/request and was stopped at 388/995 before the OOM guard) |

## Corrections recorded

- "#68 defaults are bit-identical" → greedy-identical on the probes; dense ppl +0.036 %.
- Strix Halo does ship the `vfused` GDN spine; an earlier note that it did not was wrong.
- "γ=8 is optimal / γ=16 faults" is unverified on current code (job 253).
