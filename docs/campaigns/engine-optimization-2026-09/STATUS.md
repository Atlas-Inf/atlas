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
| MTP memory on the nvidia Flash-Next pack | #70 | #78 (ready) |
| This status page | — | #76 (draft) |

**Per-box campaigns** (what each machine is running right now, and where it reports):

| box | hardware | campaign job(s) | running now |
|---|---|---|---|
| reiner | DGX Spark GB10, 119.6 GB | #58, #68/#74, #69, #70, #71, #73 | jobqueue 254-260 (below) |
| strix | Strix Halo Linux gfx1151 | #72 (27B prefill/DP4A), #56 (Flash-Next Linux) | idle: next is `w4a16_gemm_t_m128` tile analysis |
| winbox | Strix Halo Windows gfx1151, 128 GB | #57 (Flash-Next Windows port) | building `port/fnext-win-r3` (#54 rebased on main: brings #51 verify-active) |

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

- Greedy output main == #68 defaults on 3 short probe prompts; main == main (control).
  **But long generations diverge** (job 251, `lc_subset` 12k-24k prompts): kl_drift
  top-1 46.9 %, 190 greedy flips — see the corrections below; build bisect is job 259.
- Logprobs are **near-identical, not bit-identical**: dense-region ppl +0.036 %, above
  the QSA bound −0.014 % (8 × 12k windows). Kernel attribution: `NO_HC_GEMM` is not a clean control (+0.346 %); build bisect queued (job 259).
- What is in it: QSA scorer that replays the reference reduction DAG in one thread
  (bit-identical), head-grouped QSA attention, GPU prefill top-k, skipping the dense
  pass QSA overwrites, mHC skinny GEMMs routed to split-K by machine fill (prefill
  fixed cost 143 → 22 ms), MoE prefill m-tiles strided instead of ~1 M no-op CTAs, the
  MoE transpose orchestrator extended to qwen4_exp, and the `--gdn-fused-norm`
  SiLU-vs-sigmoid correctness fix.
- `ATLAS_GDN_PIPE` made the default on NVIDIA (job 243: ppl 0.000 % above and below the
  bound, needles 12/12, kl_drift top-1 100 %) and on the 27B (job 252: ppl 0.000 %, JSD 0,
  top-1 100 %, both spines confirmed from the serve log).
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

### Flash-Next decode: MTP past the QSA bound = +56 % (job 250)

MTP K=2 with per-row QSA verify (`ATLAS_QSA_VERIFY_ACTIVE=1`, #51, opt-in): code 29.6,
JSON 29.7, prose 25.1 tok/s; code at 5.1k context 18.4 → **28.7 tok/s**, accept 0.94-0.98
on code/JSON. `--mtp-vocab` slicing: no gain at K=2 (dropped). Prose under MTP is not
run-to-run deterministic (quality + determinism gate: job 257). Details on #70.

### Strix Halo 27B prefill: default is best; gap to pwilkin 1.8x (was 2.4x)

Stage-6 matrix on gfx1151 (main66, nvidia 27B): default 233 / 202 / 139.8 tok/s at
1.9k / 7.2k / 29k; `BF16_TC_PREFILL`, `FP8_M64_PREFILL`, `NO_FFN_NVFP4_MMQ` all equal or
slower, outputs identical. The long-context loss is not the GEMM tier (every arm ~140
at 29k). Kernel ranking + chunk-size knob running. Details on #72.

### MTP memory: 2.3 GB back, MTP boots at util 0.90 — PR #78 (job 256)

Pre-KV 103.4 → 101.1 GB with `--speculative`; KV at 0.93 3.1 → 5.4 GB; main66 cannot boot
at 0.90, #78 can; greedy byte-identical.

### Host memory: #68 is not a regression (job 254)

1.41 MB/req (main66) vs 1.71 MB/req (#68) over 300 tool requests after warm-up. The steady
leak is pre-existing (#73). #68 BFCL re-runs behind a watchdog (job 261).

### Strix Halo 27B prefill: one GEMM is the whole lever

Chunk 2048 vs 8192 at 29k: 144.6/138.5 vs 139.6/137.2 tok/s (no lever). fp8to4 pack at
7.2k: 212.4 tok/s under the profiler; `w4a16_gemm_t_m128` is 69.6 % of GPU time at ~27 % of
WMMA peak. Attention 7 %, GDN ≤ 4 % each.

### Flash-Next on Strix Halo Windows (winbox, `fnext-win-r2`)

Serial decode ~9 tok/s, MTP K=2 17.5-19.4 (≈2x) on short prompts; no MTP gain at 4.7k
because that branch predates #51. Prefill ~70-75 tok/s vs Gufo's 1,628 on the same silicon
(~22x): the Windows priority. Details on #57.

### PR hygiene

#46, #48, #53 closed: their content is already on main (patch-equivalent / byte-identical);
the two still-unmerged commits live in #75.

---

## In flight (reiner jobqueue)

| job | what it answers |
|---|---|
| ~~250~~ `fnext-mtp-vocab-ab2` (done, above) | Flash-Next decode: MTP K=2 (per-row QSA verify) with and without draft-head vocab slicing (`--mtp-vocab` 65536 / 131072); serial baseline 18.0-18.9 tok/s |
| ~~251~~ `campaign-drift-pin` | done: long-generation divergence found; `NO_HC_GEMM` not a clean control |
| 259 `drift-bisect` | build #68 at `cfc6dae8e` (pre-mHC) and `379d741e0` (post-mHC), dense ppl vs main + main-vs-main lc control |
| 258 `qsa-tc-split` | `QSA_ATTN_TC2` alone vs `QSA_SCORE_TC` alone (which one loses the needle) |
| ~~252~~ `gdn-pipe-27b` | done: output-identical (above) |
| 260 `dflash-gamma-resweep2` (253 cancelled: hung on a bare `wait`) | γ 8/12/16 under Option B. The old "γ=8 optimal" sweep and the γ=16 CUDA 700 both predate #33, which fixed the drafter attention reading the neighbouring head at head_dim 128 |
| ~~255~~ `cublaslt-plan-cache` | **did not exercise #77** (tests need CUTLASS_HOME; no arm set a cuBLASLt routing flag) |
| 262 `cublaslt-plan-cache2` | #77 redo: find a flag that routes the 27B through cuBLASLt BF16 (proved by per-plan debug lines), then byte-identity + TTFT |
| 257 `mtp-verify-quality` | MTP + verify-active vs serial: needles + kl_drift; determinism repeat |
| ~~256~~ `mtp-fp8-reclaim` (done, above) | #78: pre-KV with `--speculative` (main66 104.6 GB, refuses util 0.88) vs releasing the 512 MTP experts' FP8 sources (~2.5 GB est.); greedy identity at 0.93 |
| ~~254~~ `mem-slope-ab2` (done, above) | per-request host-memory growth, main66 vs #68 (the #68 BFCL run lost ~7-27 MB/request and was stopped at 388/995 before the OOM guard) |
| 261 `campaign-fnext-bfcl-wd` | #68 BFCL (n=995) with a 2.5 GB MemAvailable watchdog + per-minute memory trace |

## Corrections recorded

- Job 255 was reported as #77 validation; it exercised none of #77's code (see job 262).

- "#68 defaults are bit-identical" → greedy-identical on 3 short probes only; dense ppl +0.036 %;
  long greedy generations diverge (top-1 46.9 %). Source being bisected (job 259).
- Strix Halo does ship the `vfused` GDN spine; an earlier note that it did not was wrong.
- "γ=8 is optimal / γ=16 faults" is unverified on current code (job 260).
