# Engine optimization campaign — status (2026-09-26)

Qwen3.8-Flash-Next (`qwen4_exp`) and Qwen3.8-27B on the DGX Spark GB10 and Strix Halo
(gfx1151). Every number here is **measured** unless marked *projection*, and names the
reiner jobqueue job that produced it. All GB10 numbers: `nvidia/*-NVFP4` packs, one box,
consecutive arms, same binary where the table says so.

**Where the work lives:** the job board (label `job`) and the PRs below. Follow a
workstream by watching its job.

| workstream | job | PRs |
|---|---|---|
| Flash-Next long-context cold prefill | #74 | #68 (**merged**) |
| Flash-Next multi-turn TTFT (prefix caching) | #69 | — (warm path measured: 98 % hits, 16-token recompute in the perf leg) |
| Flash-Next decode (MTP, draft vocab, graphs) | #70 | #84 (draft: `VERIFY_ACTIVE` default ON; A/B queued) |
| GB10 tensor-core GEMM ceiling (shared) | #71 | #83 (**merged**, the lazy-BF16 OOM hazard); the ceiling itself is open |
| 27B DFlash2 concurrency | #58 | #86 (batched B×γ propose, default ON; gate run queued) |
| 27B on Strix Halo: prefill + DP4A verify | #72 | #75 |
| Per-request host-memory growth | #73 (closed) | #82 (**merged**), #92 (exact dedupe follow-up) |
| Client streaming latency | — | #67 (**merged**) |
| cuBLASLt per-call setup (shared runtime) | #71 | #77 (closed — correct, no gain) |
| MTP memory on the nvidia Flash-Next pack | #70 | #78 (**merged**) |
| GB10 boot kernel gate: the fleet refused Strix-only lookups | #87 (closed) | #89 (**merged**) |
| Qwen3.6-35B-A3B NVFP4 current revision did not load | #88 (closed) | #90 (**merged**) |
| This status page | — | #76 (draft) |

**Per-box campaigns** (what each machine is running right now, and where it reports):

| box | hardware | campaign job(s) | running now |
|---|---|---|---|
| reiner | DGX Spark GB10, 119.6 GB | post-merge: #86 concurrency gate, 27B DFlash2 perf leg, #84 A/B, prefill chunking investigation | jobs 362-368 (below) |
| strix | Strix Halo Linux gfx1151 | #72 (27B prefill/DP4A), #56 (Flash-Next Linux) | idle: next is `w4a16_gemm_t_m128` tile analysis (see atlas-audit 11_HANDOFF_WINDOWS_STRIX_2026-09-25.md) |
| winbox | Strix Halo Windows gfx1151, 128 GB | #57 (Flash-Next Windows port) | building `port/fnext-win-r3` (see atlas-audit 11_HANDOFF_WINDOWS_STRIX_2026-09-25.md) |

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
  top-1 46.9 %, 190 greedy flips — attribution below.
- What is in it: QSA scorer that replays the reference reduction DAG in one thread
  (bit-identical), head-grouped QSA attention, GPU prefill top-k, skipping the dense
  pass QSA overwrites, mHC skinny GEMMs routed to split-K by machine fill (prefill
  fixed cost 143 → 22 ms), MoE prefill m-tiles strided instead of ~1 M no-op CTAs, the
  MoE transpose orchestrator extended to qwen4_exp, and the `--gdn-fused-norm`
  SiLU-vs-sigmoid correctness fix.
- `ATLAS_GDN_PIPE` made the default on NVIDIA (job 243: ppl 0.000 % above and below the
  bound, needles 12/12, kl_drift top-1 100 %) and on the 27B (job 252: ppl 0.000 %, JSD 0,
  top-1 100 %, both spines confirmed from the serve log).

### #68: real bug found — MoE prefill grid — fixed in `1ea4e20ca` (jobs 267/269)

`9ca70af11` shortened `max_m_tiles` to 2× the average expert and fed it to EVERY MoE
prefill launch; only the two `_t_k64` kernels stride `blockIdx.y`, so the others silently
dropped hot experts' rows past the short grid. Job 267 (`ATLAS_MOE_TRANSPOSE=0`): dense
ppl +14.9 %, above-bound +174 % vs main66. The fix hands `grid_m_strided` only to the
striding launches. Job 269 (8 × 12k ppl windows, lib.sh serve profile): the fix is identical to
the pre-fix head on greedy probes AND ppl (dense 7.71124 / above 4.38496 both arms);
transposed-MoE alone reads +0.74 % dense vs untransposed — the open precision question,
tracked on #74, not a merge blocker.

### The +0.036 % dense drift is two cancelling effects (jobs 259/264)

Per-commit bisect vs main66: the MoE transpose (`cfc6dae8e`, on all 48 layers) moves dense
ppl **+0.52 %**; the mHC collapse routing (`379d741e0`) moves it about **−0.48 %** —
plausibly precision recovered, since the tensor-core hc GEMM path rounds `normed` to BF16
(not yet isolated). Net +0.036 % hid the MoE half — the
job 267 arm above is what exposed it. The transposed-vs-untransposed numerics question is
open on #74.

### `ATLAS_QSA_ATTN_TC2` is the NVIDIA default (jobs 258/263/272)

Flag split on #68's defaults (job 258): TC2 alone ppl 0.000 % dense / −0.019 % above-bound,
needles 12/12; `SCORE_TC` alone +0.222 % above-bound — it moves QSA selection and costs
quality for ~0.8 s, so it stays opt-in. TC2 TTFT (job 263): 31.5k **24.36 → 20.97 s**
(−13.8 %), 10.2k **7.42 → 6.51 s**. `qsa_attn_tc2_enabled()` flips the default on NVIDIA
(`ATLAS_QSA_ATTN_TC2=0` restores the grouped kernel; gfx1151 stays off); a once-per-process
INFO logs the dispatched variant. Job 272 on the merge-train tree: defaults reproduce the
job-258 TC2 arm exactly (0/8 windows differ, log says `tc2`); `TC2=0` reproduces the grouped
kernel bit-for-bit.

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

### γ re-sweep on current code (job 260)

The old verdicts ("γ=8 optimal", a γ=16 CUDA 700) predated the #33 head-dim fix. On main66,
γ=16 no longer faults; γ=12 wins code (+27 %) and JSON; γ=8 wins prose (+5 %) and long code
(+11 %); γ=16 never beats γ=12. No single γ wins — next item is per-sequence adaptive γ
from running mean-accepted, which fits the batched-propose design as a mask.

### Flash-Next decode: MTP past the QSA bound = +56 % (job 250); quality gate passed (job 257)

MTP K=2 with per-row QSA verify (`ATLAS_QSA_VERIFY_ACTIVE`, #51): code 29.6, JSON 29.7,
prose 25.1 tok/s; code at 5.1k context 18.4 → **28.7 tok/s**, accept 0.94-0.98 on
code/JSON. `--mtp-vocab` slicing: no gain at K=2 (dropped). Job 257 (M vs serial, M2
repeat): needles 10/12 = 10/12, short probes byte-identical, M-vs-M2 deterministic
(top-1 100 %, JSD 0) — but NOT output-identical to serial on long generations (top-1
43.7 %), the same class as existing short-context MTP divergence: the verify pass scores
K+1 rows in a different reduction shape and QSA's chaotic top-k amplifies it. Safe to
offer; default-on is the product call — now PR #84, gated by jobs 280 vs 282 (ST-995
off/on) and 281 vs 283 (perf off/on).

### MTP memory: 2.3 GB back, MTP boots at util 0.90 — PR #78 (job 256)

Pre-KV 103.4 → 101.1 GB with `--speculative`; KV at 0.93 3.1 → 5.4 GB; main66 cannot boot
at 0.90, #78 can; greedy byte-identical.

### Host memory: two real leaks found, both fixed in #82 (jobs 254/265/268/271/273)

- Thinking-off harness (jobs 254, 265, 268): ~1.4-4.2 MB/req on main66 and #68 alike —
  understated the problem (see corrections).
- Recipe profile (job 271: thinking on, MTP K=2, prefix caching, util 0.90, BFCL traffic):
  server RssAnon **+20.6 (main66) / +24.6 (#68) / +28.0 (main66 no MTP) MB per request**;
  client flat.
- jemalloc heap profile (job 273, T1 binary): the growth survives jemalloc too (not glibc
  fragmentation). Attribution: **51 %** `kernel_audit::record` ← per-step
  `GpuBackend::kernel` calls from #68's mHC split dispatcher (audit grew without bound);
  **32 %** grammar compile, half of it a ~20 MB `TokenizerInfo` deep copy per compiled
  grammar (invisible to the LRU budget); 11 % QSA snapshot aux and 6 % grammar masks, both
  bounded.
- Fix #82: `TokenizerInfo` is an Arc handle (C++ shared_ptr semantics) and the audit keeps
  one row per `(module, func, loaded)`. Validation: job 274 arm V.

### Lazy FP8→BF16 copies outside the KV budget — #83 (jobs 262/266)

`ATLAS_CUBLAS_GEMM=1` dequantises FP8 projections to BF16 lazily on first prefill, after
the KV pool is sized — on the 27B at util 0.88 that took the box to **2.1 GB MemAvailable**
(job 262). #77 (the plan cache this was being validated for) is correct but gains nothing:
the plan key contains m, and prefill m is the prompt length, so it never hits — closed,
byte-identical in 4/4 (job 266). #83 reserves the superset footprint (`numel × 2` over
resident FP8 matrices, routed experts excluded) inside KV sizing when a cuBLAS/CUTLASS/rowwise
flag is set; flags off → byte-identical budgeting. Validation: job 274 arm C71.

### Strix Halo 27B prefill: default is best; one GEMM is the whole lever

Stage-6 matrix on gfx1151 (main66, nvidia 27B): default 233 / 202 / 139.8 tok/s at
1.9k / 7.2k / 29k; `BF16_TC_PREFILL`, `FP8_M64_PREFILL`, `NO_FFN_NVFP4_MMQ` all equal or
slower, outputs identical. Chunk 2048 vs 8192 at 29k is no lever (144.6/138.5 vs
139.6/137.2). `w4a16_gemm_t_m128` is 69.6 % of GPU time at ~27 % of WMMA peak; attention
7 %, GDN ≤ 4 % each. Details on #72.

### Flash-Next on Strix Halo Windows (winbox, `fnext-win-r2`)

Serial decode ~9 tok/s, MTP K=2 17.5-19.4 (≈2x) on short prompts; no MTP gain at 4.7k
because that branch predates #51. Prefill ~70-75 tok/s vs Gufo's 1,628 on the same silicon
(~22x): the Windows priority. Details on #57.

### PR hygiene

#46, #48, #53 closed: their content is already on main (patch-equivalent / byte-identical);
the two still-unmerged commits live in #75. #77 closed (correct, no gain — above).

---

## GB10 merge train: merged (2026-09-26)

The train was #78, #67, #68, #82 and #83, followed by #89 and #90; main is at `1ab38536b`. The maintainer's merge gate was ST-995 plus the 2.5 h agentic perf leg on the combined tree. The main tree is the one the gates and the support-matrix check ran on: it's identical to the job-346 build `f8a04db1f`, and differs from the gated tree `53f4a3d3a` only by TOML `[expected_absent]` declarations.

| job | result |
|---|---|
| 274 `train2-validate` | **PASS.** All-targets build and GB10 unit tests (xgrammar 824 + 2, kernel_audit 7, lazy_bf16 + tc2 5). The recipe memory slope fell from +24.6 to +1.86 MB/request. The 27B cuBLAS boot at util 0.88 keeps 17.4 GB MemAvailable, where it previously left 2.1 GB and was killed. |
| 279 `warmturn-t2` | #69 baseline: warm turns take 1.7–1.8 s at 9–11k tokens; turn 1 (cold) takes 4.8 s. |
| 280 `st995-t2` | **PASS: 84.72 / 84.68** against bars of 82.72 / 81.65, with TC2 engaged. RssAnon stayed flat at ~1.76 GB for all 995 requests. The harness is deterministic per tree: repeat runs give identical scores. Earlier Flash-Next gate runs on other trees scored 85.73–86.03. Per-item responses are kept, so later runs can be paired (McNemar, `bfcl_paired.py`). |
| 298 `perf-serve-t2` | **VALID.** Endpoints `agentic-coding-perf-2.5h`: 1007 of 1007 turns, 0 failed, 0 missing, 89 min. TTFT median 1.16 s / p99 2.76 s, TPOT median 63 ms, score 0.4899. Warm prefix hits were 98 % with a mean recompute of 16 tokens; 0 KV-exhausted, 0 errors. The compliance checker's `all_turns_observed` fails on this dataset whatever the server does: it compares `observed` (capped at the 1006 scorable turns) with `issued` (1007). |
| 297 / 306 / 324 / 346 `moe-matrix-check` | #68 made the MoE transpose run on every MoE model, where it used to be allowlisted. These runs found:<br>• #87: the boot kernel gate refused Nemotron-3.5-Lightning and Qwen3.6-35B-A3B over Strix-only lookups.<br>• #88: the current Qwen3.6 NVFP4 revision labels its layers `W4A16_NVFP4` and was misread as FP8.<br>• A #68 regression: an undeclared marker probe.<br>After the fixes, job 346 shows all three MoE models booting with 0 unresolved lookups. Greedy output is identical to main66 + fixes, and 10k TTFT is equal or 2–3 % faster; Qwen3.6 now takes the HYBRID transpose tier. |

## #58: 27B DFlash2 batched B×γ propose (PR #86)

- **Parity (job 316, `cffb28553`):** the staged drafter backbone is **byte-identical** to serial (max|Δ| 0), and 94.8 % of draft tokens are equal. An earlier NVFP4-vs-BF16 staged-weight mismatch was fixed in `cffb28553`.
- **Throughput:** probe-harness aggregate at C=4/8/16 went from 48.8 / 55.4 / 59.5 tok/s serial to **57.3 / 68.5 / 84.2** batched (+17 / +24 / +42 %). Acceptance is equal and there were 0 errors. C=1 is untouched (floor 2).
- **Cross-sequence wy8 verify:** 60 launches become 2 per GDN layer, byte-identical, with no measurable throughput change.
- **Default on** since `8bf0539b7` (rollback `ATLAS_DFLASH_BATCHED_PROPOSE=0`); the per-sequence fallback is total.
- **Gate:** job 362 runs the repo's `concurrency-sweep` with batched propose OFF vs ON (floors 16.2 / 33.5 / 50.5 / 72). Adaptive-γ feasibility is posted on #58, and wyN now covers K=9..16.

## Prefill is chunking-sensitive on the 27B (under investigation)

- The "DFlash2 + prefix caching diverges" hazard is **not** a cache bug and not DFlash2-specific.
  - Plain decoding diverges the same way (job 317).
  - Prefix caching enabled with no cache hit is identical to a warm hit, bit for bit.
  - The trigger is that enabling caching splits prefill at token 32 to save a checkpoint.
- **Token-0 logits move by up to ~14 nats with the prefill chunking alone** (job 338):
  - single chunk vs 16-token chunks vs split@32;
  - the same happens with the independent WY4 GDN prefill.
- **Per-op dumps (job 354):** the residual stream drifts steadily between a single 53-token chunk and 16-token chunks, from cos 0.9999 at the first attention layer to 0.911 at the last. The dense FFN output is already at cos 0.992 in layer 0. The suspect is M-dependent kernel selection with different activation precision; an audit is in progress.
- Until this is resolved, output differences between chunkings, including prefix-cache splits, should not be read as a cache or speculation defect.

**Queued on reiner, in order:**
- 362: #86 concurrency gate.
- 363: first 27B DFlash2 agentic perf leg.
- 364 / 365: #84 verify-active A/B (ST paired against job 280, plus the perf leg).
- 366: #69 at ~27k tokens.
- 367: γ 8/12/16 on the wyN K=9..16 kernels.
- 368: TC2-default A/B (`ATLAS_QSA_ATTN_TC2=0`, paired against job 280).

## Corrections recorded

- Job 255 was reported as #77 validation; it exercised none of #77's code (jobs 262/266
  redid it properly).
- "#68 does not regress memory" (job 254) held only on the thinking-off harness. Under the
  recipe profile (job 271) the server grows +20-28 MB/request, and #68's mHC split
  dispatcher looking kernels up every decode step made the unbounded kernel audit the top
  consumer (51 % of job 273's growth). Fixed generically in #82.
- Job 270 was cancelled: its lib.sh profile passes `--disable-thinking`, a documented
  BFCL accuracy cliff on this checkpoint (76.95 vs 81.74, n=334), so it could not compare
  against job 225's recipe-profile numbers (85.73/85.96).
- "#68 defaults are bit-identical" → greedy-identical on 3 short probes only; dense ppl
  +0.036 % (now attributed, above); long greedy generations diverge (top-1 46.9 %).
- Strix Halo does ship the `vfused` GDN spine; an earlier note that it did not was wrong.
- "γ=8 is optimal / γ=16 faults" is superseded by job 260 (above).
