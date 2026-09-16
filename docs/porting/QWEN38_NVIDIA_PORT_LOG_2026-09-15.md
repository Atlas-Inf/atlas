# Qwen3.8 on Atlas, nvidia checkpoints — port log 2026-09-15

Two ports, one box (`gx10-e3a3`, GB10, 119.6 GB unified, driver 580.126.09):

* **A. Qwen3.8-27B dense** — `nvidia/Qwen3.8-27B-NVFP4` @ `dbb8f445` + DFlash2
  drafter `incoai/Qwen3.8-27B-DFlash2` @ `dedf8df6`. Tree: `review/pr21-fixes`
  (PR #21), `463098183` → **`a61f02634`** today.
* **B. Qwen3.8-Flash-Next** — `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54`.
  Tree: `feat/flashnext-nvidia` @ `9c1ca517` (off `review/pr23-fixes`, PR #23).

Companion: `QWEN38_PORT_GAP_ANALYSIS_2026-09-13.md` (§A–§J) carries the history
this log continues from. Every number here is an *observation under the
fingerprint named next to it* (measurement-discipline rules 1/5/10); run-record
ids are `~/.atlas/runs/<bench>/run-*.json` on the box, job artifacts are
`~/jobqueue/{done,failed}/<id>-<name>/`.

Convention for the queue ids below: `NNN` = `~/jobqueue/*/NNN-<name>`.

---

## 0. Headline (morning of 09-15)

| | Dense 27B + DFlash2 | Flash-Next |
|---|---|---|
| Blocking defect found | **γ=16 + Option B + second sequence → CUDA 700** (bisected, 040; memcheck 044 pending). Prefix-caching hypothesis **refuted** (038/039 died with pcache OFF). | **Host OOM = unified-memory pledge**, not a leak we can see yet: at util 0.93 the box sat at 870 MB `MemAvailable` (record `run-1789451775373797695`); at 0.95 the OOM killer fired (024). |
| Fix landed | `a61f02634` — `--dflash-gamma` resolves flag → MODEL.toml `[dflash].gamma` (=8) → 16. Gate-driven `--dflash` serves stop running γ=16. | Serve profile corrected to util 0.88 / seq 32768 / prefill 16384 (037 also died on `Prompt too long: 16752 > 16384` at turn 25). |
| Correctness read so far | ST-995 **PASS 87.94 / 88.24** with thinking ON (022, MTP lane). DFlash+OB temp-0 outputs: 3/8 (033) and 1/5 (041) byte-identical to serial; every first divergence sits on an **exact BF16 tie** (top1−top2 = 0.0 nats). | ST-995 **83.52 / 82.45** (023, Info — no committed floor yet). |
| Perf read so far | Option B ~1.45× legacy at bs1 (§G.6); **loses at C8** (034: 27 vs 65 agg tok/s, gate shed spec). | 037: 25 agentic turns clean before the 16K wall. |
| Queued tonight | conc-sweep gate + agentic gate + ST-995 (thinking ON) all on `a61f02634` w/ DFlash2+OB γ=8; KL v2; longctx v3; memcheck. | agentic3 (running), KL serial-vs-n-gram, ST-995 rerun at 0.88/32K + n-gram, C1/C4 probe. |

**"ST-996"**: there is no ST-996 anywhere in the tree or the recipes — the golden
BFCL draw is **ST-995** (n=995, seed 42, 62/10/10, floor 25). Everything below
that says ST-995 is what was asked for as "st-996" unless told otherwise.

---

## A. Dense 27B — nvidia + DFlash2

### A.1 Queue triage (what the overnight 09-14→15 queue actually produced)

| id | leg | result | what it means |
|---|---|---|---|
| 022 | bfcl-subset, MTP, `disable_thinking=false` + effort medium | **PASS 87.94 / 88.24** (`run-1789419056919283587`, 14924 s) | the recipe's `disable_thinking` pin was the entire 76.6 mechanism (§I) |
| 025 | Option-B parity ablations (γ=8) | `ob_fullpre` mean_na 3.0–3.7, ~18–21.5 tok/s; `ob_noctx` mean_na 0.054, 4.9 tok/s | the 5.5→3.9 mean_na dip vs legacy is inherent to OB's ctx construction, not an incremental-precompute bug |
| 026 | γ sweep under OB {6,8,10,12,16} | γ=8 **22.1–22.5**; γ=6 18.8–21.7; γ=10 19.0–19.9; γ=12 17.8–18.4; **γ=16: req0 17.8 then req1 CUDA 700** | γ=8 is the optimum; γ=16 is the crash |
| 027/028/030/031 | bfcl / agentic, gate serve, dflash + OB (pcache ON) | exit 70, CUDA 700 at the first propose of the *second* sequence | gate serves run the clap default γ=16 |
| 032 | teardown repro (OB, pcache OFF, γ=8) | 3/3 clean | — |
| 033 | 8-prompt sha parity serial vs OB (4 code / 4 prose, temp 0, penalties 0) | **4/8 identical** (c1_minheap, c2_rust, c3_lru, p4_kb); 4 diverge at a single word (`-type f -name` ↔ `-name -type f`, `resent`↔`re-sent`, …), all coherent; mean_na 3.98 → 1.00 across the batch as the gate shed spec | tie-break class divergence; prose acceptance collapse confirmed |
| 034 | C8 MinHeap, serial vs OB, bs8, auto gate | serial **65.15** agg / OB **27.03** agg (8 × ~3.4 tok/s); gate shed spec (`serial=0.69`), `DFLASH WIDTH n_active=8 verify=8 dspark_batch_ok=true` | DFlash2 loses at width even after shedding — the batched verify path is the cost |
| 035 | ~10K longctx | **no data**: python `%`-format bug → empty prompt | v2 fixed the bug but tokenized to 27,254 > 16384; **v3 (50 reps ≈ 9.7K) queued** |
| 036 | pcache 700 repro | exit 1 before serving: wrong `job_lib.sh` path + unbound `OUT` | superseded by 040 (premise refuted anyway) |
| 038/039 | bfcl / agentic, gate serve, dflash + OB, **pcache OFF** | exit 70, **same** CUDA 700 | prefix caching is **not** the trigger |
| 040 | γ=16 bisect (6 legs, §A.2) | exit 61 | γ=16 second-sequence fault, 5/5 variants; γ=15 clean |
| 041 | KL/coherence serial vs OB γ=8 | FAIL by gate thresholds — but confounded (§A.4) | v2 with penalties zeroed queued |
| 044 | compute-sanitizer memcheck of the γ=16 fault | **PENDING** | names the kernel |

Fingerprint correction for §G/§G.6 of the gap analysis: those legs passed
`--lm-head-dtype bf16`, and on the nvidia pack the flag is **ignored** —
`lm_head_setup` warns "this checkpoint ships lm_head pre-packed as NVFP4 (no
BF16 tensor exists)". Every nvidia-27B number in this campaign ran the packed
**NVFP4 head** with the padded-stride `w4a16_gemm_t` twin (248192 ≠ 248077).

### A.2 The CUDA 700 — from "prefix caching" to "γ=16, second sequence"

Signature (identical in 026-g16, 027, 028, 030, 031, 038, 039, 040×5):

```
Prefilled (single chunk): seq_len=51, remaining=255
DFlash Option B: allocated 258 blocks (4128 slots) for drafter paged cache
DFlash piecewise capture: starting (key=DflashGraphIdentity { owner: SequenceGeneration { slot: 0, generation: 2 }, ...
DFlash piecewise capture: complete ... (11/11 subgraphs captured)
WARN DFlash forward_block failed, falling back to no-spec: cuStreamSynchronize after D2H on_stream failed: CUDA_ERROR_ILLEGAL_ADDRESS (700)
ERROR copy_d2d_async failed (status 700) at: ... impl_a3_embed::embed_ctx ... decode_dispatch ... step_mtp
ERROR cuMemsetD8Async failed (status 700) ... the CUDA context is destroyed
```

Bisect 040 (serve_dflash profile: 8K seq, util 0.78, bf16 KV, pcache OFF,
`ATLAS_DFLASH_OPTION_B=1`, 3 MinHeap requests, temp 0, penalties 0, 256 out):

| leg | γ | extra | requests ok | died |
|---|---:|---|---:|---|
| A | 16 | — | 1 | yes, req1 first propose |
| B | **15** | — | **3** | **no** |
| C | 16 | `ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000` (never captures) | 1 | yes — eager path faults too |
| D | 16 | `ATLAS_DFLASH_DEBUG_FULL_PRECOMPUTE=1` | 1 | yes |
| E | 16 | `CUDA_LAUNCH_BLOCKING=1` | 1 | yes — surfaces as `cuGraphLaunch failed: status 700` |
| F | 16 | `ATLAS_DFLASH_CTX_WINDOW=4080` (257-block pool like γ=8) | 1 | yes — not the 258-block edge |

Facts that survive: (1) sequence generation 1 at γ=16 is fine for 256 tokens
with capture+replay; (2) generation ≥2's *first* propose faults, eagerly or
captured; (3) γ ≤ 15 never faults (026: γ 6/8/10/12 4/4 each; 040-B 3/3);
(4) legacy (non-Option-B) γ=16 ran 503 agentic turns overnight
(`run-1789388076481949662`). So the fault is *Option-B paged drafter path ×
γ=16 × sequence churn*. What is γ=16-specific in that path is not settled by
reading: `blocks_needed = (max_ctx_len+γ+1).div_ceil(16)`, `n_attn = γ+ctx_window`,
the γ-row `fill_slots_from_block_table` and `reshape_and_cache` all look
γ-agnostic, and the 258-vs-257 pool edge was excluded by leg F. **044 (memcheck)
is queued to name the kernel and the address.** Until then, the production
posture is: γ from MODEL.toml (=8), which is also the measured optimum.

### A.3 Fix landed — `a61f02634` on `review/pr21-fixes` (pushed)

`spark-server: resolve --dflash-gamma from MODEL.toml when the flag is absent`.
`dflash_gamma: Option<usize>`; `apply_model_default_dflash_gamma` (flag →
`ptx_set.dflash.gamma` → `DEFAULT_DFLASH_GAMMA=16`) runs immediately before
`apply_model_default_num_drafts`; `resolved_dflash_gamma()` mirrors
`resolved_num_drafts()`; boot logs `DFlash γ=8 (source: model_toml)`. 4 unit
tests (flag wins / MODEL.toml / engine default / no-op without `--dflash`);
fmt + `clippy -p spark-server --tests` clean on the pinned 1.93.1. Every
recipe-driven `--dflash` serve (bfcl / agentic / concurrency gates) now runs
γ=8 without an override. The overnight legs build this sha into
`~/code/wt-dflash2-gamma` (job `build-wt-dflash2-gamma`) so the in-use
`wt-dflash2` binary is never swapped under a running job.

### A.4 KL logit drift and coherence (041) — what it does and does not say

Harness: `kl_coherence_gate.py` math (shared-support renormalized KL over
top-10), run sequentially (one GB10 cannot host two 27B+drafter serves), 5
prompts × 256 tokens, temp 0, seed 42, `reasoning_effort:"none"`, logprobs
top-10; serial vs `--dflash` γ=8 + OB. Both legs return logprobs on every
non-tool response (the tool response carries `logprobs: null` on both legs).

| prompt | byte-identical | first divergence | serial top1−top2 there | mean KL, shared prefix | max KL, shared prefix |
|---|---|---|---:|---:|---:|
| fib | **yes** (118 tok) | — | — | 0.0057 | 0.27 |
| kv_cache | no | pos 31 ` mechanisms`→` weights` | **0.0** | 0.0299 | 0.28 |
| spec_reject | no | pos 33 ` and`→` to` | **0.0** | 0.0246 | 0.16 |
| minheap | no | pos 129 ` root`→` top` | **0.0** | 0.0143 | 0.31 |
| prose_sky | no | pos 14 ` grey`→` gray` | **0.0** | 0.0054 | 0.06 |

Gate verdict as printed: FAIL (`match_frac` 0.554 across *all* positions,
`mean_kl` 12.25 across all positions). Neither headline number is meaningful
past the first divergence — the two legs condition on different histories from
there — so the honest numbers are the shared-prefix columns.

Three findings, in confidence order:

1. **Every first divergence is an exact tie on the serial side** (e.g. serial
   pos 31: ` weights` −1.6715 vs ` mechanisms` −1.6715; serial emitted the
   *later* one, OB the *earlier*). This is the residual host last-wins vs
   kernel first-wins tie contract split (PR #23 review notes 1–2), not a
   numerics or acceptance defect. 033's 4 divergences fit the same class.
2. **Shared-prefix KL is small but non-zero (0.005–0.03 nats mean, ≤0.31
   max)** even on the byte-identical prompt — the verify-row logits (graphed
   K=γ verify at M=γ+1 rows, NVFP4 head via `w4a16_gemm_t`) differ from the
   serial decode logits (GEMV arms). Above the gate's `1e-3`. Whether that is
   acceptable is a policy call; it is the number to track.
3. **Confound found and fixed for v2:** the 041 requests inherited the
   non-thinking preset's `presence_penalty=1.5`. Serial reports *post-penalty*
   top_logprobs; the DFlash lane reports *pre-penalty* ones — 33/940 OB
   positions emitted a token that was not the argmax of its own reported
   distribution (all high-frequency tokens: `,` `\n` ` the` ` a`), serial 0/940.
   Not an acceptance bug (the per-sequence raw-argmax gate `323a7e1d` routes
   penalized requests through the pipeline and 033/041 show byte-identical
   256-token runs), but a **logprobs-reporting inconsistency under penalties**
   worth a fix in `verify_dflash_step.rs`, and it means 041's KL compares two
   different quantities. `dflash-kl-coherence-v2` (penalties explicitly zeroed)
   is queued; read that one.

### A.5 Batched concurrency (034 → gate leg tonight)

034 at bs8 (8K seq, util 0.78, bf16 KV, MinHeap ×8, 256 out): serial-multi
**65.15** agg tok/s vs DFlash+OB **27.03**. The OB serve did batch (`Captured
CUDA graph for batch size 8`, `DFLASH WIDTH n_active=8 verify=8
dspark_batch_ok=true`, 8 piecewise captures), the auto MTP gate shed spec on
69% of steps (`serial=0.69 mtp=0.31`) and per-seq throughput still landed at
3.4–3.6 tok/s vs 8.1 serial. Per-seq mean_na 3.86 says acceptance is fine at
width — it is the batched K=γ verify step that costs. `conc-sweep-dflash-ob-g8`
(gate C1/4/8/16, `mtp_gate: force` per recipe, bs16, util 0.78) is queued to
put this on a gate record next to the MTP record
(`run-1789248105378681119`: C1 20.9 / C4 41.9 / C8 64.1 / C16 85.2).

### A.6 ST-995 and agentic — what is queued and against what

* `bfcl-dflash-ob-g8-thinkon` — one variable vs 022 (MTP → DFlash2+OB γ=8):
  same recipe, `disable_thinking=false`, effort medium, same draw; util 0.75
  instead of 0.85 (drafter build OOMs at 0.85 on 119.6 GB — PR #21 finding 2).
  Reference: 022 87.94 / 88.24. Spec is hard-gated OFF inside `<think>`, so this
  is a *correctness* record for the DFlash verify lane, not a speed record.
* `agentic-dflash-ob-g8` — 50 iterations, wall budget 9000 s, recipe pins
  (pcache ON). References: legacy-DFlash γ=16 `run-1789388076481949662` 49/50 ok
  but Σwall 28878 s (57.3 s/turn, perf FAIL); MTP gate records ~5.5 s/turn.

### A.7 Open engineering (dense), ordered

1. **Name the γ=16 kernel** (044) and fix it — or fail closed at boot for
   γ>15 under Option B with the bisect as the citation.
2. **DFlash-lane logprobs under penalties** report pre-penalty distributions
   (§A.4-3). Fix: take `top_logprobs` from the same processed logits the
   pipeline emitted from.
3. **Tie contract**: decide PR #23 review notes 1–2 (kernel lane-vs-index
   merge; host last-wins in `sample_impl.rs`/`sample_step.rs`) — every
   observed DFlash-vs-serial divergence is one of these.
4. **Batched verify cost at width** (§A.5) — the C8 loss stands even with spec
   shed; profile the bs8 K=γ verify the way §G profiled bs1.
5. Prose acceptance collapse (mean_na → 1.0–1.4 on prose; 033) — γ=8 helps
   the step cost but not the acceptance; drafter-side.
6. Verify graph ~165 ms; lm_head at M=γ+1 (`w4a16_gemm` 20 ms); drafter GEMM
   fusion — §G.3 items 2–3, unchanged.
7. Recipes: un-pin `disable_thinking` for the nvidia bfcl subject and
   re-derive the 83.x bars (022 is the evidence); add `gpu_memory_utilization`
   guidance for `--dflash` (0.85 OOMs with the BF16 drafter; the NVFP4
   drafter, 1.55 GB, is the memory fix).

---

## B. Flash-Next — nvidia pack on `feat/flashnext-nvidia` @ `9c1ca517`

### B.1 Branch state (4 commits over `review/pr23-fixes`, all pushed)

| sha | what |
|---|---|
| `9923d498` | atlas-core: parse ModelOpt `quantized_layers` (MIXED_PRECISION map) |
| `dbe6f7d2` | spark-model: nvidia MTP experts (per-expert FP8-block-scale → NVFP4), MIXED_PRECISION detect excluding `mtp.*`/`*.ple.*`/`*.visual.*`; MODEL.toml `hf_id` → nvidia |
| `0902edb6` | kernels: BENCH.toml scaffold for the nvidia subjects (bootstrap protocol, no floors yet) |
| `9c1ca517` | spark-model: **PLE forward scratch bounded by `ATLAS_PLE_CHUNK` spans** — 4.7 GB → 1.17 GB at 32K prefill; the n-gram table (51.2 GB FP8, 128 shards) stays NVMe-served with a 131072-slot row cache (41.9 MB) |

Loader/serve proven: 013 smoke (MTP shard F8_E4M3 + g128 scale_inv, first
request 1538 ms), 029 MTP-on probe (`--speculative --num-drafts 1`, healthy in
80 s, 995-token generation, gate arbitrated K=1).

### B.2 The OOM, decomposed

The GB10's 119.6 GB is *unified*: `--gpu-memory-utilization` is a pledge
against the same pool the host, the page cache, and the NVMe-served PLE table
draw from. Observed:

| leg | util | seq | outcome | evidence |
|---|---:|---:|---|---|
| 014/015 | 0.85 + MTP | 32K | refused at build: 109.9 GB + 7.5 reserve > 101.7 budget | job logs |
| 024 | 0.95, no MTP | 16K | **host OOM-killed** ~04:20 after ~33/50 iterations | `journalctl`: OOM at 118/119 GB, pipewire killed too; serve log ends mid-request, no CUDA error |
| 037 | 0.93, no MTP, `9c1ca517` | 16K | 25 turns clean in 1507 s, then **HTTP 400 `Prompt too long: 16752 > 16384`** | `run-1789451775373797695`; `hardware_state.after.mem_available_kb = 870924` (**0.87 GB** available) |

Serve-side accounting at 0.93/16K (037 serve log): weights 73.33 GB on disk →
**84.0 GB device after load** (SGLang reports 83.68 GiB for the same pack —
parity), + layer construction 2.5 GB, + buffer arena 6.3 GB, + inference
reserve 4.1 GB → **95.1 GB pre-KV**; then the KV pool takes *everything left
under the pledge* (12.2 GB for a bs1 workload whose 12 full-attention layers
need < 1 GB at 32K), + Marconi 1.8 GB + rollback 0.9 GB; 5.5 GB co-tenants.
The "~25 GB gap vs SGLang" in the gap analysis is therefore: arena 6.3 +
reserve 4.1 + construction 2.5 + KV over-allocation ~11 + snapshot pools 2.7.
Two of those (KV, snapshot pools) are pledge-driven, not need-driven.

Corrected profile for tonight: **util 0.88, `--max-seq-len 32768`,
`--max-prefill-tokens 16384`** — ~6 GB handed back to the host, 32K covers the
agentic transcript growth (params `max_turns=40`, `max_tokens=8192`). Every
Flash-Next job now writes `mem-trace.tsv` (MemAvailable / serve RSS / device
used, 30 s cadence) so a future OOM has a curve.

### B.3 Records

* **023 ST-995: 83.52 / 82.45** (`run-1789433036569430053`, 13820 s, util
  0.95 / no MTP / 16K / thinking ON per MODEL.toml). Info verdict — the BENCH.toml
  scaffold has no floor yet, by design. Unsloth-class territory on the first try.
* 037 agentic: 25/40 turns of iteration 1 fine, killed by the 16K ceiling (§B.2).

### B.4 Queued tonight (all `wt-fnext-nv` @ `9c1ca517`, util 0.88 / 32K)

| job | what | reads against |
|---|---|---|
| `flashnext-nvidia-agentic3` (**running**) | 50 iterations, wall 9000 s, mem trace | 037 partial; the OOM curve |
| `flashnext-kl-coherence` | serial vs `--ngram-speculative` (PR #23's llama.cpp-style dynamic table + chains), penalties zeroed, logprobs top-10 | the n-gram lane's greedy-verify parity; engagement lines |
| `flashnext-conc-probe` | C1 / C4 MinHeap agg tok/s, bs4 | first bs>1 observation on this pack |
| `flashnext-nvidia-bfcl2` | ST-995 with n-gram spec at the corrected profile, mem trace | 023 (two variables moved → observation, not A/B) |

### B.5 Open engineering (Flash-Next), ordered

1. **Need-driven KV/snapshot sizing** for bs1 profiles (cap the KV pool at
   `max_num_seqs × max_seq_len` when that is below the pledge remainder)
   instead of absorbing the pledge — the single largest lever on the host side
   of the unified pool.
2. If agentic3's `mem-trace.tsv` shows a monotone MemAvailable decline at
   constant device usage, the PLE NVMe path is pinning pages (row-cache
   misses at 55,808-id gathers were seen in 037); if flat, the pledge was the
   whole story.
3. MTP fits only at ≤0.85 with the FP8 experts path (029 at 0.93 ran K=1) —
   measure MTP vs n-gram vs serial at the corrected profile before choosing a
   recipe default.
4. BENCH.toml floors from 023 + tonight's records, then `status = "measured"`
   so `--pull-request-gate` can serve the checkpoint.
5. Everything in gap-analysis §A.3 items 2–9 (context ceiling, `qsa_score_rows`,
   decode graph capture, `--speculative-token-map`, lm_head precision gate).

---

## C. Queue as armed (09-15 ~07:30 UTC), in dispatch order

| id | job | est | GPU |
|---|---|---|---|
| 043 | flashnext-nvidia-agentic3 | 2.5 h (running) | wt-fnext-nv |
| 044 | dflash-g16-memcheck (`--timeout 5400`) | ≤1.5 h | wt-dflash2 |
| 045 | build-wt-dflash2-gamma (`--no-gpu`, `GAMMA_SHA=a61f02634`) | ~2 min | — |
| 046 | dflash-kl-coherence-v2 | ~6 min | wt-dflash2 |
| 047 | dflash-ob-longctx-v3 | ~5 min | wt-dflash2 |
| 048 | flashnext-kl-coherence | ~10 min | wt-fnext-nv |
| 049 | conc-sweep-dflash-ob-g8 | ~10 min | wt-dflash2-gamma |
| 050 | flashnext-conc-probe | ~10 min | wt-fnext-nv |
| 051 | agentic-dflash-ob-g8 | 2.5 h | wt-dflash2-gamma |
| 052 | bfcl-dflash-ob-g8-thinkon | ~4.1 h | wt-dflash2-gamma |
| 053 | flashnext-nvidia-bfcl2 | ~4 h | wt-fnext-nv |

(ids are the expected numbering; `qctl status` is the SSOT.) Results are
appended to §D as they land.

**Actual dispatch (as of 09-16 01:10Z)** — ids drifted through the reorders
and re-queues described in §D.12: 043 → 044 → 045(fail, env) → 046 → 048(fail,
no binary) → 053 → 054 → 058 → 067 → 074 → 075 → 077 → 081 → 082 → 087 →
091(wrong sha) → 095 → 099(OOM) → 100 → 102 → 103(OOM) → **104 (running,
ST-995 + DFlash2+OB γ=8 + thinking ON, ETA ~06:00Z)** → 105 diag-every →
108 agentic MTP-27B reference → 109 conc-sweep (with drain wait) → 110
flashnext agentic4 on `45f5b355`.

---

## D. Results landed after this log was opened

*(appended in place; each row carries its run-record id)*

### D.1 043 flashnext-nvidia-agentic3 — `run-1789474170004681458` (12:09Z)

Fingerprint: `wt-fnext-nv` @ `9c1ca517`, nvidia Flash-Next pack, `--max-seq-len
32768 --max-prefill-tokens 16384 --gpu-memory-utilization 0.88 --kv-cache-dtype
bf16 --enable-prefix-caching true`, bs1, no speculation, agentic-webserver
iterations=50 wall_budget_s=9000 s_per_turn_budget=0.

| metric | value |
|---|---:|
| webserver_ok | **50 / 50** |
| followed_directions | 49 / 50 (run 4: 2/6 steps) |
| Σwall | **19,351 s** (verdict **Fail**: > 9000 s) |
| s_per_turn | 23.7 |
| decode_tps (harness) | 15.4 |
| Σ completion tokens / tool calls / turns | 297,567 / 845 / 816 |
| host: serve RssAnon | 2.0 GB after load → **~9.8–10.4 GB plateau** from ~09:00Z |
| host: MemAvailable min | **403 MB** (11:44Z) |

Reads: (1) the 32K ceiling fixed 037's failure mode — no `Prompt too long`
in 816 turns; (2) correctness is clean at this profile; (3) the perf verdict
is the serial-decode floor of this pack on Atlas (23.7 s/turn vs the dense
27B's ~5.5 s/turn MTP record) — the speculative lane (n-gram or MTP) is what
this workload needs; (4) **the host-side growth is real and workload-shaped**:
~110 MB / 5 min during long-transcript phases, ~20 MB / 5 min on short
turns, plateauing near 10 GB — at util 0.93/0.95 this is exactly the OOM
kill 024 took (`(1−util)×119.6 GB` is all the host gets). Leading mechanism
from the code read (to be confirmed by job 058, §B.6): per-boundary-token
decode-rollback **aux snapshots** — `save_decode_aux_snapshot` allocates a
fresh host `Vec` of `ingested×hd×2` bytes per QSA indexer (12 × 4.6 MB at
18K ctx) plus the PLE blob, `synchronize`s, D2Hs, and drops the previous
ring entry; live set bounded (8 ring slots) but the multi-MB churn across
threads fragments glibc arenas. Predicted by the shape (growth ∝ context
depth × boundary-token rate; plateau once max context is reached).

### D.2 044 dflash-g16-memcheck — no device error caught (12:13Z)

compute-sanitizer memcheck, γ=16 + OB, 24-token requests: request 0 ran
(mean_na 4.0, capture at generation 1 fine); at request 1 the serve process
**died outright** at "piecewise capture: starting … generation: 2" — serve
log ends there, sanitizer reports only boot-time API noise (194×
`cuMemGetInfo_v2` INVALID_CONTEXT / `cuModuleGetFunction` NOT_FOUND) and
"process didn't terminate successfully". No `Invalid __global__ read/write`.
So the tool cannot see the fault; 067 (eager + `CUDA_LAUNCH_BLOCKING=1`,
`RUST_LOG=debug`) is queued to name the single failing launch instead.

### D.3 046 dflash-ob-longctx-v3 — Option B at 9.7K context (12:5xZ)

Fingerprint: serve_dflash profile with `--max-seq-len 16384`, γ=8, OB=1,
9,754-token Rust-source prompt, 137-token answer, temp 0, 3 repeats each.

| leg | tok/s incl. TTFT | TTFT | decode tok/s (server) | p1 | serial fraction | sha |
|---|---:|---:|---:|---:|---:|---|
| serial | 5.14 ×3 | 14.42 s | 11.2 | — | 1.00 | `0560fbc9` ×3 |
| dflash+OB γ=8 | 4.52 / 4.87 / 4.51 | 14.53 s | 8.7 / 10.1 / 8.7 | 0.04–0.14 | 0.83–0.95 | `0560fbc9`, `fc9bc2e0`, `0560fbc9` |

Read: at ~10K context on a prose-style answer the drafter is rejected almost
entirely (p1 ≤ 0.14), the gate sheds spec on 83–95 % of steps, and the
residual propose cost still makes OB **slower than serial** (8.7–10.1 vs
11.2 tok/s). Option B's O(1) propose claim is *not* what this measures (the
propose cost per step is flat by construction — §G.6); what it shows is
that when acceptance collapses, the flat ~50 ms propose is pure overhead.
The acceptance-collapse problem (§A.7-5) dominates at depth. TTFT parity
(14.4 vs 14.5 s for 9.7K tokens ≈ 675 tok/s prefill) is a separate note:
prefill on the 27B at this depth is far from SGLang's 22.6k tok/s.

### D.4 053 dflash-kl-coherence-v2 — penalties zeroed (13:0xZ)

Same harness as 041 with `presence/frequency_penalty=0, repetition_penalty=1`.
**The reporting confound is gone: 0 / 2,144 positions across both legs emit
a token that is not the argmax of its own reported top-10** (041: 33 on the
OB leg). Results:

| prompt | byte-identical | first divergence | serial top1−top2 (nats) | mean KL, shared prefix |
|---|---|---|---:|---:|
| kv_cache | **yes** (65) | — | — | 0.0021 |
| fib | **yes** (118) | — | — | 0.00002 |
| minheap | no | pos 59 ` item`→` value` | **0.25** | 0.0002 |
| spec_reject | no | pos 80 ` why`→` a` | (see JSON) | 0.0033 |
| prose_sky | no | pos 14 ` grey`→` gray` | 0.0 | 0.0465 |

Two of the five divergences are exact ties (tie-contract class); **minheap's
is not** — a 0.25-nat margin flip at pos 59 means the verify-row logits
disagreed with serial decode by more than a rounding tie there. The
shared-prefix KL is small (mean 0.0104 over the five prompts, ≤0.05) — but
the gate threshold is 1e-3 and only fib clears it. Status: **measured drift,
above the fold gate, mechanism = verify-path numerics (M=γ+1 rows through
the NVFP4 head/`w4a16_gemm_t` and the K=γ graphed verify vs the M=1 GEMV
decode path).** Whether it is acceptable is a policy decision for #21; the
numbers are now clean enough to decide on.

### D.5 054 flashnext-kl-coherence — n-gram lane inert (12:3xZ)

Same harness on the nvidia Flash-Next pack (util 0.88 / 32K / pcache ON),
serial vs `--ngram-speculative`. Boot logged "N-gram speculative decoding:
ENABLED (K=2/3/4 verify, CPU proposer)", but **every request decoded at
tok_step 1.000, mean_na 0.000, and the ngram leg's wall equals serial's
(14.5 vs 14.4 s on 256-token answers)** — the lane never engaged. 3/5
outputs byte-identical to serial; the two divergences (spec_reject pos 55,
margin 0.06 nats; prose pos 14, exact tie) are then either serial run-to-run
instability across processes or the K=2 verify path running with 100 %
rejection (drafts proposed, none accepted — the accept-debug line cannot
tell these apart). Shared-prefix KL 0.0004–0.058. Not yet determined which;
the 074 debug log settles it. PR #23 measured +36 % on a 400-token essay on
the RadixArk pack; **074 (positive control: essay, 400 out, RUST_LOG=debug)
is queued** before any n-gram number is quoted for this pack.

### B.7 External Flash-Next agent report (spark box, ~16 h whiteboard-engine build) vs. our 043 evidence

Report received 09-15 (andrei, `bench/inkboard`): on the `assist/flashnext-longform-fixes`
line serving a real coding agent for ~16 h:

| item | their status | what our tree/records say |
|---|---|---|
| 32K request cap — `ATLAS_QSA_MAX_TOKENS` defaults to 32,768 in upstream `qsa.rs` and errors above | **pinned**; PR #23 clamps capacity to `--max-seq-len`; `ATLAS_QSA_MAX_TOKENS=98304` verified to 42.8K | same code on `feat/flashnext-nvidia` (`qsa.rs::new`: `max_tokens = max(env, max_seq_len)`); 043 boots with `indexer capacity 32768 (max_seq_len=32768)` — no env needed at 32K |
| prefill chunk must not exceed PLE scratch (`ATLAS_PLE_MAX_TOKENS`) | **pinned** (config coupling) | `9c1ca517` decouples it: the PLE pipeline loops over `ATLAS_PLE_CHUNK` spans (default 8192) regardless of `--max-prefill-tokens`; 043 ran 16,384-token chunks with 8192-token PLE spans, 819 prefills |
| CUDA **misaligned-address** context loss, 4× in 35 min of agent traffic, always at a D2H in layer 0 of a *continuation chunk* right after an intermediate SSM checkpoint save; 0× in 92 controlled requests | **not pinned**; needs real-agent repro + bisect of the branch's 99 commits under launch-blocking | **0 occurrences in 043**: 819 prefills, **32 multi-chunk prompts (up to 25,191 tokens = 2 × 16K continuation chunks)**, 1,562 intermediate SSM checkpoint saves / Marconi hits, 816 agent turns with tool calls and inter-turn idle gaps, 5.4 h — `grep MISALIGNED|716|700|ILLEGAL` = 0 over 17,005 log lines. Also 0 in 023 (997 single-chunk prefills) and 037. |

Read on the third row (rule 5 language): *not reproduced* under our fingerprint
(`9c1ca517`, nvidia pack, chunk 16384 + PLE span 8192, 32K cap, pcache ON,
bs1). The external fingerprint differs in at least chunk size (8192), pack,
`--max-seq-len` (≥98K) and prompt depth (their 42.8K vs our 25K), so this is
two observations, not an A/B. If the fault is chunk-boundary-shaped, fewer
and larger chunks would also hide it; the one-variable test from our side is
043's serve profile with `--max-prefill-tokens 8192` under the same agent
harness (queueable; ~5 h). Until then the site they name — D2H at layer 0 of
a continuation chunk immediately after an intermediate checkpoint save — is
the same family as the DFlash 700 (a D2H sync surfacing an earlier async
fault), and 067's launch-blocking method applies to it unchanged.

### B.6 Host-memory bisect — job 058 (landed 13:4xZ)

Fresh serve per leg (RssAnon baseline 0.98 GB after load), 6 requests each,
util 0.88 / 32K / prefill 16K profile. Δ = RssAnon after req 5 − after req 0.

| leg | shape | Δ RSS (MB) | MB per 1K tokens |
|---|---|---:|---:|
| A | 51-tok prompt, 321 out (decode-only) | +86 | 38.5 (≈14 MB/request fixed) |
| B | 27.4K distinct prompts, 11–16 out, pcache ON | +1,075 | 6.5 |
| C | the SAME 27.4K prompt ×6 (prefix hits), pcache ON | +377 then **flat** | 2.3 |
| D | = B, pcache OFF | +819 | 5.0 |
| E | = B, `ATLAS_QSA_DEVICE_TOPK=1` | +1,098 | 6.7 |
| F | 27.5K prompt + 400 out (the agentic shape) | **+1,593** | **9.5** |
| G | = F, `ATLAS_SSM_DECODE_RING=0` | +821 | 4.9 |
| H | = F, `MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576` | +590 | 3.5 |

Reads: (1) QSA host top-k is not it (E ≡ B). (2) Prefix-cache *hits* stop
the growth (C plateaus after the first save) — the per-prefill component is
the Marconi snapshot's aux blobs, retained per snapshot slot by design
(QSA raw keys 256 B/token/indexer × 12 ≈ 3 KB/token → ~84 MB per 27K
prompt; 6 distinct prompts ≈ 500 MB, matching G). (3) The decode ring
roughly doubles the long-decode growth (F vs G): the per-boundary-token aux
snapshot churn. (4) Malloc tuning trims further (H) → the remainder is
fragmentation, not retention. Two mechanisms, one root: **aux state is
serialized into fresh multi-MB host `Vec`s on every save** (ring: every
boundary token; Marconi: every checkpoint and finish-leaf). Fix in flight
on `feat/flashnext-nvidia`: caller-owned, capacity-retaining buffers for
both paths + restore-under-lock instead of `.cloned()` (§B.5-1 supersedes:
this is the lever, not KV sizing). Serve-profile mitigation available today:
`ATLAS_SSM_DECODE_RING=0` for agentic serving (the rollback watchdog is the
feature that false-positives on code anyway — PR #23 review) and the
`MALLOC_*` pair in the launcher.

### D.6 067 dflash-g16-localize + 081 memcheck-eager — the launch that fails

Eager path (`ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000`), γ=16, OB, request 2's
first propose (`Prefilled (single chunk): seq_len=51` → `allocated 258
blocks`) → 23 ms later:

| run | tool | reported failing launch | status |
|---|---|---|---|
| 067-I | `CUDA_LAUNCH_BLOCKING=1` | `grid=[32,1,1] block=[128,1,1]` = `prefill_attention_paged_dflash_bf16_indirect` (`[num_q_heads, ceil(γ/32)]`, block 128) | 700 |
| 067-J | `ATLAS_DFLASH_OPTION_B_DIAG=1` | same point, no extra info (diag is one-shot, fired on seq 1) | 700 |
| 081 | compute-sanitizer memcheck, eager | `grid=[16,1,1] block=[256,1,1]` (= γ tokens: `reshape_and_cache` or `batched_embed`); **no device memory error reported** | **719** unspecified launch failure |

A blocking launch can report the *previous* launch's execution fault, and
the two runs name different kernels, so the culprit is one of the first
kernels of the drafter's layer-0 pre-attention on the new sequence
(`batched_embed` → … → `reshape_and_cache` → paged-indirect attention) — and
memcheck seeing no bad address twice points at a launch-contract violation
(argument pointer / kernel state) rather than a plain OOB. Job
`dflash-g16-syncdebug` (in-tree `ATLAS_DEBUG_SYNC_KERNELS=1`, which syncs
after every launch and returns a **Rust backtrace of the launch site**) is
queued to name the `ops::*` wrapper unambiguously.

### D.7 074 flashnext-ngram-control — lane engages, accepts ~nothing

600-word essay prompt (PR #23's shape), 400 out, temp 0, penalties 0,
`RUST_LOG=debug`, 2 requests per leg:

| leg | req | tok/s incl. TTFT | server tok/s | tok_step | n-gram debug lines |
|---|---|---:|---:|---:|---:|
| serial | 0 / 1 | 15.6 / 16.8 | 16.2 / 17.5 | 1.000 | — |
| ngram | 0 / 1 | 16.0 / 18.7 | 16.6 / 19.5 | **1.000** | 26 / 145 (`NGRAM detail: drafts=[…] … na=0`) |

So the proposer *does* run on this pack (drafts proposed; the table warms
between requests: 26 → 145 lines) but acceptance is ~0 (`na=0` almost
everywhere) → no speedup. PR #23's +36 % / ~93 % acceptance was measured on
the RadixArk pack with the essay warm; **not reproduced on the nvidia pack**
under this fingerprint. Note also serial run-to-run is sha-identical here
(`879c300fbe0c` ×2) while the two ngram runs differ from serial and from
each other — the K=2 verify path with rejected drafts is *not* output-neutral
on this model (ties or numerics; same class as §D.4).

### D.8 075 flashnext-conc-probe — first bs>1 numbers for the nvidia pack

bs4 serve at util 0.88 / 32K (KV pool only 1.8 GB after 96.5 GB pre-KV +
7 GB reserve — 78,464 KV tokens), MinHeap ×N, 256 out, temp 0:

| C | agg tok/s | per-request tok/s | outputs |
|---|---:|---:|---|
| 1 | 17.5 / 18.5 | 17.5 / 18.5 | sha `aa7eff8d` |
| 4 | **35.2 / 36.8** | 8.8 / 9.2 | 4× `aa7eff8d` (identical across the batch) |

C1→C4 = 2.0× aggregate. Batched decode is coherent and deterministic across
lanes; the scaling matches the dense 27B's C1→C4 shape (20.9→41.9, ~2.0×)
at roughly 0.85× its absolute numbers.

### D.9 077 build — `wt-dflash2-gamma` @ `a61f0263` built; `--dflash-gamma` help shows no `default: 16`.

### D.10 095 dflash-g16-syncdebug2 — **the faulting kernel is named**

087 (syncdebug v1) could not run: `ATLAS_DEBUG_SYNC_KERNELS=1` synchronizes
after every launch, which is illegal inside the K=γ verify graph *capture*
→ 900/901 on request 0. 095 re-ran with every graph off
(`ATLAS_DFLASH_DEBUG_NO_GRAPH=1 ATLAS_DEBUG_NO_GRAPH=1`, eager propose):
request 0 completed 256 tokens (16.4 tok/s), request 1 faulted at its first
propose with the launch-site backtrace:

```
ATLAS_DEBUG_SYNC_KERNELS: async GPU fault immediately after kernel launch grid=[32,1,1] block=[128,1,1]: CUDA_ERROR_ILLEGAL_ADDRESS (700)
  KernelLaunch::launch
  ops::prefill_attn_main_a::prefill_attention_paged_dflash_bf16_indirect
  BlockDiffusionDraftHead::forward_block_layer_attention
  forward_block → propose_drafts_on_lane → DraftProposer::propose → run_mtp_propose_multi → step_mtp
```

So it is the drafter's paged-indirect attention
(`inferspark_prefill_paged_indirect_sink`, `[num_q_heads, ceil(γ/32)]`), layer
0, at the second sequence's bootstrap propose, γ=16 only. The kernel source
(`prefill_paged_compute.cuh`) bounds every block-table/Q/KV read by
`kv_len`/`q_len`/the table, so the **inputs** are the suspects: block table
contents (re-allocated in a different order after `free_state`), the 12-byte
indirect `(kv_len, q_offset, q_rope_pos)` triple, or the pool/q_buf
pointers. `0a16369df` (pushed) adds `ATLAS_DFLASH_OPTION_B_DIAG_EVERY=1`,
which dumps exactly those on every layer-0 propose; job 105 runs it on a
separate `wt-dflash2-diag` build after the overnight legs and prints request
0's vs request 1's dump side by side.

### D.11 082 flashnext-nvidia-bfcl2 — ST-995 at the corrected profile, n-gram lane inert

`run-1789501991840065669`, `wt-fnext-nv` @ `9c1ca517`, util 0.88 / 32K /
prefill 16K / bf16 KV / pcache ON / `--ngram-speculative`, thinking ON
(MODEL.toml default), n=995: **overall 83.52, normalized 82.45** — identical
to 023 (83.52 / 82.45) to two decimals. Category: hallucination 85.98, live
80.00, non_live 81.35. Info verdict (no committed floor). Two reads: (1) the
memory-profile change (0.95/16K → 0.88/32K) is accuracy-neutral; (2) the
n-gram lane never fired — **0 `NGRAM detail` lines across 995 requests**
(3 "ngram" lines total, all boot) — so the run is a serial-path record and
the identical score is expected, not a parity proof. Host memory: serve RSS
11.5 GB at the end, MemAvailable min 1.9 GB — the aux-snapshot churn again
(fix `45f5b355` verification is job 102; 091 built the wrong sha — see D.12).

### D.13 100 agentic-dflash-ob-g8 — first DFlash2+Option-B agentic record on the 27B (23:4xZ)

`run-1789513682099633281`; `.benchmarks/agentic-webserver/2026-09-15-a61f02634f-nvidia-qwen3.8-27b-nvfp4.json`
in `wt-dflash2-gamma`. Serve: gate recipe `qwen3.8-27b-nvfp4-agentic` +
`dflash=true speculative=false gpu_memory_utilization=0.75`, **`DFlash γ=8
(source: model_toml)`** — the `a61f0263` precedence fix working in the gate
path; pcache ON per recipe.

| metric | DFlash2+OB γ=8 (this) | DFlash legacy γ=16 (`run-1789388076481949662`) |
|---|---:|---:|
| webserver_ok | 48 / 50 (run 21: `/ping` timeout) | 49 / 50 |
| followed_directions | 50 / 50 | 50 / 50 |
| s_per_turn | **19.6 s** | 57.3 s |
| Σwall | 9,907 s (verdict Fail: > 9,000) | 28,878 s |
| CUDA faults | **0** in 9,907 s | 0 |
| mean_na (335 requests sampled at 22:12Z) | 1.78; spec on 41 % of steps | — |

Reads: correctness holds and the γ=16 crash is gone from the gate path;
Option B cuts the agentic wall 2.9× vs legacy DFlash, but the run is only
~10 % over the 9,000 s budget and decodes at ≈ serial speed (~12 tok/s)
because the drafter accepts 1.8 tokens/step on this text and thinking (ON in
the agentic recipe) hard-gates spec off. **The gate marked the record
INVALID for speed quoting: the sw-power-cap throttle counter advanced during
the run** — s/turn above is descriptive only (rule 1). An MTP-27B reference
leg on the same harness is queued (108); no such record existed (the 5.5
s/turn agentic figures in earlier notes are the gate's default 35B subject).

### D.14 102 flashnext-auxfix-verify — `45f5b355` measured

FINGERPRINT `45f5b355` (102 fixed 091's remote/checkout bugs). Same F/G legs
as 058 (27.5K prompt + 400 out ×6; Σtok 167,320):

| leg | binary | Δ RSS req0→5 | MB / 1K tok |
|---|---|---:|---:|
| F ring on | `9c1ca517` (058) | +1,593 | 9.52 |
| G ring off | `9c1ca517` (058) | +821 | 4.91 |
| G (repeat) | `9c1ca517` (091, wrong-sha run) | +859 | 5.13 |
| **F2 ring on** | **`45f5b355`** | **+1,324** | **7.91** |
| **G2 ring off** | **`45f5b355`** | +1,309 | 7.82 |

Read: on the fixed binary **ring on ≡ ring off** (F2 ≈ G2 within 1 %) — the
decode-ring churn component is gone, which is what the change targeted. The
residual ~1.3 GB over 6 long requests is the same on both legs and larger
than 058-G; with capacity-retaining buffers per Marconi slot the expected
steady state is 16 slots × (12 indexers × 27.5K × 256 B ≈ 84 MB) ≈ **1.3 GB**
— i.e. the retained-by-design aux footprint reaching its cap, no longer
fragmentation on top. That reading predicts a **plateau**, which 6 requests
cannot show; job 110 (agentic4: 043's exact profile on `45f5b355`, with
mem-trace) is the end-to-end test — 043 reached 9.8 GB, the prediction here
is ≲ 3.5 GB flat. Until 110 lands this is "improved and re-shaped", not
"fixed" (rule 5).

### D.15 103 conc-sweep — boot OOM again (teardown race), re-queued with a drain wait

Second boot-time `cuMemAlloc` failure (79 MB requested, 618 MB free) 40 s
after the previous job's teardown; γ resolved to 8 correctly before it died.
The dispatcher's GPU gate is process-based; on the GB10 `nvidia-smi
memory.used` reads `[N/A]`, so the re-queued script (109) waits for host
`MemAvailable > 100 GB` ×3 samples — unified memory makes the host counter
track device frees. The same wait was added to the MTP reference leg (108).

### D.12 Queue hygiene incidents (09-15 afternoon) — for the qctl README

1. **Sanitizer jobs escape teardown.** compute-sanitizer's
   `TreeLauncherSubreaper` re-parents the serve; after 044 and 081 an
   orphaned `spark` (PGID outside the job's) kept ~40 GB of device memory
   and an inherited `queue.lock` FD. 082 sat in `GPU busy — waiting` /
   lock-blocked from 14:47 to 16:04 until the strays were identified by
   `fuser`+`ps` and killed by PID. Fix to make: sanitizer jobs must record
   the launcher's pgid and sweep it in a trap; the dispatcher should not
   inherit `queue.lock` into job children (`flock` on an FD marked
   `O_CLOEXEC`).
2. **Cancel raced a start.** A batch cancel meant for pending 076 caught it
   the instant it flipped to running; its `setsid` serve escaped the pgid
   kill (same class as the §J "qctl gap"). Rule: `qctl status` immediately
   before each cancel; cancel one job at a time.
3. **`qctl sub --env` does not reach `run.sh`** (`ENV K=V` lines in
   `meta.env` are not exported by the dispatcher). Scripts now default their
   own `VAR=${VAR:-…}`.
4. **091 verified the wrong commit**: `git fetch atlasinf` (no such remote on
   reiner — it is `origin`) and a `checkout … | tail -1 || exit` whose pipe
   masked the failure; the job built `9c1ca517` and its numbers (F2 +1594 MB
   / G2 +859 MB) reproduce 058's baseline to within 2 %, which is at least a
   clean repeatability check. Re-queued as 102 with the remote fixed and the
   checkout guard un-piped.
5. **099 conc-sweep OOMed at boot** (`cuMemAlloc 635 MB with 731 MB free`)
   because it started ~30 s after 095's teardown, before the previous serve's
   memory drained; the queue's `QGPU_USED_MAX_MB` gate should also wait for
   `memory.used` to fall below a threshold for N consecutive samples.
   Re-queued as 103.
