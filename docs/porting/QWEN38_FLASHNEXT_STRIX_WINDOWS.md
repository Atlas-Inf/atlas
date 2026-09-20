# Qwen3.8-Flash-Next on Strix Halo Windows (native HIP)

Bringing `nvidia/Qwen3.8-Flash-Next-NVFP4` (GB10-only until 2026-09-19) up on
`ATLAS_TARGET_HW=strix-hip` / gfx1151 under Windows 11, ROCm/TheRock 10.0.0.
Branch `port/flashnext-strix-windows` (stacked on `feat/flashnext-nvidia`,
PR #41). Host: `winbox` — Framework Desktop, Ryzen AI MAX+ 395 / Radeon 8060S,
127.3 GB RAM, Adrenalin 32.0.31041.1004.

Source of truth for every number below: the session log at
`~/code/qwen38-port-logs/windows/flash-next/SESSION_LOG_2026-09-19.md` and the
fingerprints/artifacts under `winbox-artifacts/` beside it. Numbers carry their
labels (**PRELIMINARY**, observed, n=3) — see "Claims status" at the end.

## Quick start

```powershell
# 1. VGM must be 32 GB (see the memory model below). One-time, reboots:
vgmctl set 32          # scripts/strix-windows/vgmctl/ — build notes in its README

# 2. Build (repairs kernel symlinks first — required on fresh Windows clones):
powershell -ExecutionPolicy Bypass -File .\build-amd.ps1

# 3. Serve the measured MTP profile (writes a fingerprint sidecar):
powershell -ExecutionPolicy Bypass -File `
  scripts\strix-windows\win_serve_flashnext_nvfp4.ps1 -Tag myboot
# SERIAL=1 in the environment drops --speculative and lowers util to 0.86.

# 4. Smoke (thinking off):
$body = @{ model = 'nvidia/Qwen3.8-Flash-Next-NVFP4'; messages = @(
  @{ role='user'; content='Reply with exactly PONG' });
  max_tokens = 64; temperature = 0; reasoning_effort = 'none' } |
  ConvertTo-Json -Depth 6
Invoke-WebRequest http://127.0.0.1:8095/v1/chat/completions -Method Post `
  -ContentType 'application/json' -Body $body
```

[`win_serve_flashnext_nvfp4.ps1`](../../scripts/strix-windows/win_serve_flashnext_nvfp4.ps1)
encodes the measured recipe (util 0.90, seq 8192, prefill 2048, bs1, KV bf16,
`--ssm-cache-slots 0`, `--speculative --num-drafts 1`, commit-limit env 96) and
documents *why* each of those values is load-bearing in its header comment.
[`vgmctl`](../../scripts/strix-windows/vgmctl/README.md) lists/sets AMD Variable
Graphics Memory through ADLX.

## The memory model on a 128 GB Strix Halo under Windows HIP

This is the section that costs days if you learn it empirically. All of it is
measured on this box (ROCm 10.0.0, amdhip64_7 v7.1.52633); the ROCm bug class is
ROCm/ROCm#5940 / rocm-systems#4077 (APU `largeBar_` behaviour).

**VGM sweep** — cumulative `hipMalloc(2 GiB)` + fill kernel + D2H readback per
chunk (`hipmalloc-ladder-vgm{32,96}.txt`):

| VGM (dedicated) | host RAM left | `hipInfo` totalGlobalMem | kernel-touched ceiling |
|---|---|---|---|
| 0.5 GB (Minimum) | 127.3 GB | 76.87 GB | ~63 GB (WDDM 50%-of-RAM shared cap) |
| 96 GB (max) | 31.8 GB | 107.92 GiB | **48 GiB**, then sticky `hipErrorLaunchFailure` (719) |
| **32 GB (current)** | 95.8 GB | 89.47 GiB | past 106 GiB of virtual commit, clean `out of memory` |

The advertised "96 GB dedicated → 112 GB addressable" is **not usable on this
runtime**: past ~48 GiB the driver spills into a WDDM shared segment that is
only ~16 GB at VGM 96, and the GPU faults instead of failing the allocation.
VGM 32 is the sweet spot — local heap saturates at 32 GiB (the ROCm#5940 "32 GB
then spill" behaviour) and the rest comes out of WDDM shared memory backed by
the ~96 GB of host RAM.

**The resident commit wall is ~84.5 GiB** (32 GiB carve-out + ~52.5 GiB WDDM
shared), derived from boot 3's `cuMemAlloc_v2 failed: status 2` with tracked
usage 104 − 19.45 GiB. Working model: usable ≈ min(VGM, 32 GiB) + ~55–60% of
remaining host RAM. The hipMalloc ladder's "106 GiB" figure is *virtual*
commit — WDDM pages earlier chunks to the pagefile; it is not a resident
ceiling and must not be budgeted against.

**The over-commit-then-719 mechanism (proven, boot 7e):** `cuMemAlloc` may
*succeed* past the resident wall — the VA is reserved and backing is paged
lazily — and the **next submission** takes a sticky `hipErrorLaunchFailure`
(719) that poisons the whole context (every subsequent CUDA call returns 719;
all module unloads fail). Boot 7e ran `ATLAS_TRACE_LAUNCH=1` +
`ATLAS_TRACE_LAUNCH_SYNC=1` (a shim diagnostic that synchronizes the stream
after every launch and names the kernel on failure): **0 `ASYNC FAULT` lines
in 850,953 traced launches**. Every kernel launch synced clean; the context
died inside the allocation run right after the 42 MB MTP drafter buffer at
~85 GiB committed. A sticky 719 near the wall is a memory-backing failure,
not a bad kernel — and it is also not the unresolved-lookup list, because a
null kernel handle returns `hipErrorInvalidHandle` synchronously (see the
comment in `layers/moe/forward_prefill.rs`), not an async fault.

**Why the recipe values are what they are:**

- `ATLAS_UMA_COMMIT_LIMIT_GB=96` is an **operator-asserted** ceiling: the HIP
  shim reports it as `totalGlobalMem` (env > 0 overrides; ROCm 10's
  `hipMemGetInfo` succeeds, so the synthesized path never runs). 96 is what
  VGM 32 exposes; it is *not* a safe budget — the recipe keeps real commit
  under the 84.5 wall instead.
- `--gpu-memory-utilization 0.90` (MTP arm) / `0.86` (serial arm): the KV
  budget is **self-relative** — it absorbs whatever is free, so freeing memory
  does not buy headroom unless utilization is also lowered. Boot 7f proved
  this the hard way: `--ssm-cache-slots 0` freed the 1.8 GB Marconi pool and
  the KV cache immediately grew from 0.5 to 2.3 GB, consuming the slack.
  Serial pre-KV is ~4 GB lower than the MTP arm, so at 0.90 the serial KV
  budget balloons to ~7.1 GB and crosses the wall — hence 0.86.
- `--ssm-cache-slots 0`: the Marconi snapshot pool is dead weight without
  `--enable-prefix-caching` (`resolve_ssm_cache_slots` keeps 0 as "disabled";
  MTP verify rollback uses the separate SSM MTP pools).
- At `Server live` the serve process holds ~81 GiB WorkingSet (WDDM shared is
  charged to the process) and the host has ~12 GiB free. **Nothing else
  memory-heavy may run beside it.**

## What was ported

- `kernels/strix-hip/qwen3.8-flash-next/` target (MODEL.toml, kernel mirror),
  119 kernels compiled for `(strix-hip, qwen3.8-flash-next, nvfp4)`, 8
  model-specific overrides — `ef1e51c45`.
- `rms_norm.cu` → symlink to the gb10 kernel: the strix-hip copy was a stale
  strict subset and the fail-closed kernel audit caught
  `norm::gated_rms_norm_sigmoid` unresolved on first boot (`cd01e6115`).
- `gdn_reduce.cuh` same-dir symlink: the gb10 `#include "../../common/…"`
  escapes the flattened hip_mirror tree (`9e5026d19`, PTX byte-identical on
  the Linux reference build).
- **cuBLASLt boundary fallback** (`b8904cc71`, `622cd45d5`): `cublasLtCreate`
  is a stub on HIP (returns 1), and ~20 call sites route through
  `cublaslt::bf16_gemm_act_weight_t`. `install_bf16_fallback` installs once at
  backend init: `dense_gemm_bf16_pipelined` (gfx1151 WMMA) when `K % 8 == 0`
  and all operands are 16 B-aligned, else scalar `dense_gemm_bf16`; launches
  via `cuLaunchKernel` FFI on resolved handles. NVIDIA path unchanged.
- Launch geometry single source of truth (`225dc7d4e`): the dense BF16 GEMM
  tile/thread constants live in `spark_runtime::cublaslt` and are shared by
  the fallback and `layers/ops/gemm_dense.rs`. Behaviour-neutral.
- Launch-fault diagnostic (`c264524f0`): `ATLAS_TRACE_LAUNCH_SYNC=1` in the
  HIP shim — default-off, names the kernel behind an async fault. This is the
  tool that cleared all 850,953 launches in boot 7e.
- **expected_absent metadata** (`7af19ceea`): all 62 gb10-only optional arms
  the Flash-Next model probes but this target does not ship are declared in
  MODEL.toml `[expected_absent]` (80 total including pre-existing entries),
  each audited to a `try_kernel`/handle-gated dispatch site with its fallback
  named. The serve **no longer requires
  `--dangerously-allow-unresolved-kernel-lookups`** — boot 9's audit ran
  clean without it (`80 kernel(s) declared expected-absent … (no action)`,
  zero unresolved warnings). Boots 4–8 carried the flag only because the
  declarations didn't exist yet; it must not be re-added.
- `build-amd.ps1` runs `first_run.ps1 -Phase symlinks` before `-Phase build`
  — fresh Windows clones check kernel symlinks out as text files
  (`core.symlinks=false`) and the build fails on them.

## Boot chronology (winbox, 2026-09-19/20)

| boot | binary | outcome |
|---|---|---|
| build 1 | `ef1e51c45` | `first_run.ps1 -Phase build` doesn't repair symlinks; after `-Phase symlinks`, one real failure — `gdn_reduce.cuh` include escapes the flattened mirror |
| build 2 | `9e5026d19` | **119 kernels**, spark.exe links |
| boot 1 | `9e5026d19`, limit 100, util 0.90 | pre-flight refuse 99.33 > 89.31 (shim ignored env: `hipMemGetInfo` succeeds on ROCm 10) |
| boot 2 | `b615da369`+`d525edd18` | fail-closed audit caught `norm::gated_rms_norm_sigmoid` (stale rms_norm copy) |
| boot 3 | `cd01e6115`, limit 104, util 0.90 | all lookups resolve; 222,588 tensors, 48 layers with mHC; KV alloc `status 2` at **~84.5 GiB — the wall** |
| boot 4 | `cd01e6115`, limit 93, util 0.89 | fits; post-KV audit refuses: **62 unresolved lookups** (gb10-only arms) |
| boot 5 | + `--dangerously-allow-unresolved-kernel-lookups` | **LIVE**; every completion → HTTP 500 `QSA qk projection` — cuBLASLt is a HIP stub, ~20 call sites had no fallback |
| boot 6 | `622cd45d5` (BF16 fallback installed) | **LIVE, 4/4 completions correct**; PRELIMINARY serial 7.8–8.0 tok/s; WS 73.2 GiB, host free 14 GiB |
| boot 7 a–d | + `--speculative --num-drafts 1` | a/b: budget refuses; c: `cuMemAlloc` 48 MiB fails at ~84.6; **d: LIVE + `mtp=true`, then sticky 719 46 ms after live** — scheduler panics, zombie serve |
| boot 7e | `c264524f0` + launch-sync trace | **0 faults in 850,953 launches** — context dies in the alloc run after the 42 MB drafter buffer at ~85 GiB: **memory wall, not a kernel** |
| boot 7f | `--ssm-cache-slots 0`, util 0.92 | Marconi freed but KV absorbed the slack (2.3 GB) → `status 2` at ~83.9 |
| **boot 7g** | slots 0, **util 0.90**, MTP | **LIVE + measured** (below): KV 0.4 + drafter 0.2 GiB; WS 81.2 GB, host free 12.3 GB |
| boot 8a/8b | serial control | 8a @0.90 died (KV 7.1 GB → wall); **8b @0.86 LIVE**: KV 3.3 GB, WS 79.5 GB |
| **boot 9** | `ed8e0dd` tree (audit metadata) via `win_serve_flashnext_nvfp4.ps1` | **LIVE, zero unresolved warnings, no dangerous flag**; MinHeap probe in band (below); ST-995 launched |

## Measured

Harness fingerprint (`boot7f-fingerprint.txt` / `boot8-fingerprint.txt`):
commit `c264524f0` (binary also contains `225dc7d4e`, behaviour-neutral),
spark.exe sha256 `A2A9EF563686EB26…`, MinHeap+unittest code prompt,
`reasoning_effort:"none"`, `max_tokens 1024`, temp 0, sequential, n=3 per arm,
port 8095 loopback, VGM 32 GB, ROCm 10.0.0, 2026-09-20. One variable differs
between arms (`--speculative --num-drafts 1`) plus a forced second-order
difference — util 0.90 vs 0.86 — because the serial arm's KV budget otherwise
crosses the wall (KV capacity 0.6 vs 3.3 GB, irrelevant at bs1/641 tok).
Boots 7g/8b still carried `--dangerously-allow-unresolved-kernel-lookups`;
boot 9 re-verified the same numbers without it.

| arm | tok/s (n=3) | tok_step | mean_na | p1 | TTFT |
|---|---|---|---|---|---|
| **MTP K=2 (boot 7g)** | **17.0 / 16.7 / 17.3** | 1.95 | 0.95–0.97 | 0.95–0.97 | 3.16–3.21 s |
| serial (boot 8b), as reported | 6.7 / 6.3 / 6.4 | 1.000 | 0 | 0 | 2.93–2.98 s |
| serial (boot 8b), rework-corrected (see below) | ≈ 8.3 / 7.8 / 7.8 (derived) | 1.000 | 0 | 0 | — |
| MTP, thinking ON (1 shot, serial-floor regime) | 11.3 | 1.93 (post-think only) | 0.93 | — | 3.15 s |

HTTP wall 40–42 s vs 98–104 s for the same 641-token response. **Supersede
note (2026-09-20, after reading the UTF-16 serve logs):** the two arms are
NOT a clean one-variable A/B on this prompt. The content-loop watchdog fired
on the numeric test-data list on both arms, but the serial path rolled back
70 tokens and regenerated them twice before ending (`rolled back to boundary
… rollback=1`, `rollback=2`, `ending response early`; 8.7 s between fires ≈
70 tokens at ~8 tok/s) while the MTP/emit path has no rollback and ended at
the first fire. The serial `Done:` tok/s therefore counts 641 tokens over the
wall of ~781 generated. Derived correction: 641 ÷ (641/6.7 − 18.2 s) ≈ 8.3,
and ≈ 7.8 / 7.8 for the other two — consistent with the watchdog-free serial
observations of boot 6 (7.8–8.0 tok/s on a 368-token MinHeap answer). The
honest headline is **MTP K=2 ≈ 2.1× serial** (17.0 vs ~8), not 2.6×; GB10
sees +25% on the same model — the serial floor here is bandwidth-bound and the
second verify row is nearly free on the M=2 GEMV tiers. The A/B will be
re-run on a watchdog-fixed build with a prompt that does not trip it. Boot 9 (no dangerous flag, rebuilt audit metadata):
`Done: 641 tokens (length) 16.7 tok/s, TTFT=3281.5ms, mtp=1.00
mean_na=0.954 tok_step=1.954` — in band.

Coherence: minheap-1 is a clean `heapq`-based `MinHeap` with docstrings, type
hints, and a `unittest` class — plausible, no loops or garbage.

Parity (temp 0): serial arm byte-identical across n=3; MTP arm runs 1≡2,
run 3 diverges at char 2499 inside the unit-test data list; MTP vs serial
diverge at char 2508 (same list) — semantically equivalent test data, the
same spec-vs-serial divergence class the 27B port recorded
(`QWEN38_STRIX_PORT.md`), flagged not hidden.

Oddity to chase: `Done: 641 tokens (length)` although `max_tokens 1024` —
the client saw 641 completion tokens, cut mid-list at `values = [8, 3, 1, 6,
4, 2, 7,` on both arms. **Confirmed from the serve logs** (boot 8b, serial;
the logs are UTF-16 — `iconv -f UTF-16LE`): `Content-loop watchdog fired
(period-2…64 repeat); rolled back to boundary, re-steering content_tokens=570
dropped=70 rollback=1` → `rollback=2` → `ending response early (rollback
declined)`, three times per request, identically on all three runs. The
digit-normalising detector (`detect_content_token_loop_normalized_with`,
`scheduler/helpers.rs`) collapses every number to one sentinel, so a plain
list of distinct numbers `[8, 3, 1, 6, 4, 2, 7,` reads as a period-2
`N ,` loop with ≥4 repeats. Engine behaviour, not a port defect — it fires
on GB10 for the same request — and it also means the `Done:` tok/s of these
requests include two 70-token rollbacks each (both arms equally). Fixed
in `07ad46661` (numeric/punctuation exemption mask).

### ST-995 leg (boot 9 serve, 2026-09-20)

Golden draw, `bfcl-subset` canonical no-overrides against :8095, n=995,
seed 42, temp 0. **Overall 82.21 / normalized 82.40, wall 41,689 s
(~11.6 h), zero faults** (no `status 719` / `hipErrorLaunchFailure` /
`context is destroyed` in the serve log). Run record
`run-1789918248980432100.json` (harvested into this log dir). Fingerprint:
binary sha256 `4F1943CB4D7FE9…9174`, worktree `ed8e0dd2c`, boot-7g MTP
recipe (util 0.90, seq 8192, prefill 2048, kv bf16, drafts 1, slots 0).
MTP engagement over 998 `Done:` lines: mean_na avg 0.707 / median 0.80;
119/998 requests ran `serial=1.00` (MTP declines above the QSA inert
bound — expected, not a fault). Scorer verdict: *"not an MLPerf
submission checkpoint, so the floor does not gate this run"* (the 83.64 /
85.32 floors are MLPerf-edge numbers derived for Qwen3.6-27B).

| subset | acc |
|---|---|
| irrelevance | 91.67 |
| live_irrelevance | 88.64 |
| live_multiple | 80.95 |
| live_parallel | 56.25 |
| live_parallel_multiple | 62.50 |
| live_simple | 92.00 |
| multiple | 84.68 |
| parallel | 77.42 |
| parallel_multiple | 82.26 |
| simple_java | 58.06 |
| simple_javascript | 70.97 |
| simple_python | 90.73 |

(categories: hallucination 90.15 / live 77.65 / non_live 79.40).
Cross-platform observation only — not an A/B (OS, driver, ROCm, recipe all
differ): the same checkpoint on GB10 scored 83.52 / 82.45 serial at 16K
(see `kernels/gb10/qwen3.8-flash-next/BENCH.toml`), so Windows is −1.31
overall / −0.05 normalized against it.

### Post-leg A/B (boots 11–17, 2026-09-20, binary `59160469…` @ `f18cfffe5`)

All arms: `ATLAS_CONTENT_LOOP_WATCHDOG=0`, MinHeap+unittest prompt,
`reasoning_effort:"none"`, `max_tokens 1024`, temp 0, n=3, PONG warm-up;
every request `finish=stop` (~700 tokens). `Done:` lines are
server-attested.

| boot | binary | arm | hipBLASLt | decode tok/s | TTFT ms |
|---|---|---|---|---|---|
| 11 | new | MTP | **ON** (`GEMM live` + `pre-warmed`) | 17.4 / 17.2 / 17.4 | 3082 / 3066 / 3061 |
| 12 | new | SERIAL | ON | 8.25 / 8.44 / 8.34 | 2972 / 2991 / 2948 |
| 16 | new | MTP | OFF (`cublasLtCreate failed` → Atlas fallback) | 17.5 / 17.5 / 17.5 | 3140 / 3205 / 3201 |
| 17 | new | SERIAL | OFF | 8.13 / 8.20 / 8.14 | 3107 / 3114 / 3072 |
| 13 | old `4F1943CB` | SERIAL | n/a (stub) | 8.16 / 8.13 / 8.05 | 2979 / 2949 / 2916 |
| 14 | old `4F1943CB` | MTP | n/a | **contaminated — discard** | — |

Conclusions (observed under this fingerprint, n=3):

- **hipBLASLt ON vs OFF: no measurable effect on this workload**
  (11↔16, 12↔17 within noise). Correct — the prefill profile shows the
  time is NOT in BF16 GEMMs (below).
- **The M≤8 GEMV arm has no measurable serial effect** (17↔13: 8.1–8.2
  both) — the M=1-pipelined-GEMM hypothesis for the serial decode floor
  is REFUTED. The GEMV arm remains the more efficient route for small-M
  BF16 projections, it just isn't the bottleneck.
- **MTP vs serial is ~2.1×** (11↔12: 17.4 vs 8.3; mean_na ≈ 0.95,
  tok_step ≈ 1.95) — consistent with boot 7g/9.
- **Boot 14 is invalid.** Its serve process died mid-weight-load at
  17:06:14Z (status-719 alloc cascade during shard 6/11, backend drop
  reclaimed 83,658 allocations) and never served — the log contains zero
  `Done:`/request lines. The three `boot14-minheap-*.json` responses
  (written 17:10–17:13Z) were answered by a lingering earlier serve still
  bound to :8095, so their ~60 s walls and ~12 tok/s are unattributable.
  Boot 9 ran this same old binary at 16.7 tok/s under the same recipe —
  the GEMV-arm-vs-old comparison on the MTP path is UNMEASURED here, not
  +44%.
- Parity: 12≡17 and 17≡13 byte-identical ×3 (serial arms identical across
  binaries and backends — GEMV arm is numerics-preserving); 11v12 differ
  at char 2511 and 11v16 at char ~2499–2508 inside the test-data list
  (spec-vs-serial divergence class, same as boots 7g/8b).

### Prefill attribution — superseded by GPU-timed trace (see below)

The boot-10 `ATLAS_PROFILE_PREFILL` span numbers (`submit=2.94 s,
gpu_exec=74.5 ms`) are real host-side timings, but `gpu_exec` does **not**
measure device execution — it is ~36× smaller than the HIP-event truth
below. The phase split (`grouped_gate_up` 1,383 ms + `grouped_silu_down`
269 ms ≈ 79% of TTFT at N=20) is directionally right: those spans are
GPU-paced, not submit-paced.

### GPU-timed launch trace (`0b03c82d3`, binary `343DB554…`)

`ATLAS_TRACE_LAUNCH_SYNC=1` now brackets every `hipModuleLaunchKernel`
with hipEvents (`gpu=` per launch) and records inter-call host time
(`host_before=`). Same recipe, serial arm, hipBLASLt live.

**Decode is GPU-bound.** Steady state (63 steps, ~964 launches/step):

| | per step |
|---|---|
| wall (sync mode) | 196 ms mean / 194 median (async: 92 ms) |
| Σgpu | **89.1 ms** |
| Σhost_before | 64.2 ms |

The async step (~92 ms) ≈ Σgpu (89 ms): the ~95 µs/launch inter-launch
gaps in the plain trace were **queue backpressure mirroring kernel
durations**, not submit cost. A standalone launchbench (`launchbench.cpp`,
`hipModuleLaunchKernel`, 5,000 launches) measured HIP/WDDM submit at
**0.6 µs/launch** — the runtime is not the wall at all. Host dispatch
(~64 ms/step) hides almost entirely under GPU execution; the only exposed
per-step host item is `hc_expand`'s 14.8 ms host_before at the
emit/dispatch boundary. The ~3 D2H + 4 syncs per step (QSA top-k, PLE
host hash) cost only ~0.4 ms. **Graph replay therefore caps at ≈30%
decode gain** — it removes the host half but not the ~89 ms of GPU work;
on the prefill the same logic caps lower because GPU dominates even more.

Top decode kernels by Σgpu/step (geometry for gfx1151 occupancy, 40 CUs):

| kernel | grid | block | n/step | µs each | ms/step |
|---|---|---|---|---|---|
| `dense_gemv_bf16` | 4096×1 | 256 | 36 | 415 | 14.9 |
| `dense_gemv_bf16` | 640×1 | 256 | 36 | 267 | 9.6 |
| `dense_gemv_bf16` | 62020×1 | 256 | 1 | 5774 | 5.8 |
| `hc_pre_down` | 1×10 | 1024 | 97 | 150 | 14.6 |
| `hc_pre_finish_x4` | 1×80 | 128 | 97 | 140 | 13.6 |
| `moe_expert_silu_down_shared` | 320×11 | 128 | 48 | 138 | 6.6 |
| `moe_expert_gate_up_shared` | 80×11×2 | 128 | 48 | 134 | 6.4 |
| `gated_delta_rule_decode_f32` | 48×1 | 128 | 36 | 123 | 4.4 |
| `hc_post` | 1×1 | 256 | 96 | 23 | 2.2 |
| `hc_pre_stage` | 1×1 | 1024 | 97 | 20 | 1.9 |
| `w4a16_gemv_qg` | — | 256 | 12 | ~133 | 1.6 |
| `moe_topk_softmax` | 1×1 | 256 | 48 | 23 | 1.1 |
| `w4a16_gemv_sw` | 320×1 / 64×1 | 256 | 12/48 | 108 / 8 | 1.7 |
| `paged_decode_attn` | 24×1 | 256 | 12 | 23 | 0.3 |

`dense_gemv_bf16` = ~30 ms/step (34% of GPU time) — the 36 BF16 GDN
projections plus the 62,020-row lm_head GEMV; the `hc_*` family is real
GPU work (~32 ms/step across four kernels), not launch noise.

**Prefill is GPU-bound too.** 44-token MinHeap chunk: 8,529 launches,
wall 3,820 ms (sync trace; async 3,034 ms): Σgpu **2,754 ms (72%)**,
Σhost_before 627 ms. The grouped MoE GEMMs alone — 116
`moe_w4a16_grouped_gemm_ptrtable`/`fused_gate_up` calls — are
**Σgpu 1,754 ms (46% of prefill), ~15 ms GPU per call** with only
~105–139 µs host_before each. The cost is real device time: every routed
expert block walks the full K loop for ≤1 real row at N≤44, so the call
cost is nearly flat in token count — which is why the fix below is
routing, not kernel tuning.

The earlier `gpu_exec ≈ 77 ms` must not be quoted as device time — direct
HIP-event sums contradict it (89 ms/step decode, 2,754 ms prefill), and
the async-vs-sync step-wall discriminator (92 → 196 ms under forced sync)
proves the submit path was overlapping GPU execution all along.

First hypothesis (compute-bound on padding): M_TILE=64 with 4 warps × 16
rows means an expert holding 1–3 rows still ran all 4 warps' WMMA work
(discarded by the epilogue). `6cc12d03d` skips MMA on warps whose 16-row
slab lies entirely beyond `M_expert` in all six grouped kernels +
`common/moe_fp8_grouped_gemm.cu`. Measured (boots 18/19, binary
`8CACFAA1…`, same fingerprint as boots 11–17): **content byte-identical**
to boot 12 ×3 (correctness gate), grouped_gate_up phase total
1,383,150→1,320,037 µs (−4.6%), `submit` 2.936→2.74–2.77 s, `gpu_exec`
74.5→77.0 ms (noise), TTFT 2948–2991→2924–2943 ms, decode 8.2–8.3
(unchanged). The idle-warp MMA removed real padding work but only ~5% of
the phase — the ~15 ms/call floor is the expert-block K loop, not warp
padding. The warp-skip is kept anyway: dead-work removal, bit-identical.

### Fix 1: short-prefill MoE routing threshold (`0a512fdc8`)

For `cfg!(atlas_hip)` only, NVFP4 MoE prefills below
`ATLAS_HIP_MOE_GROUPED_MIN_TOKENS` (default 1024, measured crossover below) route to the per-token
`forward_batched` path like the bf16/fp8 arms — at N<256 the grouped
kernel pays ~15 ms/layer for ≤1 real row per expert block. NVIDIA path
byte-unchanged. Serial arm, hipBLASLt live, default GDN, temp 0
(max_tokens 256; TTFT is the metric):

| prompt tokens | thr=0 (grouped) | thr=256 | thr=4096 (batched) |
|---|---|---|---|
| 20 (PONG) | 2104 / 2003 ms | **640 / 597** | 632 / 565 |
| 44 (MinHeap) | 2975 / 2918 | **1060 / 959** | 1035 / 965 |
| 1150 (big) | **14522 / 14171** | 15457 / 15731 | 15635 / 15634 |

Batched routing cuts 20–44-token TTFT by ~66–71%; at 1,150 tokens the
grouped path wins by ~9% — crossover is between 44 and 1,150. Output parity:
grouped-vs-batched content for the same prompt diverges at char ~102–135
(docstring formatting — the known grouped/batched accumulation-order
class); thr=256 vs thr=0 content is byte-identical at 1,150 tokens (both
grouped — confirms the routing actually engaged).

Crossover pinned (`a387b38d`, same setup, boots C0/C4096 — prompt_tokens
from `usage`, n=2):

| prompt tokens | grouped (thr=0) | batched (thr=4096) |
|---|---|---|
| 266 | 7174 / 6880 ms | **4401 / 3991** |
| 708 | 10781 / 10869 | **9741 / 9849** |

Batched still leads at 708 (−10%) and loses at 1,150 (−9%) → crossover
~900 tokens; default set to **1024** (nearest power of two).

### Fix 3: `hc_pre_down` LDS staging (`3bf401256`) — bit-exact, ~neutral

`hyper_connection.cu` under strix-hip is now a real file (was a gb10
symlink): `hc_pre_down` stages `normed[t]` (40 KB FP32) into static LDS
once per block instead of letting all 320 rank rows re-read it from
L2/DRAM; per-lane FMA order and the shfl reduction are unchanged, and a
`hc_dim > 10240` runtime guard falls back to the global-read loop.
Measured (sync trace, serial, binary from `3bf401256`): decode
`hc_pre_down` 150 → **147.1 µs** (sum 14.6 → 14.3 ms/step), Σgpu/step
89.1 → 86.5 ms — **within noise; the re-reads were already being served
by L2**, so the staging bought ~2% at best. Untraced: serial 8.4/8.4/8.5
tok/s (was 8.2–8.5), MTP 17.6/17.7/17.7 mean_na 0.95 (was 17.3–17.4) —
decode unchanged. Parity gate: serial MinHeap output is byte-identical
to the pre-change batched-path output (bootB256, 1061 chars); it differs
from boot12 (grouped prefill) only at char 135, the known
grouped-vs-batched divergence — attributable to the routing threshold,
not this kernel. Kept: removes ~13 MB/launch of redundant traffic at
zero numeric cost.

### Fix 2 attempt: GDN requantization A/B — cannot boot (inconclusive)

`ATLAS_QWEN4EXP_BF16_GDN=0` requantizes the 36 GDN projections to NVFP4
at load — the hypothesis was −6 GB device plus relief on the 30 ms/step
`dense_gemv_bf16` decode cost. **On winbox it never reaches `Server
live`:** the requant arm consumes ~82.7 GB at the KV-budget point
(serial diag: free-now 13.3 GB vs ~15 GB on the default arm; the in-tree
GB10 table shows the same +5.8 GB pre-KV — requant staging is *heavier*
at build time even though final weights are smaller). Serial at util
0.86 fails the budget check outright (84.3 committed > 82.6 budget);
at 0.89 it passes the check then `cuMemAlloc_v2` fails `status 2` on a
64 MB request — the 84.5 GiB resident wall. MTP at 0.90 passes the check
then dies 719 in the alloc run. The in-code comment's GB10 measurement
(decode 2.207 vs 2.188 tok/s, i.e. ~1%) plus these load failures means
the option is unbootable here without shedding ~4 GB elsewhere (smaller
seq len, vision encoder off, or drafter off); the A/B is recorded as a
load failure, not a performance result.

### single_gpu_suite (boot 15, new binary, MTP, hipBLASLt live)

`--skip-longctx` (suite's 16000-token case > `--max-seq-len 8192`):
**coherence 3/3 PASS, tool calls 2/2 PASS, fibonacci 0/1 FAIL** — the
fibonacci failure is the Windows harness shim (`Shim: Could not determine
if target is a GUI app`) failing to exec the generated code locally, not
a model failure. Avg TPS 8.9. Results `gate-boot15.json`.

### #44 DFlash smoke (main checkout `atlas-main-verify`)

`feat/strix-windows-dflash-main` (`da943853`+`093f1ae1`), `DFLASH=1`,
drafter `C:\Users\azeez\models\dflash2` (the HF cache `refs/main` is
dangling — local path required): **`SMOKE OK  finish=length  tokens=63`**,
coherent output, `DFlash γ=8 (source: cli)`, drafter installed as the
active proposer. Repo restored to `da943853`.

## Known gaps / next levers

- **ST-995 leg**: DONE — 82.21 / 82.40, n=995, zero faults (see Measured).
- **`tests/single_gpu_suite.py`**: ran (boot 15) — coherence 3/3,
  tool calls 2/2, fibonacci harness-shim failure (not model).
- **hipBLASLt**: DONE and measured — the cuBLASLt entry points are now a
  dynamically-loaded hipBLASLt translation layer (`080813f3a`, gated by
  `ATLAS_HIPBLASLT=0`; `f18cfffe5` routes any cuBLASLt failure to the
  Atlas fallback). Verified live on Windows (`cuBLASLt BF16 GEMM live`,
  `cuBLASLt pre-warmed`). On THIS decode/short-prefill workload it is
  perf-neutral (11↔16, 12↔17 within noise) — the prefill profile shows
  the time is in the grouped-MoE kernels, not BF16 GEMMs. Its value is
  large-M prefill (AzeezStrix probe: 32–34 TFLOPS BF16 at M=2048 vs
  ~3 TFLOPS pipelined WMMA), which this workload doesn't exercise.
- **TTFT is GPU-bound in the grouped MoE GEMMs** (Σgpu 1,754 of 2,754 ms
  at N=44): FIXED for short prefills by the HIP-only routing threshold
  (`0a512fdc8`, `ATLAS_HIP_MOE_GROUPED_MIN_TOKENS`, default 1024 measured) — −66–71%
  TTFT at 20–44 tokens, crossover vs grouped between 44 and 1,150 tokens.
  The remaining lever at mid-length prompts is the grouped kernel's
  ~15 ms/call K-loop floor itself. Graph replay caps at ≈30% on decode
  (host ~64 ms/step hides under ~89 ms GPU); it is not the decode fix.
  HIP/WDDM submit itself is 0.6 µs/launch (launchbench) — never the wall.
- **GDN requant (`ATLAS_QWEN4EXP_BF16_GDN=0`) does not fit on winbox** —
  +~5 GB pre-KV over the BF16 arm leaves nothing under the 84.5 GiB wall
  at seq 8192 (serial fails the KV check at 0.86, alloc status-2 at 0.89,
  MTP 719s at 0.90). Would need ~4 GB shed elsewhere first.
- **MoE FP4 arms** (`moe_w4a16_*_fp4`, behind `ATLAS_HOLO_MOE_GATEUP_FP4`)
  and the other expected-absent entries marked "UNPORTED candidate" in
  MODEL.toml are the perf-recovery list.
- **641-token `length` oddity**: resolved — content-loop watchdog false
  positive on numeric lists; fixed in `07ad46661`.
- A 128 GB **Linux** Strix box would sidestep the WDDM wall entirely;
  AzeezStrix is a 64 GB SKU and can never run this model.

## Claims status (measurement-discipline language)

- **Measured, n=3, fingerprinted:** MTP K=2 decode 17.0/16.7/17.3 tok/s
  on the MinHeap harness (server-attested `Done:` lines with
  mean_na/tok_step), re-confirmed single-shot on a second binary (boot 9,
  audit-metadata build, no dangerous flag). The serial control's reported
  6.7/6.3/6.4 includes two watchdog rollbacks per request (see the supersede
  note); the rework-corrected ≈ 8 tok/s is DERIVED, so the ≈ 2.1× speedup is
  "observed under fingerprint F", pending a clean re-run.
- **Observed, single fingerprint:** thinking-on MTP run (11.3 tok/s,
  serial-floor regime, n=1); boot-6 serial smoke values (PRELIMINARY,
  single-shot per prompt class).
- **Derived (arithmetic shown, not separately measured):** the ~84.5 GiB
  resident wall (from `status 2` at tracked 84.5 and the alloc-failure
  pattern); the VGM-usable model `min(VGM,32) + ~55–60% of host RAM`.
- **Verified mechanism (instrumented):** the over-commit-then-719 failure
  mode — `ATLAS_TRACE_LAUNCH_SYNC` cleared every one of 850,953 launches in
  boot 7e; the context failed in allocation, before `Server live`.
- **Measured, fingerprinted:** ST-995 82.21 / 82.40 (n=995, 0 faults,
  `run-1789918248980432100`); post-leg A/B boots 11/12/16/17/13 (hipBLASLt
  perf-neutral, GEMV arm serial-neutral, MTP ≈ 2.1× serial); boot 14
  marked contaminated (serve died mid-load, responses served by a
  lingering instance).
- **Measured, fingerprinted (`0b03c82d3` + `0a512fdc8`, binary
  `343DB554…`/rebuilt):** GPU-timed decode profile (Σgpu 89 ms/step,
  kernel table above); GPU-timed prefill (Σgpu 2,754 ms, grouped MoE
  1,754 ms); HIP submit cost 0.6 µs/launch (launchbench); MoE routing
  threshold TTFT matrix (boots B0/B256/B4096, serial, n=2 per cell).
- **Negative/inconclusive results (recorded as such):** GDN requant arm
  unbootable on winbox at the standard recipe (memory, not a kernel —
  see Fix 2); launchgap-derived "submit-bound" and `gpu_exec=77 ms`
  readings are SUPERSEDED by the HIP-event trace.
- **Not yet claimed:** MTP-vs-serial parity beyond char-2508-class
  divergence (the divergences are observed, unattributed).
