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
requests include two 70-token rollbacks each (both arms equally). Fix
tracked separately: require the anchored pattern to contain at least one
token that is neither numeric nor pure punctuation (a real `1, 1, 1, …`
runaway is still caught by the exact detector).

## Known gaps / next levers

- **ST-995 leg**: running on winbox (detached, canonical
  `bfcl-subset` no-overrides against :8095; marker + fingerprint sidecars in
  the log dir). Result pending — see Claims status.
- **`tests/single_gpu_suite.py`**: not yet run on this port.
- **hipBLASLt**: TheRock 10.0.0 ships `libhipblaslt.dll`/`rocblas.dll` for
  gfx1151 — a real cuBLASLt→hipBLASLt shim would replace the Atlas GEMM
  fallback on every BF16 projection (whether hipBLASLt has gfx1151 BF16
  kernels is unverified). Biggest known perf lever.
- **MoE FP4 arms** (`moe_w4a16_*_fp4`, behind `ATLAS_HOLO_MOE_GATEUP_FP4`)
  and the other expected-absent entries marked "UNPORTED candidate" in
  MODEL.toml are the perf-recovery list.
- **TTFT ~3.2 s** through the scalar/pipelined Atlas GEMMs instead of
  cuBLASLt; the remaining prefill cost profile is unmeasured.
- **641-token `length` oddity** above.
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
- **Not yet claimed:** ST-995 accuracy on this port (leg in flight);
  MTP-vs-serial parity beyond char-2508-class divergence (the divergences
  are observed, unattributed); any TTFT claim.
