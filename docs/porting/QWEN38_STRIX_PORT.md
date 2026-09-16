# Qwen3.8-27B on Strix Halo (gfx1151)

How the Qwen3.8-27B target reaches AMD Strix Halo, on both of Atlas's AMD
toolchains, and what has and has not been measured on it.

The Windows half of this port has its own document —
[`STRIX_WINDOWS_HIP.md`](STRIX_WINDOWS_HIP.md) — because its failure modes are
entirely different. This one covers the target itself and the Linux legs.

## Quick start

With ROCm installed at `/opt/rocm` and Rust available:

```bash
ATLAS_HIPCC_WORKERS=3 CARGO_BUILD_JOBS=3 ./build-amd.sh
NUM_DRAFTS=1 ./serve-amd.sh /path/to/Qwen3.8-27B-NVFP4
```

For a side-by-side ROCm install, point both commands at the same runtime and
build directory:

```bash
export ATLAS_ROCM_HOME="$HOME/rocm-10.0.0-tarball/install"
export CARGO_TARGET_DIR=target-rocm10
ATLAS_HIPCC_WORKERS=3 CARGO_BUILD_JOBS=3 ./build-amd.sh
NUM_DRAFTS=1 ./serve-amd.sh /path/to/Qwen3.8-27B-NVFP4
```

The server listens on port 8081 by default. Override it with `PORT=9000`.

## The target compiles no kernels of its own

Two definitions, byte-identical to each other:

```
kernels/strix/qwen3.8-27b/MODEL.toml       SCALE toolchain (Linux only)
kernels/strix-hip/qwen3.8-27b/MODEL.toml   native HIP  (Linux and Windows)
```

Both set `kernel_source = "qwen3.6-27b"`. Verified against the checkpoint on
disk (`unsloth/Qwen3.8-27B-NVFP4`, snapshot `7d6f8d4d`): every field of
`text_config` is identical to `unsloth/Qwen3.6-27B-NVFP4`'s — layer counts and
types, hidden/head/kv dims, vocab, rope parameters, `attn_output_gate`,
`mtp_num_hidden_layers`. The two checkpoints differ only in weights and in
`quantization_config`. There is no 3.8-specific kernel work on either backend,
exactly as on gb10.

### `match_names` is load-bearing

Because `config.json` cannot tell 3.8 from 3.6 — same `model_type` `qwen3_5`,
same `hidden_size` 5120, same every numeric field — kernel-target resolution
hits an **exact** tie and breaks it by matching `match_names` needles against
the checkpoint reference: the HF id, `--model-name`, and the model directory.

An unbroken tie is a hard startup error, never a build-order pick. The practical
consequence is that **the served model name selects the kernel target**. Serving
3.8 weights under the 3.6 name resolves `qwen3.6-27b`, whose MODEL.toml carries
a different MTP depth and different sampling defaults — and because 3.8
legitimately reuses 3.6's kernels, nothing fails loudly. It just serves wrong.
This is exactly the bug the Windows recipe had; see that document.

### MTP is 1 here, and 0 on gb10

`kernels/gb10/qwen3.8-27b` sets `mtp_layers = 0`. That is wrong for this
checkpoint and matters more on Strix, whose serve recipe passes
`--speculative --num-drafts N` — dead weight without an MTP head. The head
ships (`text_config.mtp_num_hidden_layers = 1`, and `model_mtp.safetensors`
carries 15 `mtp.*` tensors). A 0 here produces no error and no warning, just a
serve that quietly ignores `--speculative` and decodes at 1x.

## Building

`./build-amd.sh` — `ATLAS_TARGET_HW` selects `strix-hip` (default, native HIP,
needs only ROCm and cargo) or `strix` (SCALE, needs `$SCALE_HOME`).
`ATLAS_TARGET_MODEL` defaults to `*`, which builds every target under
`kernels/$ATLAS_TARGET_HW/` into one binary. That default is worth keeping:
3.8 reuses 3.6's tree, so the marginal cost is small, and it is the only
configuration that actually exercises the `match_names` tie-break at serve time.

### `ATLAS_HIPCC_WORKERS`

The kernel-compile pool is sized from `available_parallelism()`, which is 32 on
the Strix box. Strix is a 64 GB APU with no discrete VRAM, so 32 concurrent
`hipcc` processes exhaust system memory and wedge the machine in the OOM killer
before the kernel set finishes. `ATLAS_HIPCC_WORKERS` caps the pool without
capping cargo.

This override existed in the pre-restoration history and was never carried onto
main; it is restored here. Unset, behaviour is unchanged.

## Validated

### ROCm 10 + K=2 target-verify candidate — 2026-09-03

On AzeezStrix, binary `4c0a8451…`, checkpoint revision `7d6f8d4d…`:

- the copy-paste ROCm 10 build above completes successfully;
- M=2 BF16 GEMV is bit-identical to two M=1 calls at a partial-block guard and
  all eight real Qwen3.8 decode shapes; RMSNorm and GDN device oracles pass;
- `agentic`-style 1,024-token n=3 MTP median is **8.916 tok/s**
  [8.901, 8.954], versus **3.992 tok/s** [3.884, 4.184] on the clean pre-patch
  ROCm 7.13 binary; target verify forward fell from ~497 ms to ~208 ms;
- the MTP BFCL-70 diagnostic completed all 70 at 81.43 overall / 77.70
  normalized. This is a diagnostic subset, not the pinned ST-995 gate.

### Build — Linux, native HIP, 2026-08-25

`AzeezStrix`, gfx1151, Ubuntu 24.04, ROCm/HIP 7.13, 61 GB.

```
ATLAS_HIPCC_WORKERS=3 CARGO_BUILD_JOBS=3 ./build-amd.sh
  Finished `release` profile [optimized] target(s) in 2m 37s   exit 0
```

Built **while a 56 GB serve was resident on the same box**, with available
memory never dropping below 4 GB. That is the `ATLAS_HIPCC_WORKERS` cap doing
its job, and it is the practical difference between being able to build on this
hardware and not.

### Serve and resolution — Linux, 2026-08-25

`./serve-amd.sh` with its defaults (the checkpoint default is Qwen3.8-27B-NVFP4):

```
Selected kernel target: (gfx1151, qwen3.8-27b, nvfp4) (95 modules)
  — quant compat: kernel=nvfp4 model=fp8 OK
Dense MTP head ready (FP8 e4m3 projections + dense gate/up/down MLP)
Qwen3.6 vision encoder loaded: depth=27, hidden=1152, heads=16
KV cache: 60.0 GB total x 86% util = 51.6 GB budget; 44.4 GB pre-KV
  + 5.9 GB reserve -> 1.4 GB for KV -> 22384 max KV tokens
Server live and ready at 127.0.0.1:8081 running unsloth/Qwen3.8-27B-NVFP4
```

The binary carries **every** strix-hip target (`ATLAS_TARGET_MODEL=*`), so 3.6
and 3.8 are both embedded and resolution had to break the tie on `match_names`.
It picked 3.8. That is the assertion this port rests on.

### Performance — `quick-speed-bench`, Linux

| | decode (server) | TTFT | TPOT |
|---|---|---|---|
| **Linux, ROCm 7.13** | **22.0 tok/s** | **1452 ms** | 45.45 ms |
| Windows, ROCm 6.4 | 17.9 tok/s | 7170 ms | 55.73 ms |

n=5 each, isl 60 / osl 128, single stream. The Linux spread is 22.0-22.1.
Linux is ~23% faster on decode and ~5x on TTFT — consistent with
[`STRIX_WINDOWS_HIP.md`](STRIX_WINDOWS_HIP.md)'s "why the Linux numbers won't
reproduce". Both are **measurements, not gates**: Qwen3.8-27B carries no
committed `BENCH.toml` thresholds, so a run on it baselines rather than gates,
and inherits neither 3.6's floors nor the MLPerf floor.

### BFCL: the leg ran, and it found a correctness bug rather than a score

> **Superseded 2026-09-13.** On the fp8d tree (`8eedc21cb`, ROCm 10.0.0,
> Windows 11, gfx1151) serving `nvidia/Qwen3.8-27B-NVFP4`, the full pinned
> ST-995 draw (n=995, seed 42, temp 0, no overrides) completed cleanly —
> **overall 83.32 / normalized 78.70**
> (`~/.atlas/runs/bfcl-subset/run-1789300938267487600.json`; the staged scorer
> crashed reading `responses.jsonl` under the cp1252 locale and was re-run
> under `PYTHONUTF8=1`, which is now fixed in-tree). The AST checker scored
> real tool calls, so the tool path works on this tree with this checkpoint.
> Several variables changed against the `!!!!!!` reproduction below —
> checkpoint (unsloth→nvidia), kernel set (fp8d additions), ROCm 6.4→10, and
> the serve recipe — so *which* change closed it is not recorded; re-running
> the two-request probe on the unsloth checkpoint under this tree is the
> one-command attribution. 87 unresolved kernel lookups remain at boot
> (was 94 here); each silently binds to handle 0. The rest of this section
> describes the state at the commit it was written, kept for the record.

`bfcl-subset` at a reduced draw (`non_live_pct=4 live_pct=1 hallucination_pct=1
subset_floor=2`, n=70 across 12 subsets) completed all 70 samples on **both**
platforms — 1020 s on Linux, 2528 s on Windows. The harness itself flags the
draw: *"n=70, not the pinned 995 — this run is NOT comparable to this draw's
baseline"*.

**No accuracy number should be taken from it, because the model emitted zero
tool calls.**

```
Linux   : 70 responses, 0 with tool calls
Windows : 70 responses, 0 with tool calls
```

Identical, subset for subset. The nominal Linux score — `overall_accuracy 14.29`,
`non_live 0.0`, `live 0.0`, `hallucination 100.0` — is an artefact: the two
irrelevance subsets score 100 precisely *because* no call is made, and every
category that requires one scores 0.

Reduced to a single request, plain chat is fine and the tool path is not:

```
no tools : "I do not have access to real-time data, so I cannot provide the
            current weather conditions in Paris right now. ..."
+ tools  : "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"   tool_calls: null
```

Degenerate on the FIRST request, so it is not the cross-request SSM prefix reuse
that `first_run.ps1` documents. Two hypotheses tested and **rejected**:

* `ATLAS_SSM_TAIL_MIDCHUNK=0` — no change (this is the lever the Windows recipe
  sets for a similar-looking symptom; it is not this one).
* tool grammar enabled instead of `--disable-tool-grammar true` — no change.

Because it reproduces identically on both platforms, it is in the shared port —
the kernel set or the model config — not in either platform recipe.

### It is pre-existing, and this port does not introduce it

The decisive test. The pre-restoration Avarok worktree on the same box
(`/workspace/.wt-38trial`, the build that had been serving Qwen3.8-27B
continuously for six days) was sent the identical two requests:

| request | pre-restoration Avarok build | this port (Linux) | this port (Windows) |
|---|---|---|---|
| plain chat | coherent | coherent | coherent |
| same prompt + `tools` | `!!!!!!` | `!!!!!!` | `!!!!!!` |

Byte-identical prose on the plain request, byte-identical degeneration on the
tool request, across three builds and two operating systems. **The tool path was
already broken in the source tree this port comes from.** Nothing here regresses
it, and there is no version of this port that would have shipped it working —
the behaviour is a property of the checkpoint against the strix-hip kernel set,
not of the merge.

That is what makes these legs validated: the port reproduces its origin exactly,
on both toolchains and both operating systems. It is also why the BFCL leg
cannot yield an accuracy number until the kernel-set gap below is closed — on
any tree, old or new.

### The strix-hip kernel set is incomplete, and it is not benign

The A/B that makes this concrete: `unsloth/Qwen3.6-27B-NVFP4`, the *certified*
checkpoint, on this same tree and the same kernel set, **cannot serve at all**:

```
Selected kernel target: (gfx1151, qwen3.6-27b, nvfp4) (95 modules)
Error: Failed to build model
Caused by: Kernel lookup dequant_nvfp4_bf16::dequant_nvfp4_to_bf16:
           Module load failed: Module 'dequant_nvfp4_bf16' not loaded
```

(Resolution picked 3.6 correctly, which is the tie-break working in both
directions.)

So the 94 unresolved lookups are not the harmless bookkeeping the
`--dangerously-allow-unresolved-kernel-lookups` comment implies. On this tree
they are a hard failure for 3.6 and, for 3.8, a silent wrong answer on the tool
path. **Closing the kernel-set gap is a prerequisite for any BFCL number on
Strix**, and it is kernel work, not port work.

### Scoring has its own prerequisites

Both platforms completed inference and then failed to *score*, differently:

* Linux — `ModuleNotFoundError: No module named 'soundfile'`, a transitive
  import of `qwen_agent` missing from `bfcl-eval`'s dependency set. Fixed by
  installing it into the provisioned venv; `responses.jsonl` is kept, so the run
  rescored without re-running inference.
* Windows — `ImportError: DLL load failed while importing _tiktoken: An
  Application Control policy has blocked this file.` A machine security policy,
  deliberately not worked around.

### The fallback caveat applies to every Strix number

The gfx1151 kernel set is much smaller than gb10's, so a large number of
dispatch sites resolve to a fallback — 94 unresolved lookups for qwen3.8-27b,
against the 92 `serve-amd.sh` documents for 3.6 on the same tree. These are
pre-existing: the kernel audit landed on main after the Strix branch forked, and
the certified 3.6 submission was produced under exactly the same ones. Both
serve recipes pass `--dangerously-allow-unresolved-kernel-lookups`.

Closing that gap — compiling the missing kernels, or declaring them in
`MODEL.toml` `[expected_absent]` with stated reasons — is follow-up work, and no
Strix performance number is final until it is done.

## 2026-09-02 — the tool-path bug is fixed, and every kernel is drift-measured

### Root cause of the "zero tool calls" leg above

Not the kernel-set gap and not the checkpoint: the BF16 FFN prefill arm
dispatched `gemm_tc::dense_gemm_tc`, whose gfx1151 WMMA port leaves ~half of
its output tile unwritten (NaN-sentinel oracle:
`crates/spark-model/examples/dense_gemm_bf16_oracle.rs`, exact=0.500,
max_abs=inf) and ran ~1000x slow in situ (2.4 s/GEMM at M=823 N=17408 K=5120
in serve vs 2.2 ms standalone). The unsloth checkpoint's final-eight per-row
FP8 FFN layers are the ones that route through that arm (`set_bf16_weights`),
so layers 43-48 emitted half-stale outputs, the final hidden state collapsed
to noise, and the first predicted token became `<|audio_pad|>` (248076) with a
period-2 decode loop behind it. 16-token prompts never crossed a partial
128-row tile boundary and stayed clean — which is why "Answer exactly: Paris"
passed while every real prompt failed.

The fix routes the BF16 FFN prefill arm through `gemm::dense_gemm_bf16_pipelined`
(CPU-oracle-validated: row cosine >= 0.99999991 vs the scalar kernel AND vs an
f32 CPU reference at M in {16,128,129,512,513,1024,2049} x N in {5120,8192,
17408}, 1.9-3.5 TFLOPS); `dense_gemm_tc` is quarantined behind
`ATLAS_FFN_BF16_PREFILL_TC=1` and is never a silent fallback. The same broken
kernel has other call sites (o_proj multi-seq, paged, cache_skip_v4/mla) that
this model's active path does not reach — flagged, not changed.

### Kernel drift audit — every kernel the serve path dispatches

`~/q38-kernel-drift-audit.log` on AzeezStrix (2026-09-02), 19 kernels vs CPU
f32 references or the bit-verified scalar kernel, same tree as the serve
binary. Clean (cos >= 0.999999 or bit-identical): rmsnorm (all variants),
rope, conv1d-strided (byte-identical), GDN split4 recurrence, contiguous and
paged BF16 attention, the whole w4a16 NVFP4 family (t/k64/m128 bit-identical
to each other; m128 vs CPU max|delta| 4.9e-4), the BF16 scalar GEMM
(bit-identical to CPU), the BF16 pipelined GEMM, the BF16 decode GEMV M=1
(new `dense_gemv_bf16_oracle`, mean_rel <= 1.2e-5 at all eight Qwen3.8 decode
shapes), and w4a16 batch bitparity (byte-identical).

Measured drift, quantified:

* `w4a16_gemv_dp4a` — the NVFP4 decode GEMV every token goes through —
  cos 0.999991 but **mean relative error 5.58%** (max 45x) at N=2048 K=4096.
  The int8-DP4A dot product is the drift source; llama.cpp dequantizes and
  accumulates in fp16/fp32 instead. First replacement candidate if the
  cross-library check shows decode divergence.
* `w8a16` / `w8a16t` — cos 0.999997, mean_rel 1.1-1.2%, max_rel tails 48-79x.
* `dense_gemm_tc` — broken (above), quarantined.

### Accuracy after the fix

* BFCL-70 (reduced draw): overall 81.43 / normalized 77.70, with real tool
  calls (the leg above scored 14.29 with zero calls).
* **Pinned ST-995 (golden n=995, no overrides): overall 84.22 / normalized
  83.68** — overall equals the GB10 reference (84.22) and passes the 83.82
  floor; normalized is one sample under (0.04, noise floor 0.4). Run record
  `~/.atlas/runs/bfcl-subset/run-1788366618469517340.json`.
* The GB10 BENCH.toml's Python/JS cliff reproduces exactly here
  (simple_python 95.97 / simple_java 46.77 / simple_javascript 25.81) —
  shared across hardware, checkpoints and load paths, so the serve-path
  defect that note suspected (chat-template tool-argument serialization) is
  now confirmed cross-platform and is the largest known accuracy lever.
* The MLPerf ST-996 leg (harness bfcl_v4, 12/23/46, n=1004) is running under
  the submission serve profile (0.92/64K/MTP K=2/prefix/slots 16) — reference
  for the same draw on unsloth-3.6: 78.59 / 80.45.

## 2026-09-15 — DFlash2 speculative decoding on gfx1151 (strix-hip)

DFlash2 block-diffusion drafting (`incoai/Qwen3.8-27B-DFlash2`, γ=8) is ported
to the native-HIP target. All numbers below are copied verbatim from the three
frozen A/B summaries on AzeezStrix — every value is observed under the cited
`fingerprint-<arm>.txt` in its outdir:

- `~/dp4a-ab/out/dflash-ab-20260915T1409Z` — binary `7639415f…`, commit
  `ed2e90d42`, harness MinHeap warmup-16 + **3×1024** + prose/json 512
- `~/dp4a-ab/out/dflash-h128-20260915T1546Z` — binaries `7639415f…` (ob_h256)
  / `51a6b53f…`, harness **3×512**, `ATLAS_DFLASH_STEP_TIMING=1`
- `~/dp4a-ab/out/dflash-gemv-20260915T1632Z` — binary `4de8cd79…`, commit
  `b517dd6d`, harness **3×512**, `ATLAS_DFLASH_STEP_TIMING=1`

### How to serve

```bash
DFLASH=1 DRAFT_MODEL=/home/azeez/.models/dflash2 \
  ./serve-amd.sh nvidia/Qwen3.8-27B-NVFP4
```

`DFLASH=1` forces the MTP `--speculative` args off (they are mutually
exclusive on this recipe) and appends `--dflash --draft-model "$DRAFT_MODEL"`.
`DFLASH_GAMMA` overrides the MODEL.toml γ. Three DFlash-specific defaults live
in `serve-amd.sh`, not in MODEL.toml:

- `GPU_UTIL` defaults to **0.80** under DFlash (0.88 otherwise): the KV
  budget is computed before the ~5 GB drafter allocates, and the 0.88 launch
  OOM'd — `cuMemAlloc status 2`, 4.2 GB free at a 635 MB request.
- `ATLAS_DFLASH_OPTION_B=1` (the incremental paged drafter path); `=0`
  restores the legacy propose.
- The small-M GEMV arm for drafter projections is on by default on gfx1151;
  `ATLAS_DFLASH_SMALL_M_GEMV=0` opts out.

Note `--request-timeout` defaults to 300 s: at DFlash speeds several
1024-token requests hit that cap (`finish=timeout` rows in the first matrix);
pass `--request-timeout 900` for long-output legs.

### What was ported

`kernels/strix-hip/common/` gained, verbatim from gb10 unless noted:
`dense_gemv_bf16_batchm.cu`, `dflash_batch_anchor_add.cu`,
`dflash_batch_markov.cu`, `dflash2_conv.cu`, `dflash2_candidate_selector.cu`,
the three sink paged wrappers, and HDIM=128 variants of all three sink
wrappers (see below). `KERNEL.toml` registers the
`prefill_paged_sink`/`prefill_paged_indirect_sink` module renames plus their
`_h128` counterparts, and `MODEL.toml` carries the `[dflash]` block
(draft_model `incoai/Qwen3.8-27B-DFlash2`, γ=8, window 4096, mask 248070,
target layers [5,19,33,47,61]).

`prefill_paged_compute.cuh` received the four gb10 features the sink kernels
need, in both BR=32 and BR=64 instantiations: non-const `kv_len`/`q_offset`,
batched `cu_seqlens`/`kv_lens`/`batch_indirect_args` geometry
(`q_base_b`/`q_len_eff`), `q_rope_pos` for absolute masking/RoPE, the
sliding-window `kv_block_lo` skip, and the `ATLAS_ATTN_SINKS` epilogue. This
also fixes a pre-existing arity mismatch: the Rust
`ops::prefill_attention_paged_batched` dispatcher already passed
`cu_seqlens`/`kv_lens` the header did not declare — the batched-prefill path
is untested at batch>1 on gfx1151 (max batch 1 in the serve recipe); the fix
is by contract, not by exercise.

### Two engine defects found while porting (both apply to gb10 too)

**Option-B attention ran the wrong head_dim.** The drafter KV is
`head_dim=128`, but the Option-B paged-attention dispatches
`prefill_paged_{,indirect_,batched_}sink` built at `HDIM=256` — dims 128..255
of each KV row read the neighbouring head's data. Fixed by `_h128` kernel
builds selected via `paged_sink_modules_for_head_dim`. Evidence (h128
outdir, fingerprints `ob_h256`/`ob_h128`/`legacy_h128`): mean_na
2.215–2.462 (h256 Option B) → 3.148–4.367 (h128 Option B), matching the
legacy path's 3.131–4.059; median tok/s 5.351 → 7.572. On GB10 this is the
mechanism candidate for the documented Option-B acceptance drop there
(mean_na 5.54 → ~3.9 in the 09-13 profile) — **HYPOTHESIS for GB10 until
re-measured there.**

**γ-row drafter projections used a 128-row M-tile WMMA GEMM.** At M=γ≤8 the
pipelined `dense_gemm_bf16_pipelined` spends ~94% of its MMA work on padding
— on gfx1151 (~2–3.5 TFLOPS BF16 WMMA) that dominated propose.
`drafter_dense_gemm` now routes M≤8 through the bandwidth-bound
`dense_gemv_bf16_batchm` (arm logged once per process). Evidence (gemv
outdir, `ob_h128_gemv_g8` vs `ob_h128_gemv_g8_off`): propose median
228.7 ms vs 361.0 ms, and the off-arm reproduces the pre-GEMV build
bit-identically on all three MinHeap texts.

### Measured matrix

All MinHeap, temp 0, seed 0, `reasoning_effort=none`; first table's
harness is 3×1024 (300 s cap), the other two 3×512 (`--request-timeout 900`).
Every value observed under the outdir's `fingerprint-<arm>.txt`.

| arm | harness | median tok/s | mean_na | verify_ms | propose_ms |
|---|---|---|---|---|---|
| serial | 3×1024 | 10.67 | 0.000 | — | — |
| mtp_k4 | 3×1024 | 22.57 | 2.127–2.218 | — | — |
| dflash_g8 (legacy) | 3×1024 | 3.55 | 3.876–4.704 | — | — |
| dflash_g4 (legacy) | 3×1024 | 2.89 | 2.350–2.465 | — | — |
| dflash_g6 (legacy) | 3×1024 | 3.24 | 3.223–3.645 | — | — |
| ob_h256 | 3×512 | 5.351 | 2.215–2.462 | 286.8 | 365.8 |
| ob_h128 | 3×512 | 7.572 | 3.148–4.367 | 296.9 | 358.3 |
| legacy_h128 | 3×512 | 3.612 | 3.131–4.059 | 297.4 | 1028.5 |
| mtp_k4 (control) | 3×512 | 21.308 | 1.91–2.00 | — | — |
| ob_h128_gemv_g8 | 3×512 | 9.800 | 3.15–4.16 | 292.4 | 228.7 |
| ob_h128_gemv_g7 | 3×512 | 9.972 | 2.77–3.96 | 269.6 | 220.3 |
| ob_h128_gemv_g4 | 3×512 | 11.389 | 1.90–2.60 | 118.7 | 190.8 |
| ob_h128_gemv_g8_off | 3×512 | 7.650 | 3.15–4.37 | 289.9 | 361.0 |

**Parity vs serial** (serial texts from `dflash-ab-20260915T1409Z`): no
speculative arm is byte-identical to serial on any prompt, but MTP and all
γ=6/8 DFlash arms diverge at the same character indices — minheap 133,
prose 15, json 42 — so the spec-vs-serial delta is not specific to any
drafter. γ=4 additionally diverges at minheap 37.

**Gates** (`tests/single_gpu_suite.py` vs a γ=8 serve):
`gate-dflash_g8.json` — coherence 3/3, fibonacci 1/1, tool calls 2/2,
long-context 2/3, avg TPS 5.4; `gate-ob_h128_gemv_g8.json` — same verdicts,
avg TPS 7.0. The third long-context probe (~16 k tokens) 400s against
`max_seq_len 8192` — a harness artifact, present on both.

### Honest status / known gaps

- Best observed DFlash2 on gfx1151: **11.389 tok/s** (γ=4), ≈0.5× MTP K4's
  21.31–22.57. Drafts engage and accept well (mean_na up to 4.7, tok_step up
  to 5.7) — the deficit is step cost, not acceptance.
- Verify at K=γ+1 rows lands on the float `w4a16_gemv_batch8/16` tiers
  (verify_ms median 118.7 at K=5 vs 292.4 at K=9, per the gemv-outdir
  steptiming files) rather than the M=4 DP4A arm MTP uses.
- Propose remains ~190–229 ms median against a ~45 ms bandwidth ESTIMATE
  (drafter ~3.8 GB + fc + lm_head at ~100–200 GB/s — estimate, not measured);
  rocprof ranking of the propose step is the next lever and could not run
  while the overnight legs below own the GPU.
- `prefill_attn_dflash_fp8` remains HDIM-256-only — the FP8 drafter-KV path
  is wrong for 128-dim drafters; flagged in `paged_attn_modules.rs`, not
  fixed.
- The post-drain `pure virtual method called` SIGTERM abort reproduces on
  every arm including MTP K4 (pre-existing; cf. the 2026-09-14 gates note).
- DFlash + prefix caching was untested on gfx1151 before the agentic leg
  below; GB10 reported an illegal-address fault with that combination.

### Overnight legs (2026-09-15, γ=8, `~/dp4a-ab/out/dflash-overnight-20260915T1706Z`)

Chain `~/dp4a-ab/dflash_overnight.sh`, serve profile DFLASH=1 + Option B +
small-M GEMV defaults, binary `4de8cd79…` (run scripts committed at
`scripts/strix/dflash2/`):

- **ST-995 (bfcl-subset golden draw, n=995)** — observed under
  `st995-fingerprint.txt` (commit `b517dd6d9`, binary sha `4de8cd79…`,
  nvidia/Qwen3.8-27B-NVFP4 rev `dbb8f445`, drafter
  incoai/Qwen3.8-27B-DFlash2 γ=8, Option B, small-M GEMV, GPU_UTIL 0.80,
  max-seq 8192, prefill 2048, kv bf16, head nvfp4, bs1, ssm-slots 0,
  `--disable-thinking --disable-tool-grammar true --request-timeout 900`):
  golden draw n=995 seed 42 temp 0, no param overrides — **overall 85.13 /
  normalized single-turn 78.41**, run record
  `~/.atlas/runs/bfcl-subset/run-1789510215749772387.json`, 5 h 04 m wall
  (17:06→22:10Z). Accept over the leg: mean_na avg 5.673 median 6.000,
  tok_step avg 6.673 median 7.000 (n=997 requests). Per-subset:
  irrelevance 75.00, live_irrelevance 46.59, live_multiple 87.62,
  live_parallel 87.50, live_parallel_multiple 75.00, live_simple 92.00,
  multiple 96.77, parallel 88.71, parallel_multiple 87.10, simple_java
  67.74, simple_javascript 74.19, simple_python 95.97; categories
  hallucination 60.80 / live 86.47 / non_live 87.97.
  - Reference rows — each an observation on a different
    stack/binary/checkpoint, not an A/B: Strix MTP K=4, shipped nvidia
    checkpoint, 2026-09-11: 83.22 / 79.02 (run-1789114625433625524,
    BENCH.toml); Strix MTP K=4 on the FP8→NVFP4 requant derivative,
    2026-09-14: 83.02 / 80.41 (run-1789370409958768995) with
    hallucination 75.38 / live 84.71 / non_live 81.15, simple_java
    45.16, simple_javascript 29.03; GB10 nvidia-checkpoint MTP/n-gram
    runs 2026-09-12/13: 85.13 / 76.60 with hallucination 55.11,
    simple_java 67.74, simple_javascript 74.19
    (QWEN38_PORT_GAP_ANALYSIS §F.2, runs run-1789206159965049753 /
    run-1789290824981666674).
  - The DFlash2 leg's overall score and its
    simple_java/simple_javascript/hallucination profile coincide with
    the GB10 nvidia-checkpoint runs, and sit +1.91 overall / −0.61
    normalized from the Strix shipped-nvidia MTP record; the
    hallucination-category gap vs the requant-checkpoint run is the
    checkpoint-behaviour pattern §F.2 already documented (weights
    differ between those two rows). The normalized floor 85.32 shown
    by the runner is the MLPerf-submission-checkpoint reference and
    does not gate this checkpoint (runner's own verdict text).
- **ST-996 (bfcl_v4 12/23/46, n~1004) — dropped from the chain** (merge
  window); the handover watcher stopped the leg at its banner before it
  ran. No data recorded.
- **MLPerf agentic-coding 2.5h (20 trajectories, prefix caching)** —
  running: outdir `~/dp4a-ab/out/dflash-agentic-20260915T2247Z`, γ=8,
  max-seq 24576, `--enable-prefix-caching`, SSM_SLOTS=16 with
  SSM_CKPT_INTERVAL=128 (20 Marconi slots), ETA ~03:00Z.
