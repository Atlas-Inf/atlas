# Qwen3.8-27B on Strix Halo (gfx1151)

How the Qwen3.8-27B target reaches AMD Strix Halo, on both of Atlas's AMD
toolchains, and what has and has not been measured on it.

The Windows half of this port has its own document —
[`STRIX_WINDOWS_HIP.md`](STRIX_WINDOWS_HIP.md) — because its failure modes are
entirely different. This one covers the target itself and the Linux legs.

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
