<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# GLM exact-prefill optimizations

These four opt-in paths adapt Jason McCartney / Enntity's SparkGLM work to
Atlas's native GLM implementation. They retain the existing NVFP4 checkpoint
and native KDA and sparse-MLA operators. No external prefill library or
SparkGLM runtime is introduced.

| Switch | Work removed | Eligibility |
|---|---|---|
| `ATLAS_GLM_KDA_SIDE_PROJ_LT=1` | Three separate KDA side-projection GEMMs | Packed b/f_a/g_a weights; prefill, outside graph capture and verification; exact pipelined CUTLASS twin |
| `ATLAS_GLM_KDA_CONV_PLANES=1` | Packing Q/K/V before the prefill convolution | Prefill only; convolution reads the three planes directly |
| `ATLAS_GLM_HC_PREFILL_FUSE2=1` | Separate block-input RMS normalization after the HC finalizer | GLM vanilla norm and eligible prefill finalizer; decode/verify stay separate |
| `ATLAS_GLM_SP_GATHER_OVERLAP=1` | Serial exchange/compute scheduling at SP seams | Equal rank row counts, at least 512 rows per block, matching whole/block kernels; no graph capture or determinism trace |

All switches default off. Overlap accepts `1` for four blocks, or `2` through
`8` for an explicit block count. Atlas checks the resolved switches in startup
rank agreement; differing overlap counts must not reach the collective loop.
The existing Qwen collapse/gather and routing pipelines remain intact.

The side-projection loader releases the source weights through Atlas's existing
retirement ledger. The three published weights alias one contiguous allocation.
The packed result is split into planes by an in-tree CUDA kernel; targets without
that kernel retain the pitched-copy fallback. A rejected CUTLASS shape falls back
before execution to the existing pipelined projection.

## Sources and adaptations

Original implementation: Jason McCartney / Enntity, AGPL-3.0-only,
[Enntity/atlas](https://github.com/Enntity/atlas). Exact imported changes:

- [`a7e7734c4a037c26150326d5256356cd7af50cf7`](https://github.com/Enntity/atlas/commit/a7e7734c4a037c26150326d5256356cd7af50cf7): side-projection extraction.
- [`d922a5b2127c12f6065900a0adda888defc09f6f`](https://github.com/Enntity/atlas/commit/d922a5b2127c12f6065900a0adda888defc09f6f): exact CUTLASS triple and tests.
- [`0f86a6b8070c77dd33e24da17e719b73e5170383`](https://github.com/Enntity/atlas/commit/0f86a6b8070c77dd33e24da17e719b73e5170383): KDA side packing and dispatch.
- [`26d8e827bb1119f4e1c0b06ef26f82ad97844bde`](https://github.com/Enntity/atlas/commit/26d8e827bb1119f4e1c0b06ef26f82ad97844bde): planar convolution and equivalence test.
- [`0f32741782c3cbcc2d3e11a6148c0cb30bd3c453`](https://github.com/Enntity/atlas/commit/0f32741782c3cbcc2d3e11a6148c0cb30bd3c453): HC finalizer/input-norm fusion.
- [`c7a24251d13643bf45b458d531ba13d284af6beb`](https://github.com/Enntity/atlas/commit/c7a24251d13643bf45b458d531ba13d284af6beb): longer GPU test shapes.
- [`696cfe531761525f72b18d323cea77d710a3bc37`](https://github.com/Enntity/atlas/commit/696cfe531761525f72b18d323cea77d710a3bc37): source-retirement tests.
- [`5360b8f7d87f69f156b2e320848f30558a90f415`](https://github.com/Enntity/atlas/commit/5360b8f7d87f69f156b2e320848f30558a90f415): HC row offsets.
- [`75ef79996922cf88e8f90599647d5b4bf1ffd30e`](https://github.com/Enntity/atlas/commit/75ef79996922cf88e8f90599647d5b4bf1ffd30e): SP overlap and two-rank tests.
- [`0ea7556592bbe309313c49be3457b53a62f6b6d8`](https://github.com/Enntity/atlas/commit/0ea7556592bbe309313c49be3457b53a62f6b6d8): packed-side split kernel and selected-path logging.

Local adaptations preserve current-main Qwen SP pipelines and the native
Atlas KDA recurrence, and add startup parity for the new switches. Packed
multi-prompt prefill, TTFT tracing and experimental checkpoint-native format
kernels are outside this change.

## Validation scope

CPU tests cover plane-copy geometry, source retirement, strict overlap parsing,
row partitions and byte equality of mock two-rank exchanges against the whole
seam. Ignored GPU tests compare projection, convolution/state and HC outputs
byte-for-byte at boundary and large row counts. CPU linking stubs cannot run
those kernels or establish an inference speedup. Enablement requires a fresh
native build, on/off GPU equivalence and measured prefill improvement on the
actual two-Spark recipe; historical upstream timings are not local results.

The native build at `5d652f8e0835231ead3bcac91d66e22632c08cbe` passed all six
core CUDA checks. On the existing NVIDIA NVFP4 checkpoint, switching all four
paths on reduced median client TTFT by 2.3–2.4% for the measured approximately
10K-token prose, code and structured prompts; decode was essentially unchanged.
All 11 benchmark answers and reasoning strings matched between settings.
The FP4-Local comparison also passed exact-output checks for all 13
benchmark/vision pairs; its chat TTFT decreased by 0.8–1.1%, while the single
random C1 request's TTFT increased by 1.6%.
See [the measured settings and limits](GLM_5_3_FLASH_PREFILL_RESULTS.md).

This is evidence for the bundle, not four independently measured speedups.
The switches remain opt-in pending repeated, interleaved on/off measurements
across both checkpoints at 10K and 64K input
lengths. Default promotion requires exact-output checks to pass, a median TTFT
improvement larger than the observed repeat-to-repeat variation, and no
material decode or peak-memory regression. A leave-one-out comparison must
also establish which switches contribute before enabling individual paths
by default. These additional experiments are not claimed by the current
bounded validation.
