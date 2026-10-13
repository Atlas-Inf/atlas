<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# GLM-5.3-Flash on GB10: local upstream integration

Review: [change areas and invariants](GLM_5_3_FLASH_REVIEW.md) · [measured results](glm53-validation/README.md) · [launch recipe](../../scripts/dev/glm53_native/README.md)

Status on 2026-10-09: **integrated and tested locally; not published or merged
upstream.** Native GPU and bounded end-to-end checks passed on the test pair;
see the measured scope below. This document supersedes the
September plan that described GLM as absent from the tree. It distinguishes
imported implementation, CPU checks, native compilation, and model validation.
The target is
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
on two DGX Sparks with TP2/EP2. Model weights are downloaded separately.

## Source boundaries

| Input | Exact revision | Role |
|---|---|---|
| Atlas upstream | `7ffd183f210708d5b1afba1e046049f14ad498c7` | Current upstream base used for this integration |
| Enntity upstream candidate | `619da6ff575bc914873028b62551d83aa5c09981` | GLM model, native kernels, FP8 cache, DFlash and serving changes |
| Enntity candidate's original Atlas base | `6a3d24ecb891adac1fac8edeef92de29bb6f3e15` | Historical comparison boundary |
| Local reconciliation checkpoint | `352d087e131f9ee55c2022c03835920ae22f17d2` | Candidate reconciled with the current upstream base; later work remains separate |
| Historical prefill bridge import | `a7f8de0e54a2964f22f974178e2ae150796d7b67` | Reference-only FlashKDA and NVIDIA sparse-MLA bridges; replaced by native operators |
| Enntity deployed reference | `72588541a466f53cb9d38afd74c515e14ab601db` | Comparison reference, not the identity of this local build |
| SparkGLM source build recipes | `84310dc3bbdfb99526e1b17342859241b447606d` | Provenance for adapted build scripts, FlashKDA bridge and state-slot patch |

The local branch is `feat/glm53-native-integration`. Source history and the
final review commits are authoritative; the reconciliation checkpoint above
is not a final validation tag. Display-carveout changes following the bridge
commit are deliberately excluded. No deployment recipe or benchmark result
from the reference fork is evidence that this rebased tree works.

Atlas owns model loading, scheduling, KV/SSM state, distributed execution,
workspace planning and kernel dispatch. The final recipe builds the accelerated
prefill paths inside Atlas from credited device source. It does not build or
load SparkGLM/FlashKDA/FlashInfer host APIs, adapters or model-specific shared
libraries. No SparkGLM checkout, image, Python launcher or checkpoint converter
is required. CUDA, CUTLASS, communication and ordinary Atlas dependencies remain.

## Imported model implementation

These are source capabilities, not claims of validation on the local port.
Model metadata is defined in
[`MODEL.toml`](../../kernels/gb10/glm-5.3-flash/MODEL.toml).

| Piece | Source implementation |
|---|---|
| Model | `glm5_next`, 320B total / 18B active parameters; 45 layers |
| KDA | 34 linear-attention layers, 64 heads × 128 before TP splitting, convolution width 4; FP32 recurrent state |
| MLA | 11 full-attention layers; zero-RoPE, latent width 512, semantic sparse top-2048 selection above the dense region |
| Residual / experts | mHC with multiplier 4; 288 routed experts, top-8 routing, shared expert and configured initial dense layers |
| Checkpoint loading | NVIDIA ModelOpt NVFP4 weights, dense tensor handling, TP2/EP2 integration |
| KV / serving | BF16 and `fp8_g128` latent cache, prefix/state ownership, batched decoding and cancellation paths |
| DFlash | Optional separate drafter, block verification, rollback and structured-output integration |
| Native MTP | Appended predictor paths inherited; not interchangeable with DFlash or qualified here |
| Vision | Native loader/kernels/API path; two counterbalanced 224px image probes passed; video and broader vision quality unqualified |
| Tools | `poolside_v1` parser and GLM template handling |

The metadata advertises 1,048,576 positions. This is neither an established
single-session runtime limit for this port nor a guarantee that a pair can fit
that context. Usable single-session and aggregate context must be measured with
the final weights, cache format, concurrency, buffers, and memory guards.

## Original NVIDIA checkpoint and the overlay

**The original checkpoint booted directly on two Sparks: an earlier BF16/no-drafter baseline passed, and the final native FP8/DFlash build passed the tests below.** The old SparkGLM converter rewrites 864 BF16 predictor expert matrices
under `model.language_model.layers.45.*` into an NVFP4 overlay. It does not
convert the target's ordinary decoder layers.

The imported Atlas loader skips that appended predictor when `--speculative`
is absent. Both fast and regular loaders receive the skip policy; extra-weight
loading also honors it. See `unused_glm_mtp_prefix` in
[`serve_phases/weights.rs`](../../crates/spark-server/src/main_modules/serve_phases/weights.rs)
and the skip implementations in `spark-runtime`. `--dflash` is separate from
`--speculative` at this loading boundary, so a DFlash launch also skips the
unused appended predictor.

The tested runs used the original NVIDIA snapshot directly, without an overlay.
These results do not qualify every loader/speculation combination. Native MTP loading is a
different case and is not covered by this argument.

## Cache and speculative-decoding gates

`--kv-cache-dtype=fp8_g128` selects the imported FP8 latent-cache path; the model
metadata's default remains BF16. Ordinary sparse FP8 prefill requires the Atlas
tensor-core path (`ATLAS_GLM_SPARSE_PREFILL_TC=1`) when a BF16 view or optional
native kernel does not handle the piece. Enabling the bounded native path does
not replace that fallback. The FP8 path can form a bounded BF16 view for selected
attention kernels; a BF16 kernel operand does not mean the resident KV pool has
changed to BF16.

Use ordinary decode as the initial baseline. Omitting `--dflash`, `--draft-model`
and `--speculative` is permitted with expert TP. The factory rejects expert-TP
native speculation without DFlash because the sliced experts do not have a
whole-expert MTP path. See
[`factory/build/glm.rs`](../../crates/spark-model/src/factory/build/glm.rs).

The imported environment settings are not freely interchangeable:

- `ATLAS_GLM_DFLASH=1` is the GLM DFlash lane and accompanies `--dflash` plus a
  separately supplied `--draft-model`.
- `ATLAS_GLM_MTP_LONG_CONTEXT=1` names the long verification lane despite the
  historical MTP wording. Preflight requires the GLM DFlash lane for this setting;
  do not copy it into a no-drafter recipe.
- Concurrent long-context verification requires `ATLAS_EP_PROTOCOL=v2` for owner
  slot identity. Sequence limits and memory reservations must agree on both ranks.
- `ATLAS_GLM_MULTI_SEQ_SPARSE=1` has explicit restrictions against native,
  self-, and n-gram speculation. Passing one preflight check does not establish
  that a different speculative mode or arbitrary combination is valid.
- The bounded repair-domain constants in `glm_repair_policy.rs`, the optional
  native attention domain, and the server's maximum session length serve different
  purposes. Do not derive the global context limit from any one of them.

Keep these settings explicit during qualification. A production preset must be
based on the validated combinations, not copied wholesale from a fork profile.

## Native prefill architecture and build

The [GLM Dockerfile](../../docker/gb10/glm-5.3-flash/nvfp4/Dockerfile) builds the
normal `spark` binary and embeds the selected GLM kernel target. No external
prefill builder is invoked. The only fetched accelerator build dependency is
the engine's pinned CUTLASS checkout; the licensed device subsets are in-tree.

| Component | Source and exact revision | Atlas integration |
|---|---|---|
| KDA | Three MIT device headers from MoonshotAI/FlashKDA `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` | `spark-runtime/cuda/glm_kda/launch.cu` builds one H32/D128 specialization into the runtime; `spark_runtime::glm_kda` owns checked parameters and workspace |
| Sparse MLA | Seventeen BSD-3-Clause NVIDIA device headers from FlashInfer `8eccd0c1352165302840c0e19066bc42d36dbd7a` | `kernels/gb10/glm-5.3-flash/nvfp4/glm_sparse_native*.cu` and `layers/ops/glm_sparse_native*` own target entry points, planning and launches; no FlashInfer package or host dispatcher |
| CUTLASS | `cf064d2e6bad2886238ac565b3b49007764f4939` | Shared engine build dependency through `CUTLASS_HOME`; no second FlashKDA-specific checkout |

Atlas's KDA path directly reads interleaved convolution outputs, packs beta in
caller-owned scratch and converts the canonical key-major state in shared
memory. It uses no external adapter ABI, allocator or global sequence state.
The 4096-row workspace is 113,508,352 bytes; the checked maximum is 4100 rows.
**Persistent state storage is FP32, but the accelerated recurrence rounds its
internal state to BF16.** It is not numerically identical to the existing FP32
recurrence. GPU numerical, continuation and full-model quality gates must pass
before recommending it. `ATLAS_KDA_FLASH_PREFILL=1` remains explicit opt-in;
ordinary pieces contain 2048–4100 rows, with larger eligible prefills split into
pieces. Decode, short prefills and speculative verification retain existing
recurrence paths.

`ATLAS_GLM_SPARSE_NATIVE=1` selects Atlas's native sparse operator. Its current
startup contract requires TP2/EP2, a >=4100-row arena, at least 32,768 configured context,
block size 16 and at most four active owners. Runtime admission is ordinary
continuation pieces of 2048–4100 rows, starting at or after 2048 and ending by
32,768; capture/verification remain excluded. Resident BF16 or `fp8_g128` KV
reaches the operator through the existing bounded BF16 view. The accelerator
packs attention operands to FP8 internally, so it also requires numerical
qualification. Larger configured contexts can keep this option enabled: rows
outside its admitted domain use the existing Atlas sparse path. The 523,232-token
retrieval run exercised that combined configuration; it does not extend the
native operator's 32,768-token domain.
Neither accelerator's range determines the model's global context limit.

The image retains MIT/BSD license texts, attribution and source provenance under
`/opt/atlas/notices/`, alongside the CUTLASS license and revision. Source-level
inventories are
[`KDA provenance`](../../crates/spark-runtime/cuda/glm_kda/vendor/provenance.json)
and [sparse MLA provenance](../../third_party/nvidia_sparse_mla_sm120/provenance.json).
The former external prefill bridges and their build scripts were removed after
native qualification. Their [historical source records and notices](../../third_party/nvidia_sparse_mla_sm120/history/README.md)
remain available, with revision-qualified recovery instructions. The [native
GPU tests and sampled-fixture converter](../../scripts/dev/glm_sparse_native/README.md)
remain in-tree. No `*_LIBRARY` setting is required for this deployment.

## Validation evidence and limits

The [history map](glm53-validation/history-map.json) relates original source
revisions to the review history after commit-message normalization. Human
authorship, file trees and parent relationships are preserved. Historical
measurements below still identify the original measured commit and binary.

GPU code revision: `451bbc21f3ff8df356a159cb376eb7518e26c395`, two GB10 Sparks,
CUDA 13.0, NVIDIA driver 580.126.09, TP2/EP2. The standalone GLM binary SHA256 is
`6f65cc458e142740d28f757f54f4eab14d4e96eee46a519e3d6fe7bc4c69bfcb`;
the native GLM kernel-set hash is `5fd26547c6f6`. Later documentation and recipe
commits do not change these tested code inputs. The exact environment/CLI recipe
is in `scripts/dev/glm53_native/`.

| Check | Result | Scope |
|---|---|---|
| CPU development gates | Full workspace tests, clippy, formatting, rustdoc, typos, license headers and file-size checks passed | GPU tests run separately; CPU CUDA stubs fail if called |
| ARM64 native build | Release engine and GPU-test binaries passed | 24 GiB build limit, about 7 GiB observed peak |
| Clean Docker build | `--no-cache` ARM64 image passed; CLI and bundled notices checked | Image ID starts `01ee1b6e17a8`; full-model runs used the standalone binary |
| Sparse GPU operator | Zero/constant/one-hot analytic cases passed; eight sampled reference outputs matched byte-exactly | Actual Atlas dispatch, 426,124,800-byte fixture, input/redzone/preparation/metadata checks; sampled comparison alone is not an independent oracle |
| KDA GPU operator | Independent FP64 and existing FP32 comparison, tails, chunk-aligned split and continued decode passed | Declared numerical bounds; BF16 internal recurrence is not bit-exact FP32 |
| Text and tools | 5/5 arithmetic, strict JSON, streamed text, tool and streamed-tool checks passed | Original NVIDIA weights, FP8g128 KV, FP32 SSM storage, DFlash |
| Concurrency/lifecycle | Four 128-token streams completed; 14-request disconnect/isolation sequence passed | 13 exact completions plus one client disconnect, two reuse waves; no assertion about internal cancellation acknowledgement |
| Vision |2/2 counterbalanced 224×224 spatial/color probes passed | 64 image tokens each; not a video or general vision quality gate |
| Other-model regression | Final Qwen build and all eight exact-answer requests passed | Qwen3.5-27B via the qwen3.6-27b kernel target; C2 retrieval and slot reuse |

Guarded retrieval used random-word filler, twenty independently generated codes
at 1–99% filler-word positions, and a strict JSON schema containing only property
types/names, not answer values. Prefix caching was disabled.

| Prompt tokens | Correct records | TTFT | Effective prompt tokens/s (prompt/TTFT) | Server decode tokens/s |
|---:|---:|---:|---:|---:|
|32,608|3/3|13.57s|2,403.7|54.8|
|130,016|20/20|51.54s|2,522.8|47.8|
|523,232|20/20|237.13s|2,206.5|51.7|

The last profile configured 524,288 tokens per session and 524,288 physical
aggregate KV slots shared by up to four requests. It is not four independent
524K allocations, nor the largest model metadata position. Available host memory
reached minima of 9.89 GiB on the head and 10.27 GiB on the worker; the 8 GiB
host guard never fired. These are one-seed retrieval
checks, not broad long-context accuracy certification.

Repeated structured, code and prose benchmarks now cover approximately 10K
input tokens and 1,024 generated tokens, with three trials/waves at C1, C2 and
C4. The [validation report](glm53-validation/README.md) includes the original
and native measurements, metric definitions and machine-readable results.
Aggregate decode is measured while all streams are decoding; throughput that
includes prefill is reported separately. The earlier 4K random-word speed
probe is superseded by these workload-specific comparisons.

Exact generated-output parity does **not** hold. Matching prompt token IDs and
temperature zero produced exact visible matches in 2/4 short cases and 0/9
longer cases. The short factual answers were correct, but the native explanation
exceeded its requested word limit. The cause of output divergence has not been
isolated. Kernel numerical checks and basic coherence probes do not establish
full-model accuracy equivalence.

Both native accelerators remain opt-in. Native appended MTP, prefix-sharing
profiles, video/general vision accuracy, all other model families and broader
performance/quality suites are outside these receipts. The inherited KDA CUDA
launch-bound occupancy warning remains; its tested bounded launches succeeded.
Pushing, PR publication and contributor-agreement submission remain separate
user-controlled steps.

## Provenance, credits and licensing boundaries

- **Reiner Schmidt / Mango-kid** authored the original Atlas GLM port on
  [`feat/glm53-dual-spark`](https://github.com/Mango-kid/atlas/tree/feat/glm53-dual-spark),
  including the parser, loader, KDA, semantic indexer and native MTP work. The
  historical source revision is `90b3584abc71b44b609637092b85d8423d8ff20f`.
- **Jason McCartney / Enntity** extended that work with ModelOpt handling,
  sparse-MLA and MoE optimizations, FP8 cache, DFlash, concurrency and the source
  build recipes. The exact candidate and bridge revisions are listed above.
- **Atlas contributors** supplied and evolved the surrounding runtime, model,
  communication, scheduler and kernel infrastructure. Human authorship and
  co-author credits are retained when preparing review commits.
- **MoonshotAI, FlashInfer contributors and NVIDIA/CUTLASS** supplied kernel
  device implementations used by the native accelerators. The minimal imported
  source subsets retain their original notices and exact source hashes.
- **Z.ai and NVIDIA** supplied the model/checkpoint; **IncoAI and DFlash authors**
  supplied the separate draft-model/algorithm work. Weights and their terms are
  independent of this source integration.

The detailed [GLM prior-art inventory](../glm-prior-art.md) preserves further
credits for knapcio, TensorFold/jayleaton, MiaAI-Lab, RiNGSiDE/othexmr, Reederey87,
vLLM contributors, Marconi and other referenced work. It distinguishes adapted
code/data from design references and related work. The native source inventories above identify shipped device code. The historical
[bridge/patch inventory](../../third_party/glm_prefill/provenance.json) describes
reference-only scaffolding. Review these alongside individual notices; none
is a substitute for the underlying license texts.

Atlas and the imported Enntity implementation are AGPL-3.0-only. The kernel build
also preserves applicable MIT, BSD-3-Clause and Apache-2.0 notices. This document
does not claim that all contributors have granted Atlas's separate CLA rights;
permission for any broader relicensing must be resolved with the relevant rights
holders before submission. Credit alone does not grant those rights.

The public `incoai/GLM-5.3-Flash-DFlash2` checkpoint is separately licensed
CC BY-NC-ND 4.0 according to its publisher. Optional loading does not remove its
usage restrictions. No target or draft weights are bundled here. Preserve and
review the selected checkpoint's model card, tokenizer/template files and license
terms rather than inheriting an unverified blanket template-license claim from
the earlier port plan.
