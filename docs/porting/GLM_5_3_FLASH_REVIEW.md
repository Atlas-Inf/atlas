<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# GLM-5.3-Flash review guide

This stack adds native serving of NVIDIA's GLM-5.3-Flash NVFP4 checkpoint
on two NVIDIA GB10 / DGX Sparks. Atlas owns loading, scheduling, distributed
execution, KV/SSM state and kernel dispatch. Native KDA and sparse MLA device
sources build with Atlas; serving does not invoke SparkGLM, another inference
engine, or externally built GLM prefill libraries.

The target uses FP8 KV cache. Optional DFlash loads a separate draft checkpoint.
Checkpoint-native MTP is deferred to a separate follow-up and is outside this
release. The final source also includes the rustls 0.23.46 security update in
`Cargo.lock` after `submit/glm53-decode-rc4`.
AMD Strix Halo support is outside this release. The existing Strix selector
is unchanged from the Atlas base, and its inherited host ABI is preserved.

Start with the [model integration](GLM_5_3_FLASH.md),
[launch recipe](../../scripts/dev/glm53_native/README.md), and
[validation report](glm53-validation/README.md). The recipe keeps measured
tuning explicit; changing those settings requires validation of the new profile.

## Development history

This is one combined release PR against Atlas `main`, developed from
`7ffd183f210708d5b1afba1e046049f14ad498c7`. Its frozen implementation tip is
`cb45065cb4f21c162ccd29dd837ef4b6be3aa52e` on `release/glm53-nvfp4`.
The seven local branches below preserve development history and help navigate
the diff; they are not seven independent PRs or required qualification targets.
The [history manifest](glm53-validation/submission-stack.json) records those
source boundaries without claiming independent hardware passes.

| Branch suffix after `submit/glm53-` | Change |
|---|---|
| `runtime-prerequisites` | Numerical build flags, allocation/accounting, swap records, two-rank transport and native KDA; macOS-only dependency/example gating |
| `api-compiler` | Grammar compiler, complete streamed tool/output envelopes and ordered per-choice request media |
| `native-core` | GLM loading, MLA/KDA/MoE execution, FP8 target cache, replicated KV and optional DFlash |
| `context-storage` | Token-sharded KV and optional NVMe prefix storage |
| `benchmark-tools` | Developer performance harnesses and their Cargo example declarations |
| `context-rc2` | Canonical verification, sharded prefix restoration, prompt lookup and decode improvements |
| `decode-rc4` | Later native decode, scheduling and context changes |

The core remains a large integration. Extracting prerequisites and optional
storage/benchmark work makes dependencies explicit; it does not remove the
need to review shared scheduling, allocation and numerical behavior.
Correctness tests and source/license notices stay with their implementations.

## Shared changes to review

| Area | Main paths | Required invariant |
|---|---|---|
| Model loading | `crates/atlas-core/src/config/parsers/glm5_next.rs`, `crates/spark-model/src/factory/build/glm.rs`, `crates/spark-model/src/weight_loader/` | Preserve NVIDIA layouts/scales; skip unused appended predictor weights; reject unsupported combinations before allocation |
| KV/SSM ownership | `crates/spark-model/src/factory/build/kv_budget.rs`, `crates/spark-model/src/model/ssm_pool/`, `crates/spark-runtime/src/kv_cache/` | Rank-minimum budgeting, isolated owners, rollback cleanup, and separate aggregate/per-request limits |
| Distributed execution | `crates/spark-comm/src/nccl_backend/rdma_pair/`, `crates/atlas-rdma/`, `crates/spark-comm/src/peer_lifeline.rs` | Matching rank operations, bounded failure propagation and capture/replay lifetimes |
| Speculation | `crates/spark-model/src/layers/dflash_head/`, `crates/spark-server/src/scheduler/` | Target verification, accepted-prefix state, rejected-tail rollback and per-request isolation |
| Structured output/tools | `crates/xgrammar/`, `crates/spark-server/src/tool_parser/`, `crates/spark-server/src/scheduler/strict_spec.rs` | Refuse unsupported strict schemas; preserve grammar state and complete streamed tool envelopes |
| Native prefill | `crates/spark-runtime/cuda/glm_kda/`, `crates/spark-model/src/layers/ops/glm_sparse_native*.rs`, `kernels/gb10/glm-5.3-flash/nvfp4/` | Checked geometry, bounded scratch, correct cache layout and explicit fallback domains |
| Build/distribution | `crates/atlas-kernels/build*.rs`, `crates/spark-runtime/build*.rs`, `docker/gb10/glm-5.3-flash/nvfp4/Dockerfile` | Correct numerical flags, native symbols available before cache sizing, and bundled source notices |

Compiler-only grammar changes do not independently qualify fail-closed server
behavior; policy limits and error propagation remain with the server integration.
Metal dependency and example gating preserves the existing macOS bodies. Linux
all-features compilation is compatibility evidence, not macOS GPU qualification.

## Qualification

The final combined implementation at `cb45065c` passed CPU/static checks,
ten canonical Qwen GPU gates and the bounded paired GLM campaign. See the
[release summary](glm53-validation/release-summary.json) for actual records
and limits. Intermediate branches remain development history only.

CPU checks that use build/link stubs do not execute CUDA kernels. Native builds
and GPU gates have their own source, binary, model and recipe identities.
Formal Qwen gates establish bounded cross-model regression coverage; they do
not measure GLM throughput or qualify a different intermediate branch.
GLM API, concurrency, retrieval, cancellation and storage tests are reported
separately. Failed attempts remain in the evidence history.

## Limits and reproduction

The [RC2](GLM_5_3_FLASH_RC2.md) and [RC4](GLM_5_3_FLASH_RC4.md) reports distinguish configuration
limits from tested workloads. A shared KV capacity is not a per-session context
limit. Historical near-1M single-session retrieval and approximately 1.42M
actual aggregate input are separate measurements; they do not qualify 2.5M
occupied tokens or guarantee perfect retrieval.

Native sparse-prefill acceleration has a bounded domain; longer inputs
use Atlas's fallback. Bounded API and needle tests are not broad model-quality
or sustained leak-freedom claims.

Raw records retain their measured revisions. Historical `pr/*`, `review/*`
and earlier submission records are indexed in the validation history rather
than relabeled as new runs. Commit metadata normalized at the contributor's
request uses fixed EST (UTC−05:00); test receipts retain their actual UTC times.
The evidence attachment documents reproduction inputs and exclusions. The
historical clean-clone check at `d0fe219b` verifies its committed records;
the final packaging performs a fresh canonical record check in a clean clone.

## Sources and rights

[Prior art](../glm-prior-art.md) distinguishes imported code/data, design
references and related work. Native vendor inventories preserve source pins,
hashes, license texts and human contributor credits, including Jason McCartney's
Enntity/SparkGLM work and the upstream kernel/checkpoint contributors.

The integration and validation work are AI-assisted; Reiner Schmidt is the
submitting contributor. Attribution does not establish additional licensing
rights. Target and optional draft weights are separate downloads governed by
their own terms. The [rights audit](glm53-validation/rights-audit.json) records
the separate contributor/CLA questions; technical qualification does not resolve
them, and no agreement has been signed on another contributor's behalf.
