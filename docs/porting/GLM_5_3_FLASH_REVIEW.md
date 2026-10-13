<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# GLM-5.3-Flash review guide

This reduced core follows the shared compiler, server, transport, memory and
native KDA prerequisite branches under `pr/glm53-*`. It retains native GLM
loading, prefill, decode and optional DFlash, with replicated KV storage.
Token-sharded KV and the optional NVMe prefix tier are restored separately
in `pr/glm53-context-storage`; developer performance harnesses follow in
`pr/glm53-benchmark-tools`. Correctness tests, SSM tier agreement and cleanup,
native qualification recipes and source/license notices remain with this core.
Historical GPU results below describe their original revisions, not a fresh
qualification of this reduced branch. The later RC2, RC4 and MTP stages are
unchanged in scope. The complete final stack retains the original runtime.

This integration loads NVIDIA's original NVFP4 checkpoint through Atlas and
runs TP2/EP2 across two GB10 Sparks. Atlas owns model loading, scheduling,
KV/SSM state, distributed execution and kernel dispatch. Optional DFlash loads
separate draft weights. Native prefill compiles the credited device sources
into Atlas; it does not call another inference engine or external GLM library.

Start with the [port description](GLM_5_3_FLASH.md), [validation report](glm53-validation/README.md)
and [standalone qualification recipe](../../scripts/dev/glm53_native/README.md).
The recipe deliberately keeps all qualified tuning values explicit. Changing or
collapsing those values into new defaults requires another runtime qualification.

## History and review boundaries

Atlas base is `7ffd183f210708d5b1afba1e046049f14ad498c7`. The original
integration merged Enntity candidate `619da6ff` at `352d087e`. Review history
normalizes commit-message metadata while retaining all human authors,
committers, dates, source trees and parent relationships. The [history map](glm53-validation/history-map.json)
links original revisions to their corresponding review commits. Historical
source and binary receipts keep their original identities; a source-equivalent
review commit is not a newly measured binary.

Later Atlas commits reconcile interfaces, repair regressions, replace the
external prefill bridges and qualify the native implementation.
Review the combined diff against the Atlas base, then inspect each area below.
The total branch is substantial; the small native-operator delta alone does not
represent the shared runtime changes included in this submission.

| Review area | Main paths | Invariants to examine |
|---|---|---|
| Model configuration and loading | `crates/atlas-core/src/config/parsers/glm5_next.rs`, `crates/spark-model/src/factory/build/glm.rs`, `crates/spark-model/src/weight_loader/`, `kernels/gb10/glm-5.3-flash/nvfp4/MODEL.toml` | Original NVIDIA ModelOpt scales/layouts; expert slicing; skip appended MTP weights unless used; unsupported combinations fail before allocation |
| Shared KV and SSM ownership | `crates/spark-model/src/factory/build/kv_budget.rs`, `crates/spark-model/src/model/ssm_pool/`, `crates/spark-runtime/src/kv_cache/` | Rank-minimum memory budget; rollback/compaction cleanup; owner isolation; shared pool versus per-request limit |
| Distributed execution | `crates/spark-comm/src/nccl_backend/rdma_pair/`, `crates/atlas-rdma/`, `crates/spark-comm/src/peer_lifeline.rs` | Matching rank commands; flags/order and timeouts; lifetimes across capture/replay; failure propagation |
| Optional DFlash | `crates/spark-model/src/layers/dflash_head/`, `crates/spark-server/src/scheduler/`, `crates/spark-server/src/cli/serve_args.rs` | Opt-in loading; single/batched selector ABI; first append; verification and rollback; per-request state |
| Structured output and tools | `crates/xgrammar/`, `crates/spark-server/src/tool_parser/`, `crates/spark-server/src/scheduler/strict_spec.rs` | Schema compilation and token boundaries; speculative grammar advancement; complete streamed tool envelopes |
| Native KDA | `crates/spark-runtime/src/glm_kda.rs`, `crates/spark-runtime/cuda/glm_kda/` | Checked geometry/workspace/aliasing; canonical state layout; preload before KV sizing; BF16 internal recurrence with FP32 persistent storage |
| Native sparse MLA | `crates/spark-model/src/layers/ops/glm_sparse_native*.rs`, `kernels/gb10/glm-5.3-flash/nvfp4/glm_sparse_native*.cu`, `third_party/nvidia_sparse_mla_sm120/` | Bounded admission; workspace ownership; quantization and main/tail merge; fallback beyond the native 32K domain |
| Build and distribution | `crates/atlas-kernels/build*.rs`, `crates/spark-runtime/build*.rs`, `docker/gb10/glm-5.3-flash/nvfp4/Dockerfile` | Per-source numerical flags; dependency tracking; native symbols available before sizing; bundled notices |

Inspect these groups separately when reviewing. They are review boundaries,
not claims that each group can be cherry-picked without its dependencies.
Microbenchmarks, numerical oracles and the KV-sizing campaign remain because
they explain and reproduce runtime decisions. Obsolete bridge implementations
and builders have been removed; their source history and notices remain.

## Evidence and outstanding limits

GPU execution is qualified at code commit `451bbc21`. The later polish changes
remove unused scaffolding and update documentation/evidence; they do not change
runtime algorithms or the tuned profile. The report distinguishes fresh
CPU/documentation checks from historical GPU measurements.

The tested maximum is 524,288 configured tokens, shared across up to four slots.
It is not 1M-context qualification or four fully occupied 524K requests. Native
sparse acceleration ends at 32K; the Atlas fallback handles longer contexts.
Performance is workload-dependent and output is not identical to the original
service, including a word-limit miss in one short explanation. No broad accuracy
claim follows from the basic coherence and retrieval probes.

Full other-model GPU coverage, prefix-sharing profiles, native appended MTP,
and general video/vision quality are outside this qualification. Shared code
changes therefore warrant review even when GLM's bounded tests pass.

## Attribution and distribution

[Prior art](../glm-prior-art.md) separates copied code/data, design references,
and related work. Native vendor inventories record source pins and hashes;
license texts and human contributor credits are preserved.

Target and draft weights are separate downloads. DFlash is disabled unless
explicitly enabled; selecting it does not waive the chosen weights' terms.
The public IncoAI DFlash2 model card lists CC BY-NC-ND 4.0 and a separate
commercial-licensing contact. Imported contributors' additional CLA/relicensing
rights are not established by credit or by this technical review.
