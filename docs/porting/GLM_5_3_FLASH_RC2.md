<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Native RC2 integration

This update ports the engine changes pinned by SparkGLM's October 10 release,
[Enntity/atlas f2b805e7](https://github.com/Enntity/atlas/commit/f2b805e73d36a7e5527622e24f2f8199052e01e6),
onto the native Atlas integration. The previous local review tip was `435d48f9`.
The native KDA and sparse prefill implementations remain built into Atlas.
No SparkGLM checkout, external GLM prefill library, or converted weight overlay
is required to build or serve this update.

## Changes and sources

Jason McCartney / Enntity supplied the upstream changes below. Imported commits
retain human authorship and original commit references. Atlas adaptations retain
the native operator dispatch, allocation-failure cleanup and API boundaries.
The [prior-art inventory](../glm-prior-art.md) credits the underlying work,
including the display-memory and prompt-lookup ideas.

| Area | Original source commits | Behavior |
|---|---|---|
| Optional display-memory allocation | `ec1e1519`, `57d1fe4d`, `53569cd2`, `6505d99f` | Native driver export and latent-pool placement, including sharded sizing |
| Sharded disk prefix cache | `24ffad56`, `56ec983c` | Each rank spills and restores its own latent records |
| Canonical sparse attention | `923947e9`, `b79fdf6e`, `b85befc6` | Consistent attention decomposition across sharding and verification widths |
| Canonical verification | `8f5a2af0`, `aa3de77c`, `5ad4c819`, `13cc5c7f` | Consistent row arithmetic, including dense FFNs and eligible plain decode |
| Decode kernels | `2f7010a6`, `f2b805e7` | Batched KDA record commits and persistent prefetching MoE kernels |
| Prompt-lookup drafts | `8cc50add`, `157de3d9`, `ef85f744`, `e2db357f` | Target-verified copies alongside optional DFlash; bounded host index |
| Template and tool streaming | `1a39c032`, `7838b34c`, `84454a4a`, `8c150822`, `e034500d` | Schema-preserved argument strings, corrected empty thinking block, ordinary role words retained, complete validated Poolside headers |

These are native changes to shared allocation, scheduler, tool-parser and CUDA
dispatch code, so review their effects on other model families as well as GLM.
Native review additionally fixes nullable typed arguments, scopes the canonical
tier mapping to GLM kernel capabilities, and preserves transactional allocation
cleanup. The display-memory launcher is optional and platform-specific. Its driver
allowlist and hardware checks must pass; it requires a privileged export step.
Ordinary serving and KV sharding do not require that launcher. Native additions
check the GB10 PCI identity before export, run the launcher before creating
threads, unwind partial CUDA mappings, and permit unprivileged fallback.
The test pair uses driver `580.126.09`, outside the validated driver list; the
display-memory feature remains disabled there.

## Context and candidate launch

Start with the [existing bounded recipe](../../scripts/dev/glm53_native/README.md).
In its common shell setup, replace the environment-loading block on both ranks
with the following so that the new variables are exported to the process:

```bash
set -a
. scripts/dev/glm53_native/qualification.env
. scripts/dev/glm53_native/rc2-candidate.env
set +a
```

Then use the same rank commands. Retain the initial 32K session limit,
131072-token aggregate cap, 12 GiB memory guard and disabled swap. Do not run
two loaded engines together.

The candidate explicitly overrides `ATLAS_PDL=0` and
`ATLAS_GLM_DECODE_GEMV_BATCH=0`. The initial two-machine qualification produced
nonfinite prefill values with dependent launch enabled, including with the
persistent MoE option disabled. Ordinary stream ordering passed the initial
single/four-stream checks. Per-stage diagnostic synchronization also masked the
failure, so traced runs are not correctness or performance evidence for the
untraced PDL configuration. The root cause remains unresolved; do not enable
these switches based solely on passing isolated kernel tests. All other RC2
options in the candidate, including native KV sharding, remain enabled.

The new profile uses target FP8 `fp8_g128` KV. Sharding stores each latent token
on one rank rather than both; the sparse index still needs its own memory.
One logical context spans the pair: adding the block counts of both ranks would
double-count capacity. Aggregate active capacity is the negotiated logical
block count times the block size, shared by all active sessions and resident
prefixes. Disk storage retains evicted prefixes but does not enlarge active
GPU context.

After bounded tests and memory checks, a 1,048,576-token session requires both
`--max-seq-len=1048576` and sufficient actual shared KV capacity. With block size
16, `ATLAS_KV_MAX_BLOCKS=65536` caps the shared pool at 1,048,576 tokens; this is
an upper bound, not a promise the allocation will fit. Input and generated tokens
must fit together. Raise limits only after checking both ranks' actual free
memory and allocated pool. Four sequence slots do not reserve four full windows.

SparkGLM reports approximately 1.92M shared tokens at utilization 0.89 and 2.62M
at 0.91 with its optional display-memory allocation. Those are upstream machine
measurements, not qualification of this native build or a recommended safe
setting for every pair. Its near-million-token needle test is likewise upstream
evidence. See [PR 39](https://github.com/Enntity/sparkglm/pull/39) and the
[profile change](https://github.com/Enntity/sparkglm/commit/4c8c3e4e).

## Validation boundary

The historical [validation report](glm53-validation/README.md) measures code
`451bbc21`; those throughput, retrieval and parity results do not measure RC2.
RC2 changes runtime arithmetic and scheduling. Upstream's determinism and speed
results must be reproduced locally before claiming them for this integration.
In particular, canonical verification does not prove cross-engine output
identity, and prompt lookup benefits depend on repeated text.

### Native bounded qualification (October 10)

Native source `919adb4f` was built on GB10 with CUDA 13.0.88 and Rust 1.93.1.
The [machine-readable results](glm53-validation/rc2-results.json) record the
binary digest, settings and test cases. No external GLM prefill library was used.

- Full CPU workspace tests, clippy, formatting, rustdoc and repository checks pass.
- Eleven GPU cases pass: canonical dense/W4 arithmetic, sparse attention and
  sharding, KDA record commits, persistent MoE, and asynchronous projection/MoE
  chains. Numeric checks reject unexpected nonfinite results.
- Twelve real-model requests complete. All four messages match between C1 and
  C4, and all four match a repeated C4 run, at temperature zero. These compare
  this build with itself, not with SparkGLM or another engine.
- A separate 18,658-input-token prompt recovers the exact values of all four
  buried records, with 80 generated tokens. This is a bounded retrieval smoke,
  not a long-context accuracy benchmark.
- Both ranks allocate 8192 logical blocks (131072 aggregate tokens), each storing
  4096 blocks' latents. The session setting is 32768. These are configured limits;
  the retrieval above is the largest prompt exercised in this RC2 qualification.

The real-model checks use the conservative profile above, with tracing disabled.
No new RC2 throughput comparison, million-token qualification, display-memory
allocation, disk-prefix runtime round trip or other-model GPU matrix is claimed.
Rebuild and validate the Docker image before opening the PR; this RC2 run tested
the standalone native binary. The prior production service is kept separate from
these temporary validation instances.

Target and draft weights remain separate. DFlash still requires explicit
selection, and the chosen draft weights retain their own license terms.
