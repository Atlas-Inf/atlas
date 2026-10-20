<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Native GLM prefill comparison — 2026-10-20

Runtime revision: `5d652f8e0835231ead3bcac91d66e22632c08cbe`.
The subsequent `eeadc130` changes only recipe text and one rustdoc link.
Both ranks used image
`sha256:2d7ee3e9ca756014c57e84eb777b78a66278505fe9d378d1abb7d90da5744cd2`.
[The receipt](glm53-validation/prefill-nvfp4-20261020.json) includes unrounded
metrics, rank memory observations, warnings and hashes of the raw artifacts.

## Existing NVIDIA NVFP4 checkpoint

Two DGX Sparks, TP2/EP2, FP8 `fp8_g128` target KV, FP32 KDA state,
32,768 aggregate KV tokens, four sequence slots, prefix cache disabled.
DFlash2 was explicitly enabled with gamma 8 and a 2048-token draft window.
The four [prefill switches](GLM_5_3_FLASH_PREFILL.md) changed together from
all off to all on; other serving settings and requests remained identical.

| Workload | Streams | Input tokens each | Output tokens each | TTFT off / on, seconds | Decode off / on, tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic prose | 1 | 10,007 | 512 | 5.064 / 4.948 | 41.54 / 41.56 |
| Synthetic code | 1 | 10,012 | 512 | 5.114 / 4.993 | 55.57 / 55.72 |
| Synthetic structured data | 1 | 10,005 | 512 | 5.075 / 4.959 | 74.60 / 74.57 |
| Random base-token IDs | 1 | 10,000 | 512 | 5.157 / 5.021 | 106.11 / 106.25 |
| Random base-token IDs | 4 | 1,000 | 512 | 0.822–3.321 / 0.817–3.273 | 168.18 / 168.85 aggregate |

Prose, code and structured rows report medians of two requests per setting,
using server-reported decode rates. They are repetitive synthetic content,
not uniform random-token workloads. The chat template forces thinking with
a 256-token budget; the 512 generated tokens include reasoning.

Random-token rows use `/v1/completions`, uniformly sampled valid base-token
IDs, identical seeds and one run per setting. They report client-observed
decode rates. For C1, both per-stream and aggregate decode use
`(output_tokens - 1) / (last_token_time - first_token_time)`, and are equal.
C4 aggregate uses all streams' tokens after their first token, divided by
the common interval from the earliest first token to the latest final token.
Its per-stream decode ranges were 42.42–52.93 off and 42.58–52.96 on.
End-to-end C4 throughput, which includes prefill, was 157.27 / 157.93 tokens/s.

TTFT includes request processing and first-token work. It is a prefill latency
proxy, not an isolated CUDA prefill measurement. The approximately 2.3–2.4%
chat TTFT reduction is a bounded observation from two repeats, not a confidence
interval or a claim that each individual switch improves performance.

## Correctness and memory

All 11 benchmark answers and reasoning strings matched exactly between settings.
Each setting also passed C1/C4 text, forced-tool and strict-JSON checks,
three two-turn tool-or-JSON union cases, and client cancellation followed by
a successful request. The six core CUDA tests passed, including byte equality
for packed side projections, planar convolution/state, fused HC normalization
and row-block HC computation, plus MXFP8 reference checks.

All four rank processes exited successfully. No swap, OOM or memory-watchdog
trip occurred. Minimum host memory available was 11.76 / 13.21 GiB on the two
ranks with switches off, and 11.69 / 12.15 GiB with switches on. The 32K pool
fits one full-length session, shared across four slots; it does not allocate
four independent 32K contexts.

Logs retained FP8 precision advisories, conservative memory-accounting gaps
and teardown allocation-ownership warnings. Cleanup reclaimed the allocations;
this short comparison does not establish sustained leak freedom. It also
does not qualify long-context retrieval, no-drafter parity or other models.
Earlier cross-model gate failures remain documented in the
[existing validation receipts](glm53-validation/README.md).

## FP4-Local

The official pinned checkpoint's GPU loader and live serving qualification
are still pending. The NVFP4 comparison above does not qualify FP4-Local.
