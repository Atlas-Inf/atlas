<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# GLM-5.3 release validation

## October 20 update

The native FP4-Local and prefill update at
`5d652f8e0835231ead3bcac91d66e22632c08cbe` has passed CPU qualification
(7,602 tests), six core CUDA checks, and a bounded existing-NVFP4 off/on
comparison. [Measured results and limitations](../GLM_5_3_FLASH_PREFILL_RESULTS.md)
and [machine-readable evidence](prefill-nvfp4-20261020.json) retain the actual
runtime revision. The [six real-checkpoint GPU checks](fp4local-loader-20261020.json)
also passed with the corrected test executable at `3dbb41fb`; production code
is unchanged. [FP4-Local serving checks](prefill-fp4local-20261020.json) passed
the same bounded comparison plus two positive vision probes, with all 13
answer/reasoning pairs identical between settings. The two checkpoints are
not expected to produce identical outputs to each other.
A separate [single 64K random-input request](fp4local-64k-20261020.json) passed
with a 131K aggregate pool cap and four configured slots. It establishes
bounded execution, not maximum capacity or long-context retrieval quality.

The preceding main integration at `a37b789b` passed bounded GLM checks and
seven cross-model GPU gates, but **did not pass full qualification**:
the C16 throughput and 1,004-case normalized BFCL floors failed, and the
995-case BFCL run was interrupted without a score. The C16 failure also
reproduced on unmodified main; no corresponding main BFCL control was run.
[Terminal results](merge-main-20261020.json) preserve those distinctions.
Thresholds were not lowered. Neither this update's bounded checks nor the
earlier October 15 passes below supersede those failures.

## October 15 qualification (historical)

The combined NVFP4/DFlash implementation at `cb45065cb4f21c162ccd29dd837ef4b6be3aa52e` passed the final-source
CPU/static checks, ten canonical Qwen GPU regression gates and bounded paired
GLM checks. [Release summary](release-summary.json) links the exact receipts.
The submission adds documentation and evidence only after that measured source.
This source includes the rustls 0.23.46 security update in `Cargo.lock`.
The interrupted `9269a2f` campaign is superseded; its results were not reused.

CPU tests: 7,146 passed, 0 failed, 148 ignored.
CPU build/link stubs do not execute CUDA; all-features tests were compile-only.
The GPU records retain their actual source, binary, model and recipe identities.
Checkpoint-native GLM MTP and new Strix Halo support are outside this release.

GLM checks cover C1/C4 repeated output parity, API/schema/tools, disconnect/reuse,
a 6,404-token multi-chunk needle, and resident/NVMe cache restoration.
[Raw summaries](release-glm-summaries.json) retain their narrower qualification
fields; [rank-log review](release-glm-log-review.json) records the supporting
observations and warnings. Allocation ownership sweeps, FP8 advisory warnings,
conservative cache sizing and NVMe staging alignment remain documented limits.
These results do not establish general model quality, sustained leak freedom,
larger context capacity or 2.5M occupied tokens.

## Historical measurements

The earlier report follows unchanged. Its measurements belong to the original
revisions and are not relabeled as final-release benchmarks.

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# GLM-5.3 native-port validation

This is a sanitized summary of local measurements on two DGX Sparks. It records
what was measured, including failures and limits. It is **historical evidence
for `451bbc21f3ff8df356a159cb376eb7518e26c395`**, not a release-gate result for a
later review commit. [results.json](results.json) contains the unrounded
aggregates, checks, hardware identity and hashes of the source receipts.
Raw prompts, private service logs and deployment addresses are not included.

The [history map](history-map.json) records commit-message normalization for
review. Original and mapped commits have identical source trees, author and
committer records, dates and parent topology. The measured code `451bbc21`
maps to `6114dbc21f266680598d4b25999a1a80d1527dbd`; this is source
equivalence, not a new binary measurement.

The later [RC4 update](../GLM_5_3_FLASH_RC4.md) has separate
[receipts](rc4-results.json), including bounded concurrency and near-1M retrieval.
The measurements in this historical report retain their original revision.

## Review order

1. Read the [integration and provenance notes](../GLM_5_3_FLASH.md) for model
   loading, source attribution, licensed device code and numerical boundaries.
2. Inspect the [native qualification recipe](../../../scripts/dev/glm53_native/README.md)
   and its explicit environment file. DFlash is optional engine functionality;
   this measured profile enables it with separately obtained draft weights.
3. Review the measurements below, especially the output differences and scope
   limits. Performance similarity is not exact-output or broad accuracy parity.
4. Check subsequent changes against the measured revision. Runtime changes
   require new GPU qualification; historical results remain pinned to their
   original build.

## Configuration and identity

The measured standalone binary has SHA256
`6f65cc458e142740d28f757f54f4eab14d4e96eee46a519e3d6fe7bc4c69bfcb`
and GLM kernel-set hash `5fd26547c6f6`. Hardware was two NVIDIA GB10 GPUs on
AArch64, driver 580.126.09 and CUDA 13.0. The deployed reference was Enntity
Atlas revision `72588541a466f53cb9d38afd74c515e14ab601db`.

Both profiles used NVIDIA GLM-5.3-Flash NVFP4, TP2/EP2, FP8g128 latent KV,
DFlash2 with gamma 8 and a 2048-token window, and a 524,288-token session limit.
The native profile loaded the original NVIDIA checkpoint; the reference used
its converted overlay. The native shared physical KV cap was 524,288 tokens
across at most four requests; the reference pool differed. This does not reserve
524,288 tokens for each request. Exact checkpoint snapshot revisions and NCCL
versions are not present in the receipts summarized here; repository names
alone are not immutable checkpoint pins.

Native KDA stores persistent state in FP32 but rounds its accelerated internal
recurrence to BF16. The native sparse accelerator admits pieces ending at or
before 32,768 tokens; larger contexts use Atlas's existing sparse fallback
outside that domain. The 524K end-to-end result does not extend the native
sparse kernel's domain, prove bit-exact FP32 recurrence, or qualify 1M context.

## Content throughput

Measured on 2026-10-08. Each workload used approximately 10,000 input tokens and
exactly 1,024 generated tokens. Structured prompts requested an analytical JSON
report over synthetic records; code prompts requested a Python replacement
module and tests; prose prompts requested a report over synthetic field notes.
These are seeded content workloads, not unstructured random-token prompts.
JSON decoding was unconstrained. The capped outputs are excerpts, not validated
complete JSON documents or executable modules.

Three distinct trials per workload at C1 and three waves per workload at C2/C4
ran sequentially for each build on the same pair, without other request traffic.
Both builds received identical prompt token IDs. All 18 C1 and 108 concurrent
requests completed at the requested length, with zero measured prefix hits.
The reference enabled prefix caching; native disabled it. Temperature was zero
and repetition penalty one. The GLM template used low-effort reasoning despite
`enable_thinking=false`; reasoning counts toward generated tokens.

Values below are medians. Every paired cell is **reference / native**.
Rates are tokens/s; TTFT is seconds.

| Workload | Streams | Effective prefill | Server TTFT | Per-stream decode | Aggregate decode |
|---|---:|---:|---:|---:|---:|
| Structured | 1 | 2038.28 / 2145.11 | 4.90 / 4.64 | 53.24 / 51.90 | 53.24 / 51.90 |
| Structured | 2 | 1353.12 / 1424.05 | 7.43 / 7.04 | 31.13 / 30.93 | 65.29 / 65.88 |
| Structured | 4 | 852.96 / 888.31 | 11.64 / 11.21 | 19.98 / 19.74 | 87.88 / 83.66 |
| Code | 1 | 2024.91 / 2142.77 | 4.93 / 4.65 | 56.15 / 57.44 | 56.15 / 57.44 |
| Code | 2 | 1351.71 / 1419.87 | 7.45 / 7.09 | 34.86 / 33.37 | 74.33 / 68.58 |
| Code | 4 | 743.69 / 963.36 | 13.39 / 10.37 | 20.75 / 21.17 | 92.41 / 92.77 |
| Prose | 1 | 2015.21 / 2141.46 | 4.93 / 4.64 | 40.93 / 41.46 | 40.93 / 41.46 |
| Prose | 2 | 1351.27 / 1415.65 | 7.47 / 7.12 | 25.97 / 26.45 | 54.68 / 53.97 |
| Prose | 4 | 848.22 / 860.59 | 11.77 / 11.97 | 15.97 / 16.19 | 67.87 / 68.25 |

Effective prefill is prompt tokens divided by server TTFT, including scheduling;
it is not isolated kernel throughput. Per-stream decode uses generated tokens
minus one divided by server decode duration, cross-checked against streamed
content/reasoning timestamps. At C1, aggregate decode is the same metric.

At C2/C4, aggregate decode is independently measured from
`atlas_decoded_tokens_total` sampled at 4 Hz during the interval when **all**
streams are decoding. One second is trimmed from each edge; samples with
round-trip time of 200 ms or more are excluded. The shortest retained interval
was 23.80 seconds. This is not the sum of full-request median speeds.
Whole-wave output throughput including prefill is a separate field in the JSON.
Counter deltas matched generated-token usage exactly. Maximum client/server
per-stream timing disagreement was 0.412% at C1 and 0.326% at C2/C4.
Three trials do not establish statistical performance superiority.

## Quality and numerical checks

| Check | Measured result | Limitation |
|---|---|---|
| Exact text comparison | 2/4 short visible answers exact; 0/9 long visible answers exact | Identical prompt token IDs and temperature zero did not yield identical output; cause not isolated |
| Short instruction failure | Native explanation: 122 words; requested 70–100; reference: 92 | Both explanations were coherent, but native missed the length constraint |
| Basic coherence | 8 responses, 6/6 categories passed | Manual semantic review plus deterministic formatting; short, single-stream smoke only |
| Text/tools | 5/5 arithmetic, JSON, text streaming, tool and tool-streaming checks passed | Bounded smoke cases |
| Retrieval | 3/3 at 32,608 input tokens; 20/20 at 130,016 and 523,232 | One fixture at each length, not broad long-context accuracy |
| Vision | 2/2 counterbalanced 224-pixel spatial/color probes passed | No general vision or video qualification |
| Lifecycle | 13 exact completions plus one client disconnect across 14 requests | Isolation and reuse checked; internal cancellation acknowledgement not asserted |
| Other-model regression | 8/8 Qwen3.5-27B requests passed | Used the qwen3.6-27b target; other families unqualified |
| Native sparse GPU dispatch | Analytic zero/constant/one-hot cases passed; 8 sampled reference results exact | Sampled byte matches alone are not an independent oracle |
| Native KDA GPU dispatch | Two tests covering FP64/FP32 comparison, tails, split and continuation passed | Numerical bounds, not bit-exact FP32 recurrence |

The largest retrieval used 523,232 prompt tokens, recovered 20/20 records,
and measured 237.13 s TTFT and 51.68 generated tokens/s. That corresponds to
2,206.48 effective prompt tokens/s. Memory availability minima for that profile
were 9.89 and 10.27 GiB; the 8 GiB host guards did not intervene. This does not
establish the largest safe context for other workloads or concurrent sessions.

The source receipt also records passing workspace tests, formatting, clippy,
rustdoc, typos, license headers, file-size checks and cargo-deny, plus an ARM64
release build and a clean Docker build. Full-model runs used the standalone
binary, not the Docker image. Those checks belong to the recorded source
revision; they are not claims that later edits were retested.

## Reproduction boundaries

Follow the [qualification recipe](../../../scripts/dev/glm53_native/README.md)
from a clean shell, first with its 32K limit. Use the explicit 524K override
only after bounded smoke tests and memory checks. The recipe supplies the actual
build target, dependencies, settings and rank commands. Record source and binary
hashes, immutable checkpoint revisions, driver/CUDA/NCCL versions and fabric
topology for a new run. DFlash checkpoint distribution rights are separate from
engine implementation; this directory contains no weights.

A bounded API smoke after launching the recipe is:

```bash
curl --fail-with-body http://127.0.0.1:18993/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-local-port","messages":[{"role":"user","content":"What is 17 times 23? Answer with the number only."}],"temperature":0,"max_tokens":128,"enable_thinking":false}'
```

Expected arithmetic answer: `391`. A successful smoke is not a reproduction of
the content throughput table. The [offline workload generator](../../../scripts/dev/glm53_native/generate_workloads.py)
and [manifest](../../../scripts/dev/glm53_native/workload-manifest.json) reconstruct
all 63 measured synthetic request bodies: nine C1 requests and 54 requests
across 18 concurrent waves, excluding warmup. They require only Python's
standard library and do not contact a tokenizer or service.

From the repository root, choose a new output file:

```bash
python3 scripts/dev/glm53_native/generate_workloads.py --help
python3 scripts/dev/glm53_native/generate_workloads.py --output workloads.jsonl
```

The generator refuses to overwrite an existing file and verifies each generated
body against its canonical SHA256. Each JSONL row includes `body`, suite/wave
and stream identifiers, concurrency, seed, source block count, recorded prompt
token count, and the original prompt-token-ID hash. The 63 reconstructed bodies
were compared offline against the original benchmark datasets; all matched
exactly. No full token arrays or generated model responses are bundled.

An independent measurement harness should add the selected `model` and streaming
options to `body`, submit rows grouped by suite/wave, and preserve their
`min_tokens=max_tokens=1024` settings. Re-tokenize against the chosen checkpoint
and template before comparing builds; hash token arrays using the manifest's
specified encoding. Recorded token counts are historical expectations, not a
claim that the offline generator ran a tokenizer. Warm up before measuring,
collect timings/counters as defined above, and reject competing traffic,
prefix-cache hits and incorrect output lengths.

The timing harness and raw results are not bundled. Receipt SHA256 values in
`results.json` identify the source summaries but are not independent verification
of those underlying measurements. Exact input reconstruction does not guarantee
identical timings or generated output. Native operator tests are in the
repository; see `glm_sparse_native_gpu_tests.rs` under
`crates/spark-model/src/layers/ops/` and the KDA tests linked from the integration
notes. Run ignored GPU tests only on an otherwise idle, appropriately bounded
GPU host.
