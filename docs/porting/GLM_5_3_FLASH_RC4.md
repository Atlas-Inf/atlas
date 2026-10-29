<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Native RC4 integration

The October 12 update adapts fourteen source commits after `a1c9b1bc` through
[Enntity/atlas
`0b515eec`](https://github.com/Enntity/atlas/commit/0b515eec96a9c6f637d60d9d910b18c7e7a6cb94)
into native Atlas. [SparkGLM PR 46](https://github.com/Enntity/sparkglm/pull/46)
uses the layered build
[`5c523903`](https://github.com/Enntity/atlas/commit/5c523903c886d1abf2f4bda6a0af1cdf91c28523).
That layered build is a source reference, not an Atlas runtime dependency. Atlas
retains its native KDA and sparse-MLA implementations. The external prefill
bridges are not reintroduced; the optional display-memory code was already
adapted in the [RC2 integration](GLM_5_3_FLASH_RC2.md).

## Source history and credits

Jason McCartney / Enntity authored the fourteen source commits. The [import
map](glm53-validation/rc4-import-map.json) records each full source revision,
author and corresponding native commit. The final formatting-only source change
was already represented by native formatting. Imported commits retain author
credits and original source references; subsequent native review and
qualification changes are separate commits.

| Area | Source commits | Native behavior |
|---|---|---|
| RDMA staging | `f3538416` | Optional SM staging for one-shot all-reduce in a PDL chain |
| MoE routing and dense verification | `e1e5fbdb`, `2e780044` | Fused routing chain and optional dense FFN split across the pair |
| KDA and MLA | `05b978a5`, `553ecd0c`, `7e13c1bb` | Fixed-grid record commits, fused MLA front and owner-path cleanup |
| DFlash selector and uploads | `f473f620`, `5b93d84b`, `2b3daefd`, `a6f35bf7` | Threshold top-k/parallel merges, candidate-list exchange, retained asynchronous uploads and experimental edge-weight twins |
| Weight twins | `496a77fd`, `ad047638`, `bc9d0de4` | Tiled MXFP8 projections and optional NVFP4/MXFP8 draft convolution projections |
| Formatting | `0b515eec` | Canonical-verification GPU test formatting |

These changes build on Reiner Schmidt / Mango-kid and the Atlas contributors'
model and runtime work. The [prior-art inventory](../glm-prior-art.md) retains
the original credits and distinctions between adapted source and design ideas:
MoonshotAI / FlashKDA, NVIDIA and FlashInfer contributors, knapcio,
TensorFold/ashhart, jayleaton, MiaAI-Lab, RiNGSiDE, Reederey87, vLLM and the
communication/display-memory contributors. In particular, the new selector
edge-weight experiment credits MiaAI-Lab's TensorFold recipe. Existing vendor
pins, copyright notices and licenses remain authoritative. Target and draft
weights are separate downloads; DFlash remains explicitly selected and the
chosen draft checkpoint retains its own terms.

## Validation status

The native executable at `b9bf0e6b` passed the bounded full-model baseline on
both GB10 ranks with driver `580.126.09`: 32K session limit, 8192 shared blocks
(131,072 aggregate tokens), four slots, FP8 g128 KV and optional DFlash2. All
four greedy responses matched exactly between C1 and C4, and a repeated C4 run
matched again. A synthetic 18,658-token archive recovered all four buried keys.
The minimum sampled host memory available was 12.78 GiB on the head and 13.07
GiB on the worker. No runtime memory guard fired.

Host validation passed 7,124 tests, workspace clippy, formatting, rustdoc,
license/typo checks and kernel-shadow structure. Thirty-four GPU test
invocations passed: native fusion, dense splitting, tiled projections, KDA
commits, single/split/batched selectors, canonical sharding, ABI and RDMA
staging. The additional batched-selector test at `693d8b6a` changes tests and
comments only; it ran with PDL both off and on. The RDMA NIC stress checked
65,536 operations per policy with no wrong values or stale staged bytes.

Source [`rc4-candidate.env`](../../scripts/dev/glm53_native/rc4-candidate.env)
after the historical environment and RC2 overrides on both ranks to select the
bounded baseline. Use `set -a` while sourcing so the values are exported. The
[RC4 receipts](glm53-validation/rc4-results.json) record the bounded and
long-context results. A representative throughput comparison remains pending.
The historical report for `451bbc21` and [RC2
results](glm53-validation/rc2-results.json) remain valid only for their recorded
revisions and configurations.

The bounded full-model suite also passed with `ATLAS_PDL=1`,
`ATLAS_GLM_DECODE_GEMV_BATCH=1` and `ATLAS_RDMA_ONESHOT_PDL=1` together. Its
C1/C4/repeated-C4 responses matched the disabled baseline exactly, and all four
18.7K archive keys were recovered again. This qualifies the tested RC4 option
bundle; it does not establish that RC2's previous nonfinite prefill failure is
fixed with arbitrary subsets of the new fusions disabled. The candidate file
keeps the conservative launch policy explicit. No new Docker or other-model GPU
qualification is asserted here.

## Context capacity

The checkpoint advertises **1,048,576 positions per session**, including input
and generated tokens. A configured session limit is not measured capacity. The
two ranks share one logical KV pool: aggregate active context is its negotiated
block count times 16, shared by sessions and resident prefixes. Do not add the
ranks' block counts. Four session slots do not mean four full 1M windows, and an
NVMe prefix tier does not enlarge active GPU context.

A 1M candidate needs `--max-seq-len=1048576` and sufficient allocated KV blocks;
`ATLAS_KV_MAX_BLOCKS=65536` only caps nominal aggregate capacity at 1,048,576.
Allow space for generation and speculative allocation, and inspect the actual
pool rather than treating its cap as a guarantee. Keep target `fp8_g128` KV.
Startup on the tested pair negotiated exactly 65,536 blocks on both ranks for
the smaller-workspace profile, with about 11.3 GiB host memory available on the
head after a warm request. The largest completed input contained 1,039,977
tokens and recovered 18/20 buried keys.

Begin with bounded context and measure both ranks after warming the chosen
prefill and concurrency settings. A 2048-token prefill chunk and fewer active
slots reduce workspace/state overhead. This smaller arena requires
`ATLAS_GLM_SPARSE_NATIVE=0` and `ATLAS_GLM_HC_TF32_PREWARM=0`: their bounded
native accelerator/prewarm plans require at least 4100 workspace rows. The
existing in-tree sparse tensor-core pipeline handles prefill instead; keep
`ATLAS_GLM_SPARSE_PREFILL_TC=1`, `ATLAS_GLM_SPARSE_PREFILL_KV_REUSE=1` and
`ATLAS_GLM_SPARSE_PREFILL_PIPE=1`. An initial attempt with the larger-workspace
flags left on was rejected during startup, before any memory guard fired. With
transient dense BF16 prefill (`ATLAS_GLM_DENSE_PREFILL_BF16=2`), 1024-token
chunks leave insufficient scratch for a 96 MiB weight matrix; they are not a
drop-in smaller profile. DFlash's 2048-token window already bounds its context
buffers. Removing it changes index-tail and rollback allocations, so do not
assume that alone saves memory.

The native-MTP repair domain and the 2048-token DFlash draft cache are distinct
from the target context. Do not apply native-MTP repair limits to the DFlash
lane: the 262,095-token probe still used DFlash, with 454 accepted prediction
tokens among 575 output tokens. The target attends its full cached history while
the drafter keeps its bounded window.

The CLI OOM reserve protects initial weight loading; it is not a runtime free
memory floor. Check host and CUDA free memory through warmup and the context
ramp. Keep display-memory allocation disabled on driver `580.126.09`, which is
outside the qualified allowlist. These receipts qualify one near-full
single-session input on this pair; they do not establish a larger aggregate pool
or general long-context accuracy.

## Measured context ramp

All three requests used the same C1 profile with no prefix-cache hits. TTFT
includes request processing and prefill; input/TTFT is effective prefill rate,
not an isolated kernel measurement. Decode is the server-reported output rate
for these short retrieval answers, including reasoning tokens. It is not a
representative prose/code benchmark.

| Input tokens | Output tokens | Keys recovered | TTFT (s) | Input/TTFT (tok/s) | Decode (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 63,965 | 579 | 20/20 | 34.27 | 1866.6 | 68.64 |
| 262,095 | 575 | 20/20 | 147.68 | 1774.7 | 62.83 |
| 1,039,977 | 569 | 18/20 | 868.34 | 1197.7 | 49.68 |

The minimum sampled available host memory was 10.50 GiB on rank 0 and 11.43 GiB
on rank 1; the 8 GiB guard did not fire. Speculative decoding accepted 446
prediction tokens in the largest answer (including prompt-lookup drafts); DFlash
remained active. The receipt includes request hashes and generator record
counts. The near-1M answer assigned incorrect values to KEY_04 and KEY_05; the
other 18 matched. This is a capacity result with imperfect retrieval, not a full
accuracy pass. The workload probes retrieval from synthetic filler rather than
arbitrary reasoning over a million tokens.

## Reproduce the larger profile

First qualify the bounded profile using the [rank
commands](../../scripts/dev/glm53_native/README.md). Stop both ranks before
changing their allocation. In fresh shells on both hosts, apply this environment
in order:

```bash
set -a
. scripts/dev/glm53_native/qualification.env
. scripts/dev/glm53_native/rc2-candidate.env
. scripts/dev/glm53_native/rc4-candidate.env
ATLAS_PDL=1
ATLAS_GLM_DECODE_GEMV_BATCH=1
ATLAS_RDMA_ONESHOT_PDL=1
ATLAS_KV_MAX_BLOCKS=65536
ATLAS_GLM_PREFILL_2048=1
ATLAS_GLM_PREFILL_4096=0
ATLAS_GLM_SPARSE_NATIVE=0
ATLAS_GLM_HC_TF32_PREWARM=0
set +a
SESSION_TOKENS=1048576
```

In **both** rank commands change `--max-prefill-tokens=2048`,
`--max-batch-size=1` and `--max-num-seqs=1`, retaining
`--max-seq-len="$SESSION_TOKENS"`, FP8 g128 KV and the remaining arguments.
Confirm startup actually allocates 65,536 logical blocks. The validation used an
additional host-memory monitor with an 8 GiB available-memory floor that stopped
only the test service; this monitor is not part of the Atlas CLI.

The [offline request
generator](../../scripts/dev/glm53_native/generate_context_probe.py) reproduces
the archive workload. Record counts 1,576, 6,368 and 25,388 generate the three
ramp inputs; record counts are not token counts. For example:

```bash
python3 scripts/dev/glm53_native/generate_context_probe.py --records 25388 > request.json
```

Tokenize with the same model/template before submission. Each request reserves
1,024 generated tokens and contains 20 keys at evenly spaced depths. One archive
retrieval run is not a general long-context accuracy benchmark.

## Four-session aggregate-context probe

A separate run of the same `b9bf0e6b` executable allocated **93,750 logical
blocks = 1,500,000 aggregate token positions**. It retained the smaller-arena
settings above, changing `ATLAS_KV_MAX_BLOCKS=93750`, `--max-seq-len=655360`,
`--max-batch-size=4` and `--max-num-seqs=4` on both ranks. Target KV remained
FP8 g128, DFlash remained enabled, and prefix caching and swap stayed disabled.
The [multi-session receipts](glm53-validation/rc4-multisession-results.json)
record the profile, scores and overlap evidence.

Four distinct random archives contained **1,419,934 input tokens in total**.
Each had 20 different keys and six distributed parcel events requiring custody,
location, balance and event-order tracking. All four explanations were readable
and consistent with their supplied events and arithmetic.

| Session | Input tokens | Keys | Exact state fields | Client TTFT (s) |
|---:|---:|---:|---:|---:|
| 0 | 354,979 | 20/20 | 5/5 | 434.40 |
| 1 | 354,990 | 20/20 | 5/5 | 906.09 |
| 2 | 354,992 | 20/20 | 5/5 | 666.93 |
| 3 | 354,973 | 20/20 | 4/5 | 212.40 |

The strict score was **80/80 keys and 19/20 state fields**. Session 3 repeated
unchanged holders in its custody-path array. Collapsing adjacent duplicates
produces the expected path; final holder, location, balance and event order
were correct in every session. This secondary interpretation does not replace
the strict score. The short four-session control recovered 80/80 keys with
17/20 exact state fields; its three mismatches were the same duplicate-holder
formatting issue.

GLM prefills these requests serially, so TTFT includes queueing. To establish
resident aggregate context, requests used `min_tokens=8192` and
`max_tokens=16384`, with the answer JSON first and artificial padding afterward.
Once all four answers arrived, each stream had to emit again in a common window
at least two seconds later. The clients then canceled all four streams. The
server recorded 173 four-stream verification steps and no preemption, swapping,
warnings or errors during the run. Thus the four input contexts were resident
together, rather than merely summed across completed requests. The padding is
excluded from coherence review, and canceled streams have no claimed completion
throughput. Total request wall time was 942.06 seconds. Four fresh requests
passed after cancellation, exercising reuse of all slots.

The minimum sampled available host memory was **8.16 GiB / 9.22 GiB** on the
head/worker; neither 8 GiB guard fired. This establishes the recorded workload,
not a long-duration stability guarantee. The requested approximately **2.5M**
pool was **not allocated or tested**: it would add about 4.88 GiB per rank over
the 1M pool before C4 state costs, exceeding the retained memory safety margin.
The display-memory feature remains disabled on driver `580.126.09`. Raising a
KV cap alone does not reclaim that memory or establish safe additional capacity.
