# Atlas Lightning cd2218e — Requested Throughput, Concurrency, and GSM8K-200

Status: **COMPLETE / ALL REQUESTED LANES PASS**

## Identity

- Source/canonical `main`: `cd2218ec426a907f681c602feb966611d6db7443`
- Tree: `f90190176d55b790796b05b097f96fddaf43c78b`
- Binary SHA-256: `98affc23b5f41829049a19cf702e402edcda1d7a570443203394090095405fd4`
- Image: `sha256:542653d59a7a1140c651af1e939866dcd64b5bfe9af964fc1b6f91ce9d1c9b58`
- Hardware: one GB10/SM121; ModelOpt NVFP4; DSpark gamma 4 / K=3.
- r0b0bench: `1.0.0rc2`, `core-subset --only latency,concurrency,throughput,gsm8k`.
- This selected-lane epoch is correctly marked `invalid_for_publish=true`; it is complete for the user-requested metrics, not a public full-suite claim.
- r0b0bench `report.json.started_utc` is a known naming defect in this pinned source: `cli.py` assigns it after the lane loop, so it is the report-finalization timestamp. Use source identity in `METRICS.json`, process evidence, and telemetry UTC bounds for the actual epoch.

## Model startup/load

- Application start to API ready: **33.853 s**.
- CUDA/PTX initialization: 17.475 s.
- Target weight I/O: 8.705 s.
- Target weights complete to DSpark store ready: 3.018 s.
- DSpark store ready to active proposer installed: 3.721 s.
- Proposer installed to API ready: 0.885 s.

## Latency — PASS

Five streaming reps (first dropped; four stable) plus three non-stream reps.

- Client-observed streaming TTFT mean: **375.040 ms**.
- Server-reported stable TTFT mean: **315.483 ms**.
- Mean ITL: **19.492 ms**.
- Mean per-request ITL p95: **46.264 ms**.
- Streaming E2E mean: **1,867.915 ms**.
- Non-stream E2E mean: **1,650.373 ms**.
- Failed requests: 0.

## Concurrency ladder — PASS

Each level: 3 reps, first dropped, 512 output tokens/request.

| C | Aggregate output tok/s | Speedup vs C1 | Median per-client tok/s | Mean E2E | Stable completed | Failed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.179 | 1.000x | 84.189 | 6.082 s | 4 | 0 |
| 2 | 152.837 | 1.816x | 76.477 | 6.695 s | 8 | 0 |
| 4 | **168.417** | **2.001x** | 42.129 | 12.144 s | 16 | 0 |
| 6 | 155.839 | 1.851x | 26.020 | 19.688 s | 24 | 0 |

- Peak throughput: **C4, 168.417 aggregate tok/s**.
- C6 regressed 7.47% from C4; this profile's throughput-optimal tested width is C4.
- Total failed requests: 0.

## Dedicated C1 throughput — PASS

Five 2,048-token decode requests; first dropped.

- Stable decode series: `67.696, 68.544, 67.862, 67.476 tok/s`.
- Median: **67.779 tok/s**.
- Mean: **67.895 tok/s**.
- Warmup: 57.580 tok/s.
- All HTTP 200, 2,048 completion tokens, `finish_reason=length`.

## Prefill throughput proxy — PASS

Three requests, first dropped. The harness's rough prompt constructor overshot its nominal 14K target; actual server usage was exactly **24,541 prompt tokens** on every row.

- Raw series: `1,658.401, 1,684.098, 1,710.105 prompt tok/s`.
- Stable median/mean: **1,697.102 prompt tok/s**.
- This is an end-to-end wall proxy including a 16-token decode, not a pure kernel-only prefill measurement.
- Failed requests: 0.

## GSM8K-200 quality subset — PASS

- Method: `gsm8k_0shot_flexible_extract`.
- Rows: **200** raw / 200 expected.
- Correct: **191** (independently recounted from raw rows).
- Accuracy: **95.5%**.
- Wilson 95% CI: **91.67%–97.61%**.
- Concurrency: 2; max tokens: 512; seed: 0.
- HTTP/infra errors: 0.
- Elapsed: 513.137 s.

## DSpark server internals across the epoch

These values are grouped by live scheduler width across the entire run. C2 includes the GSM8K workload as well as the C2 ladder; C4/C6 are the concurrency workload.

| Width | Verify median / p95 | Propose median / p95 | Mean accepted | Acceptance fraction | Graph captures / replay hits |
|---:|---:|---:|---:|---:|---:|
| 1 | — | — | 1.779/3 | 59.30% | per-sequence path |
| 2 | 45.2 / 47.6 ms | 8.6 / 9.2 ms | 1.952/3 | 65.07% | 1 / 8,226 |
| 4 | 61.8 / 63.3 ms | 14.4 / 15.1 ms | 2.952/3 | 98.41% | 1 / 734 |
| 6 | 94.4 / 97.4 ms | 21.1 / 22.1 ms | 2.945/3 | 98.18% | 1 / 712 |

## Continuous telemetry

- Samples: **529**.
- Coverage: **1,058.571 s / 17.643 min**.
- Collector requested 1 s; actual median interval was 2.006 s because `docker stats --no-stream` is blocking. No benchmark interval was uncovered.
- Health failures: **0**.
- Container-not-running samples during benchmark: **0**.
- Prometheus snapshots: 53.

Resource distribution:

| Metric | Min | Median | Mean | P95 | Max |
|---|---:|---:|---:|---:|---:|
| GPU utilization | 0% | 95% | 87.36% | 96% | 96% |
| GPU temperature | 45°C | 79°C | 76.17°C | 84°C | 84°C |
| GPU power | 10.72 W | 56.59 W | 53.04 W | 64.69 W | 68.02 W |
| Spark unified-memory process | 97,358 MiB | 100,472 MiB | 100,758 MiB | 106,626 MiB | 106,626 MiB |
| Host MemAvailable | 5.536 GiB | 11.600 GiB | 11.361 GiB | 14.676 GiB | 14.758 GiB |
| Swap used | 1.496 GiB | 1.499 GiB | 1.499 GiB | 1.504 GiB | 1.504 GiB |
| Host CPU busy | 0.40% | 5.67% | 5.47% | 6.42% | 10.12% |
| Container CPU | 0.06% | 100.28% | 94.25% | 100.48% | 101.97% |

Docker RSS (~0.86 GiB) excludes most GB10 unified GPU allocations; the `nvidia-smi` compute-process figure is the relevant process-memory metric.

## Reliability and teardown

- r0b0bench total elapsed: **997.962 s / 16.633 min**.
- Lane statuses: latency PASS, concurrency PASS, throughput PASS, GSM8K PASS.
- Total infrastructure errors: **0**.
- CUDA error markers: 0.
- CUDA status 700/716/900 markers: 0.
- Proposal failure markers: 0.
- Connection-reset markers: 0.
- Panic markers: 0.
- Benchmark container removed: PASS.
- Running containers after teardown: 0.
- GPU compute processes after teardown: 0.
- Canonical source remained clean at the exact SHA.

## Evidence

- r0b0bench report: `r0b0bench/requested-metrics-20260818T210217Z/report.json`
- Reduced metrics: `METRICS.json`
- Raw telemetry: `TELEMETRY-SUMMARY.json`
- Raw Prometheus snapshots: private raw snapshots (not published)
- Full server log: `SERVER-SANITIZED.log`
- Run descriptor: source identity in `METRICS.json`
