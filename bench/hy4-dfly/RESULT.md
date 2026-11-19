# HY4 DFly: code19.656 tok/s (-5.05%); spec-37057536a restored

| Code harness, n=3, commit0d3accd45 | Value |
|---|---:|
| Verify steps / width |231 /6 |
| Accepted / proposed |531 /1155 (45.974%) |
| Accepted drafts / step |2.298701 |
| Mean emitted tokens / verify step |3.298701 |
| Verify API host time |97.899009 ms |
| Commit/rollback/replay host time |47.887476 ms |
| Consecutive-step intervals |228 |
| Mean interval |165.666276 ms |
| Matched-interval verify / commit |97.928263 /47.430241 ms |
| Matched-interval accepted drafts / step |2.276316 |
| Draft + unresolved host residual |20.307772 ms |
| Separate draft CUDA / host overhead |N/A / N/A |
| Independent reference acceptance, same continuation |N/A |
|50 tok/s matched-interval budget / required saving |65.526316 /100.139961 ms per step |

| Conditional changes; not implemented or measured | Saving / estimate |
|---|---:|
| No replay; retain full-accept commit cost |47.204750 ms;27.657 tok/s |
| Also reduce verify to60–70 ms |27.928–37.928 ms;36.189–40.683 tok/s |
| Verify needed after no replay for50 tok/s |44.993053 ms |

Checkpoint publisher reports code36.5 tok/s and3.80 “DFly accepted tokens a
step”, five drafts, median of3,300 output tokens, different prompt and FP8 KV:
https://huggingface.co/atakans/Hy4-flash-NVFP4 .
Not a same-continuation acceptance comparison or an integration-bug verdict.
Publisher explain/story:2.75/1.85; the bonus-token counting convention was not
independently verified.
HY3/AngelSpec results are a different checkpoint/block size/temperature.

Separate CUDA timing and live-input reference replay were not collected before
the no-restarts instruction. No new measurement restart, optimization or serving
change. No substantiated two-change50-tok/s claim.
Full evidence: `/home/robert/hy4-dfly-20261118/analysis/numbers.json`.

## Follow-up: reference acceptance blocked by the reserve

Read-only CUDA memory probe, Nov18 Toronto, serving unchanged:

| Rank0 memory | Bytes |
|---|---:|
| Free |7405494272 |
| Required reserve (12 GiB) |12884901888 |
| Above reserve |−5479407616 |

No reference weights or target replica were loaded. Historical production logs
contain acceptance and output IDs, not the live normalized hidden/KV/index inputs.
The same-continuation reference result remains unavailable; no integration bug
has been established. No serving restart, cache eviction or reserve reduction.

Upstream vLLM's `vllm/v1/spec_decode/metrics.py` defines mean acceptance length
as `1 + num_accepted_tokens / num_drafts`, including the bonus token:
https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py .
The publisher README does not define whether its column uses that statistic;
the linked recipe source returned404, so this link is unverified.

| Counting convention | Atlas | Publisher code |
|---|---:|---:|
| Draft tokens only, measured Atlas |2.298701 |Unknown |
| Including bonus, measured Atlas |3.298701 |Unknown |
| If publisher3.80 includes bonus: draft tokens only |2.298701 |2.80 |
| If publisher3.80 excludes bonus: draft tokens only |2.298701 |3.80 |

Even a normalized publisher comparison is a different prompt/FP8-KV workload,
not evidence of a same-continuation acceptance gap.
