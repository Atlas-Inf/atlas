# Release record — `azeezish/atlas-gb10:3fcf8ff` (verified; publish pending)

- **Git**: `main` @ `3fcf8ff72` (`Merge pull request #103 … dflash2-grammar-mask-draft0`)
- **Image**: `azeezish/atlas-gb10:3fcf8ff`, image ID `sha256:88df0e8614bca0b2d4287265c35f51e66068031e6ff8591e9ede0ef09a52fb6e`, built 2026-09-27T05:28Z from `docker/gb10/Dockerfile` (all GB10 kernel targets).
- **Offline copy**: `reiner:~/jobqueue/done/521-release-finish-3fcf8ff/atlas-gb10-3fcf8ff.tar.zst` (2.04 GB, sha256 `c9f6f62e3a861d17bac8b8861dd432fedfb08a0968902e3b443a5f835d3ad727`).
- **Moving tags**: none yet. `:latest` is **not** moved; publishing is a human step (commands below).
- **Hardware**: DGX Spark GB10 (reiner, sm_121, 119.7 GB unified memory).

## Serve-matrix verdict — PASS (scoped)

`tests/run_all_models.py --roster` (reiner job 513) plus `tests/gate_results.py` (job 521): **5/5 models verified, full planned coverage.**

| model | config | coherence | fibonacci | tools | long context |
|---|---|---|---|---|---|
| nvidia/Qwen3.8-Flash-Next-NVFP4 | MTP, bf16 KV | 3/3 | 1/1 | 2/2 | 3/3 (14.6k) |
| nvidia/Qwen3.8-27B-NVFP4 | DFlash2 (`--dflash`), bf16 KV | 3/3 | 1/1 | 2/2 | 3/3 (14.6k) |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | plain, bf16 KV | 3/3 | 1/1 | 2/2 | 3/3 (14.8k) |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | bf16 KV | 3/3 | 1/1 | 2/2 | 3/3 (14.6k) |
| Qwen/Qwen3.6-35B-A3B-FP8 | fp8 KV | 3/3 | 1/1 | 2/2 | 3/3 (14.6k) |

- **Coherence probe:** `scripts/test_coherence.py` passed 89/89 against the shipped Flash-Next recipe serve with reasoning off. With thinking on, 3 "Edge Cases" checks fail by construction: they read `message.content` after `max_tokens` 10–20, and thinking spends that budget.
- **Scope limits:** the roster covers only the 5 checkpoints cached on reiner. There are no multi-node (EP/TP) phases, and Lightning DSpark isn't covered. The tok/s regression bar is **inert** because no baselines are committed in `tests/baselines`, so the check is liveness only.
- **Harness fixes this run needed:**
  - #109: follow the server's `Server live and ready` line.
  - #110: roster `env`.
  - The job ran its clients inside rootless docker's network namespace. On a rootless host, `--network host` containers aren't reachable from the real host.

## Gates on this code (reiner, 2026-09-26/27)

| gate | job | result |
|---|---|---|
| Flash-Next ST-995 | 280 | PASS, 84.72 / 84.68 |
| Flash-Next ST-995 with verify-active (now default, #84) | 464 | PASS, 84.92 / 84.93; paired vs job 280, p = 0.625 |
| Flash-Next agentic perf leg, verify-active | 465 | VALID, 1007/1007 turns, 73.3 min, TPOT median 48.9 ms, score 0.491 |
| 27B ST-995 (#86 + #96) | 383 | PASS, 88.14 / 88.68 |
| 27B agentic perf leg (MTP agentic recipe) | 457 | VALID, 1007/1007 turns, 107 min, score 0.6217 |
| 27B DFlash2 concurrency gate (#86) | 405 | C1 23.4 · C4 45 · C8 66 · C16 75; every floor met |

## Notable engine changes since `:latest` (`sha-c8b88c6`)

- **Merge train:**
  - #68: Flash-Next TTFT campaign, with QSA TC2 and GDN pipe as defaults.
  - #78: MTP FP8 reclaim.
  - #82: per-request host-heap growth fixed.
  - #83: lazy-BF16 cuBLAS copies are budgeted.
  - #67: SSE holdback.
  - #89 / #90: MoE fleet boot fixes.
  - #92: exact kernel-audit dedupe.
- **#84: speculative verify past the QSA bound is on by default on NVIDIA.**
- **#86: DFlash2 batched B×γ propose.** Also a 128-row verify pass, and Option B on by default.
- **#96: 27B dense-FFN prefill defaults to W4A16.** Outputs are chunk-invariant and perplexity improves 0.5–0.8 %.
- **#103: DFlash2 draft 0 is grammar-masked.**
- #99: op-dump fix. #109 / #110: release-harness fixes.

## Publish (maintainer)

```
docker tag azeezish/atlas-gb10:3fcf8ff azeezish/atlas-gb10:latest
docker push azeezish/atlas-gb10:3fcf8ff && docker push azeezish/atlas-gb10:latest
# if the ghcr mirror the recipes reference should follow:
docker tag azeezish/atlas-gb10:3fcf8ff ghcr.io/atlas-inf/atlas-gb10:3fcf8ff
docker tag azeezish/atlas-gb10:3fcf8ff ghcr.io/atlas-inf/atlas-gb10:latest
docker push ghcr.io/atlas-inf/atlas-gb10:3fcf8ff && docker push ghcr.io/atlas-inf/atlas-gb10:latest
```

Once `:latest` moves, change the title line to `latest @ 3fcf8ff`.
