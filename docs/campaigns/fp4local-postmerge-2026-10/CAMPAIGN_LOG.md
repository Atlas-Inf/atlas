# FP4-Local post-merge campaign: running log (autonomous)

**Goal:**
- Finish FP4-Local (`nvidia/Qwen3.8-Flash-Next-FP4-Local`) after #195: verified on main, recipe vetted, AMD build restored.
- Then work the owner's ranked list:
  1. direct exact-hit resume
  2. resident compressed CSF scales
  3. independent Qwen branch overlap
  4. device-side stochastic sampling
  5. shared scratch planning
  6. measured-cost scheduler
- Plus deployment gaps from logs and issues (#172, #180).

**Started:** 2026-10-23 06:00 UTC, for 10 h.

**Boxes:**
- **sparky** (GB10): all GPU work.
- **reiner + spark2**: running someone else's v1 gate runs. Hands off.
- **framework / strix** (gfx1151): AMD builds and soaks.

**Gate for anything that touches FP4-Local:** BFCL golden ST-995 (`bfcl-subset`, no overrides) plus `agentic-webserver` at 50 iterations, on the recipe profile.
- Recipe profile: r1short, MTP 3 drafts, EXACT_VERIFY / DECODE_FUSE / BATCH_FAST / BATCH_SMALL / HC_FAST, PLE cache 2 GB, prefix caching ON.
- Reference: 87.44 / 87.32 and 50/50, 50/50.

## Reproduce
- Bare-metal serve: `~/code/nightq/serve-flashnext.sh fp4local r1short` with `profile-r1short.env` (sparky).
- Recipe: `sparkrun run fp4l.yaml --solo --hosts localhost --image ghcr.io/atlas-inf/atlas-gb10:sha-504063a`. `fp4l.yaml` is sparkrun-recipes PR #21 at `2cbe348b`.
- BFCL: `spark benchmark run bfcl-subset --url http://127.0.0.1:8890 --model nvidia/Qwen3.8-Flash-Next-FP4-Local --format json`.
- Agentic: `spark benchmark run agentic-webserver ... --yes --param iterations=50 --param wall_budget_s=100000`. Needs `~/.cargo/env`.

## Results ledger

| When (UTC) | Build | What | Result | Run id |
|---|---|---|---|---|
| 10-20 | 5ddfabb6e (pre-merge) | BFCL ST-995 | 87.44 / 87.32 | run-1792488042130537521 |
| 10-20 | d773b218e (pre-merge) | agentic 50 | 50/50, 50/50, 11.94 s/turn | run-1792475013132909319 |
| 10-22 08:57 | main 504063ad4 + HIP fix d2d114c02 | agentic 50 | 50/50, 50/50, 12.43 s/turn (Σwall 4762 s, 379 turns) | run-1792654228906033074 |
| 10-22 10:26 | same | BFCL ST-995 | **87.44 / 87.32** (identical) | run-1792664799374893012 |
| 10-22 08:36 | strix #197 fafb7349d + fix, Qwen3.8-27B-NVFP4 | agentic 50 | 50/50, 50/50, 18.735 s/turn (first strix agentic run) | run-1792658188290604510 |
| 10-22 | framework, main + #197 + fix | strix-hip build | OK (embed_from_argmax .co for t0/t3/t4) | job 077 |
| 10-23 07:15 | recipe #21 via sparkrun, ghcr sha-504063a | BFCL ST-995 | **87.94 / 87.57** (n=995) | run-1792739730602690285 |

## Iteration log (append-only)

- **10-22.** main was verified post-merge (above). There was a second strix-hip break from #195: `__float2bfloat16_rn` in `batched_embed_nvfp4`, which HIP does not have. Fixed with `__float2bfloat16` (RNE on both) in d2d114c02.
- **10-22 → 10-23 overnight: about 14 h idle.** Pushing, stopping other workloads and queuing new sparky jobs were all denied by the auto-mode classifier, and no loop was armed. Bypass permissions have been on since 10-23.
- **10-23 04:10.** Recipe audit:
  - sparkrun-recipes #21 (FP4-Local, single GB10) is OPEN.
  - Its flags match the verified r1short profile exactly.
  - Its `container:` is `azeezish/atlas-gb10:latest` (Docker Hub, 2026-09-23), which predates FP4-Local.
  - The merged sibling recipes use `ghcr.io/atlas-inf/atlas-gb10:latest`.
  - `image.yml` builds `sha-<7>` for every main commit but never moves `:latest`; promotion is manual. On 10-23, `:latest` was b9a7eda (#174, built 10-16), so it does not have FP4-Local either.
- **10-23 04:17.** Recipe #21 served through sparkrun on `sha-504063a`:
  - Ready in about 2 min with all 7 recipe env vars present in the container.
  - Smoke: 17×23=391 correct; `get_weather{"city":"Osaka"}` tool call correct; 31.8 tok/s; MTP 66 accepted / 0 rejected.
  - 04:19: BFCL ST-995 started against the container.
- **10-23 06:00.** 10-hour queue (`~/code/q10h/queue.sh`, tmux `q10h`), in order:
  1. recipe BFCL
  2. agentic on the container
  3. exact-hit arms A–D
  4. main + #172 + #180 gate
- **10-23 06:01.** First unit-test run failed because nvcc was not on PATH (script error). Rerun:
  - fmt OK.
  - `spark-model vision_transport`: 5/5.
  - `spark-server --lib refusal`: 6/6.
  - clippy `-D warnings` clean.
  - `spark-server --bin spark` (glm_tool_boundary, chat_blocking, refusal): 43/43, including the new `blocking_refusal_prefix_keeps_content`.
  - Release build OK (sha256 `3cd769f7d3efbfd2`).
- **10-23 06:06.** CSF item 2, first experiment (`~/code/csf-micro`):
  - Layer 0 holds 1,536 CSF tensors. Compressed 81.77 MiB vs decoded 150.00 MiB, a ratio of 0.545, so **3.20 GiB** is saved over 48 layers.
  - GPU decode is **byte-identical** to the host decoder: 0 of 157,286,400 bytes differ, for both v1 (one block per tensor) and v2 (8 row-split blocks per tensor, u32 in / u64 out, each block patching only its own exceptions).
  - v2 latency per layer (**measured while BFCL was serving on the same GPU**):
    - k=10 experts: 30.5 µs, which is 1.46 ms/step over 48 layers, about 1.3% of the 111 ms decode cycle.
    - k=40 (MTP verify): 5.07 ms/step.
    - All 512 experts: 72 ms.
  - A clean rerun is queued after QUEUE DONE (`clean.log`).
- **10-23 06:12.** PR **#204**: the HIP fix (`fix/hip-embed-nvfp4-bf16`). AMD main needs #197 + #204.
- **10-23 06:20.** PR **#205**: #172, blocking refusals keep `content`. PR **#206**: #180, over-capacity vision gets a typed `VisionCapacityError` and a 400, not a retryable 500.
  - Enntity's fork already lifted the cap itself in `92dd5435`.
  - The live gate for both PRs is queue leg 4.
- **10-23 06:30.** Exact-hit code read (`prefill_b/prefix_lookup.rs`). The shortcut is bypassed by default because it was suspected of degrading agentic quality (ws ~23% cache-on vs ~60% cache-off). Arms A–D (MARCONI_EXACT 0/1 × MTP on/off) are the correctness gate for direct resume.

- **10-23 07:00.** Log dig on the recipe container's serve log: 20,498 lines, 5 WARN, 0 ERROR.
  - (a) `copy_h2d_async was handed a PAGE-LOCKED source`. Traced to `ple/verify.rs` gather_host, which uploads slot ids from the layer's own pinned staging buffer, so every gather paid a `cuStreamSynchronize`. That buffer is only rewritten after `release_prev_pins` host-waits (`cuEventSynchronize`) on the event recorded behind the previous gather kernel, so `copy_h2d_async_retained`'s contract holds.
    - Fix on `perf/ple-slots-retained` (536530e39).
    - A/B is pending. The decode gain is probably small, because the step already syncs for its readback. It also restores the prefill span overlap that the span loop's comments describe.
  - (b) KV own-footprint: 3.10 GB more "free memory fell" than Atlas's counters account for, with no co-tenant on the box. Likely the 2 GB pinned PLE row cache plus the CUDA context are not counted, which shrinks the KV pool (8.9 GB, 376,560 tokens). Not yet confirmed.
  - (c) Vision capacity clamp at 16,384 patches. This is the same cap behind #180.
  - (d) The final-norm PLACEHOLDER warning is expected (mHC).
  - (e) Listening on 0.0.0.0 is the recipe's own `host` setting.
  - NgramRowCache: 96.4% hit rate, 0 evictions, 1.65 M of 22.2 M slots resident. **Fault wall 120.9 s cumulative over ~3 h of BFCL**: serial QD=1 preads under the table mutex, per the code comment. That is a prefill-stall candidate.
- **10-23 07:10.** Qwen peer review requested on #205 and #206 (pending).
  - Own check of #205's blast radius: only `openai/encode.rs` reads `Choice.refusal`. The Anthropic `/v1/messages` and non-streaming `/v1/responses` paths never read it. If they build from the same `Choice`, they used to return an empty answer for refusal-prefixed replies, and #205 would fix them too. Not yet confirmed.
- **10-23 07:15.** **Recipe #21 BFCL ST-995: 87.94 / 87.57** through sparkrun on `ghcr.io/atlas-inf/atlas-gb10:sha-504063a`. Bare metal at the same main gave 87.44 / 87.32. The container is compiled by a different CUDA toolchain, so numerics differ slightly, and at temp 0 that moves a few samples. It passes.

## Open gaps / next
- Recipe #21: change the container to the ghcr image (`sha-504063a` or a promoted `:latest`), then merge. That needs owner/Devin action.
- Strix BFCL drift since September: 85.43/78.19 → 84.42/75.22. Not yet filed.
- A strix-hip build as a required CI check. It is missing, and its absence is why main broke twice.

### Appended 10-23 07:25
- **PR #21 comment**: posted the vetting result, plus the container blocker. Suggested pinning `ghcr…:sha-504063a` or promoting `:latest`.
- **PLE fault wall closed as "not a gap"**: the fault pool already runs 64 threads (the GB10 NVMe random-4K ceiling). The cache had 0 evictions, with 1.65 M of 22.2 M slots filled. Every miss is a first touch of a row, so the 120 s is cold-start reads at the hardware limit. Only prefetching rows ahead of need could reduce it.
- **#205 blast radius confirmed**: Anthropic `/v1/messages` blocking builds text blocks from `c.content` only (`anthropic/translate.rs`). Before #205, refusal-prefixed replies there had no text block at all. Commented on #205.
- 07:16: leg 1 agentic started on the recipe container.
