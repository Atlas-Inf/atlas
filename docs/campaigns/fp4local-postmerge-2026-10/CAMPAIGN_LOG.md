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

## Summary (living; updated as legs land)

**FP4-Local is merged, verified and recipe-vetted.**
- main (#195) + HIP fix: BFCL ST-995 **87.44 / 87.32**; agentic **50/50, 50/50, 12.43 s/turn**.
- Recipe #21 via sparkrun on `ghcr…:sha-504063a`: BFCL **87.94 / 87.57**; agentic **50/50, 50/50, 12.49 s/turn**.
- The only blocker on #21 is its `container:` line, which needs a tag that actually has #195.

**PRs opened (Atlas-Inf/atlas):**

| PR | What | Evidence |
|---|---|---|
| #204 | HIP build fix (`__float2bfloat16_rn` → `__float2bfloat16`); with #197 restores AMD main | GB10 BFCL + agentic unchanged; strix-hip build OK; strix agentic 50/50 |
| #205 | #172: blocking refusals keep `content` (also fixes Anthropic `/v1/messages` blocking returning no text) | 60 server tests; gate BFCL 87.44/87.32 + agentic 50/50 12.34 s/turn |
| #206 | #180: over-capacity vision → typed error → HTTP 400 blocking + `invalid_request_error` streaming | model 5/5; same gate PASS |
| #207 | **Item 1, direct exact-hit resume**: retries 18 ms TTFT (was ~1.2 s full recompute), 15/15 bit-exact; default `MARCONI_EXACT=1` path 500s on qwen4_exp | arms D/E/F/H/I |
| #208 | two-phase full-cache hit used embedding logits (latent) | compile/clippy |

Issue #209: session_hash collides for shared tool sets (eviction fairness; cross-conversation tail bleed checked, not reachable).

**Findings (deployment gaps):**
- **Block-aligned identical retries recompute everything:** the exact hit is declined and nothing falls back to a shallower snapshot, and the log misleadingly says "no SSM snapshot". Unaligned retries already restore the last intermediate checkpoint.
  - #207 serves exact hits directly, in 18 ms. The fallback branch covers the aligned/aux-less case (pending arm L).
  - The bigger gap is the **shared prefix with no snapshot at the fork**: agentic first turns, and cold tool requests sharing ~1.2K tokens of tool JSON.
- **Agentic first turns recompute the shared ~3.5K preamble:** p90 TTFT 3.1 s vs p50 0.64 s. Branch checkpoints exist but are off by default; A/B + gate in queue4.
- **The container KV pool is 16% smaller than bare metal**, from a 1.6 GB free-memory drop seen only in docker.
- **Neutral or closed:** PLE pinned-copy sync (27.90 vs 27.91 tok/s); PLE fault wall is cold first touch at the NVMe ceiling; RSS plateaus at ~4.0 GB under BFCL.
- **CSF resident scales (item 2):** GPU decode byte-identical; 1.39 ms/step (1.25%) at k=10, 4.75 ms at k=40, for 3.2 GiB.
- **Repo-wide CI:** `cargo deny` red since 10-21 (tool/env); the keepalive test is flaky (#191).
- **Items 3–6** deferred (item 5: no graphs in the recipe profile).

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
| 10-23 08:33 | recipe #21 via sparkrun, ghcr sha-504063a | agentic 50 | **50/50, 50/50, 12.489 s/turn** (Σwall 4585 s) | run-1792744391299397436 |

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
- **07:35. KV pool, container vs bare metal (same main):**
  - Container: Atlas-own 96.2 GB, KV pool 8.9 GB / 376,560 tokens.
  - Bare metal: Atlas-own 94.6 GB, KV pool 10.5 GB / 445,792 tokens.
  - Atlas's tracked figure is equal in both (93.1 / 93.3 GB). The container loses about 1.6 GB of free memory during load that bare metal does not, probably container page cache or processes.
  - Minor for the recipe (1 sequence, 32K context), but it caps how much prefix cache the recipe holds.
- **07:40. Item 1 design input from PR #163** (external, opt-in `--exact-prefix-cache`):
  - On main, cache-on ≠ cache-off at temp 0 (12/24 on Flash-Next), because warm replays use a different chunk layout. #163 fixes this with a 1024-token grid, at +10.6% agentic wall.
  - Arms A/C (`MARCONI_EXACT=0`: full hits recompute everything, same layout as cold) are the control and should be 15/15.
  - Arms B/D test the stashed-hidden shortcut itself. MTP drafter state is not rebuilt on a hit (that is #163's fix 7), but `EXACT_VERIFY` should keep output equal anyway; B vs D checks that.
  - Item 1 removes the redundant 1-token forward from the B/D path. It is only worth building if B and D are 15/15.
- **07:50. Likely root cause of the exact-hit quality loss on qwen4_exp.**
  - On an exact full-prompt hit (`MARCONI_EXACT=1`) the last prompt token is re-run through every layer.
  - The `finalize_last` fixup re-restores only the SSM h/conv state (`ssm_snapshots.restore`) and the stashed hidden. It never re-applies the snapshot's **aux** state (PLE n-gram history/conv carry, QSA indexer keys).
  - The prefix-hit restore does apply aux (`with_aux → apply_aux_states` in prefill_a/prefill_c). The shortcut predates PLE/QSA.
  - So decode continues from double-advanced PLE/QSA state.
  - Fix: `fix/exact-hit-restore-aux` (81655d56a) re-applies aux in the fixup.
  - Remaining unknown: the MTP drafter context carry may also consume the redundant token. That is #163's fix 7.
- **07:55. PLE prefetch correction.** `PleWarm` (layers/ple/warm.rs) already prefetches the prompt's n-gram rows ahead of the gather. The 120 s fault wall is what is left after that prefetch.
- **07:20. queue2** (`~/code/q10h/queue2.sh`, tmux `q10h2`) replaces queue.sh legs 3–4. queue.sh stops at "STOP before arm A".
  - Arms A–D: base binary.
  - Arms E/F: exp binary = main + PLE retained + aux fix, `MARCONI_EXACT=1`, MTP on/off.
  - G vs A/A2: PLE retained decode A/B, using `decode_probe.py` (10 × 512-token answers).
  - Then the fixes gate (BFCL + agentic), then the clean CSF timing.
- **08:10. Qwen peer review of #205 / #206 received. Triage:**
  - **Real gap, fixed in #206 (1abf05704):** streaming clients still got `server_error`/500. `send_invalid_request_to_sink` sent the same `StreamEvent::Error`, and `handle_error` always renders 500. #180's client (opencode) streams. Added `StreamEvent::InvalidRequest`, rendered as `invalid_request_error`/400. Only 2 `StreamEvent` consumers exist (chat_stream, completions); responses/messages inherit through chat deltas.
  - Also fixed: renamed `images` → `grids`, and added a negative test that a stringified error loses its type.
  - **Rejected with evidence:** TP hang (`validate_state` runs before the command preamble by design); missed entry points (vision state is broadcast only from chunk 0 and prefill_b); #205 double emission (only `openai/encode.rs` reads `refusal`); strict-JSON breakage (grammar-constrained, and there is no server-side `json_object` validation on blocking); "redaction" (streaming never stripped).
  - Both triages posted as PR comments. Retest on sparky (tmux `retest`, `wt-fixes/retest.log`); the gate binary is rebuilt at the PR head.
- **07:26 (sparky clock).** Times above marked `~` were corrected; earlier I had written estimates that ran ~30–40 min ahead of the box. From here on, times are taken from `date -u` on sparky.
- **07:25. Operator error:** `pkill -x cargo` on sparky, used to restart the retest, could have hit the agentic soak's sandbox builds. Checked: run-05 finished at 07:25:34, before the kill, and run-06's `cargo test` passed at 07:26:20, after it. No iteration hit. The 50-iteration verdict will confirm. Rule restated: kill tmux jobs by their own PIDs only.
- **07:26.** #206 commit amended for rustfmt (fcf3fd30d; it had not been pushed). Retest restarted.
- **07:30 heartbeat.** sparky: leg 1 agentic on the recipe container at about iteration 6; queue2 armed; retest has passed fmt and the model tests so far.
  - PRs #204, #205, #206 and recipe #21 have no reviews yet.
  - #204 CI: `cargo deny`, `kernel shadow structure` and `PR benchmark gate` are red. All three are red identically on #203 (merged), #202 and #198, so they are **repo-wide and not from #204**.
  - **New gap: `cargo deny` has been red repo-wide since 10-21.** The error is `bug[unresolved-workspace-dependency]` on every `{ workspace = true }` path dependency (atlas-core, atlas-kernels, atlas-rdma). The last green run was 10-20 at 5ddfabb6e.
    - No manifest changed between then and main, and the action SHA is the same (`3c634983`).
    - The action rebuilds its Docker image each run, so an unpinned base image or tool inside it probably moved. The advisory/license audit is effectively off until this is fixed.
    - Owner item; not chased further.
- **07:45. Strix BFCL drift numbers (85.43/78.19 → 84.42/75.22) not filed.** They come from the 10-22 Qwen summary, and I can find no source for them in BENCH.toml, docs or #197. Filing needs run ids first.
- **07:50. Latent wrong-logits bug in `prefill_d.rs::prefill_full_cache_hit`.**
  - Path: two-phase prefill (`prefill_twophase_dispatch`), with an SSM snapshot covering the whole prompt (`proc_count == 0`).
  - Behavior: it re-embeds the last token and runs final_norm + lm_head **on the embedding**, skipping every layer. It never uses the snapshot's stashed hidden, and it is not gated by `ATLAS_MARCONI_EXACT`.
  - Reachability: only when `max_prefill_tokens < len ≤ max_batch_tokens`. The recipe has max_batch_tokens = 8193 = max_prefill + 1, so only a prompt of exactly 8193 tokens. Two-phase never ran in the recipe or bare-metal logs (0 lines).
  - Fix, when picked up: restore the stashed hidden into `normed` the way `finalize_last` does, or decline the skip when the snapshot has no hidden.
- **07:35. #206 pushed (fcf3fd30d).** Retest: fmt OK, model 5/5, server bin 60/60, clippy clean. Release build running (it is the leg-3 gate binary).
- **07:36. Qwen review of the aux fix and PLE retained — triage:**
  - **PLE retained, held.** Real new hazard: the old synchronous copy implicitly ordered the H2D before *any* reader, so a cross-stream reader of `slots_dev` would now race. The main decode path uses one stream for prestage, uploads and replay (decode_a.rs), but batched decode and MTP verify are unproven. Expected gain is small (decode syncs per step anyway).
    - Plan: let arm G vs A/A2 decide. If neutral, close (neutral switches don't merge). If a win, add a per-copy `copy_done` event (Qwen's design) before a PR.
  - **Aux fix: necessary, maybe not sufficient.** Qwen lists more state the redundant forward touches.
    - Checked as harmless: KV at N-1 is a duplicate write; `seq_len` is set to total; mHC streams are per token, not carried across tokens; the decode checkpoint ring saves during decode, not prefill; the aux blob H2D is a transient copy (it syncs if pinned).
    - Open: QSA counters beyond keys; MTP drafter context (output is still protected by EXACT_VERIFY).
    - Robust design, the same as item 1: **skip the forward** (restore SSM + aux + hidden → lm_head) rather than undo it. Arms E/F show whether aux alone recovers exactness. Item 1 is the follow-up either way.
- **07:34. Item 1 prototype: `ATLAS_EXACT_DIRECT=1`** (264361a3d, on `fix/exact-hit-restore-aux`).
  - On an exact leaf hit whose snapshot has the stashed hidden (last chunk, `proc_count == 1`, no prompt logprobs), skip `prefill_b_forward_layers`. `finalize_last`'s fixup already supplies the hidden for lm_head.
  - Nothing re-runs, so nothing is double-advanced.
  - Covers the single-stream path only; the batched/multi paths keep the re-run.
  - prefill_b.rs is at 499 lines; the predicate lives in the new `prefill_b/exact_direct.rs`.
  - Known cost: the first MTP draft sees no real hidden context. Acceptance only; output is protected by EXACT_VERIFY.
  - queue2 restarted (it was only waiting) with arms H/I (exp, `EXACT=1 + DIRECT=1`, MTP on/off). The wt-exp fmt/clippy check runs after the retest build.
- **08:00 heartbeat.** Recipe agentic at about iteration 25 of 50. wt-exp fmt OK and clippy clean (spark-model). No new PR comments.
- **08:00. New deployment gap, from the agentic-phase serve log** (4,629 lines since 07:16, 0 WARN/ERROR):
  - TTFT p50 is 636 ms but p90 is 3,100 ms (n=198).
  - **51 turns** logged `Prefix cache hit: 3168|3696 tokens … but no SSM snapshot — recomputing all KV`. That is the first turn of every iteration: each new conversation shares the system prompt + tools, the radix cache matches it, but no SSM snapshot exists at that depth. Snapshots sit only at leaves and every 4,096 tokens, and 3,696 < 4,096.
  - So every new session pays a full ~3.5K-token prefill.
  - Atlas already has the fix, off by default and with no recorded measurement: `ATLAS_QWEN4EXP_PC_BRANCH=1` + `ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1` (from #174) captures an in-pass checkpoint at the fork of a cached path (`qwen4exp_points.rs`, min 1,024 tokens).
  - Added queue2 arms J (flags on) and K (control) with `preamble_probe.py`: 6 new conversations on one fixed ~3.5K-token preamble. Compare TTFT on conversations 2–6 and per-conversation output equality.
  - If J wins and matches K, the recipe (and the default) should turn it on.
- **08:30 heartbeat.** Recipe agentic at iteration 46 of 50. No PR activity.
  - Wrote the latent two-phase full-hit fix: `fix/twophase-full-hit-hidden` (3c88c09d7). It restores the snapshot's stashed hidden into `normed` before lm_head in `prefill_full_cache_hit`, and warns if there is none.
  - Compile/test pending (next sparky CPU batch). prefill_c.rs is on the file-size-cap path allowlist.
- **08:33. Recipe #21 fully vetted: agentic PASS** (50/50, 50/50, 12.489 s/turn vs 12.43 bare metal). This also confirms the 07:25 pkill hit no iteration. Posted on #21 along with the preamble-recompute finding. queue.sh stopped as planned and queue2 started (arm A booting).
- **09:00. Arm A (base, `MARCONI_EXACT=0`, MTP): 4/15 cold==warm; 0 exact-hit lines.**
  - Decode probe: 27.71 tok/s median, TTFT median 412 ms.
  - **Key finding: identical retries are almost never exact hits.** The radix cache matches whole 16-token blocks, so a warm identical prompt matches `floor(len/16)*16` tokens (e.g. 1,296 of 1,305). The leaf snapshot sits at the unaligned end, so there is no snapshot at the match: 27× `Prefix cache hit … no SSM snapshot — recomputing all KV`, i.e. **a full recompute on retry**.
  - Only the block-aligned prompts (432 and 4,144 tokens) hit fully, and both matched cold. Prompts >4,096 tokens restore from the 4,096-token intermediate checkpoint and replay the rest (12×).
  - Mismatches are tool/agent cases plus one chat case; known prefix-cache-path variance (FP4-Local doc §3, "optional preamble sentence"; #158/#163 layout).
  - **Item 1 reframed:** the exact-hit shortcut and direct resume only apply to ~1 in 16 prompt lengths. The retry lever is a checkpoint at or below the last whole block. qwen4_exp's in-pass tail checkpoint (`ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`, 64-row grid) should do that. Arm J (MIDCHUNK + PC_BRANCH) also runs the retry probe, so it measures this too.
- **08:43. Probe change for arms B onward:** `exact_hit_probe.py` is now block-aligned. Per case it uses `max_tokens=1` trials to find a pad (0–15 " ok" words) that makes `prompt_tokens % 16 == 0`, then runs cold/warm on a fresh fixed-width decimal nonce (Qwen tokenizes digits one per token, so the count is stable).
  - Every warm send is then a true full-prompt hit that exercises the exact path. It prints "not aligned" when it misses.
  - Arms A (and C) are the unaligned control. The old probe is kept as `exact_hit_probe.unaligned.py`.
- **09:05. Arm B (base, `EXACT=1`, MTP): 3/15. Uninformative.** B went live at 08:42:15 and the aligned probe was swapped in at 08:42:43, so B ran the unaligned probe and hit no prompt fully (0 exact lines).
  - B2 (same arm, aligned probe) is queued in `queue3.sh` (tmux `q10h3`) to run after QUEUE2 DONE.
  - Arms C onward use the aligned probe.
- **08:55. Arm C (base, `EXACT=0`, no MTP, aligned): 10/15**, all prompts aligned (`warm_cached == ptok`). Still 0 bypass lines, because the prompt-end snapshot was gone before the retry.
  - **Root cause of "identical retry recomputes everything":** the lifecycle trace shows `Saved SSM snapshot 9 for 1312 [chunk]` → `Saved finish-leaf SSM snapshot 5 for 1357` → retry `Prefix cache hit: 1312 tokens … no SSM snapshot`.
  - The finish-leaf goes through `insert_with_snapshot` → `SsmSnapshotIndex::insert_tail`, whose supersede sweep keeps **at most one tail (+sibling) per session**. So the prompt-end snapshot is freed the moment the turn ends.
  - Continuing conversations are fine (the next turn extends past the leaf). **Identical retries, regenerate, edit-last-message and n-branches always recompute the full prompt.** This is the real gap behind item 1.
- **08:55. Arm D (base, `EXACT=1`, no MTP): HTTP 500 on the first real exact hit.**
  ```
  Marconi SSM cache hit: 1312 tokens skipped (82 blocks), snapshot 9, replaying 0 SSM tokens to reach 1312
  ERROR Prefill start error: Prefill-as-decode layer 3 failed: QSA: decode at pos 1311 but 1312 tokens ingested — the indexer cache lost sync
  ```
  - **On qwen4_exp the `ATLAS_MARCONI_EXACT=1` shortcut fails requests**, not just degrades them: the restored QSA indexer is at N, and the redundant re-run decodes position N-1. The aux fix (E/F) still runs that forward, so it is expected to hit the same error.
  - **Direct resume (H/I) never runs it.** This confirms item 1's design: skip the forward, don't undo it.
  - The probe aborted on the 500. It now records errors and continues (swapped before E).
- **Item 1, revised plan:**
  - (1) Direct resume, so an exact hit never re-runs the last token.
  - (2) Keep the prompt-end snapshot when the finish-leaf supersedes it (one extra slot per session, first to be evicted), so retries and branches can restore at all.
  - Without (2), (1) only fires on the rare retry whose prompt-end snapshot survived.
- **09:20. CORRECTION: the 08:55 "insert_tail sweep" root cause is WRONG.** The prompt-end snapshot is not deleted. Actual cause, from the code and arm C's trace (a distinct-session chat case with no pool pressure still "lost" it):
  - `pc_policy.rs::marconi_restorable` declines it: `bypass_exact = snap_tok == matched && matched == total && !marconi_exact()`. That is the default `ATLAS_MARCONI_EXACT` bypass, applied *before* the bypass log line.
  - The caller then prints the misleading `Prefix cache hit … but no SSM snapshot — recomputing all KV`.
  - There is **no fallback to a shallower snapshot**. For a 1,312-token prompt the 1,280 intermediate existed; restoring it and replaying 32 tokens would give correct first-token logits with no double advance.
  - **Unaligned retries** (match = floor(len/16)*16): the deepest snapshot ≤ match is the mid-chunk tail capture (`ssm_tail_midchunk=true`). It carries no aux unless `ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`, and aux-carrying models decline aux-less slots, so it is declined too. Arm J (MIDCHUNK + PC_BRANCH) tests exactly this.
  - **Fix candidates** for item 1, all small and in `prefix_lookup` / `marconi_restorable`:
    - (a) When the deepest snapshot is declined (exact bypass, aux-less, session), retry the lookup capped below it and restore the next-shallower snapshot instead of recomputing from 0.
    - (b) MIDCHUNK_CKPT on by default for qwen4_exp, if J is clean.
    - (c) Log the decline reason instead of "no SSM snapshot".
- **09:20. New finding: `session_hash` collides for every tool request.** `compute_session_hash` hashes the first 1,024 prompt tokens, and with tools the chat template puts the tool JSON first. Arm C: **all 70 tool requests share `0x65871a44fdf76528`**; each chat request is distinct.
  - In multi-user agentic serving, every client with the same tool set (or a system prompt >1,024 tokens) is one "session". Session-aware eviction cannot tell them apart.
  - The tail-snapshot session gate (whose purpose is to stop cross-conversation tail restores, since "a tail's SSM state bleeds past its advertised token_count") does not separate them. Whether a wrong-state restore is reachable is not checked yet: it needs a tail whose prefix hash matches another conversation's tokens at that count.
  - A DBG run (debug logs for snapshot lifecycle; 3 cases) is queued in queue3 after QUEUE2 DONE.
- **09:05. Arm J (base + `MIDCHUNK_CKPT=1` + `PC_BRANCH=1`).**
  - **Preamble probe:** conv0 (cold, 3,285 tokens) TTFT 2,087 ms; **conv1–5 TTFT 181–186 ms** (cached 3,264), about **11× faster** first-turn TTFT for every new conversation on a shared preamble. Tool picks look sane. Output equality vs control K is pending.
  - **Aligned retry probe:** 5/15. Chat 5/5 match; tool 0/5 (mismatched in every arm, known); agent 0/5 (5/5 in C), i.e. layout changes from mid-chunk restores (#158/#163 class).
  - Restores: **55 intermediate hits** (C had 0). Mid-chunk checkpoints now carry aux, which confirms fix candidate (b).
  - Not a free win: warm ≠ cold on agent prompts, so it needs the BFCL + agentic gate before going into the recipe/default. Queue a J-profile gate after queue2.
- **09:08. queue4 armed** (tmux `q10h4`, runs after QUEUE3 DONE): J-profile gate. Base d2d114c02 + `MIDCHUNK_CKPT=1` + `PC_BRANCH=1`, agentic 50 then BFCL ST-995.
  - This decides whether the 11× preamble TTFT goes into the recipe.
  - Chain on sparky: queue2 (arms K, E/F/H/I, G/A2, then the #205/#206 gate, then CSF clean) → queue3 (DBG snapshot lifecycle, B2) → queue4 (J gate).
- **09:12. Arm K (control, default flags). CORRECTION to J's "11×".**
  - The preamble probe is equally fast with no flags: conv1–5 TTFT 181–186 ms (cached 3,264); conv0 cold 2,237 ms.
  - K restored from **snapshot 7, the default `Intermediate SSM checkpoint saved at token 3264 (block 204)`**, the last-block checkpoint before each prompt end. That carries aux.
  - The probe's questions are ~20 tokens, so that checkpoint falls *inside* the shared preamble and is reusable. **The probe didn't test the agentic case**, where long first messages put the last-block checkpoint past the fork and nothing exists at the preamble boundary (the 51 misses).
  - Outputs J vs K: conv1–5 identical tool picks; conv0, cold in both, differs (J `list_dir,search` / K `list_dir`). Either the mid-chunk capture changes cold numerics, or this is run-to-run variance.
  - K aligned retry probe: 10/15 (same as C). This is the default baseline.
  - New probe mode `long` (+~200 tokens of distinct context per first message). queue4 now runs **J2** (PC_BRANCH + MIDCHUNK), **K2** (control) and **J3** (MIDCHUNK only) with long questions *before* the J-profile gate.
- **09:12.** EXP build OK (e0158eda); arms E/F/H/I next.
- **09:25. Arm E (exp = aux fix, `EXACT=1`, MTP, aligned): 14/15 warm sends hit the exact path and all 14 returned HTTP 500** (`QSA: decode at pos N-1 but N tokens ingested`).
  - Reproducible on every exact retry with `ATLAS_MARCONI_EXACT=1` on qwen4_exp.
  - The aux fix alone cannot help: the failure is in the redundant forward. Item 1 = skip the forward; H/I test it.
  - The aux-fix-only branch is **not PR-worthy on its own**. It folds into direct resume (same branch, 264361a3d/f5dc0fb7d).
- **09:25. Qwen consult on retry fallback + session_hash — triage:**
  - **Fallback-to-shallower by default:** agreed. Implement as candidate enumeration with reject reasons (exact bypass / missing aux / session / pin failed), not "decline then give up". Pin the chosen snapshot through restore. Replay with prefill/extend semantics (never decode for a prompt token).
  - **Direct resume:** opt-in and capability-gated (snapshot has stashed hidden + aux + KV through N; disable with prompt logprobs, which I already did; first MTP step degraded). My prototype matches.
  - **Cross-session tail bleed (A):** checked, **not reachable with current captures**. The default mid-chunk tail capture saves GDN state *exactly at tb* (split kernels; midchunk_capture.rs), and qwen4_exp in-pass checkpoints capture PLE/QSA aux at that exact row. Physical == logical extent, so a cross-session restore of such a tail is as safe as an exact-prefix snapshot. The "bleeds past token_count" justification in `lookup` describes the older clamp-based `ATLAS_SSM_TAIL_CKPT` path.
  - **Session-hash collision** stays a real issue, but for eviction fairness and observability, not correctness. Proposed identity: request/conversation id for LRU affinity; prefix hash for correctness (already how non-tail entries work).
  - **Misleading log:** agreed; part of the fallback change.
- **09:35. Retry fix written: `fix/marconi-fallback-shallower` (64f33b27a).**
  - `PrefixCache::snapshot_below` (radix: the deepest resident index entry under a cap) feeds `marconi_fallback` (pc_policy.rs). When the deepest snapshot is declined, it walks down ≤4 candidates until one passes `marconi_restorable`, then the existing intermediate-hit path replays the gap.
  - Single-rank only. The bypass check now keys on the chosen depth, and the log says "no restorable SSM snapshot".
  - Unit test `snapshot_below_returns_the_next_shallower_restore_point`. All four touched files stay under the 500-line cap.
  - sparky: `wt-fallback` = main + fallback + twophase-full-hit fix. fmt/test/clippy/release running (tmux `fbcheck`).
  - **Arm L** (fallback binary, `EXACT=0`, aligned retry probe) is at the head of queue4.
  - Expected: warm sends restore at a shallower snapshot, replay < 64 tokens, warm wall time drops. Match rate vs cold should be ≥ C's 10/15 if the replay numerics hold.
- **09:28. Arm F (exp aux fix, `EXACT=1`, no MTP): 14/14 exact retries → HTTP 500**, same as E. Independent of MTP.
  - Fallback branch checks on sparky: spark-runtime snapshot tests 75/75 (new test included), clippy clean, release build OK (762bc4a5).
  - rustfmt wanted the new test's asserts wrapped; amended locally (unpushed; binary unaffected).
  - Arm L starts once queue2/queue3 drain.
- **09:45. ITEM 1 RESULT. Arm H (exp, `MARCONI_EXACT=1` + `EXACT_DIRECT=1`, MTP): direct resume works.**
  - 14/14 exact retries served with **no errors** (E/F: 14/14 HTTP 500). 28 hit/direct lines.
  - **Warm TTFT 18.1–18.8 ms**, vs 1.1–1.4 s for the full recompute the default does (about 60×).
  - Warm wall time is lower on every case, e.g. case 13: 2.12 s vs control K 5.41 s.
  - Cold==warm: **10/15, the same cases as control K** (all chat and agent match; all 5 tool cases mismatch in every arm, the known prefix-path variance). Direct resume adds no divergence.
  - Correction to my own analysis: cross-arm cold comparisons are invalid, because the probe draws a random nonce per run (E vs H, same binary, agree on only 1/15 cold). Only within-arm cold vs warm counts.
  - Next: arm I (no MTP), then PR. The PR is `fix/exact-hit-restore-aux` (aux re-apply + `ATLAS_EXACT_DIRECT` opt-in), with the fallback PR alongside.
- **09:50. Arm I (direct resume, no MTP): 14/14 served, < 100 ms TTFT, 0 errors.**
- **09:50. RE-SCORE, correcting my own probe bug.** The "tool cases always mismatch" pattern was the **tool-call `id` counter** (`call_…08` vs `call_…09`), not output divergence. With ids excluded:
  - **C 15/15, K 15/15, H 15/15, I 15/15.** Default full-recompute retries and direct resume are bit-exact against cold, with and without MTP.
  - E/F 1/15 (14 errors).
  - **A/B 5/15** (unaligned → partial hit + replay); **J 6/15** (mid-chunk restore + replay).
  - So the divergence class is "restore partway, then replay with a different pass layout" (#158/#163), and my fallback fix (arm L) will likely sit in it too. Same behavior as every existing agentic intermediate hit, but the PR must state the trade-off.
- **09:52. PR #207 opened:** direct exact-hit resume (`fix/exact-hit-restore-aux`: aux re-apply + `ATLAS_EXACT_DIRECT`). Exact retries at 18 ms TTFT, bit-exact; the default exact path 500s on qwen4_exp.
- **09:55.** PR **#208**: two-phase full-hit emits from the stashed hidden (latent; compile/clippy verified, not exercised live). Issue **#209**: session_hash collision for shared tool sets (eviction fairness; tail bleed checked and not reachable with current captures).
- **10:00. PLE retained-copy A/B: neutral, closed.** Decode probe median: A 27.71, **G (exp, retained) 27.90**, **A2 (base repeat) 27.91** tok/s. TTFT 412/414/413 ms.
  - The per-gather `cuStreamSynchronize` the startup WARN flags costs nothing measurable on decode, because the step already syncs for readback.
  - Not PR'd (neutral switches don't merge). `perf/ple-slots-retained` stays local as the record. The WARN is still real but harmless.
- queue2 now moves to the main + #205 + #206 gate (BFCL ST-995 then agentic), then the clean CSF timing.
- **10:05. Item 5 scoping (shared scratch across graph shapes).** The recipe-profile serve log (r1short: MTP 3 drafts + EXACT_VERIFY) shows **no CUDA-graph capture lines**. MODEL.toml lists graphs as served, but this profile doesn't exercise them.
  - The visible scratch in this profile: buffer arena 8,193 tokens × 3.34 GB; one shared MXFP8 BF16 prefill scratch; PLE pinned row cache 2 GB; snapshot pool 16 × 114 MB + decode-rollback 8 × 114 MB.
  - Item 5's payoff therefore isn't established for FP4-Local. Deferred, per Qwen's P5 ranking. Revisit only if a graph-enabled profile becomes the recipe.
- 10:00. Fixes gate (main + #205 + #206, binary f3fc512c) booting: BFCL ST-995 (~3 h) then agentic (~1.3 h).
- **10:15. PR CI sweep** (excluding the three checks red repo-wide):
  - **#207:** file-size cap failed (`finalize_last.rs` 506 > 500). Fixed in 4529e1a44 by moving the aux re-apply into `exact_direct.rs`; finalize_last back to 499. fmt and clippy clean on sparky. Pushed and commented.
  - **#207 / #206:** `cargo test --workspace` and `cargo llvm-cov --workspace` failed only on the flaky `spark-storage ngram_cache::keepalive::tests::ticks_only_while_active` (keepalive.rs:201; #191 fixes it). #206's job rerun; #207 re-runs on the new push.
  - #205 and #208 have no PR-specific failures.
- **10:30 heartbeat.** Fixes gate BFCL 29 min in (GPU 90%). Its serve log has only the 5 known startup WARNs and 0 errors so far.
  - #207 `cargo test --workspace` now SUCCESS on rerun (the flaky keepalive test).
  - No reviews yet on #204–#208 or recipe #21.
- **10:31.** Started a memory-trend sampler on sparky (tmux `memsample` → `~/code/q10h/memsample.log`, every 60 s: MemAvailable, serve RSS, GPU %). Baseline under BFCL: MemAvailable 12,471 MB, serve RSS 3,939 MB.
  - Goal: catch growth across the 3 h BFCL + 80 min agentic legs. No long-run memory trend has been recorded for FP4-Local before.
  - Asked Qwen for production gaps across the whole day's findings (pending).
- **10:45. Qwen production-gap review — triage against the code:**
  - **P0 auth/bind:** Atlas already defaults to `--bind 127.0.0.1` and has `--require-auth` + `--auth-tokens-file`, a 32 MB body limit (`ATLAS_MAX_BODY_BYTES`), a rate limiter and a GPU-fault middleware. Recipe #21 sets `host: 0.0.0.0` without auth. Inside a container that is needed for port mapping, so exposure depends on Docker publishing. Recipe note to add.
  - **Health/metrics:** `/health` (ready = model published), `/health/live`, `/metrics` exist.
  - **Cancellation:** implemented and unit-tested (`emit_step/cancel.rs`, `cancel_tests.rs`, `receiver_closed` in prefill and admission). Live behaviour is unverified.
  - **New `prod_probe.py`**, queued in queue4 after arm L as **P** (r1short: cancel at prefill / first chunk / mid-decode; 12 API negative cases; health during a ~30K prefill; metrics scrape) and **P2** (r2, 8 seqs: shorts vs an in-flight 30K prefill; C=2/4/8 shorts).
  - Not covered today: >32K agentic, crash/restart fault injection, overnight soak (memsample covers the remaining gate legs), MTP at C>1 with temperature.
- **11:00 heartbeat.** Fixes-gate BFCL 59 min in. No PR or issue activity.
  - memsample: serve RSS 3,939 → 3,987 MB over 27 min (+1.8 MB/min); MemAvailable flat at 12.46 GB.
  - The growth matches the PLE pinned row cache filling (1.65 M slots after 3 h in the recipe run; cap 2 GB). Check it plateaus over the remaining legs.
- **12:00 heartbeat.** Fixes-gate BFCL 1 h 59 min in. memsample: RSS 4,004 MB, plateaued since about 11:15 (+5 MB in 45 min). No leak signal under BFCL. No PR activity.
- **12:28 heartbeat.** Fixes-gate BFCL 2 h 29 min in. RSS 4,005 MB (flat). No PR activity.
- **13:00. Fixes gate (main + #205 + #206, binary 64bbdec27bd2ffea): BFCL ST-995 87.44 / 87.32** (run-1792760075267807876). Identical to main, accuracy-neutral. Posted on #205/#206. Agentic leg next.
- **12:58 heartbeat.** Fixes-gate agentic leg started (3 min in). RSS 4,012 MB. No reviewer activity on #204–#208.
- **13:10. AMD boxes put back to work** (idle since this morning's AMD soak/build):
  - **strix job 314 `s20-st995-main197-hipfix`**: build main 504063ad4 + #197 + #204 (b9ab4ed54), then BFCL ST-995 on Qwen3.8-27B-NVFP4 with the s14/s19 profile. This is the AMD train candidate gate.
    - The strix "drift" source is now found: `s14` measured **84.42 / 75.22** on this profile.
    - The strix repo lacked main; fetched origin main and bundled the integration branch. Dispatcher started.
  - **framework job 082 `f05-fnext-st995-main197`**: AMD regression check of #195 (shared qwen4_exp code) on Flash-Next NVFP4, using f22's profile (84.12 / 82.33 on the #197 train), binary main197-hipfix from f03.
    - Queued **behind another session's job 081-f40-drafter-ab** (submitted 13:07:47, running). Not touched.
  - FP4-Local itself can't run on framework: not downloaded, and 68 GB free < ~96 GB.
- 13:00. Fixes-gate agentic leg running on sparky.
- **13:28 heartbeat. All three boxes busy.**
  - sparky: fixes-gate agentic, 33 min in. Serve RSS 4,087 MB (+80 MB since agentic began; watch).
  - strix: s20 binary b9ab4ed54 (sha aaaa3c289775, reused build); BFCL ST-995 started 13:11:50.
  - framework: f05 Flash-Next ST-995 running (081 from the other session finished).
  - No PR reviews.
- **13:58 heartbeat.** Fixes-gate agentic 64 min in. Serve RSS 4,116 MB (+11 MB in 17 min, slowing). strix and framework BFCL running. No PR reviews.
- **14:13. Fixes gate PASS (main + #205 + #206):** BFCL 87.44 / 87.32 and **agentic 50/50, 50/50, 12.344 s/turn** (Σwall 4,705 s, run-1792764782219366004). Main measured 12.43. Posted on #205/#206.
- **14:13. CSF clean timing (idle GPU)**, item 2. GPU decode byte-identical (v1 and v2).
  - v2 per layer: **k=10: 28.9 µs → 1.39 ms/step** (1.25% of a 111 ms cycle); k=20: 2.49 ms; **k=40: 4.75 ms/step**; k=80: 9.39 ms; all 512: 68.5 ms.
  - The earlier contended numbers held to within ~5%. Saving: 3.20 GiB.
  - Verdict for item 2: scratch-decode costs 1.3% at one token per step, 4% at an MTP-3 verify. Worth it only when the KV budget binds (longer context or C>1). Fusing the decode into the MoE GEMV is the next step to make it free.
- queue2 DONE → queue3 (DBG snapshot lifecycle, then B2) → queue4 (L fallback, P/P2 prod probe, J2/K2/J3, J gate).
- **14:20. CORRECTION from the DBG snapshot-lifecycle run + re-reading arm A: unaligned identical retries DO restore by default.**
  - DBG: warm 854-token retries (match 848) restored from the `Intermediate SSM checkpoint saved at token 832` (warm prefill ~0.2 s vs ~0.86 s cold). My log extract had dropped the "Marconi intermediate hit" lines because they don't contain "snapshot".
  - Arm A's "no SSM snapshot" lines are mostly **cold** sends: 15 that matched only the 16/32-token template prefix, and 9 cold tool/agent sends that matched the ~1,184–1,200 tokens of shared tool JSON (the same no-snapshot-at-the-fork gap as the agentic preamble finding).
  - Arm A's warm unaligned retries: **12 intermediate hits**.
  - So "retries recompute everything" holds only for **block-aligned** prompts (exact bypass, no fallback; arm C 15/15) and tiny prompts. The fallback fix's scope is aligned prompts (~1/16) plus aux-less snapshots.
  - PR #207 body and the fallback PR draft corrected.
  - The real big-ticket item remains the shared-preamble/tool-JSON prefix with no snapshot at the fork (agentic first turns, cold tool requests). That is the J2/K2/J3 + J-gate work in queue4.
- **14:30. Arm B2 (unmodified main, `MARCONI_EXACT=1`, MTP, aligned): 14/14 exact retries → HTTP 500.** Confirms the bug is main's, not the aux branch. Posted on #207. queue3 DONE → queue4 next (arm L fallback).
- **14:30. Arm L (fallback-to-shallower, `EXACT=0`, aligned retries) — works, but NOT shipped as default.**
  - All 15 declined exact hits fell back one checkpoint (e.g. @1312 → @1280, @144 → @112) and replayed 32 SSM tokens. 14 intermediate restores, 0 errors.
  - Warm wall time drops on long prompts: case 12 9.25 → 5.62 s (K: 9.71 → 9.49); case 10 5.66 → 3.03 (K: 5.27 → 5.00).
  - **Output equality vs cold: 2/15** (ids stripped). The full recompute it replaces (C/K) and direct resume (#207) are both 15/15. Replaying through a different pass layout changes greedy output on most prompts (the #158/#163 class; cf. arm A 5/15, J 6/15).
  - Decision: **no PR as default.** It would turn a bit-exact retry into a different answer ~87% of the time to save TTFT on ~1/16 of prompt lengths. `fix/marconi-fallback-shallower` stays local with this result. The exact way to get partial restores is #163's fixed prefill grid. #207 covers exact hits, fast and exact.
  - The same data makes another point: **the default already diverges on every unaligned retry** (intermediate restore + replay; arm A 5/15). That is existing #158 behaviour, worth noting on #163.
- **14:40. Production probe P (r1short, main d2d114c02) — mostly clean.**
  - **Cancellation:** stream closed during a 24K prefill / after the first chunk / mid-decode. In each case the next request is normal (0.69–0.74 s vs 0.74 s baseline) and health is 200. The live behaviour matches the cancel code.
  - **API negatives** (all answered in 0.0 s): malformed JSON 400; max_tokens 0 → 400 "must be at least 1"; −1 → 400; empty messages 400; content-not-string 400; invalid tool schema 400; corrupt image 400 ("unrecognized image format; send JPEG, PNG…"); 33 MB body **413**. Unicode/zero-width and multi-token stop strings are fine. The server stays healthy throughout.
  - **Health:** `/health` answered in 0.0 s five times during the 24K prefill.
  - **Gaps:**
    - (1) An **unknown model name returns 200** and is served. That is a routing/compat risk when several models sit behind one gateway.
    - (2) An **invalid role ("wizard") reaches the Jinja template**: the client gets 400 "Tokenization error: Failed to render Jinja chat template…" and the server logs `ERROR spark::tokenizer::chat_render`. It should be a validation 400 naming the role.
    - (3) `/metrics` has `atlas_requests_active` but **no queue-depth / waiting metric**.
  - Gap triage:
    - (2) Not fixed. `ir::Role::Other` deliberately passes unknown roles to the template, because some templates use `developer`/`function`/`ipython`, so a hard allow-list could break them. The client already gets a 400; only the message wording and the server-side ERROR level are rough.
    - (1) Accepting any model name is common for single-model servers (vLLM returns 404). Note only.
    - (3) The missing queue-depth metric is a real ops gap for C>1 deployments. Candidate follow-up.
- **14:50. Production probe P2 (r2 profile: 8 seqs, MTP 1 draft, ATLAS_MTP_MAX_SEQS=8).**
  - **Concurrency scaling for short requests is poor:** C=1 0.89–0.97 s each; C=2 1.52–1.60 s; C=4 2.73–2.95 s; **C=8 5.6–6.05 s each**. Aggregate throughput gain at C=8 is only ~1.3× over serial. All 200s; no errors or hangs.
    - For multi-user FP4-Local deployments, batching barely helps short interactive requests on this profile.
    - Next to check: whether prefills serialize (one chunked prefill at a time) or the batched decode step itself is slow.
  - **Head-of-line test invalid (my probe):** the "long" prompt (2,000 facts ≈ 34K tokens) exceeded max-seq-len 32,768 and got an instant 400 on both P and P2. So "6 shorts during long prefill" (4.0–4.3 s) was really 6-way concurrency, and P's "health during long prefill" ran against a rejected request.
    - Rerun queued: `queue5.sh` / `prod_probe_hol.py`, 1,200 facts ≈ 20K tokens, after QUEUE4 DONE.
  - Also invalidated by the oversized prompt: P's "cancel during prefill" and "health during long prefill" (both ran against a rejected request). P's after-first-chunk and mid-decode cancellations are valid. queue5's HOL rerun covers prefill-time health and blocking; a prefill-cancel rerun is not yet queued.
- **15:00. Long-question shared-preamble A/B: the agentic first-turn case.** 6 new conversations, one ~3.3K-token system+tools preamble, ~700 tokens of distinct context per question (prompts ~3,970 tokens).
  - **J2 (`PC_BRANCH=1` + `MIDCHUNK_CKPT=1`):** conv0 cold 2,654 ms; conv1 2,555 ms (`pc branch checkpoint planned` at the fork); **conv2–5 735–782 ms** (4× `Marconi SSM cache hit`).
  - **K2 (default):** conv0 2,782 ms; **conv1–5 2,624–2,690 ms**, all 5 `no SSM snapshot — recomputing all KV`.
  - **≈3.4× faster first-turn TTFT for every new conversation after the first sibling.** This closes the 08:00 gap (agentic p90 TTFT 3.1 s).
  - **Outputs:** conv2–5 J2 (restored) == K2 (full recompute), **4/4 identical**. conv0–1 differ; those are the passes where J2 captured in-pass. Either the capture changes numerics, or it is run-to-run variance (6 samples can't tell).
  - The J-profile BFCL + agentic gate (queue4, next) is the accuracy decision for the recipe. J3 (MIDCHUNK only) is next, to see whether PC_BRANCH itself is needed.
- **15:15. J3 (`MIDCHUNK_CKPT=1` only, no PC_BRANCH):** conv0 2,658 ms, **conv1–5 2,510–2,559 ms**, 5× `no SSM snapshot`. **No speedup without PC_BRANCH**: PC_BRANCH is the lever, and MIDCHUNK is only its prerequisite.
  - **Outputs:** J3 == J2 on conv0/1 (the capture passes), and both ≠ K2 there. On conv3 J3 ≠ J2 ≠ K2. conv2/4/5 all equal.
  - So **MIDCHUNK's in-pass capture deterministically changes greedy output on ~3/6 prompts vs default** (not run-to-run noise: J2 and J3 agree with each other). Restores from the branch checkpoint reproduce the default's output.
  - Recipe decision: PC_BRANCH needs MIDCHUNK, so the numerics change comes with the 3.4× first-turn TTFT. The **J-profile gate** (BFCL + agentic, starting now) decides whether it costs accuracy.
- **14:58 (sparky clock; the ~15:00/15:15 stamps above ran ~15 min ahead).** J-profile gate agentic started 14:38 (~80 min), then BFCL (~3 h, ~19:00). strix and framework AMD BFCL ~1.5 h in. No PR reviews. The 10-hour window ends 16:00; the final report goes out then, with the J-profile BFCL and AMD results marked pending.
- **15:28 heartbeat.** J-profile agentic 49 min in (RSS 3,843 MB). strix/framework AMD BFCL 2 h 17 min in. No PR reviews.
- **15:58. J-profile gate, agentic leg: PASS. 50/50, 50/50, 12.115 s/turn** (Σwall 4,521 s, run-1792770855397417305). Default: 12.34–12.49.
  - Serve log: 50 `pc branch checkpoint planned` (one per new conversation) and 286 restores (vs ~145 in default runs).
  - **TTFT p50 500 ms / p90 2,241 ms**, vs default 636 / 3,100 ms (p90 −28%).
  - BFCL leg running (~19:00). This is the remaining accuracy check before recommending `PC_BRANCH=1 + MIDCHUNK_CKPT=1` for recipe #21.
- **16:00. 10-hour window closed.** Final report delivered.
  - Still running, with watchers armed: J-profile BFCL (sparky), strix s20 AMD candidate BFCL, framework f05 Flash-Next regression BFCL, then queue5 (HOL rerun).
- **17:12. strix s20 (main + #197 + #204): BFCL ST-995 84.12 / 74.65** (run-1792775551367679462, 0 faults) vs s14 84.42 / 75.22. Within variance; the AMD candidate does not regress. Posted on #204 and #197.
- **17:37. framework f05 (Flash-Next NVFP4, main + #197 + #204): BFCL ST-995 84.12 / 81.47** (run-1792777040459298748, 0 faults) vs f22 (#197 train) 84.12 / 82.33. Overall identical, normalized −0.86. #195's shared qwen4_exp changes show no AMD regression.

- **10-23 18:47.** J-profile BFCL ST-995 (PC_BRANCH + MIDCHUNK, d2d114c02): **87.64 / 87.38** (run-1792781263471144759), vs default 87.44 / 87.32. With agentic 50/50 at 12.115 s/turn, the J profile passes both gates. Posted the table on sparkrun-recipes #21 as an optional env and left the bake-in decision to the owner. queue5 HOL rerun (P3, r2) booted at 18:48.

- **10-23 18:49. P3 head-of-line rerun** (r2 profile, ~20K-token long prompt, fits max-seq-len):
  - Short request alone: 0.88–0.96 s.
  - **6 shorts sent during the long prefill: all finish at 18.18–18.35 s; the long finishes at 18.7 s.** So shorts wait out the whole long prefill: full head-of-line blocking on the default scheduler.
  - Concurrency (shorts only): C=2 1.60 s, C=4 2.95 s, C=8 5.65 s per request, about 1.16×, 1.26× and 1.32× aggregate over serial. Shorts are close to serialized.
  - The code has opt-in knobs aimed at this: `ATLAS_PREFILL_VARLEN` (ragged batched prefill that a late arrival can join), `ATLAS_HOLO_ALWAYS_MIXED` (fuse prefill chunks into active decode), `ATLAS_PREFILL_CODISPATCH`. queue6 A/Bs VARLEN, MIXED and both on the same probe (18:51).

- **10-23 18:57. queue6, scheduler knobs on the HOL probe: no change.** VARLEN: shorts during long 17.8–18.6 s, C=8 5.64 s. ALWAYS_MIXED: 18.2–18.4 s, C=8 5.72 s. Both: 17.7–18.5 s, C=8 5.87 s. Baseline: 18.2–18.35 s, C=8 5.65 s.
- **Why, from the serve logs (P3 and hol-varlen):**
  1. *HOL.* The long prompt (20,622 tokens) runs in 8,200-token chunks. Chunk 0 runs inline in `phase_start_prefills` (18:49:18.7 → 23.9). Shorts that arrived at 19.19 are admitted only after chunk 1 (29.48), then prefill one by one (~360 ms each). They **do not decode until the long's last chunk finishes** (35.03), and finish at 37.4. So each 48-token request waited ~18 s: two long chunks, its own prefill, and a third long chunk.
  2. *Server TTFT omits queue wait.* Those shorts log `TTFT=360 ms` after waiting ~10 s to be admitted. The long logs 16.3 s. Server-side TTFT can't see head-of-line blocking (a production-metric gap, alongside the missing queue-depth metric).
  3. *C=8.* The 8 requests arrive within 3 ms. Under VARLEN only the first two are deferred, and even those take `run_standard`, not the batched path. The rest prefill inline one at a time at ~360 ms each (≈2.9 s), then decode at width 8 for ~2.3 s. Batched prefill never engages: `want_varlen_defer` needs `active.is_empty()`, and the first finished short makes `active` non-empty.
  4. *A 48-token prefill costs ~360 ms*, split into two forwards by the tail checkpoint at token 16 (~150 ms + ~200 ms). That's about 5–6 decode steps for 48 tokens, so there's a large fixed term. queue7 sweeps prompt length 8–4000 with and without `ATLAS_SSM_TAIL_MIDCHUNK=0` (18:59).

- **10-23 19:03. queue7, prefill vs prompt length** (r2, max_tokens=1, thinking off; server TTFT ≈ client wall):

  | prompt tokens | 34 | 58 | 126 | 276 | 527 | 1027 | 2027 | 4027 |
  |---|---|---|---|---|---|---|---|---|
  | TTFT ms (default) | 300 | 368 | 410 | 469 | 617 | 865 | 1402 | 2650 |
  | `ATLAS_SSM_TAIL_MIDCHUNK=0` | 295 | 368 | 413 | 467 | 611 | 869 | 1394 | 2647 |

  Marginal ≈ 0.58 ms/token above 500 tokens, with a **~300 ms intercept**: a fixed prefill cost worth ~1,700 tokens. The tail checkpoint is not it. That term sets C=8 short-request throughput (8 serial prefills ≈ 2.9 s) and every short agentic turn's floor.
- queue8 (HOL: SLAI, max-prefill 2048, ±SLAI/MIXED) and queue9 (fixed-cost A/B: drafter context off, spec off, plus an nsys capture of 32-token prefills) are queued behind it. `serve-flashnext.sh` now honours `MAX_PREFILL` (default 8192, backup `.bak-q8`).

- **10-23 19:11. queue8, HOL levers: none fix it.** 6 shorts behind the 20.6K prompt / long wall:

  | arm | shorts (s) | long (s) | C=8 (s) |
  |---|---|---|---|
  | baseline (8192 chunk, FIFO) | 18.2–18.35 | 18.7 | 5.65 |
  | `--scheduling-policy slai` | 18.1–18.3 | 18.8 | 5.69 |
  | `--max-prefill-tokens 2048` | 19.5–20.3 | 21.4 | 5.63 |
  | 2048 + slai | 16.2–16.3 | 21.4 | 5.67 |
  | 2048 + ALWAYS_MIXED | 19.5–20.3 | 21.3 | 5.77 |

  With 2048-token chunks the shorts are admitted and prefilled after chunk 1 (19:05:54–56), then **advance one decode step per chunk** (one chunk per tick, ~1.9 s each) and finish just after the long. The design is "one chunk per scheduler iteration to bound TPOT", but a qwen4_exp chunk is 1.9–5.5 s, so a 16-token reply needs ~16 chunks. Smaller chunks only add per-chunk fixed cost (+2.7 s on the long). SLAI alternates one chunk with one decode, the same ratio.
- **Prototype** `perf/decode-steps-per-chunk` (sparky `~/code/wt-dshare`, from d2d114c0): `ATLAS_DECODE_STEPS_PER_CHUNK=k` holds a continued chunk until k standalone decode steps have run while sequences are decoding. Default 0 = byte-identical. New `phase_continue_prefills/decode_share.rs` + a 1-line gate + a 1-line counter in mod.rs. Building.

- **10-23 19:15. queue9, the ~300 ms prefill floor is not MTP:** `ATLAS_NO_MTP_DRAFTER_CONTEXT=1`: 300 / 375 / 422 / 907 ms at 34 / 58 / 126 / 1027 tokens. Spec off: 303 / 377 / 422 / 907 ms. Default: 300 / 368 / 410 / 865 ms.
- **10-23 19:24. nsys of short prefills** (58-token prompts, max_tokens 1; 7 requests by lm_head count; `~/code/q10h/nsys-short-3792654.nsys-rep`): **339 ms GPU kernel time per request**, which matches the ~370 ms TTFT, so the floor is GPU time, not host. MoE kernels take 183 ms/request (`moe_w4a16_fused_gate_up_t_k64` 0.95 ms avg, grouped GEMM 0.52 ms, `w4a16_gemm` 0.39 ms, per layer). The native MXFP8 projections take 115 ms/request (`mxfp8_proj_batchm`). Even 58 tokens × top-k route to most experts, so a prefill streams most expert weights once. Inferred: bandwidth-bound weight reads, which batching amortises and scheduling can't.
  - So for C=8, the lever is **batching the short prefills into one forward** (one weight sweep instead of eight). VARLEN batched prefill exists but never engages here: `want_varlen_defer` requires `active.is_empty()`, and the burst of 8 arrives as the first one turns active.
- **10-23 19:20. queue10, decode-steps-per-chunk v1: no effect (bug).** The counter started at `usize::MAX` and reset only on chunks that ran beside decoders, so the long prompt's last chunk (the first beside decoders) always passed. Fixed: the counter resets on every continued chunk and starts at 0. Rebuilt; queue11 runs k=8 and k=32.

- **10-23 19:35. queue11, decode-share v2 (counter fixed)** — 6 shorts / long: **k=32 → 14.46–14.57 s / 19.0 s**; k=8 → 15.2 s (4 shorts) and 18.75 s (2) / 19.8 s. Baseline 18.2 / 18.7. C=2/4/8 unchanged. The rest of the wait sits before admission: chunk 0 runs inline in `phase_start_prefills` and chunk 1 runs in the same tick (no decoders, so no hold), ~10.8 s before anything new is admitted. The scheduler already ends the tick after a new prompt's first chunk under EP (`head_just_started`). v3 extends that to non-EP when the knob is on. Built; queue13 (k=32, k=8) runs after queue12. queue12's last two arms ran beside the nice'd v3 build (-j8).

- **10-23 19:40. queue12, clean burst of 8 shorts** (6.1 s per batch on default). VARLEN 6.07–6.08 s (its first rep was 9.9 s, a cold/JIT outlier); CODISPATCH 6.3 s. **Zero "Varlen prefill waves" lines in any arm**: batched prefill never engages on a burst either.
- **10-23 19:42. queue13, decode-share v3:** k=32 14.45–14.65 s / long 19.2. The tick break works (shorts admitted at +5.2 s instead of +11 s), but the hold didn't: decode steps from earlier requests left the counter above k, and the long's inline chunk 0 never passes `hold_chunk`. So the shorts sat through chunk 1 again. v4 resets the counter on every tick with nothing decoding (`note_idle`). Rebuilt; queue14.

- **10-23 19:58. queue14, decode-share v4: works.** 6 shorts behind the 20.6K prompt:

  | arm | shorts (s) | long (s) |
  |---|---|---|
  | baseline | 18.2–18.35 | 18.7 |
  | **v4 k=32** | **9.75–9.87** | 20.0 (+1.3) |
  | v4 k=8 | 8.9 (4 of them), 14.6 (2) | 19.1 (+0.4) |

  k=8 releases a chunk before the slowest shorts finish their 16 tokens (MTP ~1.8 tok/step, 6 streams), so 2 of them wait out another chunk. A time- or completion-based release would be cleaner than a fixed k. What's left of the ~9.8 s: chunk 0 (~5.2 s, already running when the shorts arrive), 6 serial short prefills (~2.2 s), and ~2 s of decode. PREFILL_MULTI should collapse the 2.2 s (queue15). C=2/4/8 unchanged, as expected.
  - Diff, all behind `ATLAS_DECODE_STEPS_PER_CHUNK` (default 0, byte-identical):
    - new `phase_continue_prefills/decode_share.rs`;
    - a 1-line gate in `phase_continue_prefills.rs`;
    - 3 lines in `scheduler/mod.rs`: decode counter, idle reset, and `head_just_started` extended to the knob.

- **10-23 19:57. queue15/16: ROWINV, PREFILL_MULTI.** MULTI is ignored without `ATLAS_QWEN4EXP_PREFILL_ROWINV=1` (it WARNs at boot; queue15 was a no-op).

  | arm | burst of 8 (batch wall) | burst of 4 | TTFT 58 / 1027 / 4027 tok |
  |---|---|---|---|
  | default | 6.1 s | ~3.1 s | 368 / 865 / 2650 ms |
  | ROWINV | **5.18 s** | 2.66 s | **258 / 795 / 2475 ms** |
  | ROWINV + MULTI | 5.18 s | 2.65 s | 258 / 798 / 2470 ms |
  | ROWINV + MULTI + CODISPATCH | 5.13 s | 2.62 s | 271 / 813 / 2477 ms |

  - **ROWINV on its own makes short prefills 30% faster** (−110 ms at 58 tokens; −175 ms at 4K) and the C=8 burst 15% faster. It also makes chunked, batched and restored prefills bitwise-equal (its purpose). It's a candidate default for FP4-Local, pending the BFCL/agentic gates.
  - **MULTI engages but doesn't amortise.** CODISPATCH gathers the whole burst ("8 prompts, 488 rows, 2273.6 ms"), yet that's 284 ms a prompt, the same as solo ROWINV prefills (258 ms). A single 1027-row prompt takes 795 ms. So the ~260 ms term scales with prompts, not rows.
  - **This corrects the 19:24 entry.** The floor is not shared weight streaming, or MULTI would have paid it once. It's per-sequence work. queue17 takes an nsys of the multi passes to name it.

- **10-23 20:05. Why MULTI doesn't amortise (nsys, prefill only).** `nsys-multi-3862343`: max_tokens 1, ROWINV+MULTI+CODISPATCH, 1 warm prefill + 3 passes of 8.
  - Kernels: `dense_gemv_bf16` (lm_head) 25×, and the prefill MoE kernels 1200× = 25 × 48 layers. So **every prompt runs all 48 layers alone**, even though the log says "Multi-sequence prefill: 8 prompts, 488 rows".
  - Cause (`prefill_b/multi.rs`): a prompt that shares cached KV blocks leaves the pass and prefills alone afterwards, unless `ATLAS_QWEN4EXP_PREFILL_MULTI_CACHED=1`. Every chat request shares the template's leading ~32 tokens (2 blocks: "Prefix cache hit: 32 tokens"), so on a warm server **no real request ever rides the pass**.
  - GPU time: 2.29 s per 8-prompt "pass", the same as the wall. MoE prefill dominates: fused gate_up 1.87 ms + grouped 1.0 ms + w4a16 0.42 ms per layer per prompt.
  - queue19 tests `MULTI_CACHED=1`.

- **10-23 20:08. queue19, `MULTI_CACHED=1`: batching works.** ROWINV + MULTI + MULTI_CACHED + CODISPATCH:

  | | default | ROWINV only | **+MULTI+MULTI_CACHED+CODISPATCH** |
  |---|---|---|---|
  | 8-prompt prefill pass | n/a (serial) | 2.3 s (ejected → serial) | **0.61 s** |
  | burst 8, max_tokens 1 | n/a | 2.35 s | **0.65 s** |
  | burst 8, max_tokens 16 | 6.1 s | 5.18 s | **3.45 s** (−43%) |
  | burst 4, max_tokens 16 | ~3.1 s | 2.66 s | **1.89 s** |

  The 8-prompt pass costs about 2.3× a single short prefill, not 8×. So once the prompts really share a forward, the weights are streamed once. The bandwidth reading holds for the pass; the earlier "doesn't amortise" was the ejection.
- **10-23 20:10. queue20 launched:** gate of this profile on r1short (the recipe profile): agentic 50 (prefix cache on), then BFCL ST-995. Label `nq-fp4l-multiprofile-d2d114c02`. Default reference: BFCL 87.44 / 87.32; agentic 50/50 at 12.43 s/turn.
