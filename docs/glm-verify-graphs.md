# GLM DFlash verify: CUDA graphs

Status: piecewise KDA-run capture shipped behind `ATLAS_GLM_VERIFY_GRAPH=1`
(default off). This note records why the TP2 DFlash verify step is eager
today, what the flag captures, and what remains before the step can be one
graph.

## Where the time goes

C1 decode (prose, 512 tokens in 15.9 s, production profile): the verify path
launches no CUDA graphs. Each verify step issues about 1,600 kernel launches
at about 15 µs of host time each. The host does not bound the step: the RDMA
all-reduce is stream-ordered (write value, proxy job, wait value) and the
layer loop has no host sync outside diagnostics, so the host (~24 ms per
step) runs ahead of the GPU (~70-114 ms busy). The 2026-09-26 C1 nsys profile
puts 15.4 ms of GPU idle in a 129.5 ms step: ~7.5 ms is MoE-combine waits from
EP imbalance (no graph removes those) and ~4 ms is launch gaps.
The layer stack is `kkkM` repeated eleven times followed by one `k`, where
`k` is a KDA (linear attention) layer and `M` is a sparse-MLA layer: 34 KDA
layers and 11 MLA layers, each followed by the MoE block.

## Why the verify step is not captured

`decode_verify_graphed_kgamma_dispatch` (`model/trait_impl/verify_d.rs`)
already has whole-step capture, keyed by `(slot, K)`. Four things keep the
production DFlash lane off it:

1. **Admission.** Under TP, `use_graphs` needs `comm.is_none()` or one of two
   opt-ins: `ATLAS_GLM_TP_VERIFY_GRAPH` (K=5 only) or
   `ATLAS_GLM_MTP1_VERIFY_GRAPH` (repaired MTP K=2 only). DFlash widths are
   2..=8 and vary from step to step (adaptive width).
2. **The collective.** `ATLAS_RDMA_ALLREDUCE=1` (`spark-comm`
   `nccl_backend/rdma_pair.rs`) pushes a host-side `Job` to the proxy thread
   on every call, and it returns `false` on a capturing stream. A whole-step
   capture therefore falls back to NCCL send/recv on the comm stream (about
   100 µs per collective against about 8 µs over RDMA, with about 70
   collectives per step). Worse, if one rank captures and the other does not,
   the two ranks run different collective protocols for the same reduction.
3. **Sparse-MLA layers.** The fast verify path for MLA (`glm_prefill_ctx`,
   the MLA layers run as a k-row causal prefill chunk) requires `!use_graphs`.
   Its launches take the host `seq_len` as a scalar
   (`glm_chunk_pieces(meta, seq_len_start, ..)`, the native sparse plan's
   `seq_start`). The capturable alternative, `decode_multi_seq_rows`, runs one
   decode chain per row and is slower.
4. **Head.** The EOS ban (`ban.row_mask(seq_len, k)`) is passed as kernel
   parameters that change per step while `min_tokens` is active.

## What is not capture-safe

| Kind | Where | Status in KDA runs |
|---|---|---|
| Host syncs / D2H | profile timers, `ATLAS_K2_DIAG`, M16/M5/shared-FP8 oracles, `union_stats`, lightning layer trace | Diagnostics only. Gated off, or skipped via `stream_is_capturing`. The flag refuses profile/trace runs. |
| Host D2H for shapes | MoE `exact_tiles` offsets readback (`forward_prefill_routed.rs`) | Only when `rows*top_k > 64`. Verify has at most 8 rows x top-8 = 64, so it is not reached. |
| Host-dependent shapes | verify width `k`; MLA `seq_len_start`; EOS ban rows; embed token ids | `k` is part of the key. MLA, the head and embed stay eager. |
| Allocator / lazy init | derived weights, btile arena bind, lazy transposes | Covered by one eager warm-up run per key before capture. |
| Collectives | TP o_proj reduce, MoE all-reduce, K5 peer exchange | The capture splits at each one and replays it eagerly. |

## What `ATLAS_GLM_VERIFY_GRAPH=1` does

`model/verify_pieces.rs` and `model/trait_impl/verify_d_pieces.rs`:

- The verify layer loop hands each maximal run of KDA layers (12 runs:
  11 x 3 layers and 1 x 1) to `VerifyPieces::run`, keyed by
  `(slot, verify rows, first layer of the run)`. MLA layers, embed, the
  metadata upload and the head stay exactly as before.
- **First visit:** the run executes eagerly (warm-up). **Second visit:**
  capture. The layers see a `Recorder` as their communicator. Each collective
  ends the open graph, records the call with its arguments, and begins the
  next graph. Nothing executes during capture. **Later visits:** replay. The
  replay launches graph 0, runs collective 0 eagerly (the RDMA pair sees a
  non-capturing stream), launches graph 1, and so on.
- **Lossless.** `ctx.graph_capture` stays `false`, so every layer takes the
  same kernels and collective calls as the eager pass. The stream order is
  unchanged and accumulation order is unchanged. With the flag off, the loop
  is byte-identical to base; the only refactor is that the DFlash hidden
  capture moved into `kgamma_dflash_capture`.
- **TP invariant.** Collectives are never captured, so both ranks issue the
  same collective calls in the same order whether each rank replays, captures
  or runs eagerly. A capture decision that differs between ranks costs speed
  on that step, never correctness or a hang.
- **Failure.** Any error in a capture pass, such as an unrecorded collective
  (`exchange_async`, `broadcast`, send/recv), a collective on another stream,
  or a capture invalidated by a sync, ends and discards the capture. Nothing
  has executed at that point, so the run re-executes eagerly and the key is
  marked refused. The log reports `piecewise verify graph refused ...`.
- **Pointer stability.** Graphs bake in `hidden_states`/`residual`/scratch
  (fixed arenas), `dflash_hidden_save` (fixed), and KDA `h_state`,
  `conv_state`, `kda_records` and intermediates. All of these come from the
  SSM pool and depend only on the slot, hence the slot in the key. No
  per-step scalars enter a KDA run: KDA ignores `seq_len`, and the MoE block
  reads only row-local device data.
- **Retention.** Runs survive `free_sequence`: nothing they bake belongs to
  the sequence (the per-sequence `gpu.alloc` buffers `invalidate_slot_graphs`
  exists for, the PLE carry and the QSA indexer keys, are never touched by a
  GLM KDA run), so the next request on the slot replays with no warm-up or
  capture step. The slot pool is a LIFO stack, so back-to-back C1 requests
  land on the same slot. Compaction needs nothing either: a survivor moved to
  slot `n` uses slot-`n` keys, whose graphs bake slot-`n` addresses. A LoRA
  rotation destroys every run; the flag also refuses to run with LoRA loaded.
- **Diagnostics.** Rank-local switches that sync inside a KDA run keep the
  step eager (`ATLAS_DFLASH_CAPTURE_TRACE=1`, or any `tracing` DEBUG filter,
  which arms the MoE readbacks). Serve with `RUST_LOG` at info or below.

### Memory cost

The cost was measured on ennspark03 with `vgbench.cu`: 14-kernel pieces,
84 graphs per key, standalone process, unified memory:

- host RSS: about 105 KiB per graph (about 6 KiB per kernel node plus about
  16 KiB fixed);
- total (MemAvailable delta, noisy): about 170 to 250 KiB per graph.

A key has about 80 graphs. C1 at every width 2..=8 is about 7 x 84 = 588
graphs, or roughly **100 to 150 MiB**. `ATLAS_GLM_VERIFY_GRAPH_MAX_GRAPHS`
(default 600) caps the cache, which is held until a LoRA clear. A warm key
over the budget stays eager, so one slot's widths fit and other slots degrade
to eager instead of growing.

This memory is **not** part of KV-pool sizing: it comes out of the headroom
`--gpu-memory-utilization` leaves. At production's 0.93, rank 0's worst-case
free memory is about 1.9 GB and memguard trips at 1 GiB, so the default
budget uses about 15% of that margin. Check it at 0.93 (the hardware plan
forces it) before raising the budget.

### Expected gain

Only launch gaps inside KDA runs can shrink. About 1,200 of the ~1,600
launches per step sit in KDA runs, so they hold about 3 ms of the ~4 ms of
launch gaps. Replays remove most of that but add a graph launch per piece
and keep the eager collectives between pieces. Estimate: **about +1.5 to 3%
C1 decode tok/s**, the size of run-to-run noise, so the hardware plan gates
on the ranges of 5 repeats per arm. The microbench (same shape, C++ launches
at about 2 µs each: eager 2.9 ms, piecewise 1.66 ms, one-graph floor 1.24 ms
per step) bounds the host saving, not the wall time. C4 uses the
owner-batched verify (`decode_glm_long_owners`), which this change does not
touch, so C4 should not move.

## Remaining work, towards one graph per step

1. **Graph-safe collective (other workstream).** When the communicator
   exposes a capture-safe one-shot all-reduce, `Recorder::split` forwards the
   call into the open capture instead of splitting. Each KDA run then
   collapses to one graph (12 graphs per key, about 7x less memory).
   Nothing else in the cache or keys changes.
2. **MLA layers in the graph.** Either (a) take `decode_multi_seq_rows`
   under capture (already capture-safe: it reads positions and `seq_len` from
   `AttnMetadataDev`, uploaded before replay), at its row-chain cost, or
   (b) make the prefill-verify MLA path read `seq_start` from device memory.
   The index top-k saturates at 2048 once `seq_len >= 2048`, so its launch
   shape stops depending on `seq_len`. Key by a `seq_len < 2048` bucket until
   then. An intermediate step is to split the MLA layer's MoE half out of
   `prefill` so that only the attention half stays eager.
3. **Head.** Upload the EOS ban mask and ids to a fixed device buffer before
   replay (like `kgamma_upload_meta`), and read them in the argmax kernels.
4. **Embed.** Upload the k token ids to a fixed device buffer and gather
   from it, as decode already does through `buffers.token_ids()`.
5. **One graph per step.** With 1 to 4 done, the pieces collapse into the
   existing `verify_kgamma_graph` whole-step capture, keyed `(slot, k)`.
6. **C >= 2 (owner-batched verify).** Wrap the owner stage loop in the same
   `VerifyPieces::run`, keyed by the owner slot vector and rows.
7. **Memory.** Share one topology across slots with `cuGraphExecUpdate` or
   pointer indirection, which cuts the per-slot multiplier.

## Hardware check (at most 20 minutes)

See the workstream handover (`verify-graph-hwtest.sh`, `vg-probe.py`). Per
arm, same binary, flag off then on, production memory utilization 0.93: a
cold short-request cell (16 x 64 tokens, first after start, so it carries
every capture), greedy byte-equality, C1 5 x 384 forced tokens (min, median,
max), C2 2 x 384, C4 4 x 384; accepted tokens per verify step and capture
milliseconds per cell from the rank-0 log; MemAvailable minimum per host;
refused/panic counts and ERROR lines new in the on arm. Production is
restored by an EXIT trap, then checked for leftovers.
