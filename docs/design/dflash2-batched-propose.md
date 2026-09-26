# Design: batched DFlash2 propose (the 27B concurrency gap)

Status: design (2026-09-25); code not started. Tracking: job #58. Evidence from job 235 (main64, GB10, nvidia 27B,
DFlash2 γ=8 + Option B, bs16). Workstream W3.1 of the
[campaign status](../campaigns/engine-optimization-2026-09/STATUS.md) (job #58).

## 1. The measured problem

| n (sequences in step) | verify ms | propose ms | propose / n |
|---:|---:|---:|---:|
| 1 | 122.1 | 41.6 | 41.6 |
| 4 | 180.9 | 164.7 | 41.2 |
| 8 | 226.5 | 329.2 | 41.2 |
| 12 | 285.1 | 494.1 | 41.2 |

Verify already batches well (2.3x time for 12x rows). Propose is exactly linear:
generic DFlash2 declares `propose_batch_max() = 1` (`dflash_head.rs:780`) unless
`ATLAS_DFLASH_PROPOSE_LANES > 1`, so `verify_dflash_batch_step.rs:350` runs
`run_mtp_propose_multi` once per pending sequence. At n=12 propose (494 ms) is
63% of the step. Aggregate throughput: 26.9 / 36.8 / 46.6 / 47.1 tok/s at
C = 1 / 4 / 8 / 16 (job 235) vs MTP 20.9 / 41.9 / 64.1 / 85.2 on the same box and
SGLang 135-148 at C=8.

## 2. What exists

- `run_batched_layer_stage(layer, batch_rows = B*γ, batch_size = B, ...)`
  (`batch_forward.rs`): one drafter layer over B*γ stacked rows — RMSNorm, q/k/v
  as projections over all rows (weights read once), RoPE from
  `batch_position_ids`, `reshape_and_cache` via `batch_slot_mapping`,
  per-sequence paged attention (`run_staged_attention` with block tables), o_proj,
  SwiGLU MLP.
- `run_batched_tail_base`: final norm + head over B*γ rows.
- `run_batched_markov`: DSpark-only Markov sampling. **Not used by DFlash2.**
- `batch_parity` diagnostics: runs the serial oracle per sequence and compares
  tokens and hidden bytes against the batched path.
- Per-lane stream overlap (`propose_on_lanes`, `ATLAS_DFLASH_PROPOSE_LANES`):
  overlaps per-sequence proposes on streams; does NOT share weight reads.

## 3. Why the existing seam does not apply as-is

| | Lightning DSpark (what the seam serves) | DFlash2 27B |
|---|---|---|
| γ / drafts | 4 / 3, pinned by `row_contract` + `contract.rs` | 8 / 7 |
| sampling | Markov-bias rows + bonus anchor | bilinear rank-256/top-16 selector |
| attention sinks | required (hard error if absent) | none |
| window | SWA 1024 | 4096 ctx window |
| lm_head | shared | shared BF16/NVFP4 (nvidia pack head is NVFP4-packed) |

## 4. Proposed change (smallest correct step first)

1. **Measure lanes first** (staged job `lanes_conc.sh`): per-sequence propose is
   41 ms against a ~17 ms weight-read floor (3.85 GB at ~230 GB/s), so ~24 ms is
   latency/launch that stream overlap may hide. Bound: lanes cannot beat
   ~17 ms x n, because each lane re-reads the weights.
2. **Generalize the row contract**: `LightningRowContract::new(γ, drafts)` →
   a head-declared contract (`gamma`, `num_drafts`, `sampler: Markov | Selector`),
   keeping the Lightning product's exact γ=4/3 check inside the product policy,
   not in the shared row type.
3. **Make sinks and window per-head** in `run_batched_layer_stage` /
   `run_staged_attention` (sinks optional; window from the head).
4. **Add `run_batched_selector`**: the DFlash2 bilinear selector over B*γ rows,
   mirroring the serial selector kernel launch per row group.
5. **Declare** `propose_batch_max = batch_capacity` for DFlash2 once parity passes.
6. **Gate**: `batch_parity` on (tokens + hidden bytes vs the serial oracle) at
   n = 2, 4, 8, 12; then acceptance-rate parity (mean_na within noise of serial
   propose) — drafts need not be bit-identical because the target verifies; then
   `cross-contamination` 8/8 byte-identical at C=2/4/8; then the concurrency sweep.

Projection (not measured): if a batched propose at n=12 costs ~one weight read
(~17 ms) + per-sequence attention + a B*γ-row head, the n=12 step drops from
~780 ms to ~370-420 ms, roughly 2x aggregate at C=12-16.

## 4a. MEASURED: stream lanes make it worse (job 242, 2026-09-25)

Same gate as job 235 (main66, nvidia 27B, DFlash2 + Option B, bs16, util 0.70),
only `ATLAS_DFLASH_PROPOSE_LANES=4` added. Aggregate tok/s:

| C | serial propose (235) | lanes=4 (242) |
|---:|---:|---:|
| 1 | 26.9 | 27.2 |
| 4 | 36.8 | **25.6** |
| 8 | 46.6 | **31.4** |
| 16 | 47.1 | **29.2** |

Lanes overlap per-sequence proposes that each re-read ~4.5 GB of drafter + head
weights, so they contend for the same bandwidth and also stall verify. Step 1 of
§4 is answered: **lanes are not a stopgap; only B×γ batching (steps 2-6) fixes
this.**

## 4b. Qwen3.8-Max design review (2026-09-25) — projections, not measurements

Agrees with the plan above and adds bounds (assumes ~230 GB/s effective):

| path at n=12 | floor | realistic |
|---|---:|---:|
| today: serial, 41 ms x n | — | 494 ms (measured) |
| stream lanes, drafter weights only re-read per seq | 12 x 16.7 = 200 ms | ≥200 ms (max ~2.45x) |
| lanes, if each seq also reads the BF16 head (2.54 GB) | 333 ms | ≥333 ms (max ~1.5x) |
| batched B*γ, BF16 head | 16.7 + 11.1 = 27.8 ms | ~30-45 ms (16k ctx: ~45-55) |
| batched, NVFP4 head | ~20 ms | ~22-35 ms |

- The BF16 head is ~40% of batched weight traffic → move the selector to the
  NVFP4 target head (sglang#35496 precedent) once batching lands.
- Attention must read each sequence's KV once for its γ query rows (treat the 8
  rows as a query block), or long context multiplies KV traffic by up to 8x.
- Parity bar, tiered: bring-up = valid tokens, no NaN, row top-1 agreement ≥99%;
  selector = top-16 recall ≥99%; **production = acceptance parity** (mean
  accepted/step within ~0.5%) + greedy end-to-end output identical. Do not gate on
  hidden bytes: M changes 8 → 96, so kernel choice and reduction order change.
- **Checked in code:** serial propose DOES run the full `γ × 248,320` lm_head GEMM
  every call (`forward_block.rs:532`, "Phase G"). The nvidia pack's head is
  NVFP4-packed (~0.7 GB), so the lanes floor at n=12 is ~12 × 4.55 GB / 230 GB/s
  ≈ 240 ms — at best ~2x. Batching is the real fix; lanes are a data point.
- Still to verify in code before building: the mask-row RoPE offsets; the mask-row RoPE offsets; the intra-block
  mask; whether draft-row KV is persisted and rolled back.

## 4c. Gate floor note (2026-09-26)

The concurrency-sweep floors 16.2 / 33.5 / 50.5 / 72 belong to the **MTP lane**
(cut from an MTP K=4 run, C1/C4/C8/C16 = 20.9 / 41.9 / 64.1 / 85.2) — the gate
keys baselines by (hardware, checkpoint), and a DFlash2 `--serve-override
speculative=false dflash=true` sweep is judged against them. On main66
(job 240-conc, DFlash2 γ=8 + Option B, bs16, util 0.70) aggregate was
**27.1 / 36.8 / 46.7 / 48.3**: passes C1/C4, fails C8, and would fail C16 — an
honest reading of serial propose, not a bad baseline. The gate cannot express a
separate DFlash2 variant without a framework change; the behaviour is documented
in `kernels/gb10/qwen3.8-27b/BENCH.toml` instead.

## 5. Open questions

- The head at B*γ = 96 rows x 248k vocab: the lm_head read may become the new
  floor; the quantized-head selector path (sglang#35496) and `--mtp-vocab`-style
  slicing apply here too.
- ~~Why job 235 never exceeded n = 12 at C = 16~~ — resolved: verify width does
  reach 16 (`DFLASH WIDTH n_active=16 verify=16`, 193 steps). The STEP_TIMING
  buckets count sequences that needed a proposal that step, not active ones.
- Option B's inherent mean_na drop (5.5 → 3.9 single-stream; ~2.1-2.3 under
  batch in job 132) is an acceptance problem batching does not fix.
