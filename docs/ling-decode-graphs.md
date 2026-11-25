# Ling-3.2-air decode: CUDA graph capture

Status: design + non-behavioral prep (this document, prestage refactor, flag
plumbing). Capture flips only after the KDA slot-pool prerequisite merges and
the e2e oracle passes on it. Target: >= 36 tok/s single-stream greedy on one
GB10 Spark (vLLM single-node 35.8), from ~33 tok/s today — an nsys 50 ms
decode step dominated by launch overhead plus the M=2 verify projections.

Scope of the flip, in order: (1) the single-sequence decode step
(`decode_dispatch`), (2) the K=2 MTP verify of one sequence
(`decode_verify_graphed_dispatch`), (3) optionally the NEXTN draft forward.
Multi-sequence batched graphs are out of the first flip (below).

## The house machinery this rides on

- **Whole-step decode graph** — `model/trait_impl/decode_a.rs`
  `decode_dispatch`. Phase 1, outside any capture: KV block reservation,
  per-step metadata uploads to fixed device addresses (positions, slot,
  seq_len, block table at `scratch()+32768`; the token id at
  `buffers.token_ids()`), per-layer `decode_prestage`. The body
  (`decode_a3.rs`) is per-layer `decode` + final norm + LM head, so the
  graph covers embed-to-logits and `decode_logits_ptr()` stays a fixed
  buffer. The cache is `decode_graph: HashMap<slot_idx, GraphHandle>` —
  a whole-step graph bakes the row's SSM pool state pointers (a pure
  function of the slot) and reads everything else from refreshed device
  memory. Capture runs in RELAXED mode (`gpu_impl_graph.rs`), on a
  miss, deferred at `%64==` positions (the G1 rule; replay is always
  allowed). A refused capture aborts cleanly and re-runs the step eager.
- **Whole-step K=2 verify graph** — `model/trait_impl/verify_b.rs`
  `decode_verify_graphed_dispatch`: the same shape, incl. its own Phase-1
  pregraph (embed of both tokens, block reservation, metadata uploads) and
  a slot-keyed `verify2_graph` cache.
- **Admission** — both dispatches compute `layer_veto = any(layer.decode_graph_unsupported())`
  and stay eager while any layer vetoes. This is the flag the Ling family
  owns: `bailing_c1/kda.rs` and `bailing_c1/mla.rs` both return `true`
  today, kda because the per-sequence state is not pool-slot-stable (the
  pool PR's subject), MLA because of per-step host uploads and the
  late-bound V pool (this slice's subject).
- **`SsmStatePool`** — `model/ssm_pool.rs`. Fixed, process-lifetime device
  addresses per `(ssm_layer, slot)`: `h_state`, `conv_state`, verify
  intermediates, checkpoints, staging. `alloc_sequence_reset`
  (`trait_impl/meta.rs`) builds each KDA layer state as a view into the
  slot ("pre-allocate checkpoint + K=2 intermediate buffers so CUDA graph
  capture doesn't trigger lazy allocation"), and `free_sequence` zeroes
  the slot and RETAINS slot-keyed graphs (`decode_graph_key.rs`: the
  slot-vector is the exact key precisely because the slot determines every
  baked pointer).
- **Multi-sequence graphs** (`decode_a2.rs`) are default-on for the generic
  stack and keyed by slot VECTOR; they share the same `layer_veto` gate.

## What a capture must not swallow (legality inventory)

Capture-legal in the captured body: kernel launches, `cuMemsetAsync`,
device-to-device copies, memcpys from PINNED host memory. Capture-illegal
and therefore hoisted to Phase 1 or refused: pageable host→device copies
(MLA metadata, block table, `draft_word`), `gpu.alloc` + memset (KDA verify
slabs, late V-pool bind, anything per-seq), stream syncs, host readbacks
(the draft pick).

## Ling decode-path allocation inventory

| item | dynamic | graph-safe? |
|---|---|---|
| `LingScratch` (`bailing_c1/scratch.rs`) | static per serve, incl. `mla_pos/mla_slots/mla_base_len/mla_block_table` staging | yes |
| weights, embed table, LM head | load-time | yes |
| MLA layer-owned V-latent pools (`mla.rs` `v_cache_pool`) | one lazy alloc per layer, binds at first prefill use; prefill always precedes the first decode, so it is bound before any capture | yes (with the prefill-precedes invariant; assert at capture admission) |
| KDA per-seq `h_state`/`conv_state` (`alloc_state`) | per-seq `gpu.alloc` — new address per occupant | NO — the pool PR replaces this with slot-addressed pool states |
| KDA verify snapshots (`ensure_verify_slabs`) | lazy per-seq alloc on first verify, widened in place | NO — pool PR: `attach_verify_rollback` gives per-slot contiguous `k-1` H slabs (H-stride contract of `kda_recurrent_bf16_verify_snap`) + conv slabs |
| MLA per-step metadata uploads (`upload_meta`, `base_len`, block table) | per-step pageable H2D inside the layer | NO — hoisted pregraph by the prestage refactor (below) |
| `mla_block_table` staging cap | fixed `max_positions/bs+2` | yes |
| `max_blocks` kernel arg of `ling_mla_paged_attn` | equals `block_table.len()` per step | YES at single-seq: the kernel uses it only as the `seq` STRIDE (`block_tables + seq*stride`, `gated_mla.cu`); `seq` is `blockIdx.y`, 0 for every Ling call today. For multi-seq later, upload zero-padded tables and pass the fixed CAP. |
| base_len / cache walk length | read from the DEVICE word `base_len[seq]`; `kv_len = base_len + rows` | yes (content refreshes pregraph) |
| MoE path (`ffn.rs`) | device worklist + done-word memset; scratch only; 32-row sub-batching | yes (decode rows=1 → one batch) |
| `buffers.zero_all` each decode step | arena memset (MLA models) | yes (same as eager; a later no-skip for Ling is a separate perf change, not capture scope) |
| drafter `LingMtpHead` | own paged KV for 1 layer (`max_seq_len/16+2` blocks), device `draft_word`/`pick_word`s, block-table `Vec` growth via `alloc_block` (host bookkeeping of a preallocated pool) | yes with prestage; the pick readback stays eager |
| `mtp_prefill_hidden` (168 MB @ 32k × h2560) | one alloc per serve (`impl_a1_spec_init.rs`) | yes |

## Design

### Chosen mechanism: whole-step graphs, not piecewise `Pieces`

Ling serves single-node TP1 (`comm` is `None`) on one stream with no
collectives inside the step, so nothing forces the piecewise split the
GLM DFlash K=5 verify and the qwen4_exp pieces need. The whole-step decode
and K=2-verify caches already exist, are keyed by the slot, and bake only
slot-stable pointers. Kernel-arm variation (the KDA batch2-GEMV verify arm
vs the GEMM arm, GEMV-vs-GEMM projections by row count) is a function of
the STEP kind, and each step kind has its own cache, so no cross-arm
contamination is possible. Piecewise pieces remain the fallback if a
per-layer veto must stay (none is known today).

### Flags

- `ATLAS_LING_DECODE_GRAPH=1` (strict `==`, read once via `OnceLock`,
  logged through `startup_parity`): the family switch. Default OFF until
  the pool PR merges and every gate below passes; flipping the default is
  the lead's call after the numbers.
- Under the flag: `decode_graph_unsupported` returns `false` on both
  Ling layer kinds; returns `true` otherwise (today's behavior, and the
  state the e2e oracle runs in until then).
- Multi-sequence batched graphs stay OFF for this family for now
  (`decode_a2` admission gains a family gate; see risks). A failed
  capture there falls into the model-wide `suppress_graphs` poison, and
  the multi-seq Ling path still uploads metadata per row inside the
  layer loop — not prestaged by this slice.

### Where the hooks go

1. **MLA metadata prestage (non-behavioral prep, lands first).** The
   decode/verify metadata uploads move out of the layer to the dispatch
   Phase 1, exactly the GLM pattern (`decode_a`'s `meta_base` uploads):
   `decode_dispatch` Phase 1 calls a family-gated prestage that writes
   `mla_pos[0] = seq_len`, `mla_slots[0] = global_slot`,
   `mla_base_len = seq_len`, and the block table i32 upload into the fixed
   Ling staging — the same bytes the layer computed — and
   `LingMlaLayer::decode` then reads the staging instead of uploading.
   `verify_b` Phase 1 gets the two-row variant (`pos = seq_len + t`,
   the row slots, `base_len = seq_len`). Prefill and `kv_only_rows` keep
   their in-layer uploads (`upload_meta` stays for those paths). Stream
   order relative to consumers is unchanged, so the eager path is
   bit-identical: the only effect is that the bytes land earlier in the
   step.
2. **Flipping the veto** (after the pool PR): both layers return the flag;
   `decode_a`/`verify_b` admission adds the family test, the multiseq
   gate above, and keeps every existing suppression (profile, HSS, dump
   step, LoRA-eager, FP8-calibration `suppress_graphs`).
3. **Capture admission asserts** the V pool is bound (it must be, by the
   prefill-precedes invariant; a panic-free `ensure` returns eager
   otherwise) so a lazy bind can never happen under capture.

### What the KDA slot-pool PR must guarantee (their design; restated)

- Per-`(ssm_layer, slot)` fixed addresses for H `[32,128,128]` FP32
  (2 MiB/layer) and conv `[3·4096, 4]` FP32 state, zeroed on claim, never
  freed per sequence (the bedrock the decode/verify graph keys rest on).
- `attach_verify_rollback` for this family: per-slot contiguous
  `k-1 = 1` H slabs at the H stride and one conv slab per KDA layer — the
  `kda_recurrent_bf16_verify_snap` `state_inter` layout contract
  (`ensure_verify_slabs` asserts it today).
- KDA decode runs the slot-addressed state (or the
  `ling_kda_recurrent_indexed` seam they wire), and `free_sequence`
  retains slot-keyed graphs for the next occupant of the slot.

### Risk register

- **Pageable H2D under capture** — the actual `900/901` refusal source;
  solved by the prestage above; failure falls back per the existing
  abort/re-run machinery (never poisons the request).
- **Lazy alloc under capture** — verify slabs (pool PR), V pools
  (prefill-precedes invariant + admission assert), drafter block table
  (`alloc_block` is host bookkeeping of a preallocated pool — no device
  alloc, but it stays Phase-1/pregraph because it writes a host `Vec`).
- **`max_blocks` baked arg** — stride-only at `num_seqs == 1`
  (kernel-verified); documented for any later multi-seq flip.
- **GEMV batch2 lengths at M=2** — kernel-arm choice is a pure function
  of the step kind; each step kind owns its graph cache; row caps
  (`scratch.rows`, `MLA_MAX_ROWS_PAGED=8`) key no geometry surprises.
- **Sampler/argmax flow** — the verify/draft picks are device words
  (`ling_mtp_draft_pick` reads logits on device, writes `pick_word`);
  every host readback happens after `launch_graph`, off-graph. The target
  step's argmax reads `decode_logits_ptr()` after replay — unchanged.
- **Greedy nondeterminism** — serve-level greedy output is already
  nondeterministic across fresh PROCESSES; every A/B gate must compare
  in-process (see bench plan).
- **`seq64_boundary`** — the generic deferral stays; Ling's pool memsets
  have no known analog of the G1 alignment fault, but the deferral is
  lossless and stays on until bench evidence says otherwise.
- **Memory** — graph memory ≈ 0.17–0.25 MiB per ~14-kernel piece; a
  42-layer whole-step image is larger but bounded by the slot count
  (decode + verify2 + per-occupant): with a small serve pool (2–4 slots)
  this is tens of MiB against the ~35 GB MTP-serve headroom. The
  168 MB `mtp_prefill_hidden` and drafter pools are unchanged.
- **Shared GPU** — capture timing runs take short windows and coordinate
  with the kda-pool slice through the lead; static work needs no GPU.

### Bench plan (flip phase)

1. Build (`release`, GB10 toolchain: `CUDARC_CUDA_VERSION=13000`,
   `ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=ling-3.2-air`) on f341.
2. Gates: `cargo fmt --check`, `ATLAS_SKIP_BUILD=1` clippy, license
   headers, `typos`; `scripts_debug/ling_e2e_oracle.py` +
   `bailing_c1/e2e_gpu_parity_tests.rs` `#[ignore]` (2 tests); the KDA and
   MLA GPU parity harnesses; serve smoke "2+2" → "4";
   `mtp_accept_debug` acceptance ~0.82 unchanged; kernel audit
   unresolved:0.
3. Greedy 20-prompt diff, graphs ON vs eager, SAME process only (the
   eager arm is the identical binary with the family flag off or via a
   mid-serve runtime lever): effectively-identical requirement (same
   minor numeric-tolerance compare the campaign's parity harnesses use).
4. Perf: single-stream greedy tok/s with MTP on, `ATLAS_LING_DECODE_GRAPH=1`
   vs off, same process arm; nsys before/after for launch counts and
   decode-step time. Success: >= 36 tok/s sustained (20-prompt set).
5. Optional second stage: the NEXTN draft forward behind
   `ling_mtp_eh` device words — prestage `draft_word` (+ the drafter
   body's MLA metadata) pregraph, capture
   eh_norm → body → head → `ling_mtp_draft_pick` into a drafter-slot-keyed
   graph; the pick readback stays eager. Measure accept-rate unchanged +
   step-time before electing to keep it.

## Sequencing

1. (this PR) design doc + the MLA prestage refactor + flag plumbing,
   default OFF both. Gates 2–3 of the bench plan run against it with the
   flag off (bit-identical requirement).
2. The KDA pool PR lands and passes the e2e oracle on main.
3. (same branch, rebase) flip the veto under `ATLAS_LING_DECODE_GRAPH=1`,
   run the full gate + bench plan above, report numbers.
4. Lead decides default-on and the multi-seq / draft-graph follow-ups.
