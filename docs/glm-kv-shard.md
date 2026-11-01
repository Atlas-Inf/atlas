# GLM-5.3-Flash token-sharded MLA latents (`ATLAS_GLM_KV_SHARD=1`)

Status: implemented behind an opt-in flag. On the pair (2026-09-29, first
pass of the plan in the last section): retrieval correct at 16K/64K/128K,
prefix caching correct, pool 1.63M -> 2.84M tokens at the same memory
setting; decode at long context slower (see "Decode cost of the merge
form"), cold prefill TTFT 1-2% slower. Default off: with the flag unset
every allocation, kernel and launch is the one the engine ran before.

Those measurements are from the old base (0965f8d5). The port onto
integ/next (1b1f8551, branch `integ/exp-kvshard`) passes the unit tests and
builds, and has not run on the pair: see "On integ/next" for what changed
around it and what the port relies on.

## Why

GLM-5.3-Flash on a TP2 pair (two GB10, 128 GB unified memory each) stores
the full MLA latent cache and the full sparse-index cache for every token on
BOTH ranks. The latent is 90% of the per-token KV bytes, so four concurrent
512K-token contexts do not fit. The latent only has to exist once per pair.

Related work (not the source of this design): vLLM's decode context
parallelism (`decode_context_parallel_size`, Apache-2.0) also shards the KV
cache by token across ranks. See [glm-prior-art.md](glm-prior-art.md).

## Layout

- **Ownership by block-id residue, pinned to the logical index.** Block `b`
  is stored by rank `b % 2` at local slot `b / 2` of that rank's per-layer K
  pool (V aliases K). The rule lives in
  `crates/spark-runtime/src/kv_cache/latent_shard.rs` (`LatentShard`) and,
  for the device side, in `kernels/gb10/glm-5.3-flash/nvfp4/glm_kv_shard.cu`.
- **The ranks' physical ids differ; their residues do not.** Each rank runs
  its own allocator and frees in its own order (DFlash verify-row rollback,
  prefix-cache inserts and sequence frees iterate rank-local state), so after
  real traffic the same logical block has different ids on the two ranks
  (seen on hardware: a 128-block table differed from its first entry). A
  sharded pool therefore keeps one free list per id residue and draws the
  block for logical index `l` from list `l % 2`
  (`crates/spark-runtime/src/kv_cache/free_blocks.rs`,
  `PagedKvCache::try_alloc_block_at`). Every table entry then satisfies
  `b % 2 == l % 2`, so both ranks agree that logical block `l` is stored by
  rank `l % 2`, whatever the ids are. The sequence allocators
  (`block_mgmt::ensure_blocks_through_{prefill,decode}`) pass the index;
  index-less `alloc_block` / `try_alloc_block` are refused under the shard.
- **Prefix caching needs no change:** sharing is positional (a cached block
  is only matched at its own logical index), so a shared block keeps its
  residue and its owner.
- **Why residues rather than a per-rank slot map:** slot `b / 2` needs no
  map, no device table and no kernel change, and a rank can never be handed
  more blocks than its `ceil(N / 2)` slots. The cost is that a residue can
  run dry while the other still has blocks (many short sequences all at
  logical 0, or a prefix tree with more even than odd blocks):
  `num_free_blocks()` reports `2 x` the scarcer residue, which the
  scheduler's admission and prefix-cache reclaim loops already act on, and
  `block_mgmt::alloc_block_evicting` evicts from the prefix cache until the
  residue has a block. A reclaim measures its progress by every freed block
  (`num_free_in_all()`): an eviction that frees only the richer
  residue is progress, and a loop that stopped at "nothing gained" would
  give up with evictable blocks left. A slot map with spare slots would absorb such
  imbalance only by paying for the spare latents up front; the same bytes
  buy a larger pool.
- **Everything else is replicated:** block tables (in logical structure),
  refcounts, the prefix cache, eviction, the pooled semantic-index keys, the
  lent index tails and all KDA state.
- **Writes:** `write_kv_cache_bf16_latent` maps each row's global slot to the
  owner's local slot (`glm_kv_shard_map_slots`, `-1` elsewhere; every GLM
  latent writer skips `-1`). Only the owner writes a row's latent.
- **Fail closed:** under the shard, `PagedKvCache::k_pool_ptr / v_pool_ptr /
  k_cache_ptr / v_cache_ptr` panic (naming the caller) and `read_block /
  write_block` (sequence save/restore) return an error, so any latent reader
  that was not ported crashes loudly instead of reading another block.
  `zero_block(s)` and `poison_block` touch only owned local slots.

## Attention

Both ranks hold the replicated index keys, so both compute the identical
top-2048 selection locally (no top-k exchange). Every GLM MLA attention that
reads the latent cache in the DFlash serving lane goes through
`glm_chunk_attention` (prefill chunks, single-owner and owner-batched DFlash
verify, fused prefill+verify), plus the single-row eager decode
(`attention_forward_mla`). In shard mode each owner takes one of two forms
(`crates/spark-model/src/layers/qwen3_attention/prefill/paged_glm_shard.rs`):

### Merge form (owners of at most 64 rows: verify, decode, prefill tails)

1. Exchange the owner's absorbed queries (this rank's 32 heads,
   `rows x 32 x 512` BF16) with the peer.
2. `glm_kv_shard_localize`: rewrite the selected token ids to this rank's
   local token ids (read through an identity block table over the local pool);
   tokens stored by the peer become `-1`, which every GLM sparse kernel skips.
   Dense owners (sequence end <= 2048) generate causal ids instead.
3. Run the existing tensor-core split kernel
   (`glm_sparse_mla_prefill_{fp8g128,bf16}_head32_tc_kv_pad_split`) for the
   PEER's heads over this rank's tokens, merge its partitions to one FP32
   partial + natural LSE (`glm_sparse_decode_split_merge_f32`, new), and
   exchange that partial (`rows x 32 x 513 x 4` bytes) with the peer.
   Every entry point only a shard launches (this merge, the extra-partition
   merge and the counted split of the compact tuning) is compiled into the
   `glm_kv_shard` module from the unsharded kernels' bodies, so the modules
   an unsharded server loads are unchanged.
4. Run the split kernel for this rank's own heads over its tokens, append the
   peer's partial as the last partition, and merge everything in one exact LSE
   merge (the existing `glm_sparse_decode_split_merge`) into the head-sharded
   BF16 attention output. `W_uv`, `o_proj` and the TP all-reduce are
   unchanged.

A partition that owns no selected token has `LSE = -inf` and weight 0, so a
rank owning nothing contributes nothing (the unit tests check this and the
equivalence with plain softmax attention on a CPU mirror).

Why this over fetching the remote selected latents: per row the merge form
moves 32 KiB of queries + 64 KiB of FP32 partials each way, with fixed shapes
and no host sync. Fetching latents moves ~1,025 remote tokens x 528 B = 541 KiB
per row (less after de-duplicating across an owner's rows, but then the
payload size is data dependent and needs a host sync or a padded worst case).
At the served shapes (<= 4-8 owners x <= 8 rows) the merge form moves 5-10x
fewer bytes. Attention FLOPs and latent bytes read are unchanged: each rank
reads only its half of the selected tokens, for twice the heads.

### View form (owners of more than 64 rows: prefill chunks)

The chunk's queries select most of the history, so each rank assembles the
sequence's whole latent history `[0, end)` in a scratch view: it reads the
block table to the host (one sync per layer), checks that every entry sits on
its logical index's residue, swaps that verdict and its block count with the
peer (8 bytes; a rank whose table is broken, or whose peer's is, fails after
the swap, so both ranks fail instead of one leaving the other in the next
exchange), copies its own blocks into their logical positions, and exchanges
the rest with the peer in 2,048-block pieces (packed in logical order, so
both sides agree on the order). The
unchanged kernels (TC sparse prefill, the BF16 dequantized view, the dense
<= 2048 path) then read the view through an identity block table. The
chunk's new rows are written (owner-only) before the view is assembled, so
the view includes them.

## Memory (per rank, fp8_g128, 11 MLA layers, 16-token blocks)

| per token | before | after |
|---|---|---|
| MLA latent (11 x 528 B) | 5,808 B | 2,904 B |
| pooled index keys (11 x 64 B) | 704 B | 704 B |
| index tail-slot map + identity table | 0.25 B | 0.375 B |
| **total** | **6,512.25 B** | **3,608.4 B** (-44.6%, pool x1.80) |

(The task brief quoted 5,892 B for the latent; the code's layout is
16 x 528 B per block per layer = 5,808 B per token over 11 layers.)

Fixed cost of the shard scratch (allocated before KV sizing and charged to
the budget through `GlmCachePlan::latent_sharded`): at `--max-seq-len 524288`
a 32,772-block view (264 MiB for fp8_g128, 512 MiB for BF16), ~38 MiB of work
space for exchange pieces / merge partials, and small lists — about 0.3 GiB.

At the production budget (852,448 tokens x 6,512 B = 5.55 GB at
`--gpu-memory-utilization 0.90`) the sharded pool is
(5.55 GB - 0.32 GB) / 3,608 B = **~1.45M tokens** (was 852K). Four full
512K contexts need 2,097,152 tokens = 7.9 GB with the shard (13.7 GB
without), so the shard alone closes most, not all, of the gap: another
~2.3 GB of budget is still needed for 4 x 512K.

## Expected speed impact (reasoned, not measured)

- **Verify / decode (merge form):** two pair exchanges per owner per MLA
  layer (4 owners x 11 layers = 88 per step; ~15-40 us each on the RDMA pair,
  ~34 MB per step each way at 4 x 8 rows) plus a localize kernel, one extra
  split launch (same bytes read), one FP32 merge and two small copies. Expect
  roughly +1.5-3.5 ms per verify step (1-3% of today's 110-250 ms steps).
  Follow-up: batch all owners of a layer into one query exchange and one
  partial exchange (22 per step). (Measured since: see "Decode cost of the
  merge form" below — the extra split launch is not free, because a masked
  key costs the same tensor-core work as a stored one.)
- **Prefill (view form):** per layer per chunk one host sync, the 8-byte
  verdict swap and its read-back (a second sync), a local copy of
  the own half of the history, and an exchange of the peer's half
  (at 256K history ~69 MB each way per layer, ~31 ms per 8K chunk over 11
  layers at ~25 GB/s, ~3% of a chunk; ~6% at 512K). Follow-up: prefetch layer
  l+1's view while layer l computes (the history outside the chunk is known
  before the chunk starts), and plan once per chunk instead of per layer.
- **CUDA graphs are suppressed** when the cache is sharded (the exchanges and
  the view's host sync are eager). The DFlash verify lane is already eager;
  plain decode loses graphs.

## Decode cost of the merge form, and its two tunings

Measured on the pair (2026-09-29, single stream, greedy text identical to the
flag-off run): decode 142.8 -> 124.5 tok/s at 61K tokens and 132.3 -> 125.9
at 125K. Two caveats on those numbers: that profile also set
`ATLAS_GLM_KV_SHARD_CHECK=1`, and each is one request of ~65 generated tokens
(about half a second of decode), so the 61K-vs-125K difference is within the
noise of one sample. Nothing in the merge form grows with the context: the
kernel microbench below costs the same at 16K, 64K and 128K, so the relative
cost can only shrink as the step itself gets longer.

Per verify step and per MLA layer (11 of them), the merge form adds to the
unsharded `split kernel + merge`:

| extra work | where | cost at 8 rows |
|---|---|---|
| a second split launch over a row that is half `-1` | each launch still walks all 65 key tiles of the 2051-wide row; a masked key costs the same QK/PV tensor-core work as a stored one | the attention kernel time doubles: +83 us (microbench) |
| query swap, 32 KiB/row each way | two copy-engine staging copies (13 us measured) + the RDMA write (~19 us at 14 GB/s) + flag round trip, with the compute stream parked on it | ~40 us (estimated) |
| partial swap, 64 KiB/row each way | same (25 us of copies, ~38 us on the wire) | ~70 us (estimated) |
| localize, FP32 merge of the peer's heads, two device copies of the landed partial, one more merge partition | small launches | inside the +83 us |
| `..._CHECK=1` only | block-table read to the host + an 8-byte swap + its read-back: two stream syncs per layer, 22 per step, each draining the launch queue | not in the microbench; drop the flag for serving |

So an 8-row step pays roughly 0.9 ms of extra kernel time and 1.2 ms parked
on exchanges (plus the check's 22 syncs in the run above). Only the kernel
time is measured; the exchange costs are estimates until a pair run with
`ATLAS_GLM_VERIFY_PROFILE=1` reports the MLA time per step. Both tunings are
exact, opt-in, and refused at boot without `ATLAS_GLM_KV_SHARD=1` (a tuning
that silently did nothing would make an arm measure the wrong thing):

- **`ATLAS_GLM_KV_SHARD_COMPACT=1`.** `glm_kv_shard_localize_compact` packs
  each row's owned IDs to the front (a stable partition, one CTA per row, no
  host sync) and writes their number per row; the `*_split_counted` kernels
  partition that prefix instead of the whole row, so the two launches walk
  ~33 tiles each instead of 65; `glm_sparse_decode_split_merge_extra` merges
  the peer's partial where the exchange landed it (no copies). The launch
  shapes stay `rows x splits` on both ranks; only how far a CTA walks depends
  on the data. Same softmax, but the partition boundaries move: the attention
  kernel rounds each probability to BF16 relative to the running max of its
  partition, so those roundings and the FP32 summation order both differ from
  the uncompacted shard (at most one BF16 ulp of the output in the
  microbench, the same class as split vs unsplit). Greedy text can therefore
  diverge between the shard and the compact shard, as it can between flag-off
  and the shard.
- **`ATLAS_GLM_KV_SHARD_OVERLAP=1`.** Both swaps run on a side stream
  (`ExchangeLane`, fenced by two events: `glm_kv_shard::overlapped_exchange`)
  beside compute that does not need them: the partial swap beside this rank's
  own heads' partitions, the query swap beside the owner's index update and
  selection. For the latter a single verify owner takes the owner-batched
  projections, so its queries exist before its selection (the same
  projections in a different order). The window cannot use the pair, which
  orders its sends by stream order and counts landings in one place: the
  selection runs on a context whose communicator (`glm_kv_shard::WindowComm`)
  refuses every pair operation and reports no exchange support, so a later
  change that exchanges inside a window fails on both ranks with an error
  instead of corrupting the pair, and an index split (opt/index-split, owners
  of 256+ rows) stays replicated there on both ranks. Off under
  `..._CHECK=1` and under graph capture. Both ranks issue the same exchanges
  in the same order, so an overlap only moves where each rank waits. The
  lane (one stream, two events) exists only under this flag and is destroyed
  with the cache.

  Two things about the overlap are not established. Its gain is an estimate
  (at most the ~0.9 ms per 8-row step the exchanges park the compute stream
  for today); it adds eight driver calls per layer and a cross-stream wake
  whose latency nobody has measured, and at 1-2 rows the exchanges are only
  10-20 us, so it can be a net loss there. And with `ATLAS_RDMA_ONESHOT=1`
  (opt/oneshot-ar) payloads up to 1 MiB, which covers both shard payloads up
  to 8 rows, travel through a kernel that busy-waits for the peer: on the
  lane that kernel holds SMs during the very window the overlap is meant to
  free. Judge the overlap by the MLA time per step with one-shot off, and
  again with it on if the serving profile sets it.

Kernel microbench (`scripts/dev/glm_kv_shard_bench.cu`, ennspark03, fp8_g128,
one rank's launches per MLA layer, minimum of 40 batches of 50; the GPU is
shared, so compare within a row):

| rows | context | unsharded | shard | shard + compact |
|---|---|---|---|---|
| 8 | 16K | 84.1 us | 165.1 (+81.0) | 119.1 (+35.0) |
| 8 | 64K | 85.5 us | 170.5 (+85.0) | 120.1 (+34.6) |
| 8 | 128K | 86.4 us | 173.1 (+86.7) | 119.9 (+33.6) |
| 8 | 64K, this rank stores 70% of the selection | 85.5 us | 177.9 (+92.4) | 142.1 (+56.6) |
| 4 | 64K | 59.3 us | 114.0 (+54.7) | 85.3 (+26.0) |
| 2 | 64K | 53.1 us | 104.5 (+51.4) | 83.2 (+30.2) |
| 1 | 64K | 50.0 us | 102.1 (+52.1) | 81.3 (+31.4) |

All three pipelines match a double-precision CPU softmax attention over the
same dequantized latents to 1.2-1.5e-4 before BF16 rounding (output rms
0.021), and their BF16 outputs differ from each other by at most one BF16
ulp. The in-place merge is bitwise equal to copying the partial and merging
one more partition. The bench checks both ranks' heads, the localized ids
against the ownership rule, and the compacted ids against the uncompacted
ones, and exits non-zero on a failure.

The times above are minima caught while the shared GPU was otherwise idle;
under contention every pipeline doubles (8 rows, 64K: 170 / 345 / 251 us) and
the ratios hold: the shard costs 2.0x the unsharded kernels, the compact
shard 1.4-1.5x.

Its last argument takes the causal form instead (`causal` >= 0: no
selection, the shard kernels generate the causal ids), which is what every
sequence up to 2048 tokens runs — dense attention is exact there, so there is
no selection to localize. Checked at 8 rows from token 0 (rank 1 owns nothing
of any row), from 12 (four rows rank 1 owns nothing of, four it does), from
1000 and 2040, at 1 row from 0 and at 64 rows from 500: ids and outputs pass
for both ranks, and where a rank owns nothing the compact and uncompacted
shards are bitwise equal.

Flag-off identity of the kernels: the release build of this tree was
compared with a release build of a tree whose kernel sources are
integ/next's (1b1f8551; every regular file under `kernels/` and
`crates/atlas-kernels/` hashes equal). All 229 PTX modules integ/next builds
for this target are byte-identical files here, the two edited ones
(`glm_sparse_prefill_kv_reuse`, `glm_sparse_decode_split_merge`) included,
and their 694 entry points are unchanged. The port adds one module,
`glm_kv_shard`, with eight entry points (the four `glm_kv_shard_*`, the two
`*_kv_pad_split_counted`, `glm_sparse_decode_split_merge_f32` and `_extra`),
which no unsharded server looks up and so never loads. Two things make the
edited files compile to what they did: the counted split's row counts and the
merge's extra partition are compile-time dead in the old entry points (a
defaulted null argument and a template parameter), and the merge's shared
variables are declared by each entry point rather than by the shared body, so
they keep the names they had. On the old base the merge still took the extra
partition as a run-time pointer and compiled differently; there the unsharded
split + merge and the unsplit kernel were run from both trees on the same
inputs (rows 1/2/3/4/8, full / padded / holed selections, BF16 and fp8_g128,
81 MB of partials, LSEs and outputs) and compared: bitwise equal.

What is left with both tunings: ~30 us per layer of launches that have no
unsharded counterpart (compaction, the FP32 merge, a second launch's fixed
cost), and whatever part of the partial swap outlasts the own-head partitions
it hides behind — roughly 0.4-0.7 ms per 8-row step.

## On integ/next

What integ/next added since the old base, and how the shard sits with it.
None of it is measured on hardware yet.

- **Index split (`ATLAS_GLM_INDEX_SPLIT`).** The split is over an owner's
  query rows, scored against the pooled index keys, which stay replicated;
  the ranks swap finished rows of logical token ids. Nothing in it depends
  on which rank stores a latent, and both ranks end with the identical full
  selection as before. It admits owners of 256+ rows, which are always
  view-form owners; an overlap window (merge form, at most 64 rows) hands the
  selection a communicator without exchange support, so a split cannot start
  inside one.
- **Pipelined sparse prefill (`ATLAS_GLM_SPARSE_PREFILL_PIPE`).** A
  view-form owner hands the unchanged kernels its assembled history and the
  identity table where they took the pool and the sequence's table
  (`glm_owner_latents`). Under the pipe a sparse owner gets no BF16 view
  unless the native library admits it, so the pipelined kernel reads the
  assembled `fp8_g128` view directly; a BF16 view, when wanted, is
  dequantized from the assembled view. The native bridge is refused for
  sharded latents without a BF16 view. Merge-form owners never build a view
  and always run the `*_kv_pad_split` kernels.
- **KV write floor.** The floored cache write maps only the written rows'
  slots (`joint.slot + floor`) to local slots. Rows below the floor sit in
  blocks a prefix match adopted at their own logical index, whose latents
  their owner wrote when the prefix was computed.
- **Mirrored KV admission and retry.** Each rank reserves a chunk's blocks
  by logical index and the ranks vote; a residue that runs dry on one rank
  is an agreed `Exhausted`, rolled back on both (the blocks return to their
  residue's list).
- **Prefix sharing and policy.** Matches cover whole blocks, are cut to the
  ranks' agreed length and are adopted positionally (`prefix_share`), so a
  shared block keeps its residue. `ATLAS_GLM_PC_EVICT`, `..._BRANCH` and
  `..._FINISH_LEAF` act on KDA snapshots, not on latents.
- **Pool sizing.** `kv_budget` still yields a byte budget from the process's
  own footprint; `GlmCachePlan::latent_sharded` turns it into blocks with the
  halved latent and the fixed scratch. The scratch itself is allocated with
  the pool, after sizing.
- **Startup agreement.** The four variables are in the startup settings
  table (`model::startup_parity`, read through
  `glm_kv_shard::{requested, MergeTuning::get}`), so ranks that differ fail
  right after the communicator comes up, naming the variable and both
  values. The pool-size gather (`agree_kv_blocks`) checks them again, which
  holds under `ATLAS_STARTUP_PARITY=warn` too: `[shard, compact, overlap,
  check]` ride bits 28-31 of the word the ranks gather, below the NVMe spill
  tier's word in the upper half (`kv_nvme::rank_word`), and ranks that
  differ fail there, each naming both values. The block count keeps bits
  0-27. With the shard and the tier off the word is the plain block count.
- **Startup refusals.** A shard refuses, by name, the environment opt-ins
  whose attention still reads latents by global block id
  (`glm_kv_shard::unsharded_lane`): `ATLAS_GLM_MULTI_SEQ_SPARSE`,
  `ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS`, `ATLAS_GLM_MTP_REPAIR`,
  `ATLAS_GLM_C4_DECODE`, `ATLAS_GLM_C4_SPARSE`,
  `ATLAS_GLM_INDEPENDENT_DECODE` and `ATLAS_GLM_LONG_BATCH_SERIAL`. The
  production profile sets the first six to 0 and not the last. Any of the
  four shard variables with a value other than 0 or 1 fails the boot, as does
  a tuning or the check without the shard.
- **Verify graphs (`ATLAS_GLM_VERIFY_GRAPH`)** are vetoed by the shard's
  graph suppression on both ranks: verify stays eager.
- **`ATLAS_GLM_DET_TRACE`** taps the selected ids of merge-form owners like
  every other owner's (`sel`).

## Numerics

The same softmax (no quantization or approximation is added), but not
bitwise identical to flag-off: the merge form moves the partition
boundaries, which changes the kernel's per-partition BF16 probability
rounding and the FP32 summation order (as the existing split verify already
does; at most one BF16 ulp of the output measured), and dense <= 2048 owners
take the tensor-core split kernel over causal ids instead of the BF16 dense
kernel. The view form runs the same kernels on the same bytes as flag-off.

## Supported / not supported

Supported (implemented): prefill chunks (incl. sequence-parallel prefill,
pieces across the 2048 dense/sparse boundary), DFlash verify (single owner,
owner-batched, fused prefill+verify), single-sequence eager decode, BF16 and
fp8_g128 caches, prefix caching (positional sharing keeps each block's owner).

Refused at startup: a rank whose `[shard, compact, overlap, check]` differ
from its peer's, a shard variable that is not 0 or 1, a tuning or the check
without the shard, the environment lanes listed under "Startup refusals"
(multi-sequence sparse decode, the repaired MTP, C4 and independent decode
lanes, the long-verify serial diagnostic), `--high-speed-swap`, the NVMe
prefix tier (`ATLAS_KV_NVME_DIR`: it spills and restores whole blocks'
latents by physical id), a world
other than a TP2 pair with 32 heads per rank, a latent dtype other than BF16
or `fp8_g128`.

Not supported and not selected by a variable a shard could refuse (the
production profile sets `ATLAS_GLM_MLA_MULTI_SEQ=1`), so still failing closed
mid-request with the `k_pool_ptr` panic or an error: multi-sequence batched
decode (`decode_batch` / multi-seq MLA) and sequence save/restore. In the
DFlash lane plain decode is reached only for sequences whose speculation is
suspended; those go through the supported single-sequence decode, but a
batched bootstrap of several such sequences would hit the fail-closed panic.

## Hardware validation plan

1. Build the engine (the new `glm_kv_shard.cu` module and the
   `glm_sparse_decode_split_merge_f32` entry point must compile). Boot the
   production profile on both Sparks with `ATLAS_GLM_KV_SHARD=1
   ATLAS_GLM_KV_SHARD_CHECK=1`. Check the boot log on both ranks:
   - `KV latent shard: rank r of 2 stores X of N blocks' latents; ~300 MiB
     exchange scratch` (X = ceil(N/2));
   - `KV cache: ... -> N blocks x 16 tok/block = T max KV tokens` with T about
     1.7x the flag-off boot at the same `--gpu-memory-utilization`;
   - `KV cache: N blocks x 11 layers = G GB total (V aliases K)` with G about
     half the flag-off value's latent part;
   - `KV latent shard merge form: compact=.. overlap=.. check=..` naming the
     tunings in effect (a performance arm must show `check=false`).
   `..._CHECK=1` makes every merge-form attention check on the host that
   each table entry's residue matches its logical index and swap that verdict
   and the block count with the peer (both ranks fail if either is wrong);
   drop it for performance runs. The view form does this on every call
   regardless.
2. Greedy equality: the same prompts at temperature 0 with the flag off and
   on (short, 16K, 64K prompts; single sequence and 4 concurrent). Expect
   identical output for most prompts; where the text diverges, compare the
   per-token logprobs up to the first divergence (differences should be at the
   FP32-reassociation level, well below the top-1/top-2 margin at the
   divergence point). A systematic difference in the first tokens means a bug.
3. Long-context needle retrieval at 64K, 200K and 400K with the flag on vs
   off, then 4 concurrent 512K contexts (the goal) — the flag-off run cannot
   hold them.
4. DFlash acceptance: mean accepted drafts per step must match flag-off
   within noise (a broken attention collapses acceptance first).
5. Performance: the RigMark / matrix cells vs flag-off; per-step time of
   owner-batched verify at 1 and 4 owners (`ATLAS_GLM_VERIFY_PROFILE=1` gives
   the MLA vs KDA split); prefill throughput at 16K/128K/256K history.
6. Prefix caching on (`--ssm-cache-slots > 0`) with shared prefixes.
7. If a panic names `k_pool_ptr`/`v_pool_ptr`, the backtrace names a latent
   reader that is not shard-aware (see "Not supported").
