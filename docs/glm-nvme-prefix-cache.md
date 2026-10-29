# GLM-5.3-Flash: prefix cache spilled to NVMe

Status: opt-in; no behaviour change without the flags below. Ported onto
`integ/next` (whole-block prefix sharing, prefix-cache policy flags, KV
admission vote, KV sizing by own footprint) as `integ/exp-nvme`; §12 lists
what the port changed. The measurements below are from the old base.

- The tier as first built (`integ/nvme`) has run on the pair: an evicted
  25 K-token conversation's turn 2 starts in 1.53–1.86 s instead of
  11.2–11.5 s, with byte-identical answers (§9).
- `opt/nvme-fast` adds a fast I/O path and record keeping behind two more
  flags (`ATLAS_GLM_NVME_FAST`, `ATLAS_GLM_NVME_KEEP`). Those are unit-tested
  on a Mac and measured on one GB10's NVMe and GPU in isolation (§9); **they
  have not served a request on the pair yet.**

## Goal

Today, when the GPU prefix cache is full, an idle conversation's KV blocks are
deleted, and so are its KDA (SSM) snapshots once the 16 snapshot slots are
full. When the user comes back, the whole conversation is prefilled again,
which takes tens of seconds to minutes at GLM-5.3 context lengths.

With this change, those blocks and snapshots are written to the node-local
NVMe instead of being deleted. A later prefix hit reads them back in before
the prefill starts, which should take well under a second.

## 1. What existed, and what it does for GLM today

### Prefix cache (radix tree), as served

- `--enable-prefix-caching` builds a `RadixTree`
  (`crates/spark-runtime/src/radix_tree*`). There is one node per 16-token
  block, and each node's `context_hash` is chained from its parent.
- LRU eviction (`radix_tree/inner/evict.rs`) **deletes** the least recently
  used leaf node that no sequence holds, and returns its block to the free
  list. From then on, that prefix is a cache miss.
- The SSM snapshot index (`radix_tree/snapshot*.rs`) is kept separately from
  the tree nodes. With `ATLAS_MARCONI_PREFILL_ONLY=1`, only block-aligned
  prefill checkpoints (about 74 MB each) are saved. On a hit, prefill restores
  the deepest snapshot at or below the matched length and replays forward from
  there (`prefill_b/prefix_lookup.rs`). For a hybrid model like GLM, a KV hit
  with **no** snapshot is recomputed in full: "Prefix cache hit … but no SSM
  snapshot — recomputing all KV".
- Rank agreement (F83): each rank computes its own `matched_tokens`, and the
  ranks agree on the minimum with `ep_min_u32`. The snapshot restore decision
  was **not** synchronised; it was only correct because both ranks' snapshot
  pools happen to evolve identically.

### `--high-speed-swap` (HSS): not usable for this

`crates/spark-storage/src/high_speed_swap*`, `model/block_mgmt.rs`,
`serve_phases/*`:

- HSS offloads a single sequence's **live** KV during **decode**, using a
  sliding window. `--high-speed-swap-cache-blocks-per-seq` (default 64) blocks
  stay resident in GPU memory, older blocks go to per-layer NVMe files, and a
  low-rank predictor chooses which blocks to stream back for tiled attention
  (`ATLAS_HIGH_SPEED_SWAP_REPLACE`, which its own log line marks as "UNTESTED
  on real models").
- Prefill never reads from disk (Issue #31 in `block_mgmt.rs`).
- Its on-disk layout (`GroupLayout`) hard-codes BF16 GQA K/V, striped per KV
  head, sized from `num_key_value_heads`/`head_dim`. GLM's cache is different:
  - a single 512-wide latent head stored as FP8-G128 (528 B per token per
    layer), with V aliasing K;
  - a separate pooled sparse-index cache;
  - sparse MLA decode kernels that do not go through HSS's tiled path.
- Its radix-tree integration (`disk_block_id` refcounts, `matched_disk_block_ids`)
  only keeps disk IDs alive for nodes that are still in the tree. Eviction
  still deletes the node. **HSS never makes an evicted prefix findable again.**
- `kv_paging` / `rdma_kv_backend` are alternative storage backends for HSS
  (a remote RDMA peer), so the same limits apply.

Verdict: none of HSS is reused. The new tier refuses to run alongside it
(`attach_nvme_spill` errors if `cache_blocks_per_seq` is set).

### SSM snapshot spill tier: reused as-is for the KDA snapshots

`ATLAS_SSM_TIER*`, `model/ssm_tier/`, `model/ssm_snapshot_spill.rs`,
`radix_tree/snapshot_tier.rs`:

- When all 16 Marconi slots are in use, the victim snapshot is spilled
  (spill-not-drop) instead of dropped, provided it is at least
  `ATLAS_SSM_SPILL_MIN_TOKENS` deep (default 1024). The index entry stays, so
  a warm lookup finds it and faults it back into a slot (all three prefill
  paths do this, gated by `ATLAS_SSM_FAULT_MIN_TOKENS`, default 256).
- With `ATLAS_SSM_TIER_UNIFIED=1`, the store is a `Residency` with:
  - a host-RAM hot arena of `ATLAS_SSM_TIER_SLOTS` × blob, **allocated up
    front, default 64**;
  - LRU spill to an O_DIRECT swap file in `ATLAS_SSM_TIER_SWAP_DIR`;
  - a record cap from `ATLAS_SSM_TIER_DISK_GB`.
- For GLM the capability gate passes (the model has recurrent state), and the
  blob is `34 × (h + conv)` per rank. Worked out from
  `test_data/glm5_next_nvidia_nvfp4_config.json`:
  - h = 32 heads × 128 × 128 × 4 B = 2 MiB (FP32, TP2);
  - conv ≈ 3 × 12 288 × 2 B;
  - blob ≈ 73.8 MB by that arithmetic; **the pair logs `blob_bytes=77987840`**
    (78.0 MB) per rank. It is a 4 KiB multiple, and the O_DIRECT arm applies:
    the startup line is `unified SSM tier (marconi-host): O_DIRECT swap file
    in …`. If you see "not a 4 KiB multiple" or "unusable" instead, the swap
    tier is in **host RAM**.
- GB10 gotchas:
  - Host RAM is the same pool as GPU memory, so the default 64-slot hot arena
    costs **about 5 GB**. Set `ATLAS_SSM_TIER_SLOTS=2`.
  - The swap file used to be named per PID and **never unlinked** (defect #3
    in `ssm_tier/unified.rs`) — and the PID is 1 in every container, so two
    containers sharing a directory truncated each other's file. It is now
    unlinked once open, and same-tag leftovers are swept at startup.
  - If the swap directory cannot be used, the blob is not a 4 KiB multiple,
    or `ATLAS_SSM_TIER_UNIFIED` is not set at all (the legacy store), the
    spilled snapshots stay in host RAM — bounded by `ATLAS_SSM_TIER_DISK_GB`
    at best, by nothing at worst. With the KV tier on, startup refuses every
    one of those (§3, §11): the store reports where it keeps its spills
    (`SpillHome`), from what it built rather than from the environment.
- Gap closed: with a spill tier on — `ATLAS_KV_NVME_DIR` or `ATLAS_SSM_TIER`
  itself — or with `ATLAS_GLM_PC_EVICT` / `ATLAS_GLM_PC_BRANCH`, the ranks
  agree on the Marconi restore depth (see §5). Without that, a fault-in that
  succeeded on one rank and failed on the other made the ranks diverge on
  `skip_tokens`, which leads to mismatched collectives and a deadlock.

### Related finding, not fixed here

`--swap-space-gb` sequence preemption (`trait_impl/sequence/state_io.rs`)
saves and restores K/V with `read_block`/`write_block` only. **It does not
carry the GLM sparse-index cache** (pooled keys, scales, raw tails — with
slotted tails those also need re-lending), so a GLM sequence restored from a
preemption swap would have stale index keys. A TODO in `state_io.rs` records
this; until fixed, do not combine `--swap-space-gb` with GLM. The fix could
reuse the segment layout in `kv_cache/nvme_spill.rs`.

## 2. What changed

| Layer | File | What |
|---|---|---|
| tree | `spark-runtime/src/radix_tree/inner/nvme.rs` | On-disk node state, record-slot allocator under a fixed budget, exact on-disk LRU, spill victim selection, restore plan/pin/promote, subtree drop |
| tree | `radix_tree/inner.rs`, `inner/evict.rs` | `walk` stays resident-only (`block_idx != MAX`, a no-op when the tier is off). `insert` moves an on-disk node onto the recomputed block. `evict` routes to spill mode |
| tree | `radix_tree/nvme.rs`, `prefix_cache/nvme.rs` | `NvmePrefixTier` trait (`PrefixCache::nvme()`), `SpillOrder`/`DiskRef`/`RestorePlan` |
| tree | `radix_tree/snapshot_tier.rs` | `peek_deepest`: read-only deepest usable SSM anchor |
| bytes | `spark-runtime/src/kv_cache/nvme_spill.rs` | Record layout, trailer stamp/verify, attach, dispatch to one of the two I/O paths |
| bytes | `kv_cache/nvme_sync.rs` | The synchronous path (default): staged gather/scatter, one write per record, ranged reads |
| bytes | `kv_cache/nvme_fast.rs`, `kv_cache/nvme_io.rs` | The fast path (`ATLAS_GLM_NVME_FAST`): pitched copies, run-sized I/O, write-behind and read-ahead over a fixed staging ring and a worker pool (§5) |
| gpu | `spark-runtime/src/gpu.rs`, `cuda_backend/gpu_copy.rs` | `copy_h2d_pitched_async_retained` / `copy_d2h_pitched_async`: one `cudaMemcpy2DAsync` per region per run of blocks |
| store | `atlas-tier` `SwapStore::read_records` | Reads a run of consecutive records with one O_DIRECT `pread` |
| store | `atlas-tier` `ConcurrentSwapStore`, `SharedRecordFile` | Positional, lock-free record runs in either direction (`preadv`/`pwritev`), exclusive owner-only create, up-front reservation, stale-file sweep |
| model | `spark-model/src/model/block_mgmt.rs` | `apply_evicted_blocks` writes spills **before** returning blocks; a failed write falls back to plain eviction; spill-mode evict batch of 32 |
| model | `model/kv_nvme.rs` | `restore_prefix` cycle; hybrid restore policy |
| model | `prefill_b/prefix_lookup.rs`, `prefill_b/pc_policy.rs` | Restore before the chunk-0 lookup; the restore-depth agreement (`pc_agree_restore`) also runs with the tier on |
| build | `factory/build/kv_nvme.rs`, `build/glm.rs`, `build.rs` | Env parsing, record file, enable, the rank-config word on the KV-block all-gather, host-memory reserve taken out of the KV budget (§10) |

### Invariants

- **An on-disk node has only on-disk descendants.**
  - A spill only picks a resident node with no resident children.
  - A restore promotes an on-disk run from the top down, starting directly
    below the resident prefix.
  - `insert` moves nodes from the top down.
- **Every existing consumer of `PrefixMatch` still sees resident blocks
  only.** A restore happens before `lookup`, and turns on-disk nodes back into
  ordinary resident nodes that hold a single cache reference.
- **Only full blocks are spilled.** The tree holds nothing else: a partly
  filled block is never published (`model/prefix_share.rs`).

## 3. Flags

Set identical values on **every rank**. On a multi-rank world the ranks
already all-gather their KV block counts at startup; each rank's word now
carries a fingerprint of these settings in its upper half (0 with both spill
tiers off, so the collective and its bytes are unchanged without the flags),
and every rank refuses to start if the fingerprints differ. A rank whose tier
environment does not parse, or that holds a combination the tier refuses
(below), sends a sentinel, so it stops every rank in that collective. What
fails later on one rank only (directory, disk reservation, staging) stops
that rank with an error naming the variable, and its peers through the EP
peer lifeline (exit 74).

| Variable | Meaning |
|---|---|
| `ATLAS_KV_NVME_DIR=<dir>` | Enables the KV tier. Must be on the real NVMe filesystem, not tmpfs/overlay (O_DIRECT). |
| `ATLAS_KV_NVME_GB=<GiB>` | **Required** alongside the dir (strict parse, > 0). Per-rank disk budget; the coldest on-disk blocks are dropped beyond it. |
| `ATLAS_GLM_NVME_FAST=1` | The fast I/O path (§5): same records, moved in run-sized requests and pitched copies, written behind the evicting request. Also reserves the whole budget on disk at startup. Strict `1`/`0`; part of the rank fingerprint. |
| `ATLAS_GLM_NVME_KEEP=1` | A restored block keeps its record, so evicting it again writes nothing (§5). Strict `1`/`0`; part of the rank fingerprint. |
| `ATLAS_SSM_TIER=1` `ATLAS_SSM_TIER_UNIFIED=1` `ATLAS_SSM_TIER_SWAP_DIR=<dir>` `ATLAS_SSM_TIER_DISK_GB=<GiB>` `ATLAS_SSM_TIER_SLOTS=2` | Existing KDA snapshot tier (see §1). Where it keeps its spills (nowhere / an O_DIRECT file / a peer) is part of the rank fingerprint. |

`ATLAS_KV_NVME_GB` and the two `ATLAS_GLM_NVME_*` switches only shape a tier
that is on. Without `ATLAS_KV_NVME_DIR` they are not read at all: the rank
starts exactly as it does with them unset, whatever they hold, and logs one
warning per variable that is set.

Startup errors with `ATLAS_KV_NVME_DIR` set:

- without `--enable-prefix-caching`, or together with `--high-speed-swap`;
- **with `ATLAS_SSM_TIER` when the snapshot tier keeps its spills in host
  RAM**: no `ATLAS_SSM_TIER_UNIFIED=1`, no usable `ATLAS_SSM_TIER_SWAP_DIR`,
  or a blob that is not a 4 KiB multiple. The KV pool is sized around the
  host memory the tiers take (§10), and 78 MB per spilled snapshot in RAM is
  outside any such bound; the hosts hang when unified memory runs out. Every
  rank stops (the sentinel above);
- **on a multi-rank world with `ATLAS_EP_PEER_LIFELINE=0`**: a rank that fails
  in its local setup after the agreement would leave its peer waiting in the
  next collective.

First hardware test (both nodes):

```sh
# CLI, unchanged: --enable-prefix-caching --ssm-cache-slots=16
export ATLAS_MARCONI_PREFILL_ONLY=1
export ATLAS_KV_NVME_DIR=/<nvme-mount>/atlas-kv
export ATLAS_KV_NVME_GB=200
export ATLAS_SSM_TIER=1
export ATLAS_SSM_TIER_UNIFIED=1
export ATLAS_SSM_TIER_SWAP_DIR=/<nvme-mount>/atlas-ssm
export ATLAS_SSM_TIER_DISK_GB=200
export ATLAS_SSM_TIER_SLOTS=2
export ATLAS_SSM_TIER_TIMING=1      # optional: per-spill timing lines
```

## 4. Disk layout

- **One record file per rank:** `<dir>/atlas-kv-prefix.<pid>.r<rank>.swap`,
  opened O_DIRECT and **unlinked immediately**.
  - Its space is returned when the process exits for any reason, including a
    crash.
  - No other process can ever read a stale record.
  - `ls` does not show it; use `df` or `lsof` instead.
- **Record `slot` is at offset `slot × record_bytes`**, and the file grows
  sparsely.
- **Record contents** (this rank's bytes for one 16-token block):
  1. For each attention layer:
     - `K` (the full per-block stride; for FP8-G128 that includes the scales);
     - `V`, only when it does not alias K (GLM: aliased, so omitted);
     - pooled index values, plus pooled index scales when present.
  2. Zero padding.
  3. A 24-byte trailer: `magic "ATLKVNV1" | tag | checksum`.
- **Raw index tails are not stored.** They only hold the un-pooled keys and
  gates of a pool that is still being filled. `glm_index_kpool_finalize`
  pools each 4-token group when its last token is written, and only full
  blocks (all 4 pools finalised) are spilled.
  - Slotted tails (prefix caching + `ATLAS_MARCONI_PREFILL_ONLY`) change
    nothing in the record. A spilled block's lent tail returns with its last
    KV ref (`return_evicted_block`), and a restored block is published
    tail-less before the prefill reads it.
  - If the index moves to scaled FP8, the scales segment is picked up
    automatically.
- **GLM-5.3 sizes per rank:**
  - payload: 11 layers × (8448 + 1024) = 104 192 B;
  - record: **106 496 B per block** (26 × 4 KiB), which is **6.66 KB per
    token**;
  - 200 GiB ≈ 2.0 M blocks ≈ 32 M tokens.
- **Tag:** a hash of the node's causal `context_hash` and a spill epoch, held
  by the tree.
- **Checksum:** a 4-lane, 64-bit multiply-fold over the payload.
- **Host RAM:** one tree node per on-disk block — about **640 B** each, not
  the 150–250 B first estimated (§10 has the breakdown) — so a disk budget is
  also a host-memory budget: 24 GiB of records ≈ 148 MiB, 200 GiB ≈ 1.2 GiB.
  Plus the pinned staging: 3.4 MB (synchronous path) or 13.6 MB (fast path).

## 5. Spill, restore, and rank agreement

### Spill (at eviction)

1. `alloc_block_evicting` asks the tree for 32 blocks. With the tier off it
   asks for 1, as before.
2. The tree converts the LRU victims to on-disk nodes and returns
   `SpillOrder`s.
3. `apply_evicted_blocks` then:
   1. syncs the whole device (`cuCtxSynchronize`) — a victim's last writes
      can still be in flight on the prefill stream or another non-blocking
      stream;
   2. queues 22 async D2H copies per block into pinned staging;
   3. syncs once;
   4. stamps the trailers and `pwrite`s each record;
   5. **only then** returns the blocks to the free list.

A failed write triggers `spill_failed`, which drops that node, so the block
is evicted as it would be without the tier. Warnings are throttled (1, 2, 4,
8, …).

**Fast path (`ATLAS_GLM_NVME_FAST=1`).** Steps 1–2 are unchanged; step 3
becomes:

1. sync the whole device (as above);
2. sort the batch by block and gather each run of consecutive blocks with
   **one pitched copy per region** (22 copies for a 32-block run, not 704)
   into chunks of a pinned staging ring (8 chunks × 16 records);
3. sync once, hand the chunks to the worker pool, and **return the blocks to
   the free list**. Their bytes now belong to a queued write: the chunk is
   not reused until its write has finished.

A worker pads and checksums the chunk's records and writes each run of
consecutive slots with one request (`pwritev` when the run is back to front —
a chain spilled leaf first has ascending slots and descending blocks). A run
of ONE block (a fragmented pool) is moved with the synchronous path's plain
copies, one per region: a one-row pitched copy costs the same (36 µs per
block gathered, 73 µs scattered, either way — §9), so fragmentation degrades
the GPU side to the synchronous path's cost and no further.

The evicting thread waits when all 8 chunks are in flight. In a burst — a
25 K-token prefill or restore evicting 1,582 blocks — the ring fills after the
first 128, and from there the spill is throttled to the disk's write speed.
That wait is timed separately (`spill_wait_micros`) and logged. If the ring
ever has neither a free chunk nor a write to wait for (a leaked chunk; not
reachable today), the batch fails like a gather failure — those blocks are
evicted as without the tier — instead of spinning on the serving thread.

What keeps this safe:

- a restore drains every queued write before it reads;
- a spill that re-uses a slot whose earlier write is still queued drains
  first, so the later record is the one on disk;
- within one batch the last order per slot wins (the budget can re-issue a
  slot inside a batch; the earlier node is already gone);
- a write that fails is reported on the next spill or restore. Until then
  its node points at a record with the wrong tag, which fails verification —
  a recompute, never stale bytes;
- a failed RUN is reported as a whole, although its leading records may be
  on disk intact. A report for a node that is resident again is ignored: with
  kept records that node was restored from the very record in question,
  which therefore verified.

**Kept records (`ATLAS_GLM_NVME_KEEP=1`).** A restored block normally gives
its slot back. With this flag it keeps it: the node is resident AND has a
record, and evicting the block again is pure bookkeeping — no gather, no
write (`clean_evictions` in the stats). Kept records count against the
budget, so they are only kept while at most half of it is in use; an `insert`
that recomputes a block still releases its slot.

A cached full block is never appended to, but it can be **rewritten in
place**: a prefill with no usable snapshot recomputes its whole matched
prefix into the shared blocks, a full-prompt hit its last row, and a replay
the rows between its snapshot and its match wherever the KV write floor is
not honoured (`ATLAS_GLM_KV_WRITE_FLOOR_LEGACY=1`) — with equivalent, not
bit-identical, values (`model/prefix_share.rs`). A record kept for such a
block would then hold the older computation and come back on the next
restore — different bytes from what the synchronous path returns, and
possibly a different version on each rank. So the prefix lookup reports the
blocks between its resume point and its match (`nvme_forget_rewritten` →
`NvmePrefixTier::forget_kept`) and those give their records up; their next
eviction writes them again. It does not try to tell a replay that left the
rows alone from one that rewrote them: on an ordinary warm turn that is the
couple of blocks between the anchor and the match, written once more, while
everything below the anchor keeps its record.

### Restore (prefill_b, chunk 0, before `lookup`)

1. `plan_restore` walks the prompt's verified full blocks, finds the on-disk
   run below the resident prefix, and **pins** the whole path (one extra
   radix ref), so evictions caused by the restore cannot take it.
2. It decides whether to restore — the whole run, or nothing. The rule: the
   prefill must find exactly what it would have found had nothing been
   evicted, and must not page in what it is about to overwrite.
   - pure attention: the whole run;
   - hybrid (GLM), with a usable Marconi anchor, resident or spill-tiered
     (`snapshot_anchor_depth`, read-only): the whole run, **past the anchor
     too**. The pass replays `[anchor, matched)` under the KV write floor,
     which GLM honours since `integ/next`: those rows are read from the
     cache, not rewritten. A restore that stopped at the anchor (as on the
     old base, where the replay rewrote them anyway) would make the pass
     recompute rows that a resident cache serves as cached, in another chunk
     shape, and its output could differ bitwise from the same turn without
     an eviction. "Usable" mirrors what the lookup does with the anchor
     afterwards: at least `marconi_min_tokens` deep, at least
     `ATLAS_SSM_FAULT_MIN_TOKENS` deep when the snapshot tier is on, and not
     an anchor at the very end of the prompt (the exact-hit bypass recomputes
     everything);
   - hybrid, without one: nothing. That pass recomputes the prompt and
     rewrites every matched row — unless `ATLAS_GLM_PC_WRITE_FLOOR=1` keeps
     them as cached, in which case the whole run comes back as well;
   - a declined restore costs nothing: the pin is released, no block is
     allocated and nothing is read (the fast path's read would first wait
     for every queued write), and the run keeps its place in the disk LRU —
     a plan only refreshes the resident part of the path, the on-disk part
     when it is actually restored. A run nobody can use (a new conversation
     over a spilled shared system prompt, arriving again and again) is
     therefore still the budget's victim in its turn.
3. It allocates blocks (possibly spilling other conversations), then reads
   runs of consecutive slots with one `pread` each. A chain spilled leaf-first
   gets consecutive slots in descending order, and the reader handles either
   direction. It verifies every trailer, then scatters with `copy_h2d_async_retained`
   and one stream sync per 32 records.
   - **Fast path:** the target blocks are sorted first, so the restored chain
     sits on ascending blocks. Up to 8 chunks are read ahead by the workers
     (one request per slot run, checksums verified there), while the serving
     thread scatters each finished chunk with one pitched copy per region per
     run of consecutive blocks. The first record that fails stops the
     restore; read-ahead past it is discarded.
4. `complete_restore` promotes the verified prefix, drops the subtree of a
   record that failed, and unpins. Blocks that were not adopted are freed.
5. The ordinary `lookup` then matches the restored blocks. From here on the
   existing flow applies: F83 min-agreement, then Marconi lookup/fault-in and
   replay.

### Latency (per rank; both ranks run in parallel on their own NVMe)

The first estimates in this section were wrong by an order of magnitude —
they assumed 2–3 µs per small copy and 80–100 µs per synchronous write. §9
has what was measured and why. In short, per block, on the synchronous path:
about 220–250 µs for the record's own `pwrite`, 26 µs per checksum, and
22 small copies at several microseconds each in both directions.

| Context | Blocks | Bytes/rank | Synchronous path, pool full (25 K: pair run; larger: scaled at its 340–520 µs/block) | Fast path, read + scatter (projected) | Fast path, plus evicting as many unwritten victims (projected) |
|---|---|---|---|---|---|
| 25 K | 1 582 | 168 MB | 0.53–0.82 s | ~0.03–0.08 s | ~0.05–0.13 s |
| 128 K | 8 000 | 852 MB | ~2.7–4.2 s | ~0.15–0.4 s | ~0.25–0.65 s |
| 512 K | 32 000 | 3.4 GB | ~11–17 s | ~0.6–1.6 s | ~1.0–2.6 s |

The fast-path columns are projections from §9 (the disk pipeline at
10–48 µs/block across runs of the microbench, GPU scatter at 8–25 µs/block
in the isolated probes, the two overlapped; the low end needs a quiet
host, and both assume consecutive blocks and slots — the restore line's
blocks-per-run figures say how true that was). The log line reports the
parts (`evict … ms + read … ms (… MB/s)`, the writer waits inside `evict`,
and the flush before the read), which is what the pair run has to confirm.
With `ATLAS_GLM_NVME_KEEP=1`, victims that were themselves restored cost
nothing, and the last column collapses toward the one before it.

Add to either path the KDA snapshot fault-in (one 78 MB blob) and the replay
from the anchor, which are existing behaviour: on the pair they are the
remaining ~1.0 s of the 1.53–1.86 s turn-2 start.

**Spill cost on the evicting request.** Synchronous path: 250–300 µs per
evicted block on the serving thread in the pipeline microbench, before any
GPU copy cost — 0.4–0.5 s of an 11 s, 25 K-token prefill, which is
consistent with the turn-1 TTFT rise of up to 5% seen on the pair. Fast
path, projected: the gather (one device sync and 22 pitched copies per
32-block batch, about 10–30 µs per block in the isolated probes) plus, once
the 8-chunk ring is full, the wait for the writer — up to about 41 µs per
block at the 2.6 GB/s measured for 8 concurrent single-record writers, less
for run-sized writes. That is 0.02–0.11 s for the same prefill.

### Rank-agreement guarantees

- **KV:** each rank restores on its own, and the existing F83 `ep_min_u32` on
  `matched_tokens` then caps every rank to the shortest match.
  - A rank that restored more keeps the extra blocks as ordinary cache
    entries. They are released and re-looked-up, exactly as in F83 today.
  - A restore issues no collectives, so its partial failure on one rank cannot
    deadlock.
- **Marconi:** with either spill tier on (the KV tier, or the snapshot tier
  alone), a multi-rank world runs the restore-depth
  agreement of `prefill_b/pc_policy.rs` (`pc_agree_restore`, the one
  `ATLAS_GLM_PC_EVICT` / `ATLAS_GLM_PC_BRANCH` already turn on) once per
  chunk-0 lookup: one `ep_min_u32` over each rank's restorable depth (0 if
  none), and a second when that minimum is non-zero. Every rank restores at
  the minimum depth if every rank holds an exact-prefix snapshot there.
  Otherwise all ranks take the no-restore path together: a recompute, never
  a divergence.
  - With both tiers and both policy flags off this is a no-op, so there are
    no new collectives on the default path.
- **Configuration:** at startup, every multi-rank world's KV-block all-gather
  carries `(KV slots, record size, the snapshot tier's spill home, fast,
  keep)` as a 32-bit word (or a failure sentinel) beside the block count, and
  aborts on a mismatch or a failed rank, because the tiers gate paired
  collectives. No collective is added: with both tiers off the word is 0 and
  the gathered bytes are the block count alone. The snapshot tier on its own
  sends a non-zero word too, so a rank without `ATLAS_SSM_TIER` cannot start
  beside one with it.

## 6. Failure handling (never serve stale data)

| Event | Behaviour |
|---|---|
| Disk full / `pwrite` error at spill | The node is dropped and the block evicted as without the tier. Throttled warning. Later hits on that prefix recompute. On the fast path the drop happens at the next spill or restore; the record fails verification in between. |
| Disk cannot hold the budget (fast path) | Startup error naming `ATLAS_KV_NVME_GB`: the whole budget is reserved (`fallocate`) before serving, so a spill cannot run out of disk later. Every rank stops. |
| Directory on tmpfs / ramfs / overlayfs, or not writable | Startup error naming the variable (KV directory, and the snapshot tier's swap directory when it is configured) on that rank; its peers stop through the EP peer lifeline. |
| Snapshot tier's spills in host RAM (legacy store, unusable swap directory, non-4 KiB blob) beside the KV tier | Startup error on that rank, sent as the failure sentinel: every rank stops in the KV-block all-gather. |
| `ATLAS_EP_PEER_LIFELINE=0` on a multi-rank world with the KV tier | Startup error, every rank (the row above this pair depends on the lifeline). |
| A worker thread panics (fast path) | Caught; the job is reported as failed (a write: those nodes are dropped; a read: the restore stops there). The serving thread never waits on a job that cannot finish. |
| Staging ring with no free chunk and no write in flight (fast path; a leak) | The spill batch fails (plain eviction) with a warning; a restore attempts nothing and the prefix recomputes. No wait. |
| A write failure reported after its node was restored (kept records) | Ignored for that node: it was restored from that record, which verified. The failure is still counted. |
| Budget full | The coldest droppable on-disk leaf is dropped (LRU). If the eviction candidate is colder than every on-disk block, the candidate itself is deleted instead. |
| `pread` error / short read at restore | The verified prefix is kept. The failing record's node and its on-disk subtree are dropped. The rest is recomputed. |
| Wrong tag, bad magic, or checksum mismatch | Treated the same as a read error. The bytes are never scattered into a block the tree adopts. |
| No free blocks to restore into | Partial restore. The remainder is recomputed. |
| KDA snapshot fault-in miss or error | Existing behaviour (reap on miss, retain on error), and with rank agreement it is always a symmetric recompute. |
| Process crash | The record file is already unlinked, so its space is reclaimed. No state survives a restart; the cache starts empty. |

## 7. Not done / unverified

- **The fast path and kept records have not served a request.** Verified so
  far: unit tests on both I/O paths (byte-identical records either way), the
  Linux `O_DIRECT` / `preadv` / `pwritev` / `fallocate` arms and the real
  pipeline on a GB10's NVMe, and the copy shapes on a GB10 GPU in isolation —
  including that `cudaMemcpy2DAsync` with the record as host pitch is
  byte-exact in both directions for runs of 1 to 511 blocks (§9).
  Not verified: those copies between the pinned ring and the KV pools inside
  the server, the ring under a real eviction burst, a fragmented pool (the
  A/B below runs on a fresh server, where blocks and slots are consecutive),
  and every end-to-end number.
- **Restore only runs in `prefill_b`** (chunked prefill), the multi-rank path.
  `prefill_a`/`prefill_c` and the single-rank batched reservation path stay
  resident-only: correct, just no restore. On a single rank, requests that
  arrive together are admitted through that batched path: a conversation
  whose prefix is on disk is then admitted cold and recomputed, and the
  batch pre-flight (`peek_matched_tokens`, resident-only) counts its blocks
  as needed. The tier serves sequential arrivals there. A TP2/EP world always
  declines the batched reservation and is not affected.
- **A spill runs on the serving thread under a device-wide sync**, in batches
  of 32 blocks, wherever a block allocation misses — inside a decode or
  verify step too, and in the head's scheduler reclaim. The A/B gates the
  decode rate and the longest stall of one stream while another request
  evicts (§8), not every such path.
- **Assumptions to watch on hardware:**
  - all ranks build through `build_model` (as `agree_kv_blocks` already
    requires);
  - GLM sets `uses_local_mla_prefill() == false`, so the prefix skip is taken;
  - tails of full blocks are never read;
  - `cuMemAllocHost` staging is page-aligned (a warning at attach says so if
    it is not; I/O then bounces — correct, slower).
- **Worth doing next:**
  - the tree's per-block footprint (§10): ~640 B per on-disk block makes host
    RAM, not disk, the limit on the budget. An on-disk node needs neither its
    own `children` table nor a second copy of its key;
  - the ~230 µs every blocking request costs on this machine (§9): with
    requests this large it no longer limits a contiguous restore, but a
    fragmented one is bound by it. A PM-QoS latency request held for the
    duration of a restore, or an io_uring ring polled from the waiting
    thread, would remove it;
  - spilling ahead of demand when the server is idle, so a restore finds
    free blocks instead of evicting;
  - the decode ring's swap file (`atlas-decode-ring.<pid>.swap`) still leaks
    (the other half of unified.rs defect #3; GLM does not use it).
- **Not persisted across restarts, by design.**

## 8. Hardware validation plan

1. **Baseline, cache off:** N = 24 conversations with fixed seeds (distinct
   system prompts; 8–64 K tokens each, total well past GPU KV capacity; see
   the capacity line `KV cache: … blocks`). Run 2 turns each, temperature 0,
   and record each turn-2 answer and TTFT.
2. **Cache on, tiers off (control):** the same run. Revisiting a conversation
   after the others have filled the cache should log `cached_tokens=0` or no
   snapshot, and TTFT should look like a full prefill.
3. **Cache on, flags from §3:**
   - Fill: run turn 1 of all N conversations sequentially. Expect the
     `prefix cache NVMe spill tier ON` line at startup and `SSM tier disk tier
     ENGAGED` once the snapshot hot arena is full. The KV tier's spill count
     shows up in the stats on each restore line.
   - Revisit: run turn 2 of each conversation, oldest first. Expect, per
     request on **both** ranks:
     - `NVMe prefix restore: k/k blocks … in X ms`, with X under 1 s at 128 K;
     - `SSM tier fault-in: restored spilled snapshot at token D`;
     - a Marconi intermediate hit that replays only a few dozen tokens;
     - usage `cached_tokens > 0`;
     - **TTFT ≈ restore time + replay**, not a full prefill;
     - no `pc rank-agree:` lines (a rank restoring at a depth other than its
       own, or not at all).
   - Correctness: every turn-2 answer is **byte-identical** to the cache-off
     baseline at temperature 0. Also run the golden BFCL leg with the flags on
     and compare against the flags-off score.
4. **Failure drills:**
   - Tiny budget (`ATLAS_KV_NVME_GB=1`): expect disk drops, and still correct
     answers.
   - Unwritable dir: expect a startup error.
   - Env on one rank only: expect a startup abort naming the mismatch.
   - Fill the filesystem mid-run: expect throttled spill warnings, answers
     still correct, and no hang.
5. **Soak:** 2 h of mixed agentic traffic, checking that no free-block leak
   appears (`num_free_blocks` returns to baseline when idle), that RSS is
   bounded, and that there are no deadlocks.
6. **A/B on the production profile** (one server start per arm, each under
   five minutes; environment on both ranks, on top of the production profile):
   - `prod`: `ATLAS_KV_MAX_BLOCKS=7500` — the control: a pool that 8
     conversations of 25 K tokens overflow, no tier;
   - `warm`: nothing — the production profile itself. Nothing is evicted,
     every warm turn resumes from resident blocks: the byte reference;
   - `sync`: `ATLAS_KV_MAX_BLOCKS=7500` plus the flags from §3;
   - `fast`: `sync` plus `ATLAS_GLM_NVME_FAST=1`;
   - `fastkeep`: `sync` plus `ATLAS_GLM_NVME_FAST=1 ATLAS_GLM_NVME_KEEP=1`;
   - `fault`: the `fast` environment, with 20 conversations of 10 K tokens —
     more than the 16 snapshot slots, so warm turns must fault their anchor
     back from the snapshot tier.

   Four verdicts, all needed:
   - `compare prod sync` — does the tier work: exact answers, one complete
     restore per warm request on both ranks, warm turns at most 0.35× the
     control's, the evicting turn-1 prefills at most 10% slower than the
     others AND the median turn 1 at most 10% slower than the control's (a
     cost on every prefill cancels in the first), each rank's KV pool smaller
     than the control's by about the reserve line and the sizing line's own
     reserve term larger by it, the snapshot tier's startup line naming an
     O_DIRECT swap file on both ranks, a decoding stream keeping at least
     0.8× the control's rate (and stalling at most 0.5 s longer) while
     another request prefills and evicts, that request's TTFT within 1.15×,
     `MemAvailable` falling by no more than the control's fall plus the
     reserve plus 512 MiB, and the two containers differing in the tier
     variables only;
   - `compare sync fast`, `compare sync fastkeep` — the fast arms against
     `sync` (below);
   - `lossless warm <tier arm>` — every sequential output of the tier arm
     (turns 1–3, 20 requests) is byte-identical to the resident arm's. A
     restore brings the whole evicted run back, so the tier arm's prefill
     finds what the resident arm's finds — same match, same anchor, same
     cached rows — and this is the one check that sees a record that came
     back wrong in a way its checksum cannot (the checksum is taken after the gather and verified before the
     scatter). Turn 1, cold in both arms, is the control: if it already
     differs the pair is not deterministic from start to start and the
     verdict is INCONCLUSIVE, not NO-GO. A not-streamed turn 4 compares the
     generated tokens' logprobs (within 0.05) when the server returns them;
   - `faultin fault` — on both ranks: at least one `SSM tier fault-in:
     restored spilled snapshot` line and the `disk tier ENGAGED` line, none
     failed, no `no SSM snapshot`, no `pc rank-agree:` line, the same number
     of fault-ins on both ranks, every warm turn matching its prompt and
     starting within 0.5× its cold TTFT, `MemAvailable` within the reserve.

   The main plan is 27 salted requests (about 3 minutes; the control about 4):
   8 cold conversations, turn 2 of each (first restore), turn 3 of
   conversations 0–3 (second restore — with `fastkeep`, from records that
   survived a restore and a write-free eviction), turn 3 of conversations 4
   and 5 sent together, a long count decoded while a ninth cold conversation
   prefills, and turn 4 of conversations 0–1 with logprobs (third restore).
   Answers are planted text (three codes and ten ledger entries per
   conversation), checked exactly.

   Not covered by any arm: a restore that succeeds on one rank only (the
   agreement path is only ever seen not to fire), the disk budget filling
   (24 GiB is far more than five minutes of spills), and the single-rank
   batched path.

   The restore line, on both ranks (one line in the log; the numbers are an
   illustration, not a measurement):

   ```text
   NVMe prefix restore: 1582/1582 blocks (25312 tokens after 0 resident) in 96.4 ms;
     tier 5204/403298 slots, 17871 spills, 12659 restores, 0 failures;
     evict 52.0 ms + read 33.3 ms (5059 MB/s);
     spill path (fast): 17871 blocks in 702.7 ms on the serving thread,
     0 evictions without a write;
     this restore: writer wait 31.0 ms in evict, flush 9.8 ms before read,
     31.6 blocks/scatter run, 1582 spilled at 30.4 blocks/gather run;
     spill path waited 310.2 ms for the writer
   ```

   - `in X ms` is the whole restore;
   - `evict` is allocating the target blocks, which on a full pool is
     spilling that many victims: the gather plus `writer wait`;
   - `flush` is the wait for those victims' queued writes before the first
     read;
   - `read` is read + verify + scatter only, and carries the MB/s;
   - `blocks/… run` is how contiguous the blocks were (1.0 = one copy per
     region per block, the synchronous path's cost);
   - `spill path` is the cumulative time `nvme_write` has cost the serving
     thread since startup, and `spill path waited` the part of it spent
     waiting for the writer.

   GO for a fast arm needs, against the `sync` arm and on **both** ranks:
   every answer exact; every warm request matching its whole turn-1 prompt
   (`cached_tokens`) and starting within 0.3× its own cold TTFT; exactly one
   complete restore per warm request, no failure, no F83 cap, no `no SSM
   snapshot`, no rank disagreement; restore time ≤ 95 µs per block and at
   most a third of `sync`'s; read + verify + scatter ≥ 1.5 GB/s and at least
   twice `sync`'s; gather
   ≤ 80 µs per evicted block and the whole serving-thread spill cost at most
   a third of `sync`'s; the evicting turn-1 prefills no slower than `sync`'s;
   turn-2 TTFT at least 0.25 s lower; and the decode-under-eviction, memory
   and profile checks above, against `sync`. With `fastkeep`, additionally at
   least 1,000 `evictions without a write` before the second restores.

## 9. Where the time went

**How far each statement below goes.** Three kinds of evidence, not to be
mixed up:

- *Pair run* (2026-09-29, `integ/nvme` b5ce84c1, synchronous path, TP2 on
  ennspark01/02, pool capped with `ATLAS_KV_MAX_BLOCKS=7500`,
  `ATLAS_KV_NVME_GB=40`, 8 conversations of ~25 K tokens, 2 turns,
  temperature 0, thinking off; one run): the end-to-end numbers.
- *Isolated probes* on one other GB10 (ennspark03, Samsung MZALC4T0HBL1,
  ext4, kernel 7.0.0-1019; small C / CUDA programs, mostly single-shot; the
  host was serving other models throughout and its GPU read 96% utilisation,
  so the GPU figures are ratios within one probe, not absolutes).
- *Pipeline microbench* (`nvme_disk_bench`, release build in the CI sandbox
  container on ennspark03 with 10 CPUs; six runs — three at 680bb633 and
  three of the same pipeline just before it — ranges shown; the host's other
  load moves a run by 2–4×, which is why they are ranges): the real
  spill/restore code on a real `O_DIRECT` file with the mock GPU.

Nothing here is an end-to-end measurement of the fast path. Every fast-path
figure for the pair is a projection until the A/B of §8 has run.

The pair measurement: 1,582 blocks (25,312 tokens, 168 MB per rank) restored
in 534–824 ms, about 300 MB/s. The spill counter in the same log lines
advances by 1,567–1,598 between restores: the pool was full, so **every
restore also spilled about as many victims as it restored, synchronously,
inside its own allocation loop**. The 534–824 ms is both directions.

Isolated probes:

**Disk, `O_DIRECT`, one thread.** A blocking request costs ~210–250 µs
whatever its size, up to about 1 MB:

| Request | Read | Write (overwrite) |
|---|---|---|
| 4 KiB | 235 µs | 278 µs |
| 104 KiB (one record) | 212 µs — 0.50 GB/s | 256 µs — 0.42 GB/s |
| 1 MiB | 234 µs — 4.5 GB/s | 174 µs — 6.0 GB/s |
| 3.4 MB (32 records) | 565 µs — 6.0 GB/s | 542 µs — 6.3 GB/s |

A 4 KiB read taking as long as a 1 MiB one says the fixed cost is not data
transfer. It equals the exit latency the kernel reports for the CPU idle
state `LPI-2` (231 µs), which is a candidate cause, not a demonstrated one.
Either way the consequence is the same: **one request per record caps a
thread at about 0.45 GB/s; the levers are bigger requests and more of them
in flight.**

**Disk, concurrent writers** (8 threads, one record per request): 0.36 GB/s
into a growing file, 2.6 GB/s into reserved (`fallocate`) extents, 2.9 GB/s
into written ones. ext4 takes the inode lock exclusively for a direct write
that has to allocate, so writers into a growing file queue up. Hence the
up-front reservation on the fast path.

**Checksum:** 26 µs per record on one core (4.1 GB/s) — a quarter of a
2 GB/s budget if it stays on the serving thread, in each direction.

**GPU copies** (11 layers × 2 regions = 22 regions per block):

| Shape | Host → device | Device → host |
|---|---|---|
| 22 `cuMemcpyAsync` per block (synchronous path) | 160 µs/block — 0.67 GB/s | 118 µs/block — 0.91 GB/s |
| 22 pitched copies per run of 8–512 blocks | 8–25 µs/block — 4–13 GB/s | 10–31 µs/block — 3.4–11 GB/s |

One batch of 704 small copies takes 1.0 ms to enqueue and 5.0 ms to drain:
about 7 µs of copy-engine time per copy, regardless of its 1–8 KiB size.
Alternatives measured and dropped: one large copy into a device scratch
followed by device-side pitched copies (63 µs/block) and a scatter kernel
reading the pinned ring through its device alias (77 µs/block). A
region-major record layout would reach 3–8 µs/block (20–40 GB/s) but changes
the record format for no end-to-end gain at these sizes.

**Adding it up for the synchronous path,** per block restored with one victim
spilled: 222–256 µs (`pwrite`) + 2 × 26 µs (checksums) + 16 µs (`pread` in
32-record runs) + the small copies in both directions. Disk and checksums
alone are ~290–325 µs, i.e. 460–515 ms for 1,582 + 1,582 blocks — and the
pipeline microbench below, which has no GPU in it, lands at 484–535 ms for
that scenario on a different node. That is consistent with the pair's
534–824 ms being mostly one `pwrite` per record; it has not been confirmed
by timing the phases on the pair itself (the new log line does that).

The turn-1 TTFT regression fits the same arithmetic: a 25 K-token prefill on
a full pool evicts ~1,582 blocks at 250–350 µs each on the serving thread,
0.4–0.55 s of an ~11 s prefill, against the observed rise of up to 5%.

**Pipeline microbench, both paths** (4 conversations of 1,582 blocks; the
raw output of each run is kept beside the probe sources):

| Case | Synchronous path | Fast path |
|---|---|---|
| Spill, fresh tier: serving-thread time per block | 253–301 µs (0.35–0.42 GB/s) | 9–23 µs (4.7–12 GB/s) |
| Restore one conversation, contiguous slots | 54–103 ms (1.6–3.1 GB/s) | 16–76 ms (2.2–10.3 GB/s) |
| Restore + evict 1,582 victims (the pair scenario) | 478–594 ms | 35–92 ms |
| Spill, recycled slots in scattered order, per block | 250–344 µs | 35–44 µs |
| Restore one conversation, scattered slots | 431–687 ms (0.25–0.39 GB/s) | 77–90 ms (1.9–2.2 GB/s) |

Run it on an NVMe-backed checkout with
`cargo test --release -p spark-runtime --lib nvme_disk_bench -- --ignored --nocapture`.
It does not measure the GPU copies, and the mock's host `memcpy` stands in
for them on both paths. On the pair, the restore line is the judge.

**Pitched copies on the GPU** (isolated probe on the same GB10, record layout
of GLM-5.3, three runs):

- *Byte-exact.* Runs of 1, 2, 3, 16, 31, 32, 128 and 511 consecutive blocks
  were gathered with one pitched copy per region into record-pitched staging,
  scattered to other blocks the same way and read back with plain copies:
  154 MB compared per run of the probe, 0 bytes wrong, the record padding and
  the neighbouring blocks untouched.
- *A run of one block.* 512 scattered blocks, 22 copies each: 36 µs per
  block gathered and 73 µs scattered, the same whether each copy is a
  one-row pitched copy or a plain async copy (best of 7 identical to 0.1 ms
  in all three runs). The fast path uses the plain copy there. A fully
  fragmented pool therefore costs the fast path what it costs the
  synchronous one on the GPU side — about 110 µs per block restored and
  evicted — and the blocks-per-run figures in the restore line say how far
  from that a given restore was.

## 10. Memory the tier takes (per rank)

GB10 memory is one pool: host RAM, pinned host memory and device memory all
come out of the same 121.7 GiB, and the hosts run with about 2 GB free. The
tier allocates **nothing on the device**. On the host, for GLM-5.3 with the
flags of §11:

| Allocation | Kind | Size | Committed | Accounted |
|---|---|---|---|---|
| KV staging | pinned | fast: 8 chunks × 16 records × 106,496 B = **13.6 MB**; synchronous: 32 records = 3.4 MB. Fixed. | at attach, after KV sizing | reserve |
| Tree nodes + slot index of on-disk blocks | heap | **~640 B per on-disk block**: 24 GiB of records → 242 K blocks → **148 MiB**; 40 GiB → 246 MiB; 200 GiB → 1.2 GiB | as blocks spill | reserve, at a full budget |
| I/O workers (fast path) | 8 thread stacks | a few tens of KiB touched each | at attach | not reserved |
| Snapshot tier hot arena | heap, zeroed `Vec` | `ATLAS_SSM_TIER_SLOTS` × 78.0 MB = **156 MB** at 2 slots | page by page, as slots fill | reserve |
| Snapshot spill / fault-in staging | pinned | one blob = **78 MB** | at the first spill | reserve |
| Snapshot residency scratch | heap, zeroed, page-aligned | one blob = 78 MB | at construction, before KV sizing | positional (already in `used_so_far`) |
| Snapshot swap file's bounce buffer | heap | one blob of address space, never touched (callers are page-aligned) | never | — |

Where the 640 B goes, per on-disk block: the `RadixNode` (152 B, in a `Vec`
that doubles — 1.5× on average), its `parent_key` (16 tokens × 4 B plus the
allocator header), its entry in the parent's `children` map (a 4-bucket
table of 32 B buckets, plus the 64 B key), and the slot index (owner, tag,
disk-LRU entry). A unit test pins the constant to the struct size.

**KV sizing.** `build_model` sizes the KV pool from what is free at that
point, so anything committed later is invisible to it. With
`ATLAS_KV_NVME_DIR` set it now reserves the rows marked "reserve" out of the
KV budget (`kv_nvme::host_reserve_bytes`; the startup line is `NVMe spill
tiers: reserving … MiB of host memory out of the KV budget`). At the defaults
of §11 that is 13.6 + 155 + 234 ≈ **400 MB**, about 3,800 of the ~101,000
blocks the pool has at 0.93 utilisation (3.8%). Without the KV tier nothing
changes.

The reserve is a ceiling: it assumes the disk budget fills. The way to give
less of the pool away is a smaller `ATLAS_KV_NVME_GB` (6.0 MiB of host RAM per
GiB of records) or `ATLAS_SSM_TIER_SLOTS=1` (78 MB).

## 11. What a SparkGLM install needs

**A directory on each node's NVMe, bind-mounted into that node's rank
container.** For example host `/var/lib/sparkglm/nvme` → container `/nvme`,
mode 0700, owned by the container user. It must be a real disk filesystem
(ext4/xfs): startup refuses tmpfs, ramfs and overlayfs by name, and probes
that it can create an `O_DIRECT` file there. Atlas creates the `kv` and `ssm`
subdirectories itself. The two ranks do not share storage; each node uses
its own disk.

**One server per directory.** Every process opens its record file once,
exclusively and owner-only, and the startup sweep only unlinks names (a live
owner keeps its descriptor), so a second process in the same KV directory
cannot read or clobber another's records. It is still not a supported
layout: each process reserves its own `ATLAS_KV_NVME_GB`, and the snapshot
tier's file name carries the PID only — PID 1 in every container — so two
servers started at the same instant in one `ATLAS_SSM_TIER_SWAP_DIR` could
open the same file.

**Profile environment, identical on both ranks** (startup all-gathers a
fingerprint and refuses to start on a mismatch):

```sh
ATLAS_KV_NVME_DIR=/nvme/kv
ATLAS_KV_NVME_GB=24
ATLAS_GLM_NVME_FAST=1
ATLAS_GLM_NVME_KEEP=1
ATLAS_SSM_TIER=1
ATLAS_SSM_TIER_UNIFIED=1
ATLAS_SSM_TIER_SWAP_DIR=/nvme/ssm
ATLAS_SSM_TIER_DISK_GB=24
ATLAS_SSM_TIER_SLOTS=2
# unchanged: --enable-prefix-caching --ssm-cache-slots=16,
#            ATLAS_MARCONI_PREFILL_ONLY=1, ATLAS_PREFIX_SUBBLOCK=0
```

**Default disk budget: 24 GiB of KV records and 24 GiB of snapshots per
node.**

- 24 GiB of records is 242 K blocks, 3.9 M tokens — 4.8 times the GPU pool's
  ~810 K. It is host RAM, not disk, that sets this: every GiB of records
  costs 6 MiB of index that comes out of the KV pool (§10). 24 GiB costs
  148 MiB; the 200 GiB of the first test would cost 1.2 GiB.
- 24 GiB of snapshots is 330 blobs of 78 MB. A restore needs one anchor at
  or below the restored depth; without it the KV is recomputed anyway.
- The KV file's 24 GiB are **reserved at startup** on the fast path. The
  snapshot file grows on demand up to its cap. The installer should require
  60 GiB free on the mount (both budgets plus margin).

**Stale files.** None to manage. Both files are unlinked as soon as they are
open, so a crash leaves nothing and `ls` shows an empty directory (`df` shows
the space). At startup each tier also removes leftovers carrying its own name
(`atlas-kv-prefix.*.swap`, `atlas-ssm-marconi-host.*.swap`) — a process
killed between create and unlink, or a build from before the unlink. Nothing
is reused across restarts: the cache starts empty.

**Missing or unusable directory.**

| Situation | Behaviour |
|---|---|
| Directory missing | Created. If the mount itself is missing, the path resolves into the container's overlay and startup refuses it by name. |
| Not writable, read-only mount, no `O_DIRECT` | Startup error naming the variable. Every rank stops. |
| tmpfs / ramfs | Startup error: a "disk" tier there would spend the memory it exists to save. |
| Snapshot swap directory unusable, `ATLAS_SSM_TIER_UNIFIED` missing, or `ATLAS_SSM_TIER_SWAP_DIR` missing | Startup error on every rank when the KV tier is on: the spilled snapshots would stay in host RAM. (On its own, the snapshot tier still falls back to host-RAM records, bounded by `ATLAS_SSM_TIER_DISK_GB` at best — fatal on a GB10.) |
| Disk too small for `ATLAS_KV_NVME_GB` | Startup error from the reservation (fast path). |
| Disk fills later | Cannot happen to the KV file (reserved). A snapshot that cannot be written is dropped; that prefix recomputes. |
| Budget full | The coldest on-disk blocks are dropped; nothing fails. |
| I/O error or corrupt record at restore | That record and everything below it are forgotten; the prefix recomputes from there. |

**What the operator sees.** One startup line per rank (`prefix cache NVMe
spill tier ON (rank r): …, budget 24.0 GiB = 241979 blocks = 3871664 tokens;
fast I/O, restored records kept, 13.0 MiB pinned staging, up to 147.7 MiB
host index`), the reserve line of §10, and one `NVMe prefix restore:` line per
restored request. Warnings are throttled: a failing disk logs at 1, 2, 4, 8, …
failed spills.

**Wear.** Each spilled token writes 6.66 KB per rank. Without
`ATLAS_GLM_NVME_KEEP`, a conversation that is restored and evicted on every
turn is rewritten in full every turn (3.4 GB per turn at 512 K tokens); with
it, a block is written once for as long as its record is kept.

## 12. What the port onto `integ/next` changed

- **Whole blocks only.** `integ/next` removed the tree's sub-block
  (`partial_suffix`) entries, so the tier's handling of them is gone: every
  cached block is a full block of committed tokens with all four index pools
  finalised, which is exactly what a record holds.
- **No startup collective of its own.** The rank fingerprint rides the
  KV-block all-gather (§3, §5) instead of a second all-gather that also ran
  with the tier off.
- **One Marconi agreement.** The tier's own two-round agreement is replaced by
  `pc_agree_restore`, which `integ/next` already runs under the prefix-cache
  policy flags and which can also settle on a shallower common depth instead
  of recomputing. As on the branch it runs with either spill tier on — the KV
  tier or `ATLAS_SSM_TIER` alone — and the snapshot tier alone is part of the
  startup word, so its ranks cannot differ.
- **Nothing of the tier is read without `ATLAS_KV_NVME_DIR`.** On the branch
  `ATLAS_KV_NVME_GB` without the directory was a startup error; it is a
  warning now, like the two switches, so a start without the directory is
  `integ/next`'s whatever else is set.
- **Rank-local setup after the agreement.** The branch exchanged a failure
  sentinel after each rank's local setup; here the setup runs after the one
  collective and a rank that fails in it stops its peer through the EP peer
  lifeline, which the tier therefore requires.
- **The snapshot tier must be on disk.** The host reserve counts the snapshot
  tier from what the store built (`SpillHome`), and a store that keeps its
  spills in host RAM is refused beside the KV tier.
- **Restore depth and the KV write floor.** On the old base a GLM replay
  rewrote `[anchor, matched)`, so the branch restored up to the anchor and no
  further. `integ/next` honours the KV write floor in the GLM paged prefill
  (those rows are read from the cache), so the port restores the whole
  on-disk run whenever the prefill will read its cached prefix (§5). Without
  this a turn after an eviction would recompute rows that the same turn
  without an eviction reads, and the two could differ bitwise.
- **KV admission.** A restore runs in the chunk-0 prefix lookup, before the
  chunk's KV reservation is voted on (`model/kv_admission.rs`). It is
  rank-local and best effort: a rank that cannot allocate restores less, the
  F83 minimum caps every rank to the shortest match, and the vote that follows
  admits or refuses the chunk on every rank as before. A refused chunk's retry
  replays the lookup decision and does not restore again.
- **KV sizing.** The tier's host reserve is added to the reserve that
  `factory/build/kv_budget.rs` takes out of both the utilization budget and
  the free-memory floor.
- **No file moves.** `block_mgmt.rs` and `kv_cache/paged_impl.rs` keep their
  `integ/next` layout; the tier's changes are applied in place.
- **Rewritten shared blocks and kept records.** A prefill that recomputes or
  replays under its matched prefix rewrites those rows with equivalent, not
  bit-identical, values (`model/prefix_share.rs`). With
  `ATLAS_GLM_NVME_KEEP=1` a block restored and then rewritten that way would
  be evicted without a write, its record still holding the earlier
  computation. The prefix lookup now releases the kept records of the blocks
  it is about to recompute (§5), so what a restore returns is what the block
  held, with or without KEEP.

## 13. Under the latent shard (`ATLAS_GLM_KV_SHARD=1`)

The shard (`glm-kv-shard.md`) stores block `b`'s latents on rank `b % 2`
only, at local slot `b / 2`, and draws the block for logical index `l` with
`b % 2 == l % 2`, so both ranks agree which rank owns logical block `l` while
their physical ids differ. The index pools stay full on both ranks. The tier
used to be refused beside it; it now runs with it, as follows.

- **Two record classes.** The tree splits its slots into two classes
  (`NvmePrefixTier::enable_classes`): a node spilled from block `b` takes a
  slot `s` with `s % 2 == b % 2`. That residue is the node's logical index
  residue, so the class of every node, and every slot the tree hands out,
  is the same on both ranks (the trees see the same operations; only the ids
  differ, and the classes do not read them beyond their residue).
- **One record file (lane) per class and rank**
  (`kv_cache/nvme_lanes.rs`), `atlas-kv-prefix.<pid>.r<rank>.c<class>.swap`,
  record `s / 2` of class `s % 2`:
  - the rank's OWN class (`class == rank`): the full record of §4, the
    latents gathered from the local slot;
  - the PEER class: the index rows only (GLM-5.3: 11 × 1024 B + trailer =
    12 288 B per block, against 106 496 B for a full record).
  Each rank thus spills and restores only the latents it stores, and every
  block's index rows (sparse-index values and scales; raw tails are never in
  a record, as unsharded).
- **Budget.** `ATLAS_KV_NVME_GB` buys `budget / (own + peer record)` slots of
  EACH class, so the two files together stay within it. On GLM-5.3 that is
  ~59 KB per block per rank instead of 106 KB: ~1.8× the tokens per GiB.
  The host reserve counts both lanes' staging and every slot of both classes;
  the rank word fingerprints the total slots and `own + peer`, so a shard rank
  and an unsharded rank do not start together (the shard settings themselves
  are agreed in the same word).
- **Budget drops per class.** A full class drops the coldest on-disk tail of
  ITS class: a leaf, or (shard only) a node of the class whose children are
  all leaves of the other class — a chain alternates classes, so this is what
  lets a full class trim a chain whose leaf is in the other class. A victim
  colder than every droppable record of its class is deleted (cold drop) even
  when the other class has room. The rules read only the classes and the
  access order, so both ranks drop the same records.
- **Restore placement.** `restore_prefix` already allocates each restored
  block at its logical index, so its residue is its record's class on both
  ranks. `nvme_read` splits the run by class, reads each lane, and returns the
  shortest verified prefix over both (a failure in either lane ends the
  restore there). The fast path sorts a lane's blocks for pitched copies;
  within a lane they share a residue, so every block keeps its position's
  residue. Pitched runs step two ids at a time: the latents at their local
  slots (pitch = one block), the index rows at a doubled pitch.
- **Rank agreement.** Unchanged and sufficient: a restore is rank-local, and
  the F83 minimum caps both ranks to the blocks both of them hold, so the
  merge-form attention never meets a logical block whose owner lacks its
  latents. The Marconi depth agreement (`pc_agree_restore`) and the SSM
  snapshot tier work as unsharded (snapshots are per-rank state the shard does
  not touch).
- **Checks.** A spill order whose block residue is not its slot's class is
  reported failed (the node is dropped); a restore target off its slot's
  class ends the restore in front of it. Neither can happen with blocks drawn
  at their logical index.
- **Shard off.** One class, one lane, `.r<rank>.swap`, the same slots, the
  same records, the same rank word and the same reserve as before.

Not supported under the shard: nothing beyond what the shard already refuses
(`--high-speed-swap`, and the lanes in `glm-kv-shard.md`).
