// SPDX-License-Identifier: AGPL-3.0-only

//! Token-sharded latent storage for a tensor-parallel pair
//! (`ATLAS_GLM_KV_SHARD=1`, GLM-5 NoPE MLA).
//!
//! Every physical block's latents live on exactly ONE rank: block `b` is
//! owned by rank `b % world` and sits at local slot `b / world` of that
//! rank's per-layer K pool (V aliases K). The ranks' physical ids differ
//! (each allocates and frees in its own order), but the allocator draws the
//! block for logical index `l` with `b % world == l % world`
//! (`free_blocks.rs`), so both ranks agree that logical block `l` is stored
//! by rank `l % world`. Every other per-block pool (semantic index, tails)
//! is kept in full on both ranks, so only the latent bytes shrink.
//! `kernels/gb10/glm-5.3-flash/nvfp4/glm_kv_shard.cu` implements the same
//! ownership rule on device; keep the two in step.

use anyhow::{Result, ensure};

use super::free_blocks::FreeBlocks;
use super::{KvCacheConfig, PagedKvCache};
use crate::gpu::{DevicePtr, GpuBackend};

/// Topology and scratch of a latent shard, fixed at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentShardSpec {
    /// This rank (0 or 1).
    pub rank: usize,
    /// Ranks sharing the latents (only a pair is supported).
    pub world: usize,
    /// Bytes of the one device scratch allocation the model carves for its
    /// exchange buffers (see `layers::glm_kv_shard`).
    pub scratch_bytes: usize,
    /// Model scratch geometry: logical blocks one assembled view holds.
    pub view_blocks: usize,
    /// Model scratch geometry: rows one cache write may carry.
    pub write_rows: usize,
    /// Create an [`ExchangeLane`] (the model overlaps exchanges with compute).
    pub lane: bool,
}

/// A side stream for pair exchanges that overlap compute, and the two
/// events that fence it against the compute stream (`begun` is recorded on
/// the compute stream, `landed` on the side stream).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExchangeLane {
    pub stream: u64,
    pub begun: u64,
    pub landed: u64,
}

impl ExchangeLane {
    fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        let mut lane = Self {
            stream: gpu.create_stream()?,
            ..Self::default()
        };
        let events = gpu.create_event().and_then(|begun| {
            lane.begun = begun;
            gpu.create_event()
        });
        match events {
            Ok(landed) => Ok(Self { landed, ..lane }),
            Err(error) => {
                let _ = lane.destroy(gpu);
                Err(error)
            }
        }
    }

    /// Destroy the events and the stream; the first failure, after trying all.
    pub(super) fn destroy(self, gpu: &dyn GpuBackend) -> Result<()> {
        let events = [self.begun, self.landed].map(|event| gpu.destroy_event(event));
        let stream = gpu.destroy_stream(self.stream);
        events.into_iter().chain([stream]).collect()
    }
}

/// A constructed latent shard (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentShard {
    pub spec: LatentShardSpec,
    /// K-pool block slots this rank allocated per layer.
    pub local_blocks: usize,
    /// Device scratch of `spec.scratch_bytes`, followed by `identity`.
    pub scratch: DevicePtr,
    /// `u32[i] = i` for every local slot and view block: the block table of
    /// the local pool addressed by local token ids, and of assembled views.
    pub identity: DevicePtr,
    /// Where the model runs exchanges beside the compute stream, when the
    /// spec asked for it.
    pub lane: Option<ExchangeLane>,
}

/// How one rank assembles a sequence's logical blocks `[0, n)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LatentViewPlan {
    /// Local slot of every block this rank owns, in logical order.
    pub mine_slot: Vec<u32>,
    /// Logical index of each of those blocks.
    pub mine_logical: Vec<u32>,
    /// Logical index of every block the peer owns, in logical order, which
    /// is the order the peer packs and sends them.
    pub peer_logical: Vec<u32>,
}

impl LatentShard {
    /// K-pool slots each rank allocates for a pool of `num_blocks`.
    pub fn local_blocks_for(num_blocks: usize, world: usize) -> usize {
        num_blocks.div_ceil(world)
    }

    /// Entries of the identity table for a pool of `num_blocks`.
    pub fn identity_entries(num_blocks: usize, spec: &LatentShardSpec) -> usize {
        Self::local_blocks_for(num_blocks, spec.world).max(spec.view_blocks)
    }

    /// Bytes of the one scratch allocation: the model scratch, then the
    /// 256-byte-aligned identity table.
    pub fn allocation_bytes(num_blocks: usize, spec: &LatentShardSpec) -> usize {
        spec.scratch_bytes.next_multiple_of(256) + 4 * Self::identity_entries(num_blocks, spec)
    }

    /// The rank storing physical block `block`.
    pub fn owner(&self, block: u32) -> usize {
        block as usize % self.spec.world
    }

    /// This rank's local slot of `block`, if it owns it.
    pub fn local_slot(&self, block: u32) -> Option<u32> {
        (self.owner(block) == self.spec.rank).then_some(block / self.spec.world as u32)
    }

    /// `(first local slot, count)` of the blocks this rank owns among the
    /// physical ids `[first, first + count)`; they occupy consecutive slots.
    pub fn owned_run(&self, first: u32, count: usize) -> (usize, usize) {
        let (world, rank) = (self.spec.world, self.spec.rank);
        let (first, end) = (first as usize, first as usize + count);
        let owned0 = first + (rank + world - first % world) % world;
        if owned0 >= end {
            return (owned0 / world, 0);
        }
        (owned0 / world, (end - owned0).div_ceil(world))
    }

    /// Check that every entry of a sequence's block table (logical order)
    /// is owned by the rank its logical index names: the invariant that lets
    /// the ranks' differing tables agree on ownership.
    pub fn check_table(&self, table: &[u32]) -> Result<()> {
        let world = self.spec.world;
        match table
            .iter()
            .enumerate()
            .find(|&(logical, &block)| self.owner(block) != logical % world)
        {
            None => Ok(()),
            Some((logical, block)) => anyhow::bail!(
                "latent shard: logical block {logical} of a {}-block table is physical \
                 block {block}, owned by rank {} instead of {} (allocated without its \
                 logical index?)",
                table.len(),
                self.owner(*block),
                logical % world
            ),
        }
    }

    /// Split a sequence's block table (logical order) into this rank's
    /// blocks and the peer's, after [`Self::check_table`].
    pub fn plan(&self, table: &[u32]) -> Result<LatentViewPlan> {
        self.check_table(table)?;
        let mut plan = LatentViewPlan::default();
        for (logical, &block) in table.iter().enumerate() {
            match self.local_slot(block) {
                Some(slot) => {
                    plan.mine_slot.push(slot);
                    plan.mine_logical.push(logical as u32);
                }
                None => plan.peer_logical.push(logical as u32),
            }
        }
        Ok(plan)
    }
}

impl PagedKvCache {
    /// [`Self::new_with_v_alias`] (V aliasing K) with the latents sharded
    /// over a pair per [`LatentShardSpec`]: each layer's K pool holds
    /// `ceil(num_blocks / 2)` block slots, plus one scratch allocation.
    pub fn new_latent_sharded(
        config: KvCacheConfig,
        num_blocks: usize,
        gpu: &dyn GpuBackend,
        spec: LatentShardSpec,
    ) -> Result<Self> {
        ensure!(
            spec.world == 2 && spec.rank < spec.world && num_blocks > 0,
            "latent shard supports a rank pair and a non-empty pool (rank {} of {}, {num_blocks} blocks)",
            spec.rank,
            spec.world
        );
        let local_blocks = LatentShard::local_blocks_for(num_blocks, spec.world);
        let scratch = gpu.alloc(LatentShard::allocation_bytes(num_blocks, &spec))?;
        let identity = scratch.offset(spec.scratch_bytes.next_multiple_of(256));
        let table: Vec<u8> = (0..LatentShard::identity_entries(num_blocks, &spec) as u32)
            .flat_map(u32::to_le_bytes)
            .collect();
        let built = gpu.copy_h2d(&table, identity).and_then(|()| {
            let lane = spec.lane.then(|| ExchangeLane::new(gpu)).transpose()?;
            match Self::new_with_k_slots(config, num_blocks, local_blocks, gpu, true) {
                Ok(cache) => Ok((lane, cache)),
                Err(error) => {
                    if let Some(lane) = lane {
                        let _ = lane.destroy(gpu);
                    }
                    Err(error)
                }
            }
        });
        let (lane, mut cache) = match built {
            Ok(built) => built,
            Err(error) => {
                let _ = gpu.free(scratch);
                return Err(error);
            }
        };
        tracing::info!(
            "KV latent shard: rank {} of {} stores {local_blocks} of {num_blocks} blocks' latents; \
             {:.1} MiB exchange scratch",
            spec.rank,
            spec.world,
            spec.scratch_bytes as f64 / (1024.0 * 1024.0)
        );
        cache.free_blocks = FreeBlocks::new(num_blocks, spec.world);
        cache.latent_shard = Some(LatentShard {
            spec,
            local_blocks,
            scratch,
            identity,
            lane,
        });
        Ok(cache)
    }

    /// The latent shard, when this cache stores only its rank's latents.
    pub fn latent_shard(&self) -> Option<LatentShard> {
        self.latent_shard
    }

    /// This rank's K (latent) pool of `layer_idx`: every block when
    /// unsharded, else only owned blocks at their local slots.
    pub fn latent_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].k_pool
    }

    /// Where this rank stores `block`'s latents, if it does.
    pub(super) fn latent_slot(&self, block: u32) -> Option<u32> {
        match &self.latent_shard {
            Some(shard) => shard.local_slot(block),
            None => Some(block),
        }
    }

    #[track_caller]
    pub(super) fn assert_unsharded(&self, what: &str) {
        assert!(
            self.latent_shard.is_none(),
            "{what}: this KV path addresses latents by global block id, but \
             ATLAS_GLM_KV_SHARD=1 stores only this rank's blocks (unsupported path)"
        );
    }

    pub(super) fn ensure_unsharded(&self, what: &str) -> Result<()> {
        ensure!(
            self.latent_shard.is_none(),
            "{what} is unsupported with ATLAS_GLM_KV_SHARD=1 (latents are sharded)"
        );
        Ok(())
    }
}
