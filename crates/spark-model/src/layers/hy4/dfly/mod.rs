// SPDX-License-Identifier: AGPL-3.0-only

//! Five-layer DFly query forward. The caller checkpoints target KV before entry.
//! Query KV writes are temporary aliases; target verification must overwrite or
//! restore them. Drafting never advances target cache lengths or DSA pools.

use super::attention::memory::Owned;
use anyhow::{Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    weights::WeightStore,
};

mod forward;
mod loader;
mod ops;
#[cfg(test)]
mod tests;

const H: usize = 3072;
const ROWS: usize = 6;
pub const TARGET_LAYERS: [usize; 5] = [25, 29, 33, 37, 41];

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub rank: usize,
    pub world: usize,
    pub vocab: usize,
    pub mask_token_id: u32,
    pub eps: f32,
    pub theta: f32,
}
impl Config {
    pub fn validate(self) -> Result<()> {
        ensure!(
            matches!(self.world, 1 | 2)
                && self.rank < self.world
                && self.vocab > 0
                && (self.mask_token_id as usize) < self.vocab
                && self.vocab <= u32::MAX as usize
                && self.vocab.is_multiple_of(self.world)
                && self.eps.is_finite()
                && self.eps > 0.
                && self.theta.is_finite()
                && self.theta > 0.,
            "DFly invalid TP/vocabulary/norm/rotary configuration"
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct TargetCache {
    pub latent: DevicePtr,
    pub indices: DevicePtr,
    pub len: usize,
    pub capacity: usize,
}
pub struct Input {
    pub target_hidden: DevicePtr,
    pub caches: [TargetCache; 5],
    pub bonus: u32,
    pub count: usize,
}
impl Input {
    fn query_ids(&self, c: Config) -> Vec<u32> {
        std::iter::once(self.bonus)
            .chain(std::iter::repeat_n(c.mask_token_id, self.count))
            .collect()
    }

    fn validate(&self, c: Config) -> Result<()> {
        ensure!(
            (1..=5).contains(&self.count),
            "DFly draft count must be 1..5"
        );
        ensure!(
            !self.target_hidden.is_null() && (self.bonus as usize) < c.vocab,
            "DFly hidden/token missing or invalid"
        );
        let anchor = self.caches[0].len;
        ensure!(
            anchor
                .checked_add(self.count)
                .is_some_and(|n| n <= i32::MAX as usize),
            "DFly anchor exceeds sparse index range"
        );
        for cache in &self.caches {
            ensure!(
                !cache.latent.is_null()
                    && !cache.indices.is_null()
                    && cache.len == anchor
                    && anchor
                        .checked_add(self.count + 1)
                        .is_some_and(|n| n <= cache.capacity),
                "DFly target cache missing, inconsistent or exhausted"
            );
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
pub struct Head {
    pub embedding: DevicePtr,
    pub weight: DevicePtr,
}

pub type EmbeddingLookup<'a> = dyn FnMut(&[u32], DevicePtr, u64) -> Result<()> + 'a;

struct Layer {
    input_norm: DevicePtr,
    post_norm: DevicePtr,
    qa: DevicePtr,
    qan: DevicePtr,
    qb: DevicePtr,
    kva: DevicePtr,
    kvn: DevicePtr,
    wk: DevicePtr,
    wv: DevicePtr,
    gate: DevicePtr,
    out: DevicePtr,
    sink: DevicePtr,
    attn_norm: DevicePtr,
    mg: DevicePtr,
    mu: DevicePtr,
    md: DevicePtr,
}
struct Fusion {
    en: DevicePtr,
    hn: DevicePtr,
    projection: DevicePtr,
    gate: DevicePtr,
}
struct Correction {
    en: DevicePtr,
    hn: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    down: DevicePtr,
}
struct Scratch {
    hidden: DevicePtr,
    residual: DevicePtr,
    norm: DevicePtr,
    qc: DevicePtr,
    kv: DevicePtr,
    q: DevicePtr,
    absorbed: DevicePtr,
    latent: DevicePtr,
    values: DevicePtr,
    gates: DevicePtr,
    partial: DevicePtr,
    mg: DevicePtr,
    mu: DevicePtr,
    ma: DevicePtr,
    concat: DevicePtr,
    cg: DevicePtr,
    cu: DevicePtr,
    ca: DevicePtr,
    delta: DevicePtr,
    corrected: DevicePtr,
    final_hidden: DevicePtr,
    embedding: DevicePtr,
    logits: DevicePtr,
    id: DevicePtr,
    ids: DevicePtr,
}
pub struct Dfly<'a> {
    config: Config,
    layers: Vec<Layer>,
    fusion: Fusion,
    correction: Correction,
    final_norm: DevicePtr,
    scratch: Scratch,
    ops: ops::Ops<'a>,
    _memory: Owned<'a>,
}
impl<'a> Dfly<'a> {
    /// Loads owned TP weights and reclaims consumed auxiliary source tensors.
    pub fn load(store: &WeightStore, gpu: &'a dyn GpuBackend, config: Config) -> Result<Self> {
        config.validate()?;
        loader::load(store, gpu, config)
    }
    pub fn draft(
        &mut self,
        input: &Input,
        head: Head,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<Vec<u32>> {
        ensure!(!head.embedding.is_null(), "DFly shared embedding missing");
        self.validate_draft(input, head.weight, comm)?;
        self.forward(
            input,
            head,
            &mut |tokens, output, stream| {
                let bytes: Vec<_> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
                self.ops
                    .gpu
                    .copy_h2d_async_retained(&bytes, self.scratch.ids, stream)?;
                self.ops.embed(
                    head.embedding,
                    self.scratch.ids,
                    output,
                    tokens.len(),
                    stream,
                )
            },
            comm,
            stream,
        )
    }

    /// The lookup writes the requested BF16 rows to output on the supplied stream.
    pub fn draft_with_embedding_lookup(
        &mut self,
        input: &Input,
        weight: DevicePtr,
        lookup: &mut EmbeddingLookup<'_>,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.validate_draft(input, weight, comm)?;
        self.forward(
            input,
            Head {
                embedding: DevicePtr::NULL,
                weight,
            },
            lookup,
            comm,
            stream,
        )
    }

    fn validate_draft(
        &self,
        input: &Input,
        weight: DevicePtr,
        comm: Option<&dyn CommBackend>,
    ) -> Result<()> {
        input.validate(self.config)?;
        ensure!(!weight.is_null(), "DFly shared head missing");
        if self.config.world > 1 {
            let comm = comm.ok_or_else(|| anyhow::anyhow!("DFly TP communicator missing"))?;
            ensure!(
                comm.rank() == self.config.rank && comm.world_size() == self.config.world,
                "DFly TP communicator mismatch"
            );
        } else {
            ensure!(
                comm.is_none_or(|c| c.world_size() == 1),
                "DFly unexpected communicator"
            );
        }
        Ok(())
    }
}
