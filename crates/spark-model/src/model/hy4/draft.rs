// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::hy4::dfly::{self, TargetCache};

pub(super) const PROPOSE: u32 = 9;
fn validate(c: &ModelConfig) -> Result<()> {
    ensure!(
        c.hidden_size == 3072
            && c.num_hidden_layers == 42
            && c.num_attention_heads.checked_mul(c.tp_world_size) == Some(32)
            && c.q_lora_rank == 1024
            && c.kv_lora_rank == 512
            && c.tp_world_size == 2,
        "Hy4 DFly live runtime requires the validated TP2 architecture"
    );
    Ok(())
}
type DraftRuntime<'a> = dfly::Dfly<'a>;
self_cell::self_cell! {
    pub(super) struct DraftCell {owner:Arc<Shared>,#[covariant] dependent:DraftRuntime,}
}

pub(super) struct State<'a> {
    memory: Owned<'a>,
    hidden: DevicePtr,
    width: usize,
    len: usize,
    ready: bool,
}
impl<'a> State<'a> {
    pub(super) fn new(gpu: &'a dyn GpuBackend, width: usize) -> Result<Self> {
        let mut memory = Owned::new(gpu);
        let hidden = memory.alloc(width * 2 + 5 * 2048 * 4)?;
        Ok(Self {
            memory,
            hidden,
            width,
            len: 0,
            ready: false,
        })
    }
    fn indices(&self, i: usize) -> DevicePtr {
        self.hidden.offset(self.width * 2 + i * 2048 * 4)
    }
    fn capture(&mut self, hidden: DevicePtr, caches: &[TargetCache; 5], stream: u64) -> Result<()> {
        let len = caches[0].len;
        ensure!(
            !hidden.is_null()
                && len > 0
                && caches.iter().all(|c| c.len == len && !c.indices.is_null()),
            "Hy4 DFly capture missing/inconsistent target state"
        );
        self.memory
            .gpu
            .copy_d2d_async(hidden, self.hidden, self.width * 2, stream)?;
        for (i, cache) in caches.iter().enumerate() {
            self.memory
                .gpu
                .copy_d2d_async(cache.indices, self.indices(i), 2048 * 4, stream)?;
        }
        self.len = len;
        self.ready = true;
        Ok(())
    }
    pub(super) fn copy_from(&mut self, other: &Self, stream: u64) -> Result<()> {
        ensure!(
            self.width == other.width,
            "Hy4 DFly snapshot width mismatch"
        );
        self.memory.gpu.copy_d2d_async(
            other.hidden,
            self.hidden,
            self.width * 2 + 5 * 2048 * 4,
            stream,
        )?;
        self.len = other.len;
        self.ready = other.ready;
        Ok(())
    }
    pub(super) fn fork(&self, stream: u64) -> Result<Self> {
        let mut copy = Self::new(self.memory.gpu, self.width)?;
        copy.copy_from(self, stream)?;
        Ok(copy)
    }
}

impl Hy4Model {
    pub(crate) fn enable_dfly(&mut self) -> Result<()> {
        self.cell
            .as_mut()
            .context("Hy4 model torn down")?
            .with_dependent_mut(|s, r| {
                let c = &s.config;
                ensure!(
                    r.drafter.is_none() && r.requests.lock().is_empty(),
                    "Hy4 DFly must be enabled once before allocating requests"
                );
                validate(c)?;
                let cell = DraftCell::try_new(s.clone(), |s| {
                    let c = &s.config;
                    dfly::Dfly::load(
                        &s.store,
                        s.gpu.as_ref(),
                        dfly::Config {
                            rank: c.tp_rank,
                            world: c.tp_world_size,
                            vocab: c.vocab_size,
                            mask_token_id: c
                                .hy4
                                .as_ref()
                                .context("Hy4 metadata missing")?
                                .dfly_mask_token_id()?,
                            eps: c.rms_norm_eps as f32,
                            theta: c
                                .hy4
                                .as_ref()
                                .context("Hy4 metadata missing")?
                                .rope_parameters
                                .deepseek_sparse_attention
                                .rope_theta as f32,
                        },
                    )
                })?;
                r.drafter = Some(Mutex::new(cell));
                Ok(())
            })
    }
}
impl Runtime<'_> {
    pub(super) fn capture_draft_prefix(
        &self,
        data: &mut RequestData<'_>,
        hidden: DevicePtr,
        start: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        if data.draft.is_none() {
            return Ok(());
        }
        let mut targets = Self::targets(data, stream)?;
        let checkpoint = data
            .checkpoint
            .as_mut()
            .context("Hy4 draft prefix checkpoint missing")?;
        let state = data.draft.as_ref().unwrap();
        while checkpoint.draft_prefix.len() < rows {
            checkpoint
                .draft_prefix
                .push(State::new(state.memory.gpu, state.width)?);
        }
        for row in 0..rows {
            for (i, target) in targets.iter_mut().enumerate() {
                target.indices = data.layers[dfly::TARGET_LAYERS[i]]
                    .attn
                    .prefix_indices(row)?;
                target.len = start + row + 1;
            }
            checkpoint.draft_prefix[row].capture(
                hidden.offset(row * state.width * 2),
                &targets,
                stream,
            )?;
        }
        Ok(())
    }

    fn targets(data: &mut RequestData<'_>, stream: u64) -> Result<[TargetCache; 5]> {
        dfly::TARGET_LAYERS
            .iter()
            .map(|i| {
                data.layers
                    .get_mut(*i)
                    .context("Hy4 DFly target layer missing")?
                    .attn
                    .dfly_target(stream)
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("Hy4 DFly target count"))
    }
    pub(super) fn capture_draft(
        &self,
        data: &mut RequestData<'_>,
        hidden: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if data.draft.is_some() {
            let targets = Self::targets(data, stream)?;
            data.draft
                .as_mut()
                .unwrap()
                .capture(hidden, &targets, stream)?;
        }
        Ok(())
    }
    pub(super) fn propose(
        &self,
        s: &Shared,
        slot: usize,
        bonus: u32,
        count: usize,
        stream: u64,
    ) -> Result<Vec<u32>> {
        ensure!((1..=8).contains(&count), "Hy4 DFly count must be1..8");
        let drafter = self.drafter.as_ref().context("Hy4 DFly disabled")?;
        let mut requests = self.requests.lock();
        requests
            .get_mut(&slot)
            .context("Hy4 DFly slot missing")?
            .with_dependent_mut(|_, data| {
                ensure!(
                    !data.poisoned
                        && data
                            .checkpoint
                            .as_ref()
                            .is_some_and(|c| c.history == data.history
                                && c.draft
                                    .as_ref()
                                    .is_some_and(|d| d.ready && d.len == data.history.len())),
                    "Hy4 DFly checkpoint missing"
                );
                let mut caches = Self::targets(data, stream)?;
                let state = data.draft.as_ref().context("Hy4 DFly capture missing")?;
                ensure!(
                    state.ready && state.len == data.history.len(),
                    "Hy4 DFly stale hidden/index capture"
                );
                for (i, cache) in caches.iter_mut().enumerate() {
                    cache.indices = state.indices(i);
                }
                let input = dfly::Input {
                    target_hidden: state.hidden,
                    caches,
                    bonus,
                    count,
                };
                let result = drafter.lock().with_dependent_mut(|_, draft| {
                    draft.draft(
                        &input,
                        dfly::Head {
                            embedding: self.embed,
                            weight: self.vocab.weight,
                        },
                        s.comm.as_deref(),
                        stream,
                    )
                });
                if result.is_err() {
                    data.poisoned = true;
                }
                result
            })
    }
}

#[cfg(test)]
#[path = "draft_tests.rs"]
mod tests;
