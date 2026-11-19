// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(super) const CHECKPOINT: u32 = 7;
pub(super) const COMMIT: u32 = 8;

pub(super) struct Checkpoint<'a> {
    caches: Vec<AttentionCache<'a, 'a>>,
    pub(super) history: Vec<u32>,
    pub(super) draft: Option<draft::State<'a>>,
}

impl Runtime<'_> {
    pub(super) fn checkpoint(&self, slot: usize, position: usize, stream: u64) -> Result<()> {
        let mut requests = self.requests.lock();
        let cell = requests
            .get_mut(&slot)
            .context("Hy4 checkpoint slot missing")?;
        cell.with_dependent_mut(|_, data| {
            ensure!(
                !data.poisoned && data.history.len() == position,
                "Hy4 checkpoint position/state mismatch"
            );
            if let Some(checkpoint) = &mut data.checkpoint {
                if data.draft.is_some() && checkpoint.history == data.history {
                    return Ok(());
                }
                for (cache, layer) in checkpoint.caches.iter_mut().zip(&data.layers) {
                    cache.copy_state_from(&layer.attn, stream)?;
                }
                checkpoint.history.clone_from(&data.history);
                if let (Some(saved), Some(current)) = (&mut checkpoint.draft, &data.draft) {
                    saved.copy_from(current, stream)?;
                }
            } else {
                let caches = data
                    .layers
                    .iter()
                    .map(|layer| layer.attn.fork(stream))
                    .collect::<Result<Vec<_>>>()?;
                data.checkpoint = Some(Checkpoint {
                    caches,
                    history: data.history.clone(),
                    draft: data.draft.as_ref().map(|d| d.fork(stream)).transpose()?,
                });
            }
            Ok(())
        })
    }

    pub(super) fn validate_commit(
        &self,
        slot: usize,
        accepted: usize,
        rows: usize,
        canonical: Option<&crate::traits::SequenceState>,
    ) -> Result<()> {
        let requests = self.requests.lock();
        let data = requests
            .get(&slot)
            .context("Hy4 commit slot missing")?
            .borrow_dependent();
        let checkpoint = data
            .checkpoint
            .as_ref()
            .context("Hy4 verify checkpoint missing")?;
        let base = checkpoint.history.len();
        ensure!(
            !data.poisoned
                && rows > 0
                && rows <= self.multi.capacity
                && accepted > 0
                && accepted <= rows
                && data.history.len() == base + rows
                && data.history[..base] == checkpoint.history
                && canonical.is_none_or(|seq| seq.seq_len == base + accepted
                    && seq.tokens == data.history[..base + accepted]),
            "Hy4 invalid accepted verify prefix"
        );
        Ok(())
    }

    pub(super) fn commit(
        &self,
        s: &Shared,
        slot: usize,
        accepted: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        self.validate_commit(slot, accepted, rows, None)?;
        if accepted == rows {
            return Ok(());
        }
        let replay = {
            let mut requests = self.requests.lock();
            let cell = requests.get_mut(&slot).context("Hy4 commit slot missing")?;
            cell.with_dependent_mut(|_, data| {
                let checkpoint = data
                    .checkpoint
                    .as_ref()
                    .context("Hy4 verify checkpoint missing")?;
                let base = checkpoint.history.len();
                let replay = data.history[base..base + accepted].to_vec();
                data.poisoned = true;
                for (layer, cache) in data.layers.iter_mut().zip(&checkpoint.caches) {
                    layer.attn.copy_state_from(cache, stream)?;
                }
                data.history.clone_from(&checkpoint.history);
                if let (Some(current), Some(saved)) = (&mut data.draft, &checkpoint.draft) {
                    current.copy_from(saved, stream)?;
                }
                data.poisoned = false;
                Ok::<_, anyhow::Error>(replay)
            })?
        };
        if replay.len() == 1 {
            self.row(s, slot, replay[0], stream)?;
        } else {
            self.rows(s, slot, &replay, batch::BATCH, stream)?;
        }
        let position = self.requests.lock()[&slot].borrow_dependent().history.len();
        self.checkpoint(slot, position, stream)
    }
}
