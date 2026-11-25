// SPDX-License-Identifier: AGPL-3.0-only

//! Which rows of a pass a DeepSeek-V4.1 layer's rank runs: what the ranks
//! agree on at a chunk's first layer, how they share a large batch's rows
//! out between them, and how the rows come back together where the layers
//! turn to a prompt's last rows.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::owned::Owned;
use super::super::parts::PARTS;
use super::super::tail;
use super::Layer;
use crate::layer::ForwardContext;

/// The fewest rows of a batch the ranks share out ([`Owned`]): below it the
/// gathers that sharing adds cost more than the mixing it saves.
const OWNED_ROWS: usize = 1024;

impl Layer {
    /// How the ranks share out a batch of `rows` rows of a pass, when they
    /// do: a batch large enough, of a pass that reads the last layer at its
    /// last row only, on ranks that hold a part each. Every rank answers the
    /// same.
    pub(super) fn owned(&self, rows: usize, ctx: &ForwardContext) -> Option<Owned> {
        let shared = &self.block.shared;
        let whole = shared.handoff.lock().expect("handoff").whole;
        let shared_out = rows >= OWNED_ROWS
            && !whole
            && self.block.gemm.is_some()
            && shared.held.count < PARTS
            && shared.scratch.rows.is_multiple_of(PARTS)
            && ctx.comm.is_some()
            && ctx.config.ep_world_size > 1
            && std::env::var_os("ATLAS_DSV41_NO_OWNED_ROWS").is_none();
        shared_out.then(|| Owned::of(rows, shared.held))
    }

    /// Before the first layer that runs a prompt's last rows alone
    /// (`tail.rs`): every rank gets the stream of the rows from `from` of a
    /// chunk of `tokens`, which the layers before it left with the rank that
    /// keeps each row. The layers from here on are few rows, and every rank
    /// mixes them all.
    pub(super) fn replicate(
        &self,
        hidden: DevicePtr,
        tokens: usize,
        from: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let shared = &self.block.shared;
        let Some(comm) = ctx.comm.filter(|_| ctx.config.ep_world_size > 1) else {
            return Ok(());
        };
        let (copies, batch) = (shared.shape.hc.copies, shared.scratch.rows);
        let row = self.row(hidden, ctx)?;
        // A row of the stream, and of the mix it is collapsed with next.
        let buffers = [
            (ctx.buffers.hc_streams(), copies * shared.shape.hidden * 2),
            (ctx.buffers.hc_post(), copies * 4),
        ];
        for first in (0..tokens).step_by(batch) {
            let rows = (tokens - first).min(batch);
            if self.owned(rows, ctx).is_none() {
                continue;
            }
            for part in 0..PARTS {
                let of = Owned::of_part(rows, part);
                let (start, end) = ((first + of.start).max(from), first + of.end);
                if start >= end {
                    continue;
                }
                let root = part / shared.held.count;
                for (base, width) in buffers {
                    let (at, bytes) = (base.offset((row + start) * width), (end - start) * width);
                    if !comm.broadcast_async(at.0, bytes, root, stream)? {
                        ctx.gpu.synchronize(stream)?;
                        comm.broadcast(at.0, bytes, root)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// At a chunk's first layer: whether the pass reads every row of the
    /// last layer, which the rank that serves the request knows and the
    /// others are told, for the layers after this one to go by.
    pub(super) fn agree_on_rows(&self, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let shared = &self.block.shared;
        let mut whole = tail::whole_here();
        if let Some(comm) = ctx.comm.filter(|_| ctx.config.ep_world_size > 1) {
            let word = shared.scratch.word;
            let mut bytes = [0u8; 16];
            bytes[0] = u8::from(whole);
            ctx.gpu.synchronize(stream)?;
            ctx.gpu.copy_h2d(&bytes, word)?;
            comm.broadcast(word.0, bytes.len(), 0)?;
            ctx.gpu.copy_d2h(word, &mut bytes)?;
            whole = bytes[0] != 0;
        }
        shared.handoff.lock().expect("handoff").whole = whole;
        Ok(())
    }
}
