// SPDX-License-Identifier: AGPL-3.0-only

//! A block as a stage of the draft head (`DSparkBlock` and
//! `DSparkAttention` in the release's `inference/model.py`).
//!
//! A stage is a block whose attention reads two things: a window of keys
//! made from the main model's last 128 positions, and the keys of the
//! block's own rows, each of which sees all the others. Its rows are the
//! token the model just chose and the noise tokens after it, at the
//! positions that follow the sequence's end. Nothing a stage writes
//! outlasts the pass: the window is the sequence's, and the pass's caller
//! fills it ([`Block::main_keys`]).

use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::block::Block;
use super::cache_recipe::WINDOW;
use super::draft_seq::slot_runs;
use super::ops::{self, HEAD_DIM, Turn};
use super::parts;

impl Block {
    /// The window keys of `rows` positions of the main model from
    /// `position`, into their slots of `window`: `x` (`[rows, hidden]` BF16)
    /// is the head's reading of those positions in FP8. `keys` and `normed`
    /// are scratch of `[rows, 512]` BF16.
    #[allow(clippy::too_many_arguments)]
    pub fn main_keys(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        window: DevicePtr,
        position: usize,
        keys: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (s, k, w) = (&self.shared.shape, &self.shared.kernels, &self.weights.attn);
        let wkv = w.wkv.rows_from(0);
        parts::project_rows(
            gpu,
            k.parts.fp8,
            x,
            wkv.data,
            wkv.scales,
            keys,
            2,
            HEAD_DIM,
            rows,
            HEAD_DIM,
            s.hidden,
            stream,
        )?;
        self.norm(gpu, keys, w.kv_norm, normed, rows, HEAD_DIM, stream)?;
        self.at(position + rows - 1)?;
        let positions = self.at(position)?;
        let (rotate, rope) = (k.cache.rotate, s.rope);
        ops::rotate(
            gpu, rotate, normed, positions, self.freqs, rows, 1, HEAD_DIM, rope, false, stream,
        )?;
        let mut done = 0;
        for (slot, count) in slot_runs(&(position..position + rows)) {
            ops::fp8_quant_dequant(
                gpu,
                k.cache.fp8_quant_dequant,
                normed.offset(done * HEAD_DIM * 2),
                window.offset(slot * HEAD_DIM * 2),
                count * HEAD_DIM,
                stream,
            )?;
            done += count;
        }
        Ok(())
    }

    /// The stage on the `rows` rows of a draft block, which sit at
    /// `position` and after: `streams` and `pre` as [`Block::forward_rows`]
    /// takes them. Every row reads the slots of `window` that `seen` names
    /// (a window's worth, -1 for none) and the keys of all the block's rows.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_drafts(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        streams: DevicePtr,
        pre: DevicePtr,
        window: DevicePtr,
        seen: &[i32],
        position: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights,
        );
        anyhow::ensure!(
            (1..=sc.rows).contains(&rows) && seen.len() == WINDOW,
            "draft stage {}: {rows} rows of at most {}, over {} window slots",
            self.index,
            sc.rows,
            seen.len()
        );
        self.site(
            gpu,
            &w.hc_attn,
            None,
            rows,
            streams,
            pre,
            sc.site_pre,
            w.attn.norm,
            None,
            stream,
        )?;
        parts::project_two(
            gpu,
            k.parts.fp8_two,
            sc.quantized,
            (w.attn.wq_a.rows_from(0), w.attn.wq_a.n, sc.q_low),
            (w.attn.wkv.rows_from(0), w.attn.wkv.n, sc.key),
            s.hidden,
            rows,
            stream,
        )?;
        self.at(position + rows - 1)?;
        let turn = Turn {
            positions: self.at(position)?,
            freqs: self.freqs,
            rope: s.rope,
        };
        // The block's own keys, a row each, beside the window.
        let own = sc.chunk_keys;
        ops::front(
            gpu,
            k.cache.front,
            sc.q_low,
            w.attn.q_norm,
            sc.q_low_normed,
            sc.key,
            w.attn.kv_norm,
            own,
            turn,
            s.q_rank,
            rows,
            s.norm_eps,
            stream,
        )?;
        // Every row reads the same keys: the window's, then the block's.
        let row: Vec<i32> = seen.iter().copied().chain(0..rows as i32).collect();
        let bytes: Vec<u8> = (0..rows)
            .flat_map(|_| row.iter().flat_map(|i| i.to_le_bytes()))
            .collect();
        gpu.copy_h2d_async(&bytes, sc.idx, stream)?;
        self.attend_keys(gpu, rows, window, own, row.len(), turn, comm, None, stream)?;
        self.finish(gpu, rows, streams, pre, comm, None, stream)
    }
}
