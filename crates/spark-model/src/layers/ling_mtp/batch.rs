// SPDX-License-Identifier: AGPL-3.0-only
//! One draft for each of several sequences in one predictor pass: every
//! weight (merge, MLA + MoE body, head) is read once for the batch instead of
//! once per sequence. Row i is sequence i; each row attends to its own
//! sequence's predictor KV through a metadata row of its own.

use super::{LingMtp, State, state};
use crate::layer::{
    AttnMetadataDev, EmptyLayerState, ForwardContext, LayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::ProposerState;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Widest batch one pass carries: the arena's batched-verify stash width.
pub(super) const MAX_ROWS: usize = crate::layer::VERIFY_WY_TABLE_SEQS;

/// Batch kernels, resolved at load. A missing one leaves `propose_batch` to
/// the per-sequence loop.
pub(super) struct BatchKernels {
    argmax: KernelHandle,
    merge: KernelHandle,
    /// `w4a16_gemv_batch{2,3,4,8,16,32}`: each row as its own scalar GEMV.
    head: [(u32, KernelHandle); 6],
    /// Device scratch: per-row metadata, then `MAX_ROWS` logits rows.
    meta: DevicePtr,
    meta_bytes: usize,
    max_blocks: usize,
    logits: DevicePtr,
}

impl BatchKernels {
    pub(super) fn load(gpu: &dyn GpuBackend, max_blocks: usize, vocab: usize) -> Result<Self> {
        let k = |module: &str, name: &str| crate::layers::try_kernel(gpu, module, name);
        let meta_bytes = MAX_ROWS * (4 + 8 + 4 + 4 * max_blocks) + 64;
        Ok(Self {
            argmax: k("argmax", "argmax_bf16_batch"),
            merge: k("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
            head: [2, 3, 4, 8, 16, 32]
                .map(|m| (m, k("w4a16_gemv", &format!("w4a16_gemv_batch{m}")))),
            meta: gpu.alloc(meta_bytes)?,
            meta_bytes,
            max_blocks,
            logits: gpu.alloc(MAX_ROWS * vocab * 2)?,
        })
    }

    fn head_kernel(&self, rows: u32) -> Option<KernelHandle> {
        self.head
            .iter()
            .find(|&&(m, k)| m >= rows && k.0 != 0)
            .map(|&(_, k)| k)
    }
}

/// Per-row attention metadata, laid out positions | slots | seq_lens | block
/// tables (`max_blocks` entries per row): byte offsets of each array.
pub(super) fn meta_layout(rows: usize, max_blocks: usize) -> [usize; 4] {
    let slots = (rows * 4).next_multiple_of(8);
    let seq_lens = slots + rows * 8;
    let tables = seq_lens + rows * 4;
    [0, slots, seq_lens, tables + rows * max_blocks * 4]
}

pub(super) fn pack_meta(rows: &[(u32, i64, i32, &[u32])], max_blocks: usize) -> Vec<u8> {
    let [_, slots, seq_lens, end] = meta_layout(rows.len(), max_blocks);
    let tables = seq_lens + rows.len() * 4;
    let mut buf = vec![0u8; end];
    for (i, &(position, slot, seq_len, blocks)) in rows.iter().enumerate() {
        buf[i * 4..i * 4 + 4].copy_from_slice(&position.to_le_bytes());
        buf[slots + i * 8..slots + i * 8 + 8].copy_from_slice(&slot.to_le_bytes());
        buf[seq_lens + i * 4..seq_lens + i * 4 + 4].copy_from_slice(&seq_len.to_le_bytes());
        for (j, block) in blocks.iter().enumerate() {
            let at = tables + (i * max_blocks + j) * 4;
            buf[at..at + 4].copy_from_slice(&block.to_le_bytes());
        }
    }
    buf
}

impl LingMtp {
    /// Draft one token for each sequence. `Ok(None)` hands the step to the
    /// per-sequence loop: a grammar mask, a batch kernel this build lacks, or a
    /// sequence whose predictor KV is full.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_rows(
        &self,
        tokens: &[u32],
        targets: &[DevicePtr],
        positions: &[usize],
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let n = tokens.len();
        let b = &self.batch;
        let Some((head_weight, _)) = &self.draft_head else {
            return Ok(None);
        };
        let Some(head_k) = b.head_kernel(n as u32) else {
            return Ok(None);
        };
        if !(2..=MAX_ROWS).contains(&n) || b.argmax.0 == 0 || b.merge.0 == 0 {
            return Ok(None);
        }
        ensure!(
            targets.len() == n && positions.len() == n && states.len() == n,
            "Ling batched propose: {n} tokens, {} hiddens, {} positions, {} states",
            targets.len(),
            positions.len(),
            states.len()
        );
        ensure!(
            tokens.iter().all(|&t| (t as usize) < ctx.config.vocab_size),
            "invalid Ling MTP token"
        );
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let mut cache = self.cache.lock();
        let bs = cache.block_size();
        // Reserve every row's KV slot first: a full cache must leave all
        // sequences untouched for the per-sequence fallback.
        let needed = states
            .iter_mut()
            .map(|v| state(*v).map(|s| usize::from(s.rows / bs == s.blocks.len())))
            .sum::<Result<usize>>()?;
        if needed > cache.num_free_blocks()
            || states
                .iter_mut()
                .any(|v| state(*v).is_ok_and(|s| s.capacity_exhausted))
        {
            return Ok(None);
        }
        let mut rows = Vec::with_capacity(n);
        let max_blocks = b.max_blocks;
        for (v, &position) in states.iter_mut().zip(positions) {
            let s: &mut State = state(*v)?;
            if s.rows / bs == s.blocks.len() {
                s.blocks.push(cache.alloc_block()?);
            }
            ensure!(
                s.blocks.len() <= max_blocks,
                "Ling predictor block table is full"
            );
            let slot = (s.blocks[s.rows / bs] as usize * bs + s.rows % bs) as i64;
            rows.push((
                u32::try_from(position.saturating_sub(1))?,
                slot,
                i32::try_from(s.rows + 1)?,
            ));
        }
        let widest = states
            .iter_mut()
            .map(|v| state(*v).map(|s| s.blocks.len()))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(1);

        // Inputs: [embed(token) after enorm | target after model.norm, hnorm].
        let ids: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        gpu.copy_h2d_async(&ids, self.inputs, stream)?;
        ops::batched_embed(
            gpu,
            self.embed_k,
            self.inputs,
            self.embed.weight,
            self.hidden,
            n as u32,
            h as u32,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.norm_k,
            self.hidden,
            &self.enorm,
            self.residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        gpu.copy_d2d_2d_async(self.residual, 2 * h, self.pair, 4 * h, 2 * h, n, stream)?;
        for (i, &target) in targets.iter().enumerate() {
            gpu.copy_d2d_async(target, self.residual.offset(i * 2 * h), 2 * h, stream)?;
        }
        ops::rms_norm(
            gpu,
            self.norm_k,
            self.residual,
            &self.target_norm,
            self.hidden,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.norm_k,
            self.hidden,
            &self.hnorm,
            self.residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        gpu.copy_d2d_2d_async(
            self.residual,
            2 * h,
            self.pair.offset(2 * h),
            4 * h,
            2 * h,
            n,
            stream,
        )?;
        let max_m = ops::DENSE_GEMV_BATCHM_MAX_M as usize;
        for first in (0..n).step_by(max_m) {
            let m = max_m.min(n - first);
            ops::dense_gemv_batchm(
                gpu,
                b.merge,
                self.pair.offset(first * 4 * h),
                &self.merge,
                self.hidden.offset(first * 2 * h),
                m as u32,
                h as u32,
                (2 * h) as u32,
                h as u32,
                stream,
            )?;
        }

        // The body: one MLA + MoE pass over the n rows.
        let mut tables: Vec<Vec<u32>> = Vec::with_capacity(n);
        for v in states.iter_mut() {
            tables.push(state(*v)?.blocks.clone());
        }
        let packed: Vec<_> = rows
            .iter()
            .zip(&tables)
            .map(|(&(p, slot, len), t)| (p, slot, len, t.as_slice()))
            .collect();
        let data = pack_meta(&packed, widest);
        ensure!(
            data.len() <= b.meta_bytes,
            "Ling batched predictor metadata overflow"
        );
        gpu.copy_h2d_async(&data, b.meta, stream)?;
        let [_, slots, seq_lens, _] = meta_layout(n, widest);
        let meta = AttnMetadataDev {
            positions: b.meta,
            positions_h: b.meta,
            positions_w: b.meta,
            slot: b.meta.offset(slots),
            seq_len: b.meta.offset(seq_lens),
            block_table: b.meta.offset(seq_lens + n * 4),
            max_blocks_per_seq: widest as u32,
            num_seqs: n as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let context = ForwardContext {
            attn_metadata: Some(meta),
            ssm_batch: None,
            graph_capture: false,
            comm: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
            ..*ctx
        };
        // The MLA body keeps no per-sequence layer state.
        let mut empty: Vec<EmptyLayerState> = (0..n).map(|_| EmptyLayerState).collect();
        let mut body_states: Vec<&mut (dyn LayerState + 'static)> = empty
            .iter_mut()
            .map(|s| s as &mut (dyn LayerState + 'static))
            .collect();
        let seq_lens_host: Vec<usize> = rows.iter().map(|r| r.2 as usize - 1).collect();
        self.body.decode_multi_seq(
            self.hidden,
            self.residual,
            n,
            n,
            &mut body_states,
            &mut cache,
            &seq_lens_host,
            &tables,
            &context,
            stream,
        )?;
        for (v, &(position, ..)) in states.iter_mut().zip(&rows) {
            let s = state(*v)?;
            s.rows += 1;
            s.pair = Some(position as usize);
        }
        drop(cache);

        // Head and argmax over the n rows.
        let vocab = ctx.config.vocab_size;
        ops::rms_norm(
            gpu,
            self.norm_k,
            self.hidden,
            &self.norm,
            self.residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            gpu,
            head_k,
            self.residual,
            head_weight,
            b.logits,
            n as u32,
            vocab as u32,
            h as u32,
            stream,
        )?;
        ops::argmax_bf16_batch(
            gpu,
            b.argmax,
            b.logits,
            self.inputs,
            vocab as u32,
            n as u32,
            vocab as u32,
            stream,
        )?;
        let mut raw = vec![0u8; 4 * n];
        gpu.copy_d2h_on_stream(self.inputs, &mut raw, stream)?;
        Ok(Some(
            raw.chunks_exact(4)
                .map(|c| vec![u32::from_le_bytes([c[0], c[1], c[2], c[3]])])
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_rows_land_where_the_layout_says() {
        let rows = [(7u32, 33i64, 8i32, &[2u32, 5][..]), (40, 81, 41, &[9][..])];
        let buf = pack_meta(&rows, 2);
        let [_, slots, seq_lens, end] = meta_layout(2, 2);
        assert_eq!(buf.len(), end);
        let u32_at = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
        assert_eq!((u32_at(0), u32_at(4)), (7, 40));
        assert_eq!(
            i64::from_le_bytes(buf[slots + 8..slots + 16].try_into().unwrap()),
            81
        );
        assert_eq!((u32_at(seq_lens), u32_at(seq_lens + 4)), (8, 41));
        let tables = seq_lens + 8;
        // Row 1's table starts a full row width after row 0's; its unused
        // entry stays zero.
        assert_eq!([u32_at(tables), u32_at(tables + 4)], [2, 5]);
        assert_eq!([u32_at(tables + 8), u32_at(tables + 12)], [9, 0]);
        assert_eq!(slots % 8, 0, "the i64 slots are 8-byte aligned");
    }
}
