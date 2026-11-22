// SPDX-License-Identifier: AGPL-3.0-only

use super::{AttentionCache, Hy4Attention, memory::Owned};
use crate::layers::hy4::batch::{self, MAX_ROWS};
use crate::layers::hy4::timing;
use anyhow::{Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

const CAUSAL_ROWS: usize = 8;
const INDEX_KEYS: usize = 2048;

fn causal_rows(window: Option<usize>, slots: usize) -> usize {
    window.map_or(CAUSAL_ROWS, |w| (slots - (w - 1)).min(CAUSAL_ROWS))
}

pub(super) struct BatchScratch<'a> {
    _memory: Owned<'a>,
    shares: DevicePtr,
    qc: DevicePtr,
    kv: DevicePtr,
    q: DevicePtr,
    absorbed: DevicePtr,
    latent: DevicePtr,
    values: DevicePtr,
    gates: DevicePtr,
    partial: DevicePtr,
    key: DevicePtr,
    query: DevicePtr,
    hw: DevicePtr,
    indices: DevicePtr,
    softmax_parts: DevicePtr,
    quant: DevicePtr,
    scales: DevicePtr,
    scores: DevicePtr,
    max_groups: usize,
}
impl<'a> BatchScratch<'a> {
    fn new(a: &Hy4Attention<'a>) -> Result<Self> {
        let mut m = Owned::new(a.memory.gpu);
        let max_groups = a.max_context.div_ceil(4);
        let (quant, scales, scores) = match &a.index {
            Some(ix) => (
                m.alloc(CAUSAL_ROWS * ix.heads * 128)?,
                m.alloc(CAUSAL_ROWS * ix.heads * 4)?,
                m.alloc(CAUSAL_ROWS * max_groups * 4)?,
            ),
            None => (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL),
        };
        let mut alloc = |n| m.alloc(MAX_ROWS * n * 2);
        Ok(Self {
            shares: alloc(a.qrank + 576)?,
            qc: alloc(a.qrank)?,
            kv: alloc(576)?,
            q: alloc(a.heads * 256)?,
            absorbed: alloc(a.heads * 576)?,
            latent: alloc(a.heads * 512)?,
            values: alloc(a.heads * 256)?,
            gates: alloc(a.heads * 256)?,
            partial: alloc(a.hidden)?,
            key: alloc(128)?,
            query: alloc(a.index.as_ref().map_or(1, |ix| ix.heads * 128))?,
            hw: alloc(a.index.as_ref().map_or(1, |ix| ix.heads))?,
            indices: m.alloc(CAUSAL_ROWS * INDEX_KEYS * 4)?,
            softmax_parts: m.alloc(
                CAUSAL_ROWS * a.heads * a.window.unwrap_or(INDEX_KEYS).div_ceil(128) * 514 * 4,
            )?,
            quant,
            scales,
            scores,
            max_groups,
            _memory: m,
        })
    }
}

impl Hy4Attention<'_> {
    fn batch_linear(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        dimensions: [usize; 9],
        stream: u64,
    ) -> Result<()> {
        let name = if w == self.qa {
            "attn_q_a"
        } else if w == self.qb {
            "attn_q_b"
        } else if w == self.kva {
            "attn_kv_a"
        } else if w == self.wk {
            "attn_absorb"
        } else if w == self.wv {
            "attn_value"
        } else if w == self.gate {
            "attn_gate"
        } else if w == self.output {
            "attn_output"
        } else {
            "dsa_projection"
        };
        let _phase = timing::span(name, stream)?;
        let scale = self.scales.get(&w).copied().unwrap_or(DevicePtr::NULL);
        batch::gemm(
            self.memory.gpu,
            [x, w, scale, y],
            dimensions,
            [u32::from(!scale.is_null()), 0],
            stream,
        )
    }
    fn batch_norm(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        n: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let _norm = timing::span("attn_norm", stream)?;
        KernelLaunch::new(self.memory.gpu, self.kernels.norm)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .arg_f32(self.eps)
            .launch(stream)
    }
}

impl AttentionCache<'_, '_> {
    pub(super) fn forward_batch(
        &mut self,
        input: DevicePtr,
        output: DevicePtr,
        positions: &[[i64; 3]],
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            positions.len() <= MAX_ROWS,
            "Hy4 attention batch exceeds workspace"
        );
        let scratch = match self.batch.take() {
            Some(s) => s,
            None => BatchScratch::new(self.layer)?,
        };
        let result = self.batch_rows(input, output, positions, comm, &scratch, stream);
        self.batch = Some(scratch);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_rows(
        &mut self,
        input: DevicePtr,
        output: DevicePtr,
        positions: &[[i64; 3]],
        comm: Option<&dyn CommBackend>,
        b: &BatchScratch<'_>,
        stream: u64,
    ) -> Result<()> {
        let a = self.layer;
        let gpu = a.memory.gpu;
        let rows = positions.len();
        let linear =
            |x, w, y, n, k| a.batch_linear(x, w, y, [n, k, rows, 1, 0, 0, k, n, k], stream);
        if let Some(comm) = comm {
            let stride = (a.qrank + 576) / a.world;
            let share = b.shares.offset(a.rank * rows * stride * 2);
            a.batch_linear(
                input,
                a.qa,
                share,
                [
                    a.qrank / 2,
                    a.hidden,
                    rows,
                    1,
                    0,
                    0,
                    a.hidden,
                    stride,
                    a.hidden,
                ],
                stream,
            )?;
            a.batch_linear(
                input,
                a.kva,
                share.offset(a.qrank),
                [288, a.hidden, rows, 1, 0, 0, a.hidden, stride, a.hidden],
                stream,
            )?;
            let exchange = timing::span("attn_qkv_exchange", stream)?;
            crate::layers::glm_kv_shard::pair_exchange(
                comm,
                share,
                b.shares.offset((1 - a.rank) * rows * stride * 2),
                rows * stride * 2,
                stream,
            )?;
            KernelLaunch::new(
                gpu,
                gpu.op_cache()
                    .kernel(gpu, "hy4_mla", "hy4_batch_join_qkv")?,
            )
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(b.shares)
            .arg_ptr(b.qc)
            .arg_ptr(b.kv)
            .arg_u32(a.qrank as u32)
            .arg_u32(rows as u32)
            .launch(stream)?;
            drop(exchange);
        } else {
            linear(input, a.qa, b.qc, a.qrank, a.hidden)?;
            linear(input, a.kva, b.kv, 576, a.hidden)?;
        }
        a.batch_norm(b.qc, a.qan, b.qc, a.qrank, rows, stream)?;
        linear(b.qc, a.qb, b.q, a.heads * 256, a.qrank)?;
        a.batch_linear(
            b.q,
            a.wk,
            b.absorbed,
            [
                512,
                192,
                rows,
                a.heads,
                256,
                576,
                a.heads * 256,
                a.heads * 576,
                192,
            ],
            stream,
        )?;
        linear(input, a.gate, b.gates, a.heads * 256, a.hidden)?;
        if let Some(ix) = &a.index {
            linear(input, ix.key, b.key, 128, a.hidden)?;
            linear(b.qc, ix.query, b.query, ix.heads * 128, a.qrank)?;
            linear(input, ix.head, b.hw, ix.heads, a.hidden)?;
        }
        let causal_rows = causal_rows(a.window, self.layout.table.len() * self.layout.page_size);
        for first in (0..rows).step_by(causal_rows) {
            let end = (first + causal_rows).min(rows);
            let start_position = self.len;
            let _causal = timing::span("attn_causal_rows", stream)?;
            let page_size = self.layout.page_size as u32;
            let columns = self.layout.table.len() as u32;
            let circular = u32::from(a.window.is_some());
            let batched = end - first;
            let base = positions.get(first).copied().unwrap_or([0; 3]);
            {
                let _rope = timing::span("attn_norm", stream)?;
                KernelLaunch::new(
                    gpu,
                    gpu.op_cache().kernel(gpu, "hy4_mla", "hy4_batch_kv_norm")?,
                )
                .grid([batched as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.kv.offset(first * 576 * 2))
                .arg_ptr(a.kvn)
                .arg_ptr(self.latent)
                .arg_ptr(self.table)
                .arg_u32(start_position as u32)
                .arg_u32(page_size)
                .arg_u32(columns)
                .arg_u32(circular)
                .arg_f32(a.eps)
                .launch(stream)?;
            }
            {
                let _rope = timing::span("attn_rope", stream)?;
                KernelLaunch::new(
                    gpu,
                    gpu.op_cache().kernel(gpu, "hy4_mla", "hy4_batch_rotary")?,
                )
                .grid([a.heads as u32, batched as u32, 1])
                .block([32, 1, 1])
                .arg_ptr(b.q.offset(first * a.heads * 256 * 2))
                .arg_ptr(b.absorbed.offset(first * a.heads * 576 * 2))
                .arg_u32(256)
                .arg_u32(576)
                .arg_u32(192)
                .arg_u32(512)
                .arg_u32(a.heads as u32 * 256)
                .arg_u32(a.heads as u32 * 576)
                .arg_ptr(DevicePtr::NULL)
                .arg_u32(start_position as u32)
                .arg_u32(page_size)
                .arg_u32(columns)
                .arg_u32(circular)
                .arg_u32(0)
                .arg_u64(base[0] as u64)
                .arg_u64(base[1] as u64)
                .arg_u64(base[2] as u64)
                .arg_f32(a.rope.rope_theta as f32)
                .arg_u32(a.rope.mrope_section[1] as u32)
                .arg_u32(a.rope.mrope_section[2] as u32)
                .arg_u32(u32::from(a.rope.mrope_interleaved))
                .arg_u32(0)
                .launch(stream)?;
                KernelLaunch::new(
                    gpu,
                    gpu.op_cache().kernel(gpu, "hy4_mla", "hy4_batch_rotary")?,
                )
                .grid([1, batched as u32, 1])
                .block([32, 1, 1])
                .arg_ptr(b.kv.offset(first * 576 * 2))
                .arg_ptr(self.latent)
                .arg_u32(576)
                .arg_u32(576)
                .arg_u32(512)
                .arg_u32(512)
                .arg_u32(576)
                .arg_u32(576)
                .arg_ptr(self.table)
                .arg_u32(start_position as u32)
                .arg_u32(page_size)
                .arg_u32(columns)
                .arg_u32(circular)
                .arg_u32(1)
                .arg_u64(base[0] as u64)
                .arg_u64(base[1] as u64)
                .arg_u64(base[2] as u64)
                .arg_f32(a.rope.rope_theta as f32)
                .arg_u32(a.rope.mrope_section[1] as u32)
                .arg_u32(a.rope.mrope_section[2] as u32)
                .arg_u32(u32::from(a.rope.mrope_interleaved))
                .arg_u32(0)
                .launch(stream)?;
            }
            if a.index.is_some() {
                self.run_index_tile(
                    &positions[first..end],
                    b.key,
                    b.query,
                    b.hw,
                    b.quant,
                    b.scales,
                    b.scores,
                    b.indices,
                    b.max_groups,
                    stream,
                )?;
            } else {
                for row in first..end {
                    if a.window.is_none() {
                        let count = (self.len / 4).min(INDEX_KEYS / 4 - 1) * 4 + self.len % 4 + 1;
                        gpu.copy_d2d_async(
                            self.scratch.indices,
                            b.indices.offset((row - first) * INDEX_KEYS * 4),
                            count * 4,
                            stream,
                        )?;
                    }
                    self.len += 1;
                    self.record_prefix(stream)?;
                }
            }
            drop(_causal);
            let count = a.window.map_or_else(
                || ((self.len - 1) / 4).min(INDEX_KEYS / 4 - 1) * 4 + (self.len - 1) % 4 + 1,
                |w| self.len.min(w),
            );
            let splits = count.div_ceil(128);
            let mla = timing::span("latent_attention", stream)?;
            // ATLAS_HY4_MLA_MMA=1: one CTA per (row, 128-key partition)
            // gathers K into shared once and runs both attention GEMMs on
            // tensor cores for all local heads. Requires heads <= 16 and
            // ~99KB opt-in dynamic shared (sm_120+).
            // EXPERT_MMA+MLA_MMA is decode-unsafe on VERIFY rows: accepted
            // drafts collapse (144 vs 180 on the Step 1 harness). Keep MLA on
            // the tensor-core path only for prefill-scale row counts when the
            // expert MMA path is also enabled; VERIFY stays on the proven MLA
            // kernel while expert MMA still serves the FFN phases.
            let expert_mma = std::env::var("ATLAS_HY4_EXPERT_MMA").ok().as_deref() == Some("1");
            let mla_mma = a.heads <= 16
                && std::env::var("ATLAS_HY4_MLA_MMA").as_deref() == Ok("1")
                && !(expert_mma && rows < 32);
            let (mla_kernel, mla_grid, mla_smem) = if mla_mma {
                (
                    "hy4_mla_batch_partial_mma",
                    [splits as u32, (end - first) as u32, 1],
                    100544,
                )
            } else {
                (
                    "hy4_mla_batch_partial",
                    [a.heads as u32, splits as u32, (end - first) as u32],
                    0,
                )
            };
            KernelLaunch::new(gpu, gpu.op_cache().kernel(gpu, "hy4_mla", mla_kernel)?)
                .grid(mla_grid)
                .block([256, 1, 1])
                .shared_mem(mla_smem)
                .arg_ptr(b.absorbed.offset(first * a.heads * 576 * 2))
                .arg_ptr(self.latent)
                .arg_ptr(self.table)
                .arg_ptr(if a.window.is_some() {
                    DevicePtr::NULL
                } else {
                    b.indices
                })
                .arg_ptr(b.softmax_parts)
                .arg_u32(start_position as u32)
                .arg_u32(self.layout.page_size as u32)
                .arg_u32(self.layout.table.len() as u32)
                .arg_u32(a.window.unwrap_or(0) as u32)
                .arg_u32(count as u32)
                .arg_u32(splits as u32)
                .arg_u32(a.heads as u32)
                .arg_u32(INDEX_KEYS as u32)
                .launch(stream)?;
            KernelLaunch::new(
                gpu,
                gpu.op_cache()
                    .kernel(gpu, "hy4_mla", "hy4_mla_batch_merge")?,
            )
            .grid([a.heads as u32, (end - first) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(b.softmax_parts)
            .arg_ptr(a.sink)
            .arg_ptr(b.latent.offset(first * a.heads * 512 * 2))
            .arg_u32(splits as u32)
            .arg_u32(a.heads as u32)
            .launch(stream)?;
            drop(mla);
        }
        a.batch_linear(
            b.latent,
            a.wv,
            b.values,
            [
                256,
                512,
                rows,
                a.heads,
                512,
                256,
                a.heads * 512,
                a.heads * 256,
                512,
            ],
            stream,
        )?;
        KernelLaunch::new(gpu, a.kernels.gate)
            .grid([(rows * a.heads * 256).div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(b.values)
            .arg_ptr(b.gates)
            .arg_u32((rows * a.heads * 256) as u32)
            .launch(stream)?;
        linear(b.values, a.output, b.partial, a.hidden, a.heads * 256)?;
        if let Some(comm) = comm {
            let _reduce = timing::span("all_reduce", stream)?;
            comm.all_reduce_async(b.partial.0, rows * a.hidden * 2, stream)?;
        }
        a.batch_norm(b.partial, a.norm, output, a.hidden, rows, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::hy4_reference::paged::PagedLayout;

    #[test]
    fn causal_tiles_keep_the_earliest_query_window_resident() {
        for window in [1, 2, 3, 127, 512] {
            for page in [1, 4, 16, 128] {
                let capacity = PagedLayout::window_pages(window, page).unwrap() * page;
                let rows = causal_rows(Some(window), capacity);
                assert!((1..=CAUSAL_ROWS).contains(&rows));
                assert!(window - 1 + rows <= capacity);
            }
        }
        assert_eq!(causal_rows(None, 1), CAUSAL_ROWS);
    }
}
