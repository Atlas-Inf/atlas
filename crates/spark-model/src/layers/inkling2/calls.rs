// SPDX-License-Identifier: AGPL-3.0-only

//! The Inkling 2 kernel calls a visit's forward makes over its `n` rows.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::layer::Inkling2Layer;
use super::ops::{self, BlockRes, HeadNorm, HeadNormArgs, MoeDecode, MsgGate, ShortConv};
use super::weights::KvConv;

impl Inkling2Layer {
    /// One short conv over the batch's rows; `states` is this visit's state
    /// table, `state_offset` the conv's place (fp32 elements) in each state.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn conv(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        num_seqs: usize,
        x: DevicePtr,
        out: DevicePtr,
        states: DevicePtr,
        state_offset: usize,
        conv: KvConv,
        dyn_taps: DevicePtr,
        channels: usize,
        head_dim: usize,
        stream: u64,
    ) -> Result<()> {
        let (weight, dyn_taps) = match conv {
            KvConv::Static { weight } => (weight, DevicePtr::NULL),
            KvConv::Dynamic { static_w, .. } => (static_w, dyn_taps),
        };
        ops::short_conv(
            gpu,
            &self.k,
            &ShortConv {
                x,
                out,
                states,
                state_offset,
                rows_per_seq: n / num_seqs,
                weight,
                dyn_taps,
                rows: n,
                num_seqs,
                channels,
                k: self.p.conv_k,
                x_stride: channels,
                out_stride: channels,
                dyn_stride: self.p.kv_heads * self.p.conv_k,
                head_dim,
            },
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_res(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        blocks: DevicePtr,
        nb: usize,
        partial_f32: DevicePtr,
        partial_bf16: DevicePtr,
        queries: DevicePtr,
        nq: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ops::block_res(
            gpu,
            &self.k,
            &BlockRes {
                blocks,
                partial_f32,
                partial_bf16,
                queries,
                out,
                rows: n,
                hidden: self.dims.hidden,
                num_blocks: nb,
                row_stride: self.dims.hidden,
                block_stride: self.dims.block_stride(),
                num_queries: nq,
                sums: self.shared.res_sums,
                eps: 1e-6,
            },
            stream,
        )
    }

    /// Message gates `which` (consecutive, at most three) of this loop layer;
    /// strides in fp32 elements between the gates' `h`, `m` and `out` rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn msg_gates(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        which: std::ops::Range<usize>,
        (h, h_stride): (DevicePtr, usize),
        (m, m_stride): (DevicePtr, usize),
        (out, out_stride): (DevicePtr, usize),
        stream: u64,
    ) -> Result<()> {
        let gates = self
            .w
            .messages
            .context("inkling2: message gate on a layer outside the loop")?;
        ops::msg_gate(
            gpu,
            &self.k,
            &MsgGate {
                gates: &gates[which],
                h,
                h_stride,
                m,
                m_stride,
                out,
                out_stride,
                z: self.shared.gate_z,
                rows: n,
                hidden: self.dims.hidden,
                rank: self.p.gate_rank,
                eps: self.p.eps,
            },
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn head_norm(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        x: DevicePtr,
        y: DevicePtr,
        scale: DevicePtr,
        // the rows' positions for the log-length scale; NULL = none
        positions: DevicePtr,
        heads: usize,
        head_dim: usize,
        norm: HeadNorm,
        stream: u64,
    ) -> Result<()> {
        ops::head_norm(
            gpu,
            self.k.head_norm,
            &HeadNormArgs {
                x,
                y,
                scale,
                positions,
                rows: n,
                heads,
                head_dim,
                in_stride: heads * head_dim,
                out_stride: heads * head_dim,
                norm,
                log_n_floor: self.p.log_n_floor,
                log_alpha: self.p.log_alpha,
            },
            stream,
        )
    }

    /// The routed experts over this forward's `n` rows (router output in
    /// `ids` / `weights`, input `mlp_normed`, output `moe_out`): grouped GEMMs
    /// for prefill-sized row counts, the decode GEMVs otherwise.
    pub(super) fn routed_experts(&self, gpu: &dyn GpuBackend, n: usize, stream: u64) -> Result<()> {
        let (s, w, d, p, h) = (&self.shared, &self.w, &self.dims, &self.p, self.dims.hidden);
        if n > 1 {
            let args = ops::MoePrefill {
                x: s.mlp_normed,
                experts: w.experts,
                expert_ids: s.ids,
                weights: s.weights,
                gu: s.gu,
                act: s.act,
                y: s.y,
                out: s.moe_out,
                sort: s.moe_sort,
                rows: n,
                top_k: d.top_k,
                hidden: h,
                inter: d.inter,
                num_experts: d.experts,
                gate_softcap: p.gate_softcap,
                up_softcap: p.up_softcap,
            };
            // AMD (strix-hip): the grouped GEMMs are mma.sync / block-scaled MMA PTX (stubs
            // there); the grouped GEMVs take any row count.
            if !cfg!(atlas_scale) && n >= ops::MOE_GROUPED_MIN_ROWS {
                ops::moe_prefill(gpu, &self.k, &args, stream)?;
            } else {
                ops::moe_batched(gpu, &self.k, &args, stream)?;
            }
        } else {
            ops::moe_decode(
                gpu,
                &self.k,
                &MoeDecode {
                    x: s.mlp_normed,
                    experts: w.experts,
                    expert_ids: s.ids,
                    weights: s.weights,
                    act: s.act,
                    y: s.y,
                    out: s.moe_out,
                    rows: n,
                    top_k: d.top_k,
                    hidden: h,
                    inter: d.inter,
                    out_stride: h,
                    gate_softcap: p.gate_softcap,
                    up_softcap: p.up_softcap,
                },
                stream,
            )?;
        }
        Ok(())
    }
}
