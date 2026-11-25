// SPDX-License-Identifier: AGPL-3.0-only

//! The Inkling 2 kernel calls a visit's forward makes over its `n` rows.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::layer::Inkling2Layer;
use super::ops::{self, BlockRes, HeadNorm, HeadNormArgs, MoeDecode, MsgGate, ShortConv};
use super::weights::KvConv;

/// Optional folds for the AMD one-launch block residual (`block_res_fused`).
/// `DevicePtr::NULL` disables each piece; `resid.3` is the boundary restart.
#[derive(Clone, Copy)]
pub(super) struct BlockResFold {
    /// `(x, c, resid_out, restart)` — an `inkling2_residual` to run first.
    pub resid: Option<(DevicePtr, DevicePtr, DevicePtr, bool)>,
    /// bf16 destination of the block push after `resid` (row 0's `x`).
    pub push_dst: DevicePtr,
    /// Up to 3 rmsnorm weight pointers to run on this row's `out`.
    pub norm_ws: [DevicePtr; 3],
    pub n_norm: usize,
    /// Fp32 destination for the norms: row `q`'s at `q * hidden`.
    pub normed: DevicePtr,
    /// `p.eps` for the folded norm (the block residual's own eps is `BlockRes::eps`).
    pub norm_eps: f32,
}

impl BlockResFold {
    /// No folds: just the one-launch sums + out.
    pub(super) fn none() -> Self {
        Self::norms(&[], DevicePtr::NULL, 0.0)
    }

    /// Only the rmsnorms fold (empty `ws` = none).
    pub(super) fn norms(ws: &[DevicePtr], normed: DevicePtr, eps: f32) -> Self {
        let w = |i: usize| if i < ws.len() { ws[i] } else { DevicePtr::NULL };
        Self {
            resid: None,
            push_dst: DevicePtr::NULL,
            norm_ws: [w(0), w(1), w(2)],
            n_norm: ws.len(),
            normed,
            norm_eps: eps,
        }
    }

    /// The MLP site: residual + optional boundary push + optional norm fold.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resid(
        x: DevicePtr,
        c: DevicePtr,
        resid_out: DevicePtr,
        restart: bool,
        push_dst: DevicePtr,
        ws: &[DevicePtr],
        normed: DevicePtr,
        eps: f32,
    ) -> Self {
        let mut f = Self::norms(ws, normed, eps);
        f.resid = Some((x, c, resid_out, restart));
        f.push_dst = push_dst;
        f
    }
}

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

    /// The AMD one-launch block residual (`block_res_fused`), same argument
    /// shape as `block_res` plus the optional folds (`resid`, `push_dst`, the
    /// rmsnorms that follow). Bit-identical to the launches it replaces.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_res_fused(
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
        fold: &BlockResFold,
        stream: u64,
    ) -> Result<()> {
        ops::block_res_fused(
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
            fold.resid,
            fold.push_dst,
            fold.norm_ws,
            fold.n_norm,
            fold.normed,
            fold.norm_eps,
            stream,
        )
    }

    /// The MLP-input block residual on AMD: the elementwise residual, the
    /// boundary visit's block push and (unless `norm` is off — gated visits)
    /// the rmsnorm that follows, all inside the one-launch kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_res_mlp_fused(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        nb: usize,
        x: DevicePtr,
        conv_out: DevicePtr,
        boundary: bool,
        norm: bool,
        stream: u64,
    ) -> Result<()> {
        let (s, w) = (&self.shared, &self.w);
        let mlp_norm_ws = [w.mlp_norm];
        self.block_res_fused(
            gpu,
            n,
            s.blocks,
            nb,
            s.resid,
            DevicePtr::NULL,
            w.mlp_query,
            1,
            s.mlp_in,
            &BlockResFold::resid(
                x,
                conv_out,
                s.resid,
                boundary,
                if boundary {
                    s.blocks.offset((nb - 1) * self.dims.block_stride() * 2)
                } else {
                    DevicePtr::NULL
                },
                if norm { &mlp_norm_ws } else { &[] },
                s.mlp_normed,
                self.p.eps,
            ),
            stream,
        )
    }

    /// Message gates `which` (consecutive, at most three) of this loop layer;
    /// strides in fp32 elements between the gates' `h`, `m` and `out` rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn msg_gates(
        &self,
        gpu: &dyn GpuBackend,
        (n, exact): (usize, bool),
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
                decode_rows: exact,
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
    /// for prefill-sized row counts, the decode GEMVs otherwise and for every
    /// verify run (`exact`: the decode numerics, BF16 expert inputs).
    pub(super) fn routed_experts(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        exact: bool,
        stream: u64,
    ) -> Result<()> {
        let (s, w, d, p, h) = (&self.shared, &self.w, &self.dims, &self.p, self.dims.hidden);
        if n > 1 {
            let grouped = !exact && n >= ops::MOE_GROUPED_MIN_ROWS;
            let args = ops::MoePrefill {
                x: s.mlp_normed,
                experts: if grouped {
                    w.prefill_experts
                } else {
                    w.experts
                },
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
            if grouped {
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
