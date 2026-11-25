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

    /// The fused attention prologue (`inkling2_attn_prologue`, AMD only): the
    /// k/v convs, the four head norms and the KV write in one launch (verify
    /// rows keep their `sconv_state` rolls inside it).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_prologue(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        seqs: usize,
        states: DevicePtr,
        kvs: usize,
        log_pos: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        slots: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (s, w, p) = (&self.shared, &self.w, &self.p);
        let taps = |conv: KvConv, dyn_taps: DevicePtr| match conv {
            KvConv::Static { weight } => (weight, DevicePtr::NULL),
            KvConv::Dynamic { static_w, .. } => (static_w, dyn_taps),
        };
        let (k_w, k_dyn) = taps(w.k_conv, s.dyn_k);
        let (v_w, v_dyn) = taps(w.v_conv, s.dyn_v);
        ops::attn_prologue(
            gpu,
            &self.k,
            &ops::AttnPrologue {
                k_lin: s.k_lin,
                v_lin: s.v_lin,
                q_in: s.q,
                r_in: s.r,
                kc: s.kc,
                vc: s.vc,
                qn: s.qn,
                kn: s.kn,
                vn: s.vn,
                rn: s.rn,
                states,
                state_off_k: 0,
                state_off_v: kvs,
                k_w,
                v_w,
                k_dyn,
                v_dyn,
                q_scale: w.q_scale,
                rel_scale: w.rel_scale,
                positions: log_pos,
                k_cache: k_pool,
                v_cache: v_pool,
                slots,
                rows: n,
                rows_per_seq: n / seqs,
                num_seqs: seqs,
                heads: p.heads,
                kv_heads: p.kv_heads,
                head_dim: p.head_dim,
                d_rel: p.d_rel,
                conv_k: p.conv_k,
                dyn_stride: p.kv_heads * p.conv_k,
                qk_clip: p.qk_clip,
                rel_clip: p.rel_clip,
                eps: p.eps,
                log_n_floor: p.log_n_floor,
                log_alpha: p.log_alpha,
            },
            stream,
        )
    }

    /// The residual + boundary push folded into the MLP block-residual sums
    /// (`inkling2_block_res_sums_resid`, AMD only), then `_out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_res_mlp(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        nb: usize,
        x: DevicePtr,
        conv_out: DevicePtr,
        resid: DevicePtr,
        out: DevicePtr,
        push: bool,
        stream: u64,
    ) -> Result<()> {
        let (w, d, s) = (&self.w, &self.dims, &self.shared);
        ops::block_res_mlp(
            gpu,
            &self.k,
            &ops::BlockRes {
                blocks: s.blocks,
                partial_f32: DevicePtr::NULL,
                partial_bf16: DevicePtr::NULL,
                queries: w.mlp_query,
                out,
                rows: n,
                hidden: d.hidden,
                num_blocks: nb,
                row_stride: d.hidden,
                block_stride: d.block_stride(),
                num_queries: 1,
                sums: s.res_sums,
                eps: 1e-6,
            },
            x,
            conv_out,
            resid,
            push,
            stream,
        )
    }

    /// `inkling2_mlp_tail` (AMD decode rows): combine (single row only) + the
    /// post-MLP conv + layer_out. `combine` says `y` still holds per-slot
    /// expert outputs to combine; otherwise `moe_out` is already final.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mlp_tail(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        states: DevicePtr,
        state_off: usize,
        combine: bool,
        hidden: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (s, w, d, p) = (&self.shared, &self.w, &self.dims, &self.p);
        ops::mlp_tail(
            gpu,
            &self.k,
            &ops::MlpTail {
                y: if combine { s.y } else { DevicePtr::NULL },
                weights: if combine { s.weights } else { DevicePtr::NULL },
                moe_out: s.moe_out,
                mlp_conv: s.mlp_conv,
                states,
                state_off,
                conv_w: w.mlp_conv,
                resid: s.resid,
                hidden,
                rows: n,
                hidden_dim: d.hidden,
                top_k: if combine { d.top_k } else { 0 },
                conv_k: p.conv_k,
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
    /// `combine = false` (AMD fused MLP tail): the decode-GEMV path leaves `y`
    /// per expert slot for `inkling2_mlp_tail`'s own combine.
    pub(super) fn routed_experts(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        exact: bool,
        combine: bool,
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
                    combine,
                },
                stream,
            )?;
        }
        Ok(())
    }
}
