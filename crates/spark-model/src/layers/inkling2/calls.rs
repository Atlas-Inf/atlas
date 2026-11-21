// SPDX-License-Identifier: AGPL-3.0-only

//! The Inkling 2 kernel calls a visit's forward makes over its `n` rows.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::layer::Inkling2Layer;
use super::ops::{self, BlockRes, HeadNorm, HeadNormArgs, MsgGate, ShortConv};
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
        let s = &self.shared;
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
                row_seq: s.row_seq,
                seq_start: s.seq_start,
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
            self.k.block_res,
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
                eps: 1e-6,
            },
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn msg_gate(
        &self,
        gpu: &dyn GpuBackend,
        n: usize,
        which: usize,
        h: DevicePtr,
        m: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let g = self
            .w
            .messages
            .context("inkling2: message gate on a layer outside the loop")?[which];
        ops::msg_gate(
            gpu,
            self.k.msg_gate,
            &MsgGate {
                h,
                m,
                down: g.down,
                forget: g.forget,
                gain: g.gain,
                out,
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
        log_scaled: bool,
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
                positions: if log_scaled {
                    self.shared.positions
                } else {
                    DevicePtr::NULL
                },
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
}
