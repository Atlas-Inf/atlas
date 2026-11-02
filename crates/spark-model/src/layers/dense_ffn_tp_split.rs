// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_DENSE_TP_SPLIT=1` (default off): the GLM dense FFN of a
//! canonical verify block (layers 0..=2, replicated rows) split by output
//! columns across the pair, joined by two exchanges.
//!
//! Unsplit, both ranks read all three 28 MB NVFP4 projections per layer
//! (about 0.45 ms a layer at a three-row verify), or all three of
//! FP4-Local's 50 MB MXFP8 ones (`dense_ffn_native.rs`). Split, rank `r` runs gate
//! and up over intermediate columns `[r * inter/2, +inter/2)`, the SiLU on
//! its half, swaps halves with the peer, then down over hidden columns
//! `[r * h/2, +h/2)` with the whole intermediate, and swaps those. A
//! tensor-core tier's output column depends only on its own weight row and
//! the activation (`canonical_verify`), and the exchanges copy, so every
//! column, and the joined `[rows, h]` output, is the unsplit FFN's word.
//!
//! Both ranks take the same lane: the switch is in the startup agreement,
//! and the rest (the rows, graph capture, the pair's exchange capacity) is
//! the same on both.

use super::*;
use crate::weight_map::Mxfp8Weight;

/// `ATLAS_GLM_DENSE_TP_SPLIT` as set: `0` (or unset) or `1`. Read once.
pub(crate) fn dense_tp_split_requested() -> Result<bool> {
    static ON: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(
        || match std::env::var("ATLAS_GLM_DENSE_TP_SPLIT").as_deref() {
            Err(_) | Ok("0") => Ok(false),
            Ok("1") => Ok(true),
            Ok(value) => Err(format!(
                "ATLAS_GLM_DENSE_TP_SPLIT requires 0 or 1, got {value:?}"
            )),
        },
    )
    .clone()
    .map_err(anyhow::Error::msg)
}

/// Rows `[first, first + n)` of an NVFP4 `[_, k]` weight: packed bytes and
/// group scales are row-major, the tensor scale is the weight's.
fn rows_from(w: &QuantizedWeight, first: u32, k: u32) -> QuantizedWeight {
    let first = first as usize;
    QuantizedWeight {
        weight: w.weight.offset(first * k as usize / 2),
        weight_scale: w.weight_scale.offset(first * k as usize / 16),
        ..*w
    }
}

/// Rows `[first, first + n)` of a row-major MXFP8 `[_, k]` weight: E4M3
/// bytes and E8M0 group scales are row-major.
fn mx_rows_from(w: &Mxfp8Weight, first: u32, k: u32) -> Mxfp8Weight {
    let first = first as usize;
    Mxfp8Weight {
        data: w.data.offset(first * k as usize),
        scales: w.scales.offset(first * k as usize / 32),
        ..*w
    }
}

impl DenseFfnLayer {
    /// The split FFN of `rows` verify rows on `proj` into `moe_output`, or
    /// false (nothing launched) when it does not apply (or under SP prefill).
    pub(super) fn glm_verify_rows_split(
        &self,
        input: DevicePtr,
        rows: u32,
        proj: SplitProj<'_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let Some(comm) = ctx.comm else {
            return Ok(false);
        };
        if !dense_tp_split_requested()?
            || ctx.graph_capture
            || crate::layers::glm_sp::current().is_some()
            || !proj.row_major()
            || comm.world_size() != 2
            || !inter.is_multiple_of(32)
            || !h.is_multiple_of(32)
            || !comm.supports_exchange_async((rows * inter / 2) as usize * 2)
        {
            return Ok(false);
        }
        let split = SplitFfn {
            gpu: ctx.gpu,
            comm,
            proj,
            act_mul: self.act_mul,
            h,
            inter,
        };
        let b = &ctx.buffers;
        split.run(
            input,
            rows,
            [b.expert_gate_out(), b.expert_up_out(), b.moe_output()],
            stream,
        )?;
        Ok(true)
    }
}

/// The projections a split runs: the NVFP4 weights on the W4A16
/// tensor-core tier `tc`, or the native MXFP8 ones (`dense_ffn_native.rs`).
#[derive(Clone, Copy)]
pub(super) enum SplitProj<'a> {
    Nvfp4 {
        weights: &'a DenseFfnWeights,
        tc: KernelHandle,
    },
    Mxfp8(&'a DenseNative),
}

impl SplitProj<'_> {
    /// Whether a rank's rows are a contiguous slice (not MXFP8's tiled layout).
    fn row_major(self) -> bool {
        match self {
            Self::Nvfp4 { .. } => true,
            Self::Mxfp8(n) => [n.gate, n.up, n.down].iter().all(|w| !w.tiled),
        }
    }
}

/// One rank's half of a dense FFN `[inter, h]` / `[h, inter]`.
pub(super) struct SplitFfn<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub comm: &'a dyn spark_comm::CommBackend,
    pub proj: SplitProj<'a>,
    pub act_mul: KernelHandle,
    pub h: u32,
    pub inter: u32,
}

impl SplitFfn<'_> {
    /// `[rows, h]` FFN rows of `input` into `out`. Scratch `g` (`rows *
    /// inter` BF16): this rank's gate (then gated activation) half, the
    /// peer's half after it, later the two down halves. `u` (as large): this
    /// rank's up half, then the joined activation `[rows, inter]`.
    pub(super) fn run(
        &self,
        input: DevicePtr,
        rows: u32,
        [g, u, out]: [DevicePtr; 3],
        stream: u64,
    ) -> Result<()> {
        let (h, inter, gpu) = (self.h, self.inter, self.gpu);
        let (ih, hh, r) = (inter / 2, h / 2, self.comm.rank() as u32);
        self.gate_up(input, rows, r * ih, ih, [g, u], stream)?;
        ops::silu_mul(gpu, self.act_mul, g, u, g, rows * ih, stream)?;
        self.swap_join(g, [rows, inter], u, stream)?;
        self.down(u, rows, r * hh, hh, g, stream)?;
        self.swap_join(g, [rows, h], out, stream)
    }

    /// Gate and up rows `[first, first + n)` of `rows` rows of `input` into
    /// `[gate, up]` (`[rows, n]` each).
    fn gate_up(
        &self,
        input: DevicePtr,
        rows: u32,
        first: u32,
        n: u32,
        [gate_out, up_out]: [DevicePtr; 2],
        stream: u64,
    ) -> Result<()> {
        let (gpu, h) = (self.gpu, self.h);
        match self.proj {
            SplitProj::Nvfp4 { weights: w, tc } => {
                let (gate, up) = (
                    rows_from(&w.gate_proj, first, h),
                    rows_from(&w.up_proj, first, h),
                );
                let gate_up = [(&gate, gate_out), (&up, up_out)];
                match ops::w4a16_pair_touch(gpu, tc, input, gate_up, rows, n, h, stream) {
                    Some(done) => done,
                    None => {
                        for (weight, o) in gate_up {
                            ops::w4a16_gemv_batchm(gpu, tc, input, weight, o, rows, n, h, stream)?;
                        }
                        Ok(())
                    }
                }
            }
            SplitProj::Mxfp8(native) => {
                let (gate, up) = (
                    mx_rows_from(&native.gate, first, h),
                    mx_rows_from(&native.up, first, h),
                );
                let parts = [
                    ops::MxPart::mx(&gate, gate_out, n as usize),
                    ops::MxPart::mx(&up, up_out, n as usize),
                ];
                let tiers = &ops::native_kernels(gpu).mx_multi;
                ops::mx_multi_rows(
                    gpu,
                    tiers,
                    false,
                    input,
                    &parts,
                    rows as usize,
                    h as usize,
                    stream,
                )
            }
        }
    }

    /// Down rows `[first, first + n)` of `rows` activation rows `[rows,
    /// inter]` at `input` into `out` (`[rows, n]`).
    fn down(
        &self,
        input: DevicePtr,
        rows: u32,
        first: u32,
        n: u32,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (gpu, inter) = (self.gpu, self.inter);
        match self.proj {
            SplitProj::Nvfp4 { weights: w, .. } => {
                let down = rows_from(&w.down_proj, first, inter);
                crate::layers::canonical_verify::w4a16(
                    gpu, input, &down, out, rows, n, inter, stream,
                )
            }
            SplitProj::Mxfp8(native) => {
                let down = mx_rows_from(&native.down, first, inter);
                let tiers = &ops::native_kernels(gpu).mx.row_major;
                let (rows, n) = (rows as usize, n as usize);
                ops::mxfp8_gemv_rows_touched(
                    gpu,
                    tiers,
                    &down,
                    input,
                    out,
                    rows,
                    n,
                    inter as usize,
                    n,
                    stream,
                )
            }
        }
    }

    /// Send this rank's `[rows, n/2]` half at `own`, land the peer's right
    /// after it, and write the `[rows, n]` rows at `dst`, rank 0's columns
    /// first: two pitched copies.
    fn swap_join(
        &self,
        own: DevicePtr,
        [rows, n]: [u32; 2],
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (rows, half) = (rows as usize, n as usize / 2);
        let peer = own.offset(rows * half * 2);
        anyhow::ensure!(
            self.comm
                .exchange_async(own.0, peer.0, rows * half * 2, false, stream)?,
            "dense TP split: pair exchange refused {} bytes",
            rows * half * 2
        );
        let rank = self.comm.rank();
        for (src, r) in [(own, rank), (peer, 1 - rank)] {
            self.gpu.copy_d2d_2d_async(
                src,
                half * 2,
                dst.offset(r * half * 2),
                n as usize * 2,
                half * 2,
                rows,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "dense_ffn_tp_split_test_gpu.rs"]
mod gpu_tests;
