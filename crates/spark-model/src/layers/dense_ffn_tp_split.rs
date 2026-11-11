// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_DENSE_TP_SPLIT=1` (default off): the GLM dense FFN of a
//! canonical verify block (layers 0..=2, replicated rows) split by output
//! columns across the TP ranks (the pair, or the four-rank chain), joined by
//! two exchanges (on the chain, two gathers).
//!
//! Unsplit, every rank reads all three 28 MB NVFP4 projections per layer
//! (about 0.45 ms a layer at a three-row verify), or all three of FP4-Local's
//! 50 MB MXFP8 ones (`dense_ffn_native.rs`, about 0.73 ms). Split over `n`
//! ranks, rank
//! `r` runs gate and up over intermediate columns `[r * inter/n, +inter/n)`,
//! the SiLU on its part, joins the parts, then down over hidden columns
//! `[r * h/n, +h/n)` with the whole intermediate, and joins those. A
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
    /// false (nothing launched) when it does not apply.
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
        let ranks = comm.world_size();
        let part = (rows * inter) as usize / ranks.max(1) * 2;
        if !dense_tp_split_requested()?
            || ctx.graph_capture
            || crate::layers::glm_sp::current().is_some()
            || !proj.row_major()
            || !matches!(ranks, 2 | 4)
            || !inter.is_multiple_of(16 * ranks as u32)
            || !h.is_multiple_of(16 * ranks as u32)
            || !if ranks == 2 {
                comm.supports_exchange_async(part)
            } else {
                comm.supports_gather_async(part)
            }
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

/// One rank's part of a dense FFN `[inter, h]` / `[h, inter]`.
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
    /// inter` BF16): the ranks' gate (then gated activation) parts, later
    /// their down parts; this rank's goes first on the pair (the peer's
    /// after it), at its rank's place on a chain. `u` (as large): this
    /// rank's up part, then the joined activation `[rows, inter]`.
    pub(super) fn run(
        &self,
        input: DevicePtr,
        rows: u32,
        [g, u, out]: [DevicePtr; 3],
        stream: u64,
    ) -> Result<()> {
        let (h, inter, gpu) = (self.h, self.inter, self.gpu);
        let ranks = self.comm.world_size() as u32;
        let (ih, hh, r) = (inter / ranks, h / ranks, self.comm.rank() as u32);
        let own = |cols: u32| {
            let at = if ranks == 2 { 0 } else { r * rows * cols };
            g.offset(at as usize * 2)
        };
        self.gate_up(input, rows, r * ih, ih, [own(ih), u], stream)?;
        ops::silu_mul(gpu, self.act_mul, own(ih), u, own(ih), rows * ih, stream)?;
        self.join(g, [rows, inter], u, stream)?;
        self.down(u, rows, r * hh, hh, own(hh), stream)?;
        self.join(g, [rows, h], out, stream)
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

    /// Join the ranks' `[rows, n/ranks]` parts in `parts` (this rank's set,
    /// as `run` places it) into the `[rows, n]` rows at `dst`, rank 0's
    /// columns first: a pair swap or a chain gather, then one join launch
    /// (else a pitched copy per part).
    fn join(
        &self,
        parts: DevicePtr,
        [rows, n]: [u32; 2],
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let ranks = self.comm.world_size();
        if ranks == 2 {
            return self.swap_join(parts, [rows, n], dst, stream);
        }
        let (rows, part) = (rows as usize, n as usize / ranks);
        let bytes = rows * part * 2;
        anyhow::ensure!(
            self.comm
                .gather_async(parts.0, bytes as i64, bytes, stream)?,
            "dense TP split: chain gather refused {bytes} bytes"
        );
        // Equal parts in whole 16s: `rank_join_bf16`'s cuts.
        let kernel = super::ops::rank_join_kernel(self.gpu);
        if kernel.0 != 0 {
            return super::ops::rank_join_bf16(
                self.gpu,
                kernel,
                dst,
                parts,
                bytes,
                [rows, n as usize],
                ranks,
                stream,
            );
        }
        for r in 0..ranks {
            self.gpu.copy_d2d_2d_async(
                parts.offset(r * bytes),
                part * 2,
                dst.offset(r * part * 2),
                n as usize * 2,
                part * 2,
                rows,
                stream,
            )?;
        }
        Ok(())
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
