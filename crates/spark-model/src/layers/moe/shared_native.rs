// SPDX-License-Identifier: AGPL-3.0-only

//! GLM shared experts on FP4-Local's own FP8 block bytes
//! (`ATLAS_GLM_FP4L_SHARED_FP8`, loaded by `weight_loader::glm5::components`).
//!
//! A layer with a native shared expert keeps only the checkpoint's E4M3
//! gate / up `[inter, h]` and down `[h, inter]` with their f32 128 x 128
//! grids; its runtime NVFP4 copies, their transposed twins and the
//! `ATLAS_GLM_TARGET_SHARED_FP8` E4M3 predecode are never kept. Every MoE call
//! of the layer runs the grouped `forward_prefill` (as expert TP does: the
//! fused routed+shared decode kernels read the NVFP4 shared slot), whose
//! shared phase lands here:
//!
//! - up to 64 rows (32 a launch): gate + up in one `fp8b_gemv_tc*_multi`
//!   launch, SiLU-mul, down on `fp8b_gemv_tc*`: exact weights, row bits
//!   independent of the launch width (decode == verify row);
//! - `ATLAS_GLM_SHARED_TP_SPLIT=1` (replicated rows, EP2 or EP4, up to 64
//!   rows): each of the `ranks` ranks runs intermediate columns
//!   `[rank * inter/ranks, +inter/ranks)`: gate / up rows of the same twins (a
//!   pointer offset) and down's K-slice on the strided `fp8b_gemv_tc*_ld`
//!   tiers; the EP all-reduce sums the parts, as production's NVFP4 split
//!   does;
//! - longer chunks: the `ATLAS_GLM_FP4L_FP8_PREFILL` GEMM (`w8a8`, or the exact
//!   `w8a16` for `w8a16` and `bf16`: a shared expert has no BF16 copy).
//!
//! Decode bytes per rank per layer: 25.2 M params x 1.0002 B (12.6 MB under
//! the pair's split, 6.3 MB the chain's) against NVFP4's x 0.5625 (7.1 MB
//! the pair's split). Resident: 25.2 MB a
//! layer against production's NVFP4 + transposed twin + E4M3 cache (~53.5 MB).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::layers::ops::{Fp8bPrefill, MxPart, native_kernels};
use crate::weight_map::{ExpertWeight, FP8_BLOCK, Fp8BlockWeight};

/// Rows up to which the shared expert runs the GEMV tiers.
pub(crate) const SHARED_NATIVE_GEMV_ROWS: u32 = 64;

/// A layer's native shared expert: the checkpoint's FP8 block projections.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SharedNative {
    /// `[inter, h]`.
    pub gate: Fp8BlockWeight,
    /// `[inter, h]`.
    pub up: Fp8BlockWeight,
    /// `[h, inter]`.
    pub down: Fp8BlockWeight,
    pub prefill: Fp8bPrefill,
}

/// One rank's part of a native shared expert: gate / up rows
/// `[col0, col0 + inter/ranks)` (twins of their own) and down's matching
/// K-slice (values and grid at the slice's first column; rows `inter` apart).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SharedSplit {
    pub gate: Fp8BlockWeight,
    pub up: Fp8BlockWeight,
    pub down_data: DevicePtr,
    pub down_scales: DevicePtr,
}

impl SharedNative {
    /// Rank `rank`'s part of `ranks` (`col0 = rank * inter/ranks`, on the
    /// 128 grid): a row range of gate / up is contiguous in values and grid
    /// rows (also in the 16-row value tiling); down's columns start `col0`
    /// values and `col0 / 128` grid entries into each row.
    pub(crate) fn split(&self, rank: usize, ranks: usize, h: usize, inter: usize) -> SharedSplit {
        let col0 = rank * (inter / ranks);
        let rows_of = |w: &Fp8BlockWeight| Fp8BlockWeight {
            data: w.data.offset(col0 * h),
            scales: w.scales.offset((col0 / FP8_BLOCK) * (h / FP8_BLOCK) * 4),
            tiled: w.tiled,
        };
        SharedSplit {
            gate: rows_of(&self.gate),
            up: rows_of(&self.up),
            down_data: self.down.data.offset(col0),
            down_scales: self.down.scales.offset((col0 / FP8_BLOCK) * 4),
        }
    }

    /// The prefill GEMM mode: `w8a8`, else exact `w8a16` (no BF16 copy).
    fn gemm_mode(self) -> Fp8bPrefill {
        match self.prefill {
            Fp8bPrefill::W8a8 => Fp8bPrefill::W8a8,
            _ => Fp8bPrefill::W8a16,
        }
    }

    pub(crate) fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        self.gate.free(gpu)?;
        self.up.free(gpu)?;
        self.down.free(gpu)
    }
}

/// Free a runtime NVFP4 copy's buffers (`weight`, `weight_scale` and any
/// per-row scale2): loader allocations, never store tensors.
fn free_nvfp4(gpu: &dyn GpuBackend, w: &QuantizedWeight) -> Result<()> {
    for p in [w.weight, w.weight_scale, w.weight_scale_2_vec] {
        if !p.is_null() {
            gpu.free(p)?;
        }
    }
    Ok(())
}

impl MoeLayer {
    /// Whether this layer's shared expert is native (`shared_native.rs`).
    pub(crate) fn has_shared_native(&self) -> bool {
        self.shared_native.is_some()
    }

    /// Whether every MoE call of this layer must take the grouped path:
    /// expert TP (sliced routed width) or a native shared expert (no NVFP4
    /// shared slot for the fused decode kernels).
    pub(super) fn grouped_only(&self, ctx: &ForwardContext) -> bool {
        ctx.config.expert_tp || self.shared_native.is_some()
    }

    /// Install the native shared expert and free every copy derived from the
    /// runtime NVFP4 one it replaces: the NVFP4 originals, their transposed
    /// twins (and the unified layout's gate/up receipt over them) and any
    /// E4M3 predecode (`shared_*_fp8`, the `ATLAS_GLM_TARGET_SHARED_FP8` cache
    /// or the FP8 prefill predecode), leaving null weights so no derived
    /// shared path can be taken. Runs at load; the deferred cache then skips
    /// this layer. A checkpoint-native BF16 / FP8 shared expert of another
    /// model is authoritative and refused. The layer must be on legacy B-tile
    /// storage (the resident B-tile arena binds the shared NVFP4 copies).
    pub(crate) fn install_shared_native(
        &mut self,
        gpu: &dyn GpuBackend,
        native: SharedNative,
    ) -> Result<()> {
        self.btile_storage.require_legacy()?;
        ensure!(
            self.separate_shared_expert.is_none() && self.fp8_shared_expert.is_none(),
            "native shared expert: the layer holds a checkpoint-native BF16 / MXFP8 / FP8 shared expert"
        );
        // Native takes precedence over every derived copy.
        self.shared_gate_up_receipt = None;
        for p in [
            self.shared_gate_fp8.take(),
            self.shared_up_fp8.take(),
            self.shared_down_fp8.take(),
        ]
        .into_iter()
        .flatten()
        {
            gpu.free(p)?;
        }
        self.shared_fp8_cache.layer = None;
        self.shared_fp8_cache.verify = false;
        let shared = std::mem::replace(
            &mut self.weights.shared_expert,
            ExpertWeight {
                gate_proj: QuantizedWeight::null(),
                up_proj: QuantizedWeight::null(),
                down_proj: QuantizedWeight::null(),
            },
        );
        for w in [&shared.gate_proj, &shared.up_proj, &shared.down_proj] {
            free_nvfp4(gpu, w)?;
        }
        for t in [
            self.shared_gate_t.take(),
            self.shared_up_t.take(),
            self.shared_down_t.take(),
        ]
        .into_iter()
        .flatten()
        {
            free_nvfp4(gpu, &t)?;
        }
        self.shared_native = Some(native);
        Ok(())
    }

    /// Whether the native shared expert runs TP-split for `rows` replicated
    /// rows (`ATLAS_GLM_SHARED_TP_SPLIT=1` over the pair or the chain, parts
    /// on the 128 grid).
    pub(super) fn shared_native_split_ready(
        &self,
        ctx: &ForwardContext,
        rows: u32,
        inter: u32,
    ) -> bool {
        self.shared_native.is_some()
            && super::forward_pair_shared::shared_split_ranks(ctx)
                .is_some_and(|ranks| (inter as usize).is_multiple_of(ranks * FP8_BLOCK))
            && rows <= SHARED_NATIVE_GEMV_ROWS
    }

    /// The whole shared expert over `n` rows of `input` (`[n, h]`) into
    /// `attn_output`, on `stream`.
    pub(super) fn run_shared_native(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let native = self
            .shared_native
            .expect("run_shared_native without a native shared expert");
        let (gate_out, up_out, down_out) = (
            ctx.buffers.ssm_deinterleaved(),
            ctx.buffers.ssm_qkvz(),
            ctx.buffers.attn_output(),
        );
        let (n, h, inter) = (n as usize, h as usize, inter as usize);
        let kernels = native_kernels(ctx.gpu);
        let gemv = n <= SHARED_NATIVE_GEMV_ROWS as usize;
        if gemv {
            ops::mx_multi_rows(
                ctx.gpu,
                &kernels.fp8b_multi,
                false,
                input,
                &[
                    MxPart::fp8b(&native.gate, gate_out, inter),
                    MxPart::fp8b(&native.up, up_out, inter),
                ],
                n,
                h,
                stream,
            )?;
        } else {
            let mode = native.gemm_mode();
            for (w, out) in [(&native.gate, gate_out), (&native.up, up_out)] {
                ops::fp8b_prefill_gemm(
                    ctx.gpu,
                    &kernels,
                    ops::Fp8ActScratch::of(ctx.buffers),
                    mode,
                    w,
                    input,
                    out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
            }
        }
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            (n * inter) as u32,
            stream,
        )?;
        if gemv {
            ops::fp8b_gemv_rows(
                ctx.gpu,
                &kernels.fp8b,
                &native.down,
                gate_out,
                down_out,
                n,
                h,
                inter,
                h,
                stream,
            )
        } else {
            ops::fp8b_prefill_gemm(
                ctx.gpu,
                &kernels,
                ops::Fp8ActScratch::of(ctx.buffers),
                native.gemm_mode(),
                &native.down,
                gate_out,
                down_out,
                n,
                h,
                inter,
                stream,
            )
        }
    }

    /// This rank's part of the native shared expert (see the module doc):
    /// `attn_output` then holds this rank's partial; the EP all-reduce sums
    /// the ranks'.
    pub(super) fn run_shared_native_split(
        &self,
        input: DevicePtr,
        rows: u32,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let native = self
            .shared_native
            .expect("run_shared_native_split without a native shared expert");
        let (rows, h, inter) = (rows as usize, h as usize, inter as usize);
        let ranks = ctx.config.ep_world_size;
        let part = inter / ranks;
        let split = native.split(ctx.config.ep_rank, ranks, h, inter);
        let (gate, up) = (split.gate, split.up);
        let (gate_out, up_out, down_out) = (
            ctx.buffers.ssm_deinterleaved(),
            ctx.buffers.ssm_qkvz(),
            ctx.buffers.attn_output(),
        );
        let kernels = native_kernels(ctx.gpu);
        ops::mx_multi_rows(
            ctx.gpu,
            &kernels.fp8b_multi,
            native.gate.tiled,
            input,
            &[
                MxPart::fp8b(&gate, gate_out, part),
                MxPart::fp8b(&up, up_out, part),
            ],
            rows,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            (rows * part) as u32,
            stream,
        )?;
        ensure!(
            !native.down.tiled,
            "the split shared down reads a row-major twin"
        );
        ops::fp8b_gemv_ld_rows(
            ctx.gpu,
            &kernels.fp8b_ld,
            split.down_data,
            split.down_scales,
            gate_out,
            down_out,
            rows,
            h,
            part,
            inter,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "shared_native_tests.rs"]
mod tests;
