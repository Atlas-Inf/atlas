// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-only native EXL3 overlay for a single packed linear (the LM head):
//! the 1- and 2-row forwards run the int8 sq GEMV on the PACKED EXL3 weight,
//! so decode and MTP verify never touch the BF16 materialized copy. Anything
//! wider (prompt scoring, wide verify) keeps the BF16 weight. Installed by the
//! factory under `ATLAS_EXL3_NATIVE_DECODE=1` (see `weight_map::exl3::
//! materialize`); the pattern mirrors `Exl3GdnDecode`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{
    Exl3Int8Kernels, Exl3Int8Workspace, Exl3Kernels, exl3_convert, exl3_int8_gemv,
    exl3_int8_linear_bf16_rows, sq_grid, sq_plan,
};
use crate::weight_map::exl3::Exl3Weight;

/// Scratch alignment: allocate halves as bytes with room for an offset.
const ALIGN: usize = 256;

/// One packed EXL3 linear + the int8 sq machinery it runs through.
pub struct Exl3LinearDecode {
    w: Exl3Weight,
    k8: Exl3Int8Kernels,
    k: Exl3Kernels,
    ws: Exl3Int8Workspace,
    x_f16: DevicePtr,
    a_had: DevicePtr,
    c_f32: DevicePtr,
    grid: u32,
}

impl Exl3LinearDecode {
    /// Workspace/scratch are sized for m = 2 so the MTP-verify (2-row) call
    /// needs no rebuild.
    pub fn new(gpu: &dyn GpuBackend, w: Exl3Weight) -> Result<Self> {
        let k8 = Exl3Int8Kernels::resolve(gpu)?;
        let k = Exl3Kernels::resolve(gpu)?;
        let grid = sq_grid(gpu.sm_count()?);
        let plan = sq_plan(
            w.shape.in_features,
            w.shape.out_features,
            2,
            w.shape.bits,
            grid,
        )?;
        let ws = Exl3Int8Workspace::new(gpu, plan.ws_ints)?;
        let (max_in, max_out) = (w.shape.in_features, w.shape.out_features);
        let x_f16 = gpu.alloc(2 * max_in * 2 + ALIGN)?;
        let a_had = gpu.alloc(2 * max_in * 2 + ALIGN)?;
        let c_f32 = gpu.alloc(2 * max_out * 4 + ALIGN)?;
        Ok(Self {
            w,
            k8,
            k,
            ws,
            x_f16,
            a_had,
            c_f32,
            grid,
        })
    }

    /// Rows of the packed weight (the logits width).
    pub fn out_features(&self) -> usize {
        self.w.shape.out_features
    }

    /// `y[m, ..] = x[m, k] · W` into BF16 rows `out_stride` elements apart.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_bf16(
        &self,
        gpu: &dyn GpuBackend,
        x_bf16: DevicePtr,
        m: u32,
        out_bf16: DevicePtr,
        out_stride: usize,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            m == 1 || m == 2,
            "EXL3 native decode: m = {m} (want 1 or 2)"
        );
        exl3_int8_linear_bf16_rows(
            gpu, &self.k8, &self.k, &self.ws, x_bf16, m, &self.w, self.x_f16, self.a_had,
            self.c_f32, out_bf16, out_stride, self.grid, stream,
        )
    }

    /// Same, but the fp32 accumulator is written straight to `out_f32`
    /// (contiguous `[m, n]`) — the FP32-logits decode path.
    pub fn forward_f32(
        &self,
        gpu: &dyn GpuBackend,
        x_bf16: DevicePtr,
        m: u32,
        out_f32: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            m == 1 || m == 2,
            "EXL3 native decode: m = {m} (want 1 or 2)"
        );
        let kdim = self.w.shape.in_features;
        exl3_convert(
            gpu,
            self.k.bf16_to_f16,
            x_bf16,
            self.x_f16,
            (m as usize * kdim) as u32,
            stream,
        )?;
        exl3_int8_gemv(
            gpu, &self.k8, &self.ws, self.x_f16, m, &self.w, self.a_had, out_f32, self.grid, stream,
        )
    }
}
