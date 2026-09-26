// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-only native EXL3 overlay for GDN layers: the single-token
//! in_proj_qkv / in_proj_z / out_proj projections run the int8 sq GEMV on the
//! PACKED EXL3 weights (int8 activations), so decode never touches the BF16
//! materialized copies. PREFILL is unchanged and keeps the BF16 weights.
//! Installed by `build_linear_attention_dense_bf16` under
//! `ATLAS_EXL3_NATIVE_DECODE=1` (see `weight_map::exl3::materialize`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{
    Exl3Int8Kernels, Exl3Int8Workspace, Exl3Kernels, exl3_int8_linear_bf16,
    exl3_int8_linear_bf16_rows, sq_grid, sq_plan,
};
use crate::weight_map::exl3::Exl3Weight;

use super::Qwen3SsmLayer;

/// Scratch alignment: allocate halves as bytes with room for an offset.
const ALIGN: usize = 256;

/// The three packed GDN projections + the int8 sq machinery they share.
pub struct Exl3GdnDecode {
    qkv: Exl3Weight,
    z: Exl3Weight,
    out: Exl3Weight,
    k8: Exl3Int8Kernels,
    k: Exl3Kernels,
    ws: Exl3Int8Workspace,
    x_f16: DevicePtr,
    a_had: DevicePtr,
    c_f32: DevicePtr,
    grid: u32,
}

impl Exl3GdnDecode {
    /// Workspace/scratch are sized for m = 2 so M6d (batched verify) can
    /// reuse this overlay without a rebuild.
    pub fn new(
        gpu: &dyn GpuBackend,
        qkv: Exl3Weight,
        z: Exl3Weight,
        out: Exl3Weight,
    ) -> Result<Self> {
        anyhow::ensure!(
            qkv.shape.in_features == z.shape.in_features
                && out.shape.out_features == qkv.shape.in_features,
            "EXL3 native decode: shape mismatch (qkv {}x{}, z {}x{}, out {}x{})",
            qkv.shape.out_features,
            qkv.shape.in_features,
            z.shape.out_features,
            z.shape.in_features,
            out.shape.out_features,
            out.shape.in_features
        );
        let k8 = Exl3Int8Kernels::resolve(gpu)?;
        let k = Exl3Kernels::resolve(gpu)?;
        let grid = sq_grid(gpu.sm_count()?);
        let mut ws_ints = 0usize;
        let (mut max_in, mut max_out) = (0usize, 0usize);
        for w in [&qkv, &z, &out] {
            let plan = sq_plan(
                w.shape.in_features,
                w.shape.out_features,
                2,
                w.shape.bits,
                grid,
            )?;
            ws_ints = ws_ints.max(plan.ws_ints);
            max_in = max_in.max(w.shape.in_features);
            max_out = max_out.max(w.shape.out_features);
        }
        let ws = Exl3Int8Workspace::new(gpu, ws_ints)?;
        // Two independent m = 2 scratch streams: x_f16 feeds the convert,
        // a_had is the gemv's own scratch, c_f32 the fp32 accumulator.
        let x_f16 = gpu.alloc(2 * max_in * 2 + ALIGN)?;
        let a_had = gpu.alloc(2 * max_in * 2 + ALIGN)?;
        let c_f32 = gpu.alloc(2 * max_out * 4 + ALIGN)?;
        Ok(Self {
            qkv,
            z,
            out,
            k8,
            k,
            ws,
            x_f16,
            a_had,
            c_f32,
            grid,
        })
    }

    /// Total rows of the [Q|K|V|Z] projection (qkv rows then z rows,
    /// sequential — matches the deinterleaved buffer layout).
    pub fn qkvz_rows(&self) -> usize {
        self.qkv.shape.out_features + self.z.shape.out_features
    }

    /// `deinterleaved[0 .. qkv.out] = normed · W_qkv`, then the z rows at
    /// `deinterleaved.offset(qkv.out * 2)` (BF16).
    pub fn qkvz(
        &self,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        deinterleaved: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        exl3_int8_linear_bf16(
            gpu,
            &self.k8,
            &self.k,
            &self.ws,
            normed,
            1,
            &self.qkv,
            self.x_f16,
            self.a_had,
            self.c_f32,
            deinterleaved,
            self.grid,
            stream,
        )?;
        exl3_int8_linear_bf16(
            gpu,
            &self.k8,
            &self.k,
            &self.ws,
            normed,
            1,
            &self.z,
            self.x_f16,
            self.a_had,
            self.c_f32,
            deinterleaved.offset(self.qkv.shape.out_features * 2),
            self.grid,
            stream,
        )
    }

    /// `out[0 .. out.out] = normed_out · W_out` (BF16).
    pub fn out_proj(
        &self,
        gpu: &dyn GpuBackend,
        normed_out: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        exl3_int8_linear_bf16(
            gpu, &self.k8, &self.k, &self.ws, normed_out, 1, &self.out, self.x_f16, self.a_had,
            self.c_f32, out, self.grid, stream,
        )
    }

    /// Batched (MTP-verify) qkvz: `m` contiguous input rows `[m, h]` → rows at
    /// `row_stride` BF16 elements apart in `dst`, qkv rows then z rows at
    /// `dst.offset(qkv.out * 2)` — the m = 1..2 generalization of [`Self::qkvz`]
    /// (row 0 of an m = 2 sq call is bitwise equal to the m = 1 call).
    pub fn qkvz_rows_batched(
        &self,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        dst: DevicePtr,
        m: u32,
        row_stride: usize,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            m == 1 || m == 2,
            "EXL3 native decode: m = {m} (want 1 or 2)"
        );
        exl3_int8_linear_bf16_rows(
            gpu, &self.k8, &self.k, &self.ws, normed, m, &self.qkv, self.x_f16, self.a_had,
            self.c_f32, dst, row_stride, self.grid, stream,
        )?;
        exl3_int8_linear_bf16_rows(
            gpu,
            &self.k8,
            &self.k,
            &self.ws,
            normed,
            m,
            &self.z,
            self.x_f16,
            self.a_had,
            self.c_f32,
            dst.offset(self.qkv.shape.out_features * 2),
            row_stride,
            self.grid,
            stream,
        )
    }

    /// Batched (MTP-verify) out_proj: contiguous `[m, value_dim]` in,
    /// contiguous `[m, h]` out — the m = 1..2 generalization of
    /// [`Self::out_proj`].
    pub fn out_proj_batched(
        &self,
        gpu: &dyn GpuBackend,
        normed_out: DevicePtr,
        out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(m <= 2, "EXL3 native decode: m = {m} (want <= 2)");
        exl3_int8_linear_bf16(
            gpu, &self.k8, &self.k, &self.ws, normed_out, m, &self.out, self.x_f16, self.a_had,
            self.c_f32, out, self.grid, stream,
        )
    }
}

impl Qwen3SsmLayer {
    /// Install the decode-only native EXL3 overlay (mirror of
    /// `set_fp8_decode_weights`): decode runs the packed int8 sq GEMV,
    /// prefill keeps the BF16 weights.
    pub fn set_exl3_decode(&mut self, d: Exl3GdnDecode) {
        self.exl3_decode = Some(Box::new(d));
    }

    /// Install the overlay from the packed GDN projections of `lp` if they
    /// are still in `store` (they only survive materialize under
    /// `ATLAS_EXL3_NATIVE_DECODE=1`). Returns whether it was installed.
    pub fn attach_exl3_decode_from_store(
        &mut self,
        store: &spark_runtime::weights::WeightStore,
        lp: &str,
        gpu: &dyn GpuBackend,
    ) -> Result<bool> {
        let p = format!("{lp}.linear_attn");
        if !store.contains(&format!("{p}.in_proj_qkv.trellis")) {
            return Ok(false);
        }
        let w = |name: &str| {
            crate::weight_map::exl3::exl3_from_store(store, &format!("{p}.{name}"), gpu)
        };
        let d = Exl3GdnDecode::new(gpu, w("in_proj_qkv")?, w("in_proj_z")?, w("out_proj")?)?;
        self.set_exl3_decode(d);
        Ok(true)
    }
}
