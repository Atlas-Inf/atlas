// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-only native EXL3 overlay for GDN layers: the single-token
//! in_proj_qkv / in_proj_z / out_proj projections run the int8 sq GEMV on the
//! PACKED EXL3 weights (int8 activations), so decode never touches the BF16
//! materialized copies. PREFILL is unchanged and keeps the BF16 weights.
//! Installed by `build_linear_attention_dense_bf16` under
//! `--exl3-native-decode` (see `weight_map::exl3::materialize`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{
    Exl3Bf16Sizes, Exl3Int8Kernels, Exl3Int8Workspace, Exl3Kernels, exl3_bf16_scratch,
    exl3_dense_bf16_nk_with_scratch, exl3_int8_linear_bf16, exl3_int8_linear_bf16_rows, sq_grid,
    sq_plan,
};
use crate::weight_map::DenseWeight;
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
    /// M6f: the BF16 copies of `qkv`/`z`/`out` are freed; prefill and the wide
    /// paths ask the layer to rebuild them on demand (see
    /// [`Self::rebuild_qkvz_bf16`]). False keeps the resident copies.
    lazy: bool,
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
            lazy: false,
        })
    }

    /// M6f-b flips this on after freeing the BF16 copies; until then the
    /// resident weights keep serving the BF16 paths.
    pub fn set_lazy(&mut self, on: bool) {
        self.lazy = on;
    }

    pub fn is_lazy(&self) -> bool {
        self.lazy
    }

    /// The scratch regions [`Self::rebuild_qkvz_bf16`] /
    /// [`Self::rebuild_out_bf16`] need (bytes).
    pub fn bf16_sizes(&self) -> Exl3Bf16Sizes {
        let bytes = |w: &Exl3Weight| w.shape.in_features * w.shape.out_features * 2;
        Exl3Bf16Sizes {
            qkvz_bytes: bytes(&self.qkv) + bytes(&self.z),
            out_bytes: bytes(&self.out),
            tmp_bytes: bytes(&self.qkv).max(bytes(&self.z)).max(bytes(&self.out)),
        }
    }

    /// Rebuild the fused qkvz BF16 `[qkv.out + z.out, in]` into the stream's
    /// pooled scratch and return it. Stream-ordered: the caller's later GEMM on
    /// the same stream reads it without any sync.
    pub fn rebuild_qkvz_bf16(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<DevicePtr> {
        let s = exl3_bf16_scratch(gpu, stream, &self.bf16_sizes())?;
        exl3_dense_bf16_nk_with_scratch(gpu, &self.k, &self.qkv, s.qkvz, s.tmp_a, s.tmp_b, stream)?;
        let z_rows = self.qkv.shape.out_features * self.qkv.shape.in_features * 2;
        exl3_dense_bf16_nk_with_scratch(
            gpu,
            &self.k,
            &self.z,
            s.qkvz.offset(z_rows),
            s.tmp_a,
            s.tmp_b,
            stream,
        )?;
        Ok(s.qkvz)
    }

    /// Rebuild the out_proj BF16 `[out.out, out.in]` into the stream's pooled
    /// scratch and return it. See [`Self::rebuild_qkvz_bf16`].
    pub fn rebuild_out_bf16(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<DevicePtr> {
        let s = exl3_bf16_scratch(gpu, stream, &self.bf16_sizes())?;
        exl3_dense_bf16_nk_with_scratch(gpu, &self.k, &self.out, s.out, s.tmp_a, s.tmp_b, stream)?;
        Ok(s.out)
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
    /// `--exl3-native-decode`). Returns whether it was installed.
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
        let mut d = Exl3GdnDecode::new(gpu, w("in_proj_qkv")?, w("in_proj_z")?, w("out_proj")?)?;
        // M6f-b frees the BF16 copies here (ATLAS_EXL3_LAZY_BF16); until then
        // the resident weights keep serving every BF16 path.
        d.set_lazy(crate::weight_map::exl3::exl3_lazy_bf16());
        self.set_exl3_decode(d);
        Ok(true)
    }

    /// M6f-b: free this layer's resident BF16 GDN copies (the fused qkvz and
    /// the store's BF16 out_proj) and switch the overlay to lazy rebuild.
    /// Only called from the loader when `attach_exl3_decode_from_store`
    /// installed the overlay AND `exl3_lazy_bf16()`; every runtime read goes
    /// through [`Self::qkvz_bf16`] / [`Self::out_proj_bf16`] afterwards, so a
    /// site that was missed dereferences NULL and faults loudly.
    pub(crate) fn drop_bf16_copies(
        &mut self,
        gpu: &dyn GpuBackend,
        store: &spark_runtime::weights::WeightStore,
        lp: &str,
    ) -> Result<()> {
        let Some(e) = self.exl3_decode.as_mut() else {
            return Ok(());
        };
        e.set_lazy(true);
        // Decode may be CUDA-graph captured, and capture must never allocate:
        // pre-grow the pool for the model's default stream now.
        let sizes = e.bf16_sizes();
        exl3_bf16_scratch(gpu, gpu.default_stream(), &sizes)?;
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);

        // The fused qkvz is layer-owned (`gpu_concat_rows` allocated it; tp = 1,
        // so the shard is the same pointer).
        let qkvz_bytes = if self.ssm.in_proj_qkvz.weight.is_null() {
            0
        } else {
            let bytes = sizes.qkvz_bytes;
            gpu.free(self.ssm.in_proj_qkvz.weight)?;
            self.ssm.in_proj_qkvz.weight = DevicePtr::NULL;
            bytes
        };
        // The BF16 out_proj is a STORE tensor: reclaim frees it and records the
        // pointer so teardown will not free it again. Keep the field as
        // Some(null): several dispatch checks test `out_proj_dense.is_some()`.
        let out_name = format!("{lp}.linear_attn.out_proj.weight");
        let out_bytes = if self.out_proj_dense.is_some() {
            store.reclaim(gpu, &out_name)?;
            self.out_proj_dense = Some(DenseWeight {
                weight: DevicePtr::NULL,
            });
            sizes.out_bytes
        } else {
            0
        };
        tracing::info!(
            "EXL3 lazy BF16: freed {:.1} MB (qkvz {:.1} + out_proj {:.1}) for {lp}",
            mb(qkvz_bytes + out_bytes),
            mb(qkvz_bytes),
            mb(out_bytes),
        );
        Ok(())
    }

    /// The BF16 qkvz weight for the wide (prefill / verify) paths: the resident
    /// copy, or a rebuild into the stream's scratch once the overlay is lazy.
    /// The returned pointer's contents are stream-ordered against the caller's
    /// work on `stream`. M6f-b routes the BF16 call sites here.
    pub(crate) fn qkvz_bf16(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<DenseWeight> {
        if let Some(e) = self.exl3_decode.as_deref()
            && e.is_lazy()
        {
            return Ok(DenseWeight {
                weight: e.rebuild_qkvz_bf16(gpu, stream)?,
            });
        }
        Ok(self.ssm.in_proj_qkvz)
    }

    /// The BF16 out_proj weight, lazy variant of [`Self::qkvz_bf16`].
    pub(crate) fn out_proj_bf16(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<DenseWeight>> {
        if let Some(e) = self.exl3_decode.as_deref()
            && e.is_lazy()
        {
            return Ok(Some(DenseWeight {
                weight: e.rebuild_out_bf16(gpu, stream)?,
            }));
        }
        Ok(self.out_proj_dense)
    }

    /// Whether a BF16 qkvz is obtainable: the resident copy, or a lazy overlay
    /// that can rebuild one.
    pub(crate) fn has_qkvz_bf16(&self) -> bool {
        self.exl3_decode.as_deref().is_some_and(|e| e.is_lazy())
            || !self.ssm.in_proj_qkvz.weight.is_null()
    }
}
