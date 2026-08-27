// SPDX-License-Identifier: AGPL-3.0-only

//! Attention sub-block + the per-layer block driver: the Gemma-4
//! four-norm sandwich (norm → attention → norm → residual; norm → GeGLU →
//! norm → residual). M-agnostic GEMMs/norms run once over Σp; the attention
//! stage loops per image over its disjoint q/k/v slices.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{GemmaVisionEncoder, GemmaVisionLayerWeights};
use super::launch_optional;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl GemmaVisionEncoder {
    /// Run one image's attention: `q`/`k` are already QK-normed; this stage
    /// applies the rotary (Wave-3 `gemma_vision_rope_rotate`) and the MHA
    /// itself (Wave-3 `gemma_vision_attention`), writing `[p, hidden]` O
    /// states to `o`.
    ///
    /// Both kernels are Wave-3 stubs today: null handle → no-op, so the
    /// orchestration is shape- and order-correct while the PTX lands. The
    /// documented arg layouts:
    /// - rotate: `(q, k, cos, sin, seq, heads, head_dim)`
    /// - attention: `(q, k, v, o, cos, sin, seq, heads, head_dim)` — MHA,
    ///   12 heads × head_dim 64, no GQA, attention scale 1.0 (QK-norm
    ///   replaces 1/√head_dim).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_stage(
        &self,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        o: DevicePtr,
        cos: DevicePtr,
        sin: DevicePtr,
        p: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        launch_optional(
            gpu,
            self.k_rope_rotate,
            "gemma_vision_rope_rotate",
            stream,
            |kl| {
                kl.grid([p, 1, 1])
                    .block([self.head_dim as u32, 1, 1])
                    .arg_ptr(q)
                    .arg_ptr(k)
                    .arg_ptr(cos)
                    .arg_ptr(sin)
                    .arg_u32(p)
                    .arg_u32(self.num_heads as u32)
                    .arg_u32(self.head_dim as u32)
            },
        )?;
        launch_optional(gpu, self.k_attn, "gemma_vision_attention", stream, |kl| {
            kl.grid([p, self.num_heads as u32, 1])
                .block([32, 1, 1])
                .arg_ptr(q)
                .arg_ptr(k)
                .arg_ptr(v)
                .arg_ptr(o)
                .arg_ptr(cos)
                .arg_ptr(sin)
                .arg_u32(p)
                .arg_u32(self.num_heads as u32)
                .arg_u32(self.head_dim as u32)
        })
    }

    /// Attention sub-block over M=Σp rows plus per-image attention: residual
    /// save → `input_layernorm` → clipped q/k/v GEMMs into `buf_qkv` →
    /// per-head QK-Norm → per-image rope + MHA → clipped `o_proj` →
    /// `post_attention_layernorm` → residual add. `buf_h1` holds the hidden
    /// states on entry and exit.
    pub(super) fn attn_sub_block(
        &self,
        blk: &GemmaVisionLayerWeights,
        pt: u32,
        p_i: &[usize],
        p_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let n_h = pt * h;
        // Residual save for the attention sub-block.
        gpu.copy_d2d(self.buf_h1, self.buf_h2, (n_h * 2) as usize)?;
        // input_layernorm, in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.input_layernorm,
            self.buf_h1,
            pt,
            h,
            self.norm_eps,
            stream,
        )?;
        // Clipped q/k/v GEMMs → buf_qkv [Σp, 3×hidden]: q, k, v planes.
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: after input_layernorm");
        }
        let q = self.buf_qkv;
        let k = self.buf_qkv.offset((pt * h * 2) as usize);
        let v = self.buf_qkv.offset((2 * pt * h * 2) as usize);
        self.clipped_gemm(gpu, self.buf_h1, &blk.q_proj, q, pt, h, h, stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: after q_proj");
        }
        self.clipped_gemm(gpu, self.buf_h1, &blk.k_proj, k, pt, h, h, stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: after k_proj");
        }
        self.clipped_gemm(gpu, self.buf_h1, &blk.v_proj, v, pt, h, h, stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: after v_proj");
        }
        // Per-head QK-Norm on q and k (before rotary + attention).
        self.qk_norm_inplace(q, &blk.q_norm, pt, gpu, stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: after qk_norm_q");
        }
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            // HF q_norm row0 norm ~3.2 (RMSNorm over head_dim) — a large
            // deviation here means the q plane layout or the qk-norm launch
            // is wrong.
            gpu.synchronize(stream).ok();
            let mut buf = vec![0u8; self.head_dim * 2];
            let _ = gpu.copy_d2h(q, &mut buf);
            let v: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!("GVE: q[0,0] |x|={n:.4}");
        }
        self.qk_norm_inplace(k, &blk.k_norm, pt, gpu, stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: after qk_norm_k");
        }
        // V-norm: HF Gemma4VisionAttention applies RMSNorm(head_dim,
        // with_scale=False) to v after the projection — the checkpoint ships
        // no v_norm weight, so the head-dim ones buffer is the weight.
        self.qk_norm_inplace(
            v,
            &DenseWeight {
                weight: self.head_norm_unit_w,
            },
            pt,
            gpu,
            stream,
        )?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: after v_norm");
        }
        // Attention per image over its disjoint slices. q/k/v planes are
        // packed per-PATCH (row-major [Σp, hidden] each, in image order):
        // image i's q plane starts at p_off[i]*hidden (NOT qkv_row — the
        // planes are NOT interleaved per image; a qkv_row stride here reads
        // the k/v regions and corrupts multi-image batches).
        let hd_bytes = self.head_dim * 2;
        let plane_bytes = pt as usize * self.hidden_size * 2;
        for (i, &p) in p_i.iter().enumerate() {
            let base = p_off[i] * self.hidden_size * 2;
            let qi = self.buf_qkv.offset(base);
            let ki = self.buf_qkv.offset(base + plane_bytes);
            let vi = self.buf_qkv.offset(base + 2 * plane_bytes);
            let oi = self.buf_mlp.offset(p_off[i] * self.hidden_size * 2);
            let cos = self.buf_rope_cos.offset(p_off[i] * hd_bytes);
            let sin = self.buf_rope_sin.offset(p_off[i] * hd_bytes);
            self.attention_stage(qi, ki, vi, oi, cos, sin, p as u32, gpu, stream)?;
        }
        // Clipped o_proj: buf_mlp → buf_h1 (normed input fully consumed by
        // q/k/v).
        self.clipped_gemm(
            gpu,
            self.buf_mlp,
            &blk.o_proj,
            self.buf_h1,
            pt,
            h,
            h,
            stream,
        )?;
        // post_attention_layernorm + residual add, both in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.post_attention_layernorm,
            self.buf_h1,
            pt,
            h,
            self.norm_eps,
            stream,
        )?;
        spark_runtime::kernel_args::KernelLaunch::new(gpu, self.k_add)
            .grid([(n_h).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h)
            .launch(stream)
    }

    /// One ViT block over the packed batch: attention sub-block then FFN
    /// sub-block, both in place on `buf_h1`.
    pub(super) fn vit_block_batched(
        &self,
        blk: &GemmaVisionLayerWeights,
        p_total: usize,
        p_i: &[usize],
        p_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let pt = p_total as u32;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            tracing::info!("GVE: block start pt={pt}");
        }
        self.attn_sub_block(blk, pt, p_i, p_off, gpu, stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: attn_sub done");
        }
        self.ffn_sub_block(blk, pt, gpu, stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!("GVE: ffn_sub done");
        }
        Ok(())
    }
}
