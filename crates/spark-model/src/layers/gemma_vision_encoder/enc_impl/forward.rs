// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level `GemmaVisionEncoder::forward_batched`: drives the full
//! image → soft-token pipeline (patch embed → 16 ViT blocks → pool →
//! embed_vision projection) over N images, packing the final 1536-wide
//! features into `buf_out` in image order and returning per-image
//! soft-token counts.
//!
//! Wave 2: orchestration contract only — per-image geometry (RoPE cos/sin,
//! position gather) is precomputed on the host; the shared kernels (GEMM /
//! RMSNorm / GeGLU) run for real; the gemma-specific kernels (rope-rotate,
//! attention, clamp, pool) are Wave-3 stubs that no-op until their PTX lands.

use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;

use super::super::GemmaVisionEncoder;
use crate::media::gemma_vision::GemmaImageInput;

impl GemmaVisionEncoder {
    /// Batched forward over N images. Returns per-image soft-token counts
    /// (image order); the projected features are packed into
    /// [`Self::buf_out`] `[Σsoft, OUT_HIDDEN_SIZE]` BF16, image-order, with
    /// [`Self::total_soft_tokens`] set to Σsoft.
    ///
    /// M-agnostic stages (patch embed, all per-layer GEMMs/norms/GeGLU) run
    /// ONCE over M=Σpatches; per-image-geometry stages (host pos/rope prep,
    /// attention, pool, embed_vision projection) loop per image over its
    /// disjoint slices — the Qwen3-VL `forward_batched` contract.
    ///
    /// IN-BOUNDS INVARIANT: the packed path requires Σpatches ≤ `p_max` (all
    /// row capacities are one-image); beyond that a per-image fallback loops
    /// and REFUSES a batch whose Σsoft_tokens would overrun `buf_out` (fail
    /// fast — the splice wave may raise the cap).
    pub fn forward_batched(
        &self,
        images: &[GemmaImageInput],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<usize>> {
        let n_img = images.len();
        let pks2 = self.pooling_kernel_size * self.pooling_kernel_size;

        // Per-image patch counts / soft counts / running row offsets.
        let mut p_i = Vec::with_capacity(n_img);
        let mut s_i = Vec::with_capacity(n_img);
        let mut p_off = Vec::with_capacity(n_img);
        let mut s_off = Vec::with_capacity(n_img);
        let (mut p_total, mut s_total) = (0usize, 0usize);
        for img in images {
            let p = img.grid_h * img.grid_w;
            ensure!(
                p > 0,
                "gemma vision: image with empty grid ({}×{})",
                img.grid_h,
                img.grid_w
            );
            ensure!(
                img.pixels.len() == p * self.patch_dim,
                "gemma vision: {} pixel floats for {p} patches, expected {} per patch \
                 (patch_size {}) — checkpoint/preprocessor geometry mismatch",
                img.pixels.len(),
                self.patch_dim,
                self.patch_dim / 3
            );
            ensure!(
                img.pos_ids.len() == p,
                "gemma vision: {} pos ids for {p} patches",
                img.pos_ids.len()
            );
            let soft = p / pks2;
            ensure!(
                img.soft_token_count == soft,
                "gemma vision: soft_token_count {} != {p}/{pks2} = {soft}",
                img.soft_token_count
            );
            p_off.push(p_total);
            s_off.push(s_total);
            p_i.push(p);
            s_i.push(soft);
            p_total += p;
            s_total += soft;
        }

        if p_total > self.p_max {
            return self.forward_oversized_fallback(images, &p_i, &s_i, &s_off, gpu, stream);
        }

        // 1. Per-image host prep, packed into the SHARED buffers at row
        //    offsets (rope cos/sin + position gather).
        for (i, img) in images.iter().enumerate() {
            let cos_dst = self.buf_rope_cos.offset(p_off[i] * self.head_dim * 2);
            let sin_dst = self.buf_rope_sin.offset(p_off[i] * self.head_dim * 2);
            self.build_rope_cossin_into(img, cos_dst, sin_dst, gpu, stream)?;
            if std::env::var("ATLAS_VISION_TIMING").is_ok() && i == 0 {
                gpu.synchronize(stream).ok();
                let mut buf = vec![0u8; self.head_dim * 2];
                let _ = gpu.copy_d2h(self.buf_rope_cos.offset(self.head_dim * 2), &mut buf);
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                tracing::info!("GVE: cos[1] first5={:?}", &v[..5]);
            }
            let pos_dst = self.buf_pos.offset(p_off[i] * self.hidden_size * 2);
            self.gather_pos_emb_into(img, pos_dst, gpu, stream)?;
        }

        // 2. Patch embed over M=Σp: upload (scale+convert) → input_proj GEMM
        //    → +pos → buf_h1.
        let _t0 = std::time::Instant::now();
        self.patch_embed_batched(images, &p_off, p_total, gpu, stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!(
                "GVE: patch_embed {p_total} patches {:.1}ms",
                _t0.elapsed().as_secs_f64() * 1000.0
            );
            // Compare |x| of the first patch row against the HF oracle
            // (patch_embed norm ~3132 for mona_lisa at 2304 patches).
            let mut buf = vec![0u8; self.hidden_size * 2];
            let _ = gpu.copy_d2h(self.buf_h1, &mut buf);
            let v: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!("GVE: patch_embed[0] |x|={n:.4} first5={:?}", &v[..5]);
        }

        // 3. ViT blocks: M-agnostic GEMMs/norms/GeGLU once; attention per
        //    image over its disjoint buf_qkv slice.
        for (bi, blk) in self.layers.iter().enumerate() {
            let _tb = std::time::Instant::now();
            self.vit_block_batched(blk, p_total, &p_i, &p_off, gpu, stream)?;
            if std::env::var("ATLAS_VISION_TIMING").is_ok() {
                gpu.synchronize(stream).ok();
                if matches!(bi, 0 | 4 | 8 | 12 | 15) {
                    let mut buf = vec![0u8; self.hidden_size * 2];
                    let _ = gpu.copy_d2h(self.buf_h1, &mut buf);
                    let v: Vec<f32> = buf
                        .chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect();
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    tracing::info!(
                        "GVE: block {bi} row0 |x|={n:.4} first5={:?} {:.1}ms",
                        &v[..5],
                        _tb.elapsed().as_secs_f64() * 1000.0
                    );
                } else {
                    tracing::info!(
                        "GVE: block {bi} {:.1}ms",
                        _tb.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
        }

        // 4. Pool + embed_vision projection per image → packed buf_out rows.
        let _tp = std::time::Instant::now();
        for (i, img) in images.iter().enumerate() {
            let pooled = self.buf_pool.offset(s_off[i] * self.hidden_size * 2);
            self.pool_stage(
                img,
                self.buf_h1.offset(p_off[i] * self.hidden_size * 2),
                pooled,
                gpu,
                stream,
            )?;
            if std::env::var("ATLAS_VISION_TIMING").is_ok() {
                gpu.synchronize(stream).ok();
                let mut buf = vec![0u8; self.hidden_size * 2];
                let _ = gpu.copy_d2h(pooled, &mut buf);
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!("GVE: pooled[{i}] |x|={n:.4} first5={:?}", &v[..5]);
            }
            let out_slice = self
                .buf_out
                .offset(s_off[i] * super::super::OUT_HIDDEN_SIZE * 2);
            self.embed_vision_project(s_i[i], pooled, out_slice, gpu, stream)?;
        }
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            tracing::info!(
                "GVE: pool+project {:.1}ms",
                _tp.elapsed().as_secs_f64() * 1000.0
            );
        }

        self.total_soft_tokens
            .store(s_total, std::sync::atomic::Ordering::Relaxed);
        Ok(s_i)
    }

    /// Fallback for Σpatches > p_max: encode each image ALONE (full
    /// single-image kernel sequence) writing its projected rows into the
    /// packed `buf_out` at `s_off[i]`. Refuses a batch whose Σsoft_tokens
    /// exceeds the `buf_out` row capacity rather than overflowing it.
    #[allow(clippy::too_many_arguments)]
    fn forward_oversized_fallback(
        &self,
        images: &[GemmaImageInput],
        p_i: &[usize],
        s_i: &[usize],
        s_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<usize>> {
        let s_total: usize = s_i.iter().sum();
        let out_rows = s_off.last().map(|o| o + s_i.last().unwrap()).unwrap_or(0);
        ensure!(
            out_rows <= self.s_max,
            "gemma vision: Σsoft_tokens {s_total} > buf_out rows {} — oversized batch refused \
             (the splice wave sizes buf_out per request)",
            self.s_max
        );
        for (i, img) in images.iter().enumerate() {
            self.build_rope_cossin_into(img, self.buf_rope_cos, self.buf_rope_sin, gpu, stream)?;
            self.gather_pos_emb_into(img, self.buf_pos, gpu, stream)?;
            let p = p_i[i];
            self.patch_embed_batched(std::slice::from_ref(img), &[0], p, gpu, stream)?;
            for blk in &self.layers {
                self.vit_block_batched(blk, p, &[p], &[0], gpu, stream)?;
            }
            self.pool_stage(img, self.buf_h1, self.buf_pool, gpu, stream)?;
            let out_slice = self
                .buf_out
                .offset(s_off[i] * super::super::OUT_HIDDEN_SIZE * 2);
            self.embed_vision_project(s_i[i], self.buf_pool, out_slice, gpu, stream)?;
        }
        self.total_soft_tokens
            .store(s_total, std::sync::atomic::Ordering::Relaxed);
        Ok(s_i.to_vec())
    }
}
