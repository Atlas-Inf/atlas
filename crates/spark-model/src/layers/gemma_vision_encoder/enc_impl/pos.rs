// SPDX-License-Identifier: AGPL-3.0-only

//! Per-image position geometry, precomputed on the host and uploaded (the
//! Qwen3-VL idiom): 2D rotary cos/sin tables from `pos_ids`, and the
//! `x_emb + y_emb` position-embedding gather from the learned table.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::GemmaVisionEncoder;
use super::f32_to_bf16_bits;
use crate::media::gemma_vision::GemmaImageInput;

impl GemmaVisionEncoder {
    /// Build per-patch 2D rotary cos/sin in row-major patch order, mirroring
    /// the Qwen3-VL `build_rope_cossin_into` host-precompute idiom. Each
    /// token's row of `head_dim` is `[x_freq; y_freq; x_freq; y_freq]` where
    /// `x_freq[k] = cos/sin(x·inv_freq[k])` with `inv_freq[k] = θ^(−2k/hd)`,
    /// k in `[0, hd/4)`. Uploads BF16 to `cos_dst`/`sin_dst` (per-image
    /// slices in the batched path). Wave 3's `gemma_vision_rope_rotate`
    /// consumes these; the exact HF 2D layout is pinned by the numerical
    /// oracle.
    pub(super) fn build_rope_cossin_into(
        &self,
        img: &GemmaImageInput,
        cos_dst: DevicePtr,
        sin_dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // HF Gemma4VisionRotaryEmbedding (verified against modeling_gemma4.py):
        //   spatial_dim = head_dim / 2 (=32); inv_freq = 1/base^(arrange(0,
        //   spatial_dim, 2)/spatial_dim) — ONE frequency per 2 channels, i.e.
        //   16 distinct frequencies. Each spatial axis produces emb = cat(
        //   (freqs, freqs)) over its position → 32 channels ([f0..f15, f0..f15],
        //   the rotate_half-ready duplication). cos/sin per axis are 32 wide;
        //   the kernel table row = [x(32); y(32)] = 64.
        // apply_multidimensional_rope then rotates each 32-channel segment
        // independently (rotate_half on half=16 within the segment) — the
        // kernel's `partner = d ± head_dim/4` handles that.
        let hd = self.head_dim;
        let half = hd / 2; // 32 — channels per spatial axis
        let inv_n = half / 2; // 16 — distinct frequencies per axis
        debug_assert_eq!(self.rope_inv_freq.len(), inv_n);
        let p = img.pos_ids.len();

        let mut cos_bf16 = vec![0u16; p * hd];
        let mut sin_bf16 = vec![0u16; p * hd];
        for (p_idx, &(x, y)) in img.pos_ids.iter().enumerate() {
            // Padding patches (unused slots) get zero pos embedding — also
            // zero rotary.
            let (xf, yf) = if x < 0 || y < 0 {
                (0.0f32, 0.0f32)
            } else {
                (x as f32, y as f32)
            };
            let off = p_idx * hd;
            // Axis layout: channels [0..32) = x, [32..64) = y; within each,
            // [f0..f15, f0..f15] duplication (rotate_half pairing).
            for k in 0..inv_n {
                let fr = xf * self.rope_inv_freq[k];
                let fc = yf * self.rope_inv_freq[k];
                let (x_s, x_c) = (fr.sin(), fr.cos());
                let (y_s, y_c) = (fc.sin(), fc.cos());
                // x axis segment [0..32): [f_k, f_k+16] both from x.
                cos_bf16[off + k] = f32_to_bf16_bits(x_c);
                sin_bf16[off + k] = f32_to_bf16_bits(x_s);
                cos_bf16[off + inv_n + k] = f32_to_bf16_bits(x_c);
                sin_bf16[off + inv_n + k] = f32_to_bf16_bits(x_s);
                // y axis segment [32..64): [f_k, f_k+16] both from y.
                let y_off = off + half;
                cos_bf16[y_off + k] = f32_to_bf16_bits(y_c);
                sin_bf16[y_off + k] = f32_to_bf16_bits(y_s);
                cos_bf16[y_off + inv_n + k] = f32_to_bf16_bits(y_c);
                sin_bf16[y_off + inv_n + k] = f32_to_bf16_bits(y_s);
            }
        }
        // SAFETY (both): `cos_bf16`/`sin_bf16` are live `vec![u16; p*hd]`
        // whose byte lengths are derived from the same Vecs; every element
        // was zero-initialised; u16 has no invalid bit patterns and u8 has
        // alignment 1. The views are read-only and die before their Vecs.
        let cos_b: &[u8] = unsafe {
            std::slice::from_raw_parts(cos_bf16.as_ptr() as *const u8, cos_bf16.len() * 2)
        };
        let sin_b: &[u8] = unsafe {
            std::slice::from_raw_parts(sin_bf16.as_ptr() as *const u8, sin_bf16.len() * 2)
        };
        gpu.copy_h2d_async(cos_b, cos_dst, stream)?;
        gpu.copy_h2d_async(sin_b, sin_dst, stream)
    }

    /// Gather each patch's learned position embedding from the position
    /// table `[2, position_embedding_size, hidden]` (host copy, BF16):
    /// `pos_emb[p] = table[slot][x_p] + table[slot][y_p]` with the
    /// image/frame slot 0 for static images; padding patches (pos (-1,-1))
    /// get an all-zero row. Uploads BF16 to `dst` (per-image slice in the
    /// batched path).
    ///
    /// Indexing note: the checkpoint's table first dim is the image/frame
    /// index slot per the verified weight naming; Wave 3's numerical oracle
    /// confirms the x/y slot convention against HF exactly.
    pub(super) fn gather_pos_emb_into(
        &self,
        img: &GemmaImageInput,
        dst: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // HF Gemma4VisionPatchEmbedder._position_embeddings: the 2D table's
        // FIRST slot indexes the x-axis, the SECOND the y-axis —
        // x_emb = table[0][x], y_emb = table[1][y], sum per patch
        // (verified against modeling_gemma4.py). Padding (x,y < 0) → zero row.
        let h = self.hidden_size;
        let (x_base, y_base) = (0usize, self.position_embedding_size * h);
        let mut out_bf16 = vec![0u16; img.pos_ids.len() * h];
        for (p_idx, &(x, y)) in img.pos_ids.iter().enumerate() {
            if x < 0 || y < 0 {
                continue; // zero row for padding patches
            }
            let (x, y) = (x as usize, y as usize);
            debug_assert!(x < self.position_embedding_size && y < self.position_embedding_size);
            let x_row = x_base + x * h;
            let y_row = y_base + y * h;
            let out_off = p_idx * h;
            for d in 0..h {
                // Decode BF16 → f32, add, re-encode (u16 bit addition would be
                // garbage — BF16 patterns are not integers).
                let xv = f32::from_bits((self.position_table_host[x_row + d] as u32) << 16);
                let yv = f32::from_bits((self.position_table_host[y_row + d] as u32) << 16);
                out_bf16[out_off + d] = f32_to_bf16_bits(xv + yv);
            }
        }
        // SAFETY: `out_bf16` is a live `vec![u16; p*h]`; byte length derived
        // from the same Vec; zero-initialised; u16/u8 have no invalid bit
        // patterns. Read-only view dies before the Vec.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(out_bf16.as_ptr() as *const u8, out_bf16.len() * 2)
        };
        gpu.copy_h2d_async(bytes, dst, stream)
    }
}
