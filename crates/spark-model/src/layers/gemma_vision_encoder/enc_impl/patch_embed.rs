// SPDX-License-Identifier: AGPL-3.0-only

//! Patch-embed step: scale + convert pixels → `input_proj` GEMM → +pos.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::GemmaVisionEncoder;
use super::f32_to_bf16_bits;
use crate::layers::ops;
use crate::media::gemma_vision::GemmaImageInput;
use crate::weight_map::DenseWeight;

impl GemmaVisionEncoder {
    /// Batched patch embed over N images packed at `p_off[i]` (rows): each
    /// image's `[P, patch_dim]` f32 pixels are scaled `2×(x−0.5)` and
    /// converted to BF16 on the host, uploaded into `buf_h2` at its row
    /// slice; then ONE `input_proj` GEMM (768→768, no bias) over M=Σp from
    /// `buf_h2` into `buf_h1`, and ONE pos-add `buf_h1 += buf_pos` (packed
    /// per-image gathers). `buf_h1` therefore holds the hidden states the
    /// ViT blocks consume.
    ///
    /// The scale+convert is the Wave-2 host-side stand-in for the Wave-3
    /// fused `gemma_vision_pixel_scale` kernel — elementwise and exact in
    /// f32, so the upload bytes are identical either way.
    pub(super) fn patch_embed_batched(
        &self,
        images: &[GemmaImageInput],
        p_off: &[usize],
        p_total: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let pt = p_total as u32;
        for (i, img) in images.iter().enumerate() {
            let p = img.grid_h * img.grid_w;
            let mut bf16 = vec![0u16; p * self.patch_dim];
            for (e, &v) in img.pixels.iter().enumerate() {
                bf16[e] = f32_to_bf16_bits(2.0 * (v - 0.5));
            }
            // SAFETY: `bf16` is a live `vec![u16; p*patch_dim]`; byte length
            // derived from the same Vec; every element written by the loop;
            // u16/u8 have no invalid bit patterns. Read-only, dies first.
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(bf16.as_ptr() as *const u8, bf16.len() * 2) };
            gpu.copy_h2d_async(
                bytes,
                self.buf_h2.offset(p_off[i] * self.patch_dim * 2),
                stream,
            )?;
        }
        // input_proj: [Σp, patch_dim] @ [768, patch_dim]ᵀ → buf_h1 [Σp, hidden].
        ops::dense_gemm(
            gpu,
            self.k_gemm,
            self.buf_h2,
            &DenseWeight {
                weight: self.input_proj_w,
            },
            self.buf_h1,
            pt,
            h,
            self.patch_dim as u32,
            stream,
        )?;
        // pos-add: buf_h1 += buf_pos (packed gathers).
        let n_pe = (p_total * self.hidden_size) as u32;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_pe, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_pos)
            .arg_u32(n_pe)
            .launch(stream)
    }
}
