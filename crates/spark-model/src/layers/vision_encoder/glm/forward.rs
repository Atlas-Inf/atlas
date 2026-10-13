// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 visual forward path. Every image/video group is kept as its own
//! attention sequence, while its final merger rows are packed in input order.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::GlmVisionEncoder;

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

fn rope_table(grid_h: usize, grid_w: usize, head_dim: usize) -> (Vec<u8>, Vec<u8>) {
    let rotary_dim = head_dim / 2;
    let half = rotary_dim / 2;
    let theta = 10_000.0f32;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32))
        .collect();
    // `get_cos_sin` returns `rotary_dim / 2` values per axis.  GLM's
    // `rotary_dim` is half the head size, so flattening the two position ids
    // yields one `head_dim / 2`-wide row per patch: H frequencies followed by
    // W frequencies.  The ApplyRotaryEmb Neox path then pairs that row with
    // the two halves of the full head.
    let mut cos = Vec::with_capacity(grid_h * grid_w * rotary_dim * 2);
    let mut sin = Vec::with_capacity(grid_h * grid_w * rotary_dim * 2);
    // vLLM's 2-D position ids visit each spatial-merge block, then its 2x2
    // members.  This is also the order expected by the post-ViT conv2d.
    for block_h in 0..grid_h / 2 {
        for block_w in 0..grid_w / 2 {
            for inner_h in 0..2 {
                for inner_w in 0..2 {
                    let positions = [block_h * 2 + inner_h, block_w * 2 + inner_w];
                    for &position in &positions {
                        for dim in 0..half {
                            let angle = position as f32 * inv[dim];
                            cos.extend_from_slice(&bf16_bits(angle.cos()).to_le_bytes());
                            sin.extend_from_slice(&bf16_bits(angle.sin()).to_le_bytes());
                        }
                    }
                }
            }
        }
    }
    (cos, sin)
}

impl GlmVisionEncoder {
    pub(crate) fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        self.encode_sequences(images, gpu, stream)
    }

    /// Encode vision items, one attention sequence per temporal group. The
    /// reference supplies one cu_seqlens segment per temporal group; joining a
    /// video's groups into one attention sequence would let later frames
    /// attend to earlier frames and diverge from the checkpoint even though
    /// the final row count matched.
    pub(crate) fn forward_items(
        &self,
        items: &[&crate::VisionItem],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let mut sequences = Vec::new();
        for item in items {
            ensure!(
                !item.groups.is_empty(),
                "GLM vision item has no temporal groups"
            );
            for group in &item.groups {
                sequences.push((group.as_slice(), item.grid_h, item.grid_w));
            }
        }
        self.encode_sequences(&sequences, gpu, stream)
    }

    /// Validate every sequence and the packed output capacity before the
    /// first launch, then encode each sequence into consecutive `buf_out`
    /// rows. Rejecting late would leave earlier sequences' work queued on
    /// `stream` behind an error the caller has already reported.
    fn encode_sequences(
        &self,
        sequences: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let mut out = Vec::with_capacity(sequences.len());
        let mut out_rows = 0usize;
        for &(pixels, grid_h, grid_w) in sequences {
            ensure!(grid_h > 0 && grid_w > 0, "GLM vision grid must be non-zero");
            ensure!(
                grid_h % 2 == 0 && grid_w % 2 == 0,
                "GLM vision grid must be divisible by spatial merge 2"
            );
            let patches = grid_h * grid_w;
            let expected = patches
                .checked_mul(self.patch_dim)
                .ok_or_else(|| anyhow::anyhow!("GLM vision pixel count overflow"))?;
            ensure!(
                pixels.len() == expected,
                "GLM vision pixel geometry mismatch: got {}, expected {expected}",
                pixels.len()
            );
            ensure!(
                patches <= self.p_max,
                "GLM vision image has {patches} patches, capacity is {}",
                self.p_max
            );
            let merged = (grid_h / 2) * (grid_w / 2);
            out_rows += merged;
            ensure!(
                out_rows <= self.out_rows,
                "GLM vision request needs more than {} merged rows",
                self.out_rows
            );
            out.push((grid_h / 2, grid_w / 2, merged));
        }
        let mut row = 0usize;
        for (&(pixels, grid_h, grid_w), &(_, _, merged)) in sequences.iter().zip(&out) {
            self.forward_sequence(
                pixels,
                grid_h,
                grid_w,
                self.buf_out.offset(row * self.out_hidden_size * 2),
                gpu,
                stream,
            )?;
            row += merged;
        }
        Ok(out)
    }

    fn forward_sequence(
        &self,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
        output: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let patches = grid_h * grid_w;
        // SAFETY: reinterpreting a live `&[f32]` as its exact byte span; u8 has
        // alignment 1.
        let pixel_bytes =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
        gpu.copy_h2d_async(pixel_bytes, self.buf_f32, stream)?;
        KernelLaunch::new(gpu, self.k_f32_bf16)
            .grid([div_ceil((patches * self.patch_dim) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_f32)
            .arg_ptr(self.buf_pixels)
            .arg_u32((patches * self.patch_dim) as u32)
            .launch(stream)?;
        self.gemm_bias(
            gpu,
            self.buf_pixels,
            self.patch_embed_w,
            self.patch_embed_b,
            self.buf_h1,
            patches,
            self.hidden_size,
            self.patch_dim,
            stream,
        )?;
        let (cos, sin) = rope_table(grid_h, grid_w, self.head_dim);
        gpu.copy_h2d_async(&cos, self.buf_rope_cos, stream)?;
        gpu.copy_h2d_async(&sin, self.buf_rope_sin, stream)?;
        for block in &self.blocks {
            self.rms_norm(
                gpu,
                self.buf_h1,
                block.norm1_w,
                self.buf_norm,
                patches,
                self.hidden_size,
                stream,
            )?;
            self.gemm_bias(
                gpu,
                self.buf_norm,
                block.qkv_w,
                block.qkv_b,
                self.buf_wide,
                patches,
                3 * self.hidden_size,
                self.hidden_size,
                stream,
            )?;
            self.attention(
                gpu,
                self.buf_wide,
                block.q_norm_w,
                block.k_norm_w,
                self.buf_rope_cos,
                self.buf_rope_sin,
                self.buf_attn,
                patches,
                stream,
            )?;
            self.gemm_bias(
                gpu,
                self.buf_attn,
                block.proj_w,
                block.proj_b,
                self.buf_norm,
                patches,
                self.hidden_size,
                self.hidden_size,
                stream,
            )?;
            self.add(
                gpu,
                self.buf_h1,
                self.buf_norm,
                patches * self.hidden_size,
                stream,
            )?;
            self.rms_norm(
                gpu,
                self.buf_h1,
                block.norm2_w,
                self.buf_norm,
                patches,
                self.hidden_size,
                stream,
            )?;
            self.gemm_bias(
                gpu,
                self.buf_norm,
                block.gate_up_w,
                block.gate_up_b,
                self.buf_wide,
                patches,
                2 * self.intermediate_size,
                self.hidden_size,
                stream,
            )?;
            self.swiglu(
                gpu,
                self.buf_wide,
                self.buf_act,
                patches,
                self.intermediate_size,
                stream,
            )?;
            self.gemm_bias(
                gpu,
                self.buf_act,
                block.down_w,
                block.down_b,
                self.buf_norm,
                patches,
                self.hidden_size,
                self.intermediate_size,
                stream,
            )?;
            self.add(
                gpu,
                self.buf_h1,
                self.buf_norm,
                patches * self.hidden_size,
                stream,
            )?;
        }
        self.rms_norm(
            gpu,
            self.buf_h1,
            self.post_layernorm_w,
            self.buf_norm,
            patches,
            self.hidden_size,
            stream,
        )?;
        let merged = (grid_h / 2) * (grid_w / 2);
        self.conv2d(
            gpu,
            self.buf_norm,
            self.buf_wide,
            self.buf_conv,
            grid_h,
            grid_w,
            stream,
        )?;
        self.gemm(
            gpu,
            self.buf_conv,
            self.merger.proj_w,
            self.buf_merger,
            merged,
            self.out_hidden_size,
            self.out_hidden_size,
            stream,
        )?;
        self.layer_norm(
            gpu,
            self.buf_merger,
            self.merger.post_norm_w,
            self.merger.post_norm_b,
            merged,
            self.out_hidden_size,
            stream,
        )?;
        self.gelu(gpu, self.buf_merger, merged * self.out_hidden_size, stream)?;
        self.gemm(
            gpu,
            self.buf_merger,
            self.merger.gate_up_w,
            self.buf_wide,
            merged,
            2 * self.projection_intermediate_size,
            self.out_hidden_size,
            stream,
        )?;
        self.swiglu(
            gpu,
            self.buf_wide,
            self.buf_act,
            merged,
            self.projection_intermediate_size,
            stream,
        )?;
        self.gemm(
            gpu,
            self.buf_act,
            self.merger.down_w,
            output,
            merged,
            self.out_hidden_size,
            self.projection_intermediate_size,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_to_f32(pair: &[u8]) -> f32 {
        f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16)
    }

    #[test]
    fn rope_table_tracks_h_and_w_in_merge_block_order() {
        let (cos, sin) = rope_table(2, 2, 64);
        assert_eq!(cos.len(), 2 * 2 * (64 / 2) * 2);
        assert_eq!(sin.len(), cos.len());
        let row = |bytes: &[u8], index: usize| bf16_to_f32(&bytes[index * 2..index * 2 + 2]);
        // The first patch is at (h=0,w=0), so every rotary cosine is one.
        assert!((row(&cos, 0) - 1.0).abs() < 0.01);
        // The second patch remains on h=0 and advances w; the first axis is
        // unchanged while the second axis has a non-trivial sine.
        assert!((row(&cos, 32) - 1.0).abs() < 0.01);
        assert!(row(&sin, 48).abs() > 0.1);
        // The third patch advances h and returns to w=0.
        assert!(row(&sin, 64).abs() > 0.1);
        assert!(row(&sin, 80).abs() < 0.01);
    }
}
