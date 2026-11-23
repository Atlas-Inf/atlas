// SPDX-License-Identifier: AGPL-3.0-only

//! One image through the tower.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::host::{bf16_to_f32, f32_to_bf16, patch_order, position_rows, rope_rows};
use super::{Block, Inkling2VisionTower};

impl Inkling2VisionTower {
    /// `pixels` `[grid_h * grid_w, patch_dim]` in tower order; writes
    /// `grid_h * grid_w / merge²` rows of `buf_out` from `out_row`.
    pub(super) fn encode(
        &self,
        gpu: &dyn GpuBackend,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
        out_row: usize,
        stream: u64,
    ) -> Result<()> {
        let (n, hid, m2) = (grid_h * grid_w, self.hidden, self.merge * self.merge);
        ensure!(
            n > 0
                && n <= self.p_max
                && grid_h.is_multiple_of(self.merge)
                && grid_w.is_multiple_of(self.merge),
            "image grid {grid_h}x{grid_w} does not fit the tower ({} patches, merge {})",
            self.p_max,
            self.merge
        );
        ensure!(
            pixels.len() == n * self.patch_dim,
            "image has {} pixel values, expected {}",
            pixels.len(),
            n * self.patch_dim
        );
        // The uploads below are host copies into scratch an earlier image's
        // kernels may still be reading.
        gpu.synchronize(stream)?;
        let seq = n + 1;
        let order = patch_order(grid_h, grid_w, self.merge);
        let head_dim = hid / self.heads;

        // patches -> rows 1..=n of x, plus their resampled positions
        let x = self.buf_x;
        let patches = x.offset(hid * 2);
        gpu.copy_h2d(&f32_to_bf16(pixels), self.buf_pixels)?;
        self.gemm(
            gpu,
            self.buf_pixels,
            self.patch_w,
            patches,
            [n, hid, self.patch_dim],
            stream,
        )?;
        let pos = position_rows(&self.pos_table, self.side, hid, &order, grid_h, grid_w);
        gpu.copy_h2d(&f32_to_bf16(&pos), self.buf_pos)?;
        self.elementwise(gpu, self.k.add, &[patches, self.buf_pos], n * hid, stream)?;
        gpu.copy_h2d(&self.cls_row, x)?;
        self.layer_norm(gpu, x, x, self.ln_pre, seq, stream)?;

        let (cos, sin) = rope_rows(&order, head_dim);
        gpu.copy_h2d(&f32_to_bf16(&cos), self.buf_cos)?;
        gpu.copy_h2d(&f32_to_bf16(&sin), self.buf_sin)?;
        for block in &self.blocks {
            self.block(gpu, block, seq, stream)?;
        }

        // merger: a tile's merge² patch states are adjacent rows
        let tokens = n / m2;
        self.gemm(
            gpu,
            patches,
            self.proj_in_w,
            self.buf_mlp,
            [tokens, self.proj_hidden, m2 * hid],
            stream,
        )?;
        self.elementwise(
            gpu,
            self.k.gelu,
            &[self.buf_mlp],
            tokens * self.proj_hidden,
            stream,
        )?;
        let out = self.buf_out.offset(out_row * self.out_hidden_size * 2);
        self.gemm(
            gpu,
            self.buf_mlp,
            self.proj_out_w,
            out,
            [tokens, self.out_hidden_size, self.proj_hidden],
            stream,
        )?;
        if tracing::enabled!(tracing::Level::DEBUG) {
            // For comparison with the reference's `pooler_output`.
            gpu.synchronize(stream)?;
            let mut bytes = vec![0u8; tokens * self.out_hidden_size * 2];
            gpu.copy_d2h(out, &mut bytes)?;
            let rows = bf16_to_f32(&bytes);
            let norm: f32 = rows
                .chunks_exact(self.out_hidden_size)
                .map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt())
                .sum::<f32>()
                / tokens as f32;
            tracing::debug!(
                "inkling2 vision: grid {grid_h}x{grid_w}, {tokens} tokens, mean row norm {norm:.4}, first {:?}, last {:?}",
                &rows[..4],
                &rows[rows.len() - 4..]
            );
        }
        Ok(())
    }

    fn block(&self, gpu: &dyn GpuBackend, b: &Block, seq: usize, stream: u64) -> Result<()> {
        let (hid, x) = (self.hidden, self.buf_x);
        self.layer_norm(gpu, x, self.buf_norm, b.ln1, seq, stream)?;
        self.linear(
            gpu,
            self.buf_norm,
            b.qkv,
            self.buf_qkv,
            [seq, 3 * hid, hid],
            stream,
        )?;
        self.attention(gpu, seq, stream)?;
        // out_proj carries ls_1
        self.linear(
            gpu,
            self.buf_o,
            b.out,
            self.buf_norm,
            [seq, hid, hid],
            stream,
        )?;
        self.elementwise(gpu, self.k.add, &[x, self.buf_norm], seq * hid, stream)?;
        self.layer_norm(gpu, x, self.buf_norm, b.ln2, seq, stream)?;
        self.linear(
            gpu,
            self.buf_norm,
            b.fc1,
            self.buf_mlp,
            [seq, self.inter, hid],
            stream,
        )?;
        self.elementwise(gpu, self.k.gelu, &[self.buf_mlp], seq * self.inter, stream)?;
        // c_proj carries ls_2
        self.linear(
            gpu,
            self.buf_mlp,
            b.fc2,
            self.buf_norm,
            [seq, hid, self.inter],
            stream,
        )?;
        self.elementwise(gpu, self.k.add, &[x, self.buf_norm], seq * hid, stream)
    }

    /// Dense SDPA over `buf_qkv` `[seq, 3*H*D]` into `buf_o`, as
    /// `VisionEncoder::vit_attention_gemm`: rotate and regroup by head once,
    /// then per head Q·Kᵀ, a row softmax (scale folded in), P·V and a scatter
    /// back to the head's columns. One stream orders the heads' shared scratch.
    fn attention(&self, gpu: &dyn GpuBackend, seq: usize, stream: u64) -> Result<()> {
        let (heads, d, s) = (self.heads, (self.hidden / self.heads) as u32, seq as u32);
        KernelLaunch::new(gpu, self.k.rope_deint)
            .grid([div_ceil(s * d, 256), heads as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_qkv)
            .arg_ptr(self.buf_qr)
            .arg_ptr(self.buf_kr)
            .arg_ptr(self.buf_vt)
            .arg_ptr(self.buf_cos)
            .arg_ptr(self.buf_sin)
            .arg_u32(s)
            .arg_u32(heads as u32)
            .arg_u32(d)
            .launch(stream)?;
        let head_bytes = seq * d as usize * 2;
        for h in 0..heads {
            let at = |p: DevicePtr| p.offset(h * head_bytes);
            KernelLaunch::new(gpu, self.k.gemm_f32)
                .grid([div_ceil(s, 16), div_ceil(s, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(at(self.buf_qr))
                .arg_ptr(at(self.buf_kr))
                .arg_ptr(self.buf_scores)
                .arg_u32(s)
                .arg_u32(s)
                .arg_u32(d)
                .launch(stream)?;
            KernelLaunch::new(gpu, self.k.softmax)
                .grid([s, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_scores)
                .arg_ptr(self.buf_probs)
                .arg_u32(s)
                .arg_u32(d)
                .launch(stream)?;
            self.gemm(
                gpu,
                self.buf_probs,
                at(self.buf_vt),
                self.buf_o_stage,
                [seq, d as usize, seq],
                stream,
            )?;
            KernelLaunch::new(gpu, self.k.scatter)
                .grid([div_ceil(s * d, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_o_stage)
                .arg_ptr(self.buf_o.offset(h * d as usize * 2))
                .arg_u32(s)
                .arg_u32(d)
                .arg_u32(self.hidden as u32)
                .launch(stream)?;
        }
        Ok(())
    }
}
