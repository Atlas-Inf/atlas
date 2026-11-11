// SPDX-License-Identifier: AGPL-3.0-only

//! RADIO vision encoder (nemotron_h_omni) — stage by stage against the
//! checkpoint's own `modeling_radio.py`, goldens in
//! `n35vl-runs/vision_oracle/`.
//!
//! Stage 1: the embedder — patch-embed Linear (768→1280, no bias) +
//! bilinear-resampled learned absolute pos-embed + 10-token cls/register
//! prefix. Attention blocks and the projector land in later stages.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Radio embedder for one image tile grid.
pub struct RadioEmbedder {
    /// `embeddings.patch_projection.weight` [1280, 768] BF16.
    pub patch_w: DevicePtr,
    /// `embeddings.position_embedding` [max_rows*max_cols, 1280] BF16 on
    /// device, plus the host f32 copy for per-tile resampling.
    pub pos_embed_dev: DevicePtr,
    pub pos_embed_host: Vec<f32>,
    /// `embeddings.cls_register_token` [10, 1280] BF16 (3 cls + 7 register).
    pub cls_register: DevicePtr,
    pub max_rows: usize,
    pub max_cols: usize,
    pub hidden: usize,
    pub patch: usize,
    pub prefix_tokens: usize,
    /// Kernels (common tree): GEMM + elementwise add.
    pub k_gemm: KernelHandle,
    pub k_add: KernelHandle,
    /// Scratch: resampled pos table `[grid*hidden]` BF16, reuse across calls.
    pub buf_pos: DevicePtr,
    pub buf_pos_cap: usize,
}

impl RadioEmbedder {
    /// Bilinear-resample the learned pos grid `(max_rows, max_cols)` to
    /// `(h, w)` per the checkpoint's `_interpolate_position_embedding`:
    /// interpolate the square table to (max_dim, max_dim), crop if the tile
    /// is smaller, else interpolate again. Returns host f32 `[h*w*hidden]`.
    pub fn resample_pos_embed(&self, h: usize, w: usize) -> Vec<f32> {
        let c = self.hidden;
        let (mr, mc) = (self.max_rows, self.max_cols);
        let table = &self.pos_embed_host; // [mr*mc*c] row-major (r,c,ch)
        let max_dim = h.max(w);
        // step 1: bilinear (mr,mc) -> (max_dim,max_dim)
        let sq = bilinear_2d(table, mr, mc, c, max_dim, max_dim);
        // step 2: crop or re-interpolate to (h,w)
        if h < max_dim || w < max_dim {
            // crop rows/cols
            if h <= max_dim && w <= max_dim {
                let mut out = vec![0f32; h * w * c];
                for i in 0..h {
                    for j in 0..w {
                        out[(i * w + j) * c..(i * w + j) * c + c]
                            .copy_from_slice(&sq[(i * max_dim + j) * c..(i * max_dim + j) * c + c]);
                    }
                }
                return out;
            }
        }
        bilinear_2d(&sq, max_dim, max_dim, c, h, w)
    }

    /// Patch-embed one image's preprocessed pixels: `pixels` is
    /// [n_patches, 3*patch*patch] BF16 rows (patchified by the caller),
    /// `out` is [(prefix + n_patches) * hidden] BF16.
    /// Layout: [10 cls/register rows] then patch rows + pos embed.
    pub fn embed(
        &self,
        gpu: &dyn GpuBackend,
        pixels_bf16: DevicePtr,
        grid_h: usize,
        grid_w: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let n = grid_h * grid_w;
        let hid = self.hidden as u32;
        let kdim = (3 * self.patch * self.patch) as u32;
        // prefix rows: copy cls_register then patches land after row 10.
        let prefix_bytes = self.prefix_tokens * self.hidden * 2;
        gpu.copy_d2d(self.cls_register, out, prefix_bytes)?;
        let patches_out = out.offset(prefix_bytes);
        // C[n,1280] = A[n,768] @ W[1280,768]^T
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(hid, 128), div_ceil(n as u32, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(pixels_bf16)
            .arg_ptr(self.patch_w)
            .arg_ptr(patches_out)
            .arg_u32(n as u32)
            .arg_u32(hid)
            .arg_u32(kdim)
            .launch(stream)?;
        // + resampled pos embed
        let pos = self.resample_pos_embed(grid_h, grid_w);
        let pos_bf16: Vec<u16> = pos.iter().map(|&f| f32_to_bf16(f)).collect();
        let bytes = bytemuck_u16(&pos_bf16);
        if self.buf_pos_cap < bytes.len() {
            return Err(anyhow::anyhow!("pos scratch too small"));
        }
        gpu.copy_h2d(bytes, self.buf_pos)?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil((n * self.hidden) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(patches_out)
            .arg_ptr(self.buf_pos)
            .arg_u32((n * self.hidden) as u32)
            .launch(stream)
    }
}

/// Bilinear interpolate an `(ih, iw, c)` f32 table to `(oh, ow)` rows —
/// `F.interpolate(mode="bilinear", align_corners=False)` semantics:
/// src = (dst + 0.5) * (in/out) - 0.5, clamped.
fn bilinear_2d(t: &[f32], ih: usize, iw: usize, c: usize, oh: usize, ow: usize) -> Vec<f32> {
    let mut out = vec![0f32; oh * ow * c];
    for y in 0..oh {
        let sy = ((y as f32 + 0.5) * ih as f32 / oh as f32 - 0.5).max(0.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(ih - 1);
        let fy = sy - y0 as f32;
        for x in 0..ow {
            let sx = ((x as f32 + 0.5) * iw as f32 / ow as f32 - 0.5).max(0.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(iw - 1);
            let fx = sx - x0 as f32;
            for ch in 0..c {
                let v00 = t[(y0 * iw + x0) * c + ch];
                let v01 = t[(y0 * iw + x1) * c + ch];
                let v10 = t[(y1 * iw + x0) * c + ch];
                let v11 = t[(y1 * iw + x1) * c + ch];
                out[(y * ow + x) * c + ch] =
                    (v00 * (1.0 - fy) + v10 * fy) * (1.0 - fx) + (v01 * (1.0 - fy) + v11 * fy) * fx;
            }
        }
    }
    out
}

fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

fn bytemuck_u16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}
// SPDX-License-Identifier: AGPL-3.0-only
// Appended to radio_encoder.rs — stage 2: one ViT block (pre-LN → MHA →
// ×λ → +res → pre-LN → MLP(GELU) → ×λ → +res). λ=1.0 in this checkpoint.

/// One RADIO ViT block's weights (BF16 on device).
pub struct RadioBlock {
    pub norm1_w: DevicePtr,
    pub norm1_b: DevicePtr,
    pub qkv_w: DevicePtr,
    pub qkv_b: DevicePtr, // [3H*D, H*D], [3H*D]
    pub proj_w: DevicePtr,
    pub proj_b: DevicePtr, // [H*D, H*D]
    pub norm2_w: DevicePtr,
    pub norm2_b: DevicePtr,
    pub fc1_w: DevicePtr,
    pub fc1_b: DevicePtr, // [4H*D? no — 4*hidden, hidden]
    pub fc2_w: DevicePtr,
    pub fc2_b: DevicePtr,
}

/// Kernel set shared by embedder+blocks (vision_encoder module, common
/// GEMM for the big matmuls).
pub struct RadioKernels {
    pub k_norm: KernelHandle,           // vision_layer_norm (w,b)
    pub k_gemm_bias: KernelHandle,      // vision_gemm_bias (scalar, correct-any-M)
    pub k_gemm_pipelined: KernelHandle, // dense_gemm_bf16_pipelined (big-M fast path)
    pub k_add_bias: KernelHandle,       // vision_add_bias
    pub k_add: KernelHandle,            // vision_add_inplace
    pub k_gelu: KernelHandle,           // vision_gelu (tanh approx — see test note)
    pub k_attn: KernelHandle,           // vision_attention_rope, driven with
    // cos=1/sin=0 (identity) for RADIO.
    pub k_gemm_f32: KernelHandle, // dense_gemm_bf16_f32out (QK^T scores)
    pub k_softmax: KernelHandle,  // vit_softmax_rows
    pub k_scatter: KernelHandle,  // vit_scatter_head
    pub k_rope_deint: KernelHandle, // vit_rope_deinterleave (identity ropes)
    pub k_f32_bf16: KernelHandle, // vision_f32_to_bf16
    pub k_copy: KernelHandle,     // vision_bf16_copy (pre-LN staging)
}

/// x: [seq, hidden] bf16 in `x`. All scratch is caller-owned (sized at
/// the caller's max num_patches). `seq` includes the prefix tokens —
/// attention is dense over the full row set (RADIO has no masking).
#[allow(clippy::too_many_arguments)]
pub fn radio_block_forward(
    gpu: &dyn GpuBackend,
    kk: &RadioKernels,
    blk: &RadioBlock,
    x: DevicePtr, // [seq, hidden] in/out
    scratch_norm: DevicePtr,
    scratch_qkv: DevicePtr,
    scratch_o: DevicePtr,
    scratch_mlp: DevicePtr,
    rope_cos: DevicePtr, // [seq, head_dim] all-ones
    rope_sin: DevicePtr, // [seq, head_dim] all-zeros
    seq: usize,
    hidden: usize,
    heads: usize,
    intermediate: usize,
    stream: u64,
) -> Result<()> {
    let n = seq * hidden;
    // LN1 -> scratch_norm (vision_layer_norm is in-place: copy x, then norm)
    KernelLaunch::new(gpu, kk.k_copy)
        .grid([div_ceil(n as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(scratch_norm)
        .arg_u32(n as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, kk.k_norm)
        .grid([seq as u32, 1, 1])
        .block([hidden.min(1024) as u32, 1, 1])
        .arg_ptr(scratch_norm)
        .arg_ptr(blk.norm1_w)
        .arg_ptr(blk.norm1_b)
        .arg_u32(seq as u32)
        .arg_u32(hidden as u32)
        .arg_f32(1e-6)
        .launch(stream)?;
    // QKV = norm @ W^T + b   → scratch_qkv [seq, 3*hidden]
    vit_gemm_bias(
        gpu,
        kk,
        scratch_norm,
        blk.qkv_w,
        blk.qkv_b,
        scratch_qkv,
        seq as u32,
        (3 * hidden) as u32,
        hidden as u32,
        stream,
    )?;
    // attention: identity rope → O [seq, hidden]
    let head_dim = hidden / heads;
    KernelLaunch::new(gpu, kk.k_attn)
        .grid([seq as u32, heads as u32, 1])
        .block([32, 1, 1])
        .shared_mem((seq * 4 + 2 * head_dim * 4) as u32)
        .arg_ptr(scratch_qkv)
        .arg_ptr(scratch_o)
        .arg_ptr(rope_cos)
        .arg_ptr(rope_sin)
        .arg_u32(seq as u32)
        .arg_u32(heads as u32)
        .arg_u32(head_dim as u32)
        .launch(stream)?;
    // proj + bias, then residual
    vit_gemm_bias(
        gpu,
        kk,
        scratch_o,
        blk.proj_w,
        blk.proj_b,
        scratch_norm,
        seq as u32,
        hidden as u32,
        hidden as u32,
        stream,
    )?;
    KernelLaunch::new(gpu, kk.k_add)
        .grid([div_ceil(n as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(scratch_norm)
        .arg_u32(n as u32)
        .launch(stream)?;
    // LN2 -> scratch_norm
    KernelLaunch::new(gpu, kk.k_copy)
        .grid([div_ceil(n as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(scratch_norm)
        .arg_u32(n as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, kk.k_norm)
        .grid([seq as u32, 1, 1])
        .block([hidden.min(1024) as u32, 1, 1])
        .arg_ptr(scratch_norm)
        .arg_ptr(blk.norm2_w)
        .arg_ptr(blk.norm2_b)
        .arg_u32(seq as u32)
        .arg_u32(hidden as u32)
        .arg_f32(1e-6)
        .launch(stream)?;
    // fc1+gelu
    vit_gemm_bias(
        gpu,
        kk,
        scratch_norm,
        blk.fc1_w,
        blk.fc1_b,
        scratch_mlp,
        seq as u32,
        intermediate as u32,
        hidden as u32,
        stream,
    )?;
    KernelLaunch::new(gpu, kk.k_gelu)
        .grid([div_ceil((seq * intermediate) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(scratch_mlp)
        .arg_u32((seq * intermediate) as u32)
        .launch(stream)?;
    // fc2 + residual
    vit_gemm_bias(
        gpu,
        kk,
        scratch_mlp,
        blk.fc2_w,
        blk.fc2_b,
        scratch_norm,
        seq as u32,
        hidden as u32,
        intermediate as u32,
        stream,
    )?;
    KernelLaunch::new(gpu, kk.k_add)
        .grid([div_ceil(n as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(scratch_norm)
        .arg_u32(n as u32)
        .launch(stream)
}

/// C[m,n] = A[m,k] @ B[n,k]^T + bias[n] via vision_gemm_bias
/// (scalar-but-correct; the pipelined GEMM + add_bias fast path lands once
/// the stage verifies).
fn vit_gemm_bias(
    gpu: &dyn GpuBackend,
    kk: &RadioKernels,
    a: DevicePtr,
    b: DevicePtr,
    bias: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kk.k_gemm_bias)
        .grid([div_ceil(n, 32), div_ceil(m, 32), 1])
        .block([32, 32, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bias)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
