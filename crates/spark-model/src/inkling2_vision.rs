// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2's vision tower (`model.visual.*`, transformers
//! `Inkling2VisionModel`): one image token per 40x40 pixel tile.
//!
//! ```text
//! patches [n, 3*20*20] -> Linear (no bias) + learned position (the 32x32
//!   table resampled bilinearly to the image's patch grid)
//! [CLS + cls position] ++ patches -> LayerNorm
//! 23 x { x += ls1 * out(attn(LN(x)));  x += ls2 * fc2(gelu(fc1(LN(x)))) }
//! patches (CLS dropped), each 2x2 tile concatenated [4*1024]
//!   -> proj_in -> gelu -> proj_out [2304]   (no biases, no final norm)
//! ```
//!
//! Attention is dense over the image's patches and its CLS token, with axial
//! 2D RoPE on Q and K: of a head's 64 channels, pairs `(2i, 2i+1)` rotate by
//! `col * f_i` for `i < 16` and `row * f_(i-16)` after, `f_j = theta^(-j/16)`,
//! patch rows and columns counted from 1 and the CLS token at (0, 0).
//!
//! The shared ViT kernels (`vision_encoder`) rotate pairs `(d, d + 32)` and
//! know no layer scale, so the weights are rearranged once at load:
//!  * `q_proj` / `k_proj` rows are permuted inside each head so reference
//!    channel `2i` sits at `i` and `2i+1` at `i + 32` (Q·K is unchanged by a
//!    permutation applied to both), then fused with `v_proj` as the kernels'
//!    `[seq, 3*H*D]` QKV;
//!  * `ls_1` is folded into `out_proj`'s rows and bias, `ls_2` into `c_proj`'s.
//!
//! Patches arrive tile by tile (`vision_preprocess::inkling2`), so a tile's
//! four states are adjacent rows and the 2x2 concatenation is the same memory
//! read as `[n/4, 4*hidden]`.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::WeightStore;

const ROPE_THETA: f64 = 10_000.0;
/// Requests may carry several images; `buf_out` holds this many full ones.
const OUT_IMAGES: usize = 4;

struct Block {
    ln1: (DevicePtr, DevicePtr),
    qkv: (DevicePtr, DevicePtr),
    out: (DevicePtr, DevicePtr),
    ln2: (DevicePtr, DevicePtr),
    fc1: (DevicePtr, DevicePtr),
    fc2: (DevicePtr, DevicePtr),
}

struct Kernels {
    norm: KernelHandle,
    gemm: KernelHandle,
    gemm_f32: KernelHandle,
    add_bias: KernelHandle,
    add: KernelHandle,
    gelu: KernelHandle,
    rope_deint: KernelHandle,
    softmax: KernelHandle,
    scatter: KernelHandle,
    copy: KernelHandle,
}

pub struct Inkling2VisionTower {
    patch_w: DevicePtr,
    /// The learned `[side*side, hidden]` position table, on the host.
    pos_table: Vec<f32>,
    side: usize,
    /// `class_embedding + cls_position_embedding`, BF16 bytes.
    cls_row: Vec<u8>,
    ln_pre: (DevicePtr, DevicePtr),
    blocks: Vec<Block>,
    proj_in_w: DevicePtr,
    proj_out_w: DevicePtr,
    k: Kernels,
    buf_pixels: DevicePtr,
    buf_x: DevicePtr,
    buf_norm: DevicePtr,
    buf_qkv: DevicePtr,
    buf_o: DevicePtr,
    buf_mlp: DevicePtr,
    buf_pos: DevicePtr,
    buf_cos: DevicePtr,
    buf_sin: DevicePtr,
    buf_qr: DevicePtr,
    buf_kr: DevicePtr,
    buf_vt: DevicePtr,
    buf_scores: DevicePtr,
    buf_probs: DevicePtr,
    buf_o_stage: DevicePtr,
    /// `[out_rows, out_hidden]` BF16: the packed image tokens, the splice source.
    pub buf_out: DevicePtr,
    pub out_hidden_size: usize,
    /// Patches one image may have (a multiple of `merge²`).
    pub p_max: usize,
    hidden: usize,
    heads: usize,
    inter: usize,
    patch_dim: usize,
    merge: usize,
    proj_hidden: usize,
    eps: f32,
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

impl Inkling2VisionTower {
    pub fn new(store: &WeightStore, config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<Self> {
        let v = config
            .vision
            .as_ref()
            .filter(|v| v.is_inkling2)
            .context("inkling2 vision tower needs config.vision")?;
        let (hidden, heads, inter, merge) = (
            v.hidden_size,
            v.num_heads,
            v.intermediate_size,
            v.spatial_merge_size,
        );
        let head_dim = hidden / heads;
        let patch_dim = v.in_channels * v.patch_size * v.patch_size;
        let proj_hidden = v.projection_intermediate_size;
        let host = |name: &str, shape: &[usize]| -> Result<Vec<u8>> {
            let t = store.get(name)?;
            ensure!(
                t.shape == shape,
                "{name} has shape {:?}, expected {shape:?}",
                t.shape
            );
            let mut bytes = vec![0u8; t.byte_size()];
            gpu.copy_d2h(t.ptr, &mut bytes)?;
            Ok(bytes)
        };
        let dev = |name: &str, shape: &[usize]| -> Result<DevicePtr> {
            let t = store.get(name)?;
            ensure!(
                t.shape == shape,
                "{name} has shape {:?}, expected {shape:?}",
                t.shape
            );
            Ok(t.ptr)
        };
        let pair = |name: &str, n: usize, k: usize| -> Result<(DevicePtr, DevicePtr)> {
            Ok((
                dev(&format!("{name}.weight"), &[n, k])?,
                dev(&format!("{name}.bias"), &[n])?,
            ))
        };
        let norm = |name: &str| -> Result<(DevicePtr, DevicePtr)> {
            Ok((
                dev(&format!("{name}.weight"), &[hidden])?,
                dev(&format!("{name}.bias"), &[hidden])?,
            ))
        };

        let emb = "model.visual.backbone.embeddings";
        let pos_shape = store
            .get(&format!("{emb}.position_embedding"))?
            .shape
            .clone();
        let side = (pos_shape[0] as f64).sqrt() as usize;
        ensure!(
            pos_shape == [side * side, hidden],
            "inkling2 position table {pos_shape:?} is not a square grid of {hidden}-wide rows"
        );
        let pos_table = bf16_to_f32(&host(&format!("{emb}.position_embedding"), &pos_shape)?);
        let cls: Vec<f32> = bf16_to_f32(&host(&format!("{emb}.class_embedding"), &[hidden])?)
            .iter()
            .zip(bf16_to_f32(&host(
                &format!("{emb}.cls_position_embedding"),
                &[hidden],
            )?))
            .map(|(a, b)| a + b)
            .collect();

        let mut blocks = Vec::with_capacity(v.depth);
        for i in 0..v.depth {
            let b = format!("model.visual.backbone.layers.{i}");
            let wh = [hidden, hidden];
            let mut qkv_w = Vec::with_capacity(3 * hidden * hidden * 2);
            let mut qkv_b = Vec::with_capacity(3 * hidden * 2);
            for part in ["q_proj", "k_proj"] {
                let w = host(&format!("{b}.self_attn.{part}.weight"), &wh)?;
                let bias = host(&format!("{b}.self_attn.{part}.bias"), &[hidden])?;
                qkv_w.extend(permute_head_rows(&w, heads, head_dim, hidden));
                qkv_b.extend(permute_head_rows(&bias, heads, head_dim, 1));
            }
            qkv_w.extend(host(&format!("{b}.self_attn.v_proj.weight"), &wh)?);
            qkv_b.extend(host(&format!("{b}.self_attn.v_proj.bias"), &[hidden])?);
            let scaled = |name: &str, gamma: &str, k: usize| -> Result<(DevicePtr, DevicePtr)> {
                let g = bf16_to_f32(&host(&format!("{b}.{gamma}.gamma"), &[hidden])?);
                let w = host(&format!("{b}.{name}.weight"), &[hidden, k])?;
                let bias = host(&format!("{b}.{name}.bias"), &[hidden])?;
                Ok((
                    upload(gpu, &scale_rows(&w, &g, k))?,
                    upload(gpu, &scale_rows(&bias, &g, 1))?,
                ))
            };
            blocks.push(Block {
                ln1: norm(&format!("{b}.ln_1"))?,
                qkv: (upload(gpu, &qkv_w)?, upload(gpu, &qkv_b)?),
                out: scaled("self_attn.out_proj", "ls_1", hidden)?,
                ln2: norm(&format!("{b}.ln_2"))?,
                fc1: pair(&format!("{b}.mlp.c_fc"), inter, hidden)?,
                fc2: scaled("mlp.c_proj", "ls_2", inter)?,
            });
            // The rearranged copies replace these.
            for name in [
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                "mlp.c_proj.weight",
                "mlp.c_proj.bias",
            ] {
                store.reclaim(gpu, &format!("{b}.{name}"))?;
            }
        }

        let p_max = {
            let p = crate::layers::vision_encoder::enc_impl::init::derive_max_patches(
                v.max_pixels,
                v.patch_size,
            )
            .0;
            (p / (merge * merge)).max(1) * merge * merge
        };
        let seq = p_max + 1;
        let out_rows = OUT_IMAGES * p_max / (merge * merge);
        let ve = "vision_encoder";
        Ok(Self {
            patch_w: dev(
                &format!("{emb}.patch_embedding.weight"),
                &[hidden, patch_dim],
            )?,
            pos_table,
            side,
            cls_row: f32_to_bf16(&cls),
            ln_pre: norm("model.visual.backbone.ln_pre")?,
            blocks,
            proj_in_w: dev(
                "model.visual.proj_in.weight",
                &[proj_hidden, merge * merge * hidden],
            )?,
            proj_out_w: dev(
                "model.visual.proj_out.weight",
                &[config.hidden_size, proj_hidden],
            )?,
            k: Kernels {
                norm: gpu.kernel(ve, "vision_layer_norm")?,
                gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
                gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
                add_bias: gpu.kernel(ve, "vision_add_bias")?,
                add: gpu.kernel(ve, "vision_add_inplace")?,
                gelu: gpu.kernel(ve, "vision_gelu_erf")?,
                rope_deint: gpu.kernel(ve, "vit_rope_deinterleave")?,
                softmax: gpu.kernel(ve, "vit_softmax_rows")?,
                scatter: gpu.kernel(ve, "vit_scatter_head")?,
                copy: gpu.kernel(ve, "vision_bf16_copy")?,
            },
            buf_pixels: gpu.alloc(p_max * patch_dim * 2)?,
            buf_x: gpu.alloc(seq * hidden * 2)?,
            buf_norm: gpu.alloc(seq * hidden * 2)?,
            buf_qkv: gpu.alloc(seq * 3 * hidden * 2)?,
            buf_o: gpu.alloc(seq * hidden * 2)?,
            buf_mlp: gpu.alloc(seq * inter.max(proj_hidden) * 2)?,
            buf_pos: gpu.alloc(p_max * hidden * 2)?,
            buf_cos: gpu.alloc(seq * head_dim * 2)?,
            buf_sin: gpu.alloc(seq * head_dim * 2)?,
            buf_qr: gpu.alloc(seq * hidden * 2)?,
            buf_kr: gpu.alloc(seq * hidden * 2)?,
            buf_vt: gpu.alloc(seq * hidden * 2)?,
            buf_scores: gpu.alloc(seq * seq * 4)?,
            buf_probs: gpu.alloc(seq * seq * 2)?,
            buf_o_stage: gpu.alloc(seq * head_dim * 2)?,
            buf_out: gpu.alloc(out_rows * config.hidden_size * 2)?,
            out_hidden_size: config.hidden_size,
            p_max,
            hidden,
            heads,
            inter,
            patch_dim,
            merge,
            proj_hidden,
            eps: v.rms_norm_eps as f32,
        })
    }

    /// Image tokens `buf_out` holds for one request.
    pub fn output_rows(&self) -> usize {
        OUT_IMAGES * self.p_max / (self.merge * self.merge)
    }

    /// Encode every `(pixels, grid_h, grid_w)` into `buf_out`, packed in
    /// order; per image `(tiles high, tiles wide, tokens)`.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let m2 = self.merge * self.merge;
        let mut out = Vec::with_capacity(images.len());
        let mut row = 0usize;
        for &(pixels, gh, gw) in images {
            let tokens = gh * gw / m2;
            ensure!(
                row + tokens <= self.output_rows(),
                "request's images need more than the tower's {} image tokens",
                self.output_rows()
            );
            self.encode(gpu, pixels, gh, gw, row, stream)?;
            out.push((gh / self.merge, gw / self.merge, tokens));
            row += tokens;
        }
        Ok(out)
    }

    fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        w: DevicePtr,
        c: DevicePtr,
        mnk: [usize; 3],
        stream: u64,
    ) -> Result<()> {
        let [m, n, k] = mnk.map(|v| v as u32);
        KernelLaunch::new(gpu, self.k.gemm)
            .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(w)
            .arg_ptr(c)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)
    }

    fn linear(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        wb: (DevicePtr, DevicePtr),
        c: DevicePtr,
        mnk: [usize; 3],
        stream: u64,
    ) -> Result<()> {
        self.gemm(gpu, a, wb.0, c, mnk, stream)?;
        let (m, n) = (mnk[0] as u32, mnk[1] as u32);
        KernelLaunch::new(gpu, self.k.add_bias)
            .grid([div_ceil(m * n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c)
            .arg_ptr(wb.1)
            .arg_u32(m)
            .arg_u32(n)
            .launch(stream)
    }

    /// `dst = LayerNorm(src)` over `rows` (`dst == src`: in place).
    fn layer_norm(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst: DevicePtr,
        wb: (DevicePtr, DevicePtr),
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let n = (rows * self.hidden) as u32;
        if src != dst {
            KernelLaunch::new(gpu, self.k.copy)
                .grid([div_ceil(n, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(src)
                .arg_ptr(dst)
                .arg_u32(n)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.k.norm)
            .grid([rows as u32, 1, 1])
            .block([self.hidden.min(1024) as u32, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(wb.0)
            .arg_ptr(wb.1)
            .arg_u32(rows as u32)
            .arg_u32(self.hidden as u32)
            .arg_f32(self.eps)
            .launch(stream)
    }

    fn elementwise(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        ptrs: &[DevicePtr],
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let mut launch = KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1]);
        for &p in ptrs {
            launch = launch.arg_ptr(p);
        }
        launch.arg_u32(n as u32).launch(stream)
    }
}

#[path = "inkling2_vision/encode.rs"]
mod encode;
#[path = "inkling2_vision/host.rs"]
mod host;
use host::*;
#[cfg(test)]
#[path = "inkling2_vision/tests.rs"]
mod tests;
