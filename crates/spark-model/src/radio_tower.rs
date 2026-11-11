// SPDX-License-Identifier: AGPL-3.0-only

//! Production RADIO vision tower for `nemotron_h_omni`: embedder +
//! 32 ViT blocks + `vision_final_layernorm` + pixel-shuffle v2 + `mlp1`
//! (RMSNorm → Linear → ReLU² → Linear). Oracle-verified per stage
//! (`radio_encoder_tests.rs` / `radio_encoder_pipe_tests.rs`).

use anyhow::Result;
use atlas_core::config::OmniVisionConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::radio_encoder::{RadioBlock, RadioEmbedder, radio_block_forward};

/// The whole RADIO tower for one model: weights + kernels + scratch.
pub struct RadioVisionTower {
    pub embedder: RadioEmbedder,
    pub blocks: Vec<RadioBlock>,
    pub final_ln_w: DevicePtr,
    pub final_ln_b: DevicePtr,
    pub mlp1_norm_w: DevicePtr, // [5120] vanilla RMSNorm weight
    pub mlp1_l1_w: DevicePtr,   // [20480, 5120]
    pub mlp1_l2_w: DevicePtr,   // [4096, 20480]
    pub kk_norm: KernelHandle,
    pub kk_gemm_bias: KernelHandle,
    pub kk_gemm_pipelined: KernelHandle,
    pub kk_add_bias: KernelHandle,
    pub kk_add: KernelHandle,
    pub kk_gelu: KernelHandle,  // vision_gelu_erf
    pub kk_attn: KernelHandle,  // vision_attention_rope w/ identity ropes
    pub kk_rms: KernelHandle,   // radio_rms_norm
    pub kk_relu2: KernelHandle, // relu_squared_inplace
    pub kk_copy: KernelHandle,
    /// Scratch: token stream [(p_max+prefix) × hid], norm/qkv/o/mlp
    /// staging at p_max rows, plus shuffled rows.
    pub buf_tokens: DevicePtr,
    pub buf_norm: DevicePtr,
    pub buf_qkv: DevicePtr,
    pub buf_o: DevicePtr,
    pub buf_mlp: DevicePtr,
    pub buf_shuffled: DevicePtr,
    pub buf_pixels: DevicePtr,  // [p_max * 768] bf16 — host patch upload
    pub buf_rho_cos: DevicePtr, // [p_max * head_dim] all-ones
    pub buf_rho_sin: DevicePtr, // [p_max * head_dim] all-zeros
    /// [p_max/4 × 4096] bf16 — packed output rows, the splice source.
    pub buf_out: DevicePtr,
    pub out_hidden_size: usize,
    pub p_max: usize,
    pub head_dim: usize,
    pub intermediate: usize, // ViT MLP inner (5120)
    pub shuffled_w: usize,   // post-shuffle row width (5120)
    pub mlp1_hidden: usize,  // 20480
    pub cfg: OmniVisionConfig,
}

impl RadioVisionTower {
    /// One image: `pixels` [N, 768] f32 host (from
    /// `radio_preprocess::preprocess_image_omni`), `grid` (h_p, w_p).
    /// Writes `h_p*w_p/4` rows of `[4096]` BF16 into `buf_out` at `offset`.
    pub fn encode_image_into(
        &self,
        gpu: &dyn GpuBackend,
        pixels_f32: &[f32],
        grid_h: usize,
        grid_w: usize,
        out_offset_rows: usize,
        stream: u64,
    ) -> Result<()> {
        let n = grid_h * grid_w;
        let hid = self.embedder.hidden;
        let prefix = self.embedder.prefix_tokens;
        anyhow::ensure!(
            n <= self.p_max,
            "image grid {grid_h}x{grid_w}={n} exceeds p_max={} — refuse rather than truncate mid-image",
            self.p_max
        );
        // pixels → bf16 upload
        let bf16: Vec<u16> = pixels_f32
            .iter()
            .map(|&f| ((f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16) as u16)
            .collect();
        gpu.copy_h2d(
            unsafe { std::slice::from_raw_parts(bf16.as_ptr() as *const u8, bf16.len() * 2) },
            self.buf_pixels,
        )?;
        // embedder → x [(prefix+n) × hid] on the dedicated token buffer
        let x = self.buf_tokens;
        self.embedder
            .embed(gpu, self.buf_pixels, grid_h, grid_w, x, stream)?;
        for blk in &self.blocks {
            radio_block_forward(
                gpu,
                &self.kernels(),
                blk,
                x,
                self.buf_norm,
                self.buf_qkv,
                self.buf_o,
                self.buf_mlp,
                self.buf_rho_cos,
                self.buf_rho_sin,
                prefix + n,
                hid,
                16,
                self.intermediate,
                stream,
            )?;
        }
        // strip prefix rows → feats [n × hid]; final LN in-place on a copy
        let feats = self.buf_norm; // reuse [p_max × hid]
        gpu.copy_d2d(x.offset(prefix * hid * 2), feats, n * hid * 2)?;
        KernelLaunch::new(gpu, self.kk_norm)
            .grid([n as u32, 1, 1])
            .block([hid.min(1024) as u32, 1, 1])
            .arg_ptr(feats)
            .arg_ptr(self.final_ln_w)
            .arg_ptr(self.final_ln_b)
            .arg_u32(n as u32)
            .arg_u32(hid as u32)
            .arg_f32(self.cfg.layer_norm_eps as f32)
            .launch(stream)?;
        // pixel-shuffle v2: [h,w,1280] → [h/2,w/2,5120] — host-computed
        // permutation drives a device gather? The perm table is small;
        // do the gather on host into the shuffled buffer.
        let perm = crate::radio_preprocess::pixel_shuffle_v2_permutation(grid_h, grid_w, hid);
        let mut got = vec![0u8; n * hid * 2];
        gpu.copy_d2h(feats, &mut got)?;
        let mut sh = vec![0u8; perm.len() * 2];
        for (dst, &src) in perm.iter().enumerate() {
            let si = src as usize * 2;
            sh[dst * 2] = got[si];
            sh[dst * 2 + 1] = got[si + 1];
        }
        gpu.copy_h2d(&sh, self.buf_shuffled)?;
        let n_tok = perm.len() / self.shuffled_w;
        // mlp1: rms_norm in-place on buf_shuffled → l1 → relu² → l2.
        // buf_mlp (p_max × intermediate = p_max/4 × 20480) holds m1.
        KernelLaunch::new(gpu, self.kk_rms)
            .grid([n_tok as u32, 1, 1])
            .block([self.shuffled_w.min(1024) as u32, 1, 1])
            .arg_ptr(self.buf_shuffled)
            .arg_ptr(self.mlp1_norm_w)
            .arg_ptr(self.buf_shuffled)
            .arg_u32(n_tok as u32)
            .arg_u32(self.shuffled_w as u32)
            .arg_f32(self.cfg.projector_norm_eps as f32)
            .launch(stream)?;
        let m1 = self.buf_mlp;
        KernelLaunch::new(gpu, self.kk_gemm_pipelined)
            .grid([
                div_ceil(self.mlp1_hidden as u32, 128),
                div_ceil(n_tok as u32, 128),
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(self.buf_shuffled)
            .arg_ptr(self.mlp1_l1_w)
            .arg_ptr(m1)
            .arg_u32(n_tok as u32)
            .arg_u32(self.mlp1_hidden as u32)
            .arg_u32(self.shuffled_w as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.kk_relu2)
            .grid([div_ceil((n_tok * self.mlp1_hidden) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(m1)
            .arg_u32((n_tok * self.mlp1_hidden) as u32)
            .launch(stream)?;
        let out = self
            .buf_out
            .offset(out_offset_rows * self.out_hidden_size * 2);
        KernelLaunch::new(gpu, self.kk_gemm_pipelined)
            .grid([
                div_ceil(self.out_hidden_size as u32, 128),
                div_ceil(n_tok as u32, 128),
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(m1)
            .arg_ptr(self.mlp1_l2_w)
            .arg_ptr(out)
            .arg_u32(n_tok as u32)
            .arg_u32(self.out_hidden_size as u32)
            .arg_u32(self.mlp1_hidden as u32)
            .launch(stream)?;
        Ok(())
    }

    fn kernels(&self) -> crate::radio_encoder::RadioKernels {
        crate::radio_encoder::RadioKernels {
            k_norm: self.kk_norm,
            k_gemm_bias: self.kk_gemm_bias,
            k_gemm_pipelined: self.kk_gemm_pipelined,
            k_add_bias: self.kk_add_bias,
            k_add: self.kk_add,
            k_gelu: self.kk_gelu,
            k_attn: self.kk_attn,
            k_gemm_f32: KernelHandle(0),
            k_softmax: KernelHandle(0),
            k_scatter: KernelHandle(0),
            k_rope_deint: KernelHandle(0),
            k_f32_bf16: KernelHandle(0),
            k_copy: self.kk_copy,
        }
    }
}

use crate::weight_map::dense;
use anyhow::Context;
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

fn bf16_bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((((c[1] as u32) << 8) | c[0] as u32) << 16))
        .collect()
}

impl RadioVisionTower {
    /// Load the whole RADIO tower for `nemotron_h_omni`: timm-style
    /// `vision_model.radio_model.*` block weights, `vision_projector.*`
    /// final-LN, `mlp1.{0,1,3}` projector. All dims come from the
    /// checkpoint shapes / `config.vision` — never constants.
    pub fn new(store: &WeightStore, config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<Self> {
        let vp = "vision_model.radio_model.model";
        let vcfg = config
            .vision
            .as_ref()
            .context("nemotron_h_omni requires config.vision")?;
        let omni = vcfg
            .omni
            .clone()
            .context("nemotron_h_omni vision config missing omni params")?;
        let hid = vcfg.hidden_size;
        let patch = vcfg.patch_size;
        let inter = vcfg.intermediate_size;
        let heads = vcfg.num_heads.max(1);
        let out_h = config.hidden_size;

        let patch_w = dense(store, &format!("{vp}.patch_generator.embedder.weight"))?.weight;
        let pos_shape = store
            .get(&format!("{vp}.patch_generator.pos_embed"))?
            .shape
            .clone();
        let n_pos = pos_shape
            .iter()
            .copied()
            .find(|&d| d > 1 && d != hid)
            .context("pos_embed shape has no position dim")?;
        let pos_embed_dev = dense(store, &format!("{vp}.patch_generator.pos_embed"))?.weight;
        let cls_register = dense(store, &format!("{vp}.patch_generator.cls_token.token"))?.weight;
        let side = (n_pos as f64).sqrt() as usize;
        let mut pe_bytes = vec![0u8; n_pos * hid * 2];
        gpu.copy_d2h(pos_embed_dev, &mut pe_bytes)?;
        let pos_embed_host = bf16_bytes_to_f32(&pe_bytes);

        let mlp1_hidden = store
            .get("mlp1.1.weight")?
            .shape
            .first()
            .copied()
            .context("mlp1.1.weight missing rows")?;

        let p_max = omni.max_num_patches.max(1);
        let prefix = omni.prefix_tokens;
        let head_dim = hid / heads;

        let blocks = (0..vcfg.depth)
            .map(|i| {
                let bp = format!("{vp}.blocks.{i}");
                Ok(RadioBlock {
                    norm1_w: dense(store, &format!("{bp}.norm1.weight"))?.weight,
                    norm1_b: dense(store, &format!("{bp}.norm1.bias"))?.weight,
                    qkv_w: dense(store, &format!("{bp}.attn.qkv.weight"))?.weight,
                    qkv_b: dense(store, &format!("{bp}.attn.qkv.bias"))?.weight,
                    proj_w: dense(store, &format!("{bp}.attn.proj.weight"))?.weight,
                    proj_b: dense(store, &format!("{bp}.attn.proj.bias"))?.weight,
                    norm2_w: dense(store, &format!("{bp}.norm2.weight"))?.weight,
                    norm2_b: dense(store, &format!("{bp}.norm2.bias"))?.weight,
                    fc1_w: dense(store, &format!("{bp}.mlp.fc1.weight"))?.weight,
                    fc1_b: dense(store, &format!("{bp}.mlp.fc1.bias"))?.weight,
                    fc2_w: dense(store, &format!("{bp}.mlp.fc2.weight"))?.weight,
                    fc2_b: dense(store, &format!("{bp}.mlp.fc2.bias"))?.weight,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let final_ln_w = dense(store, "vision_projector.vision_final_layernorm.weight")?.weight;
        let final_ln_b = dense(store, "vision_projector.vision_final_layernorm.bias")?.weight;
        let mlp1_norm_w = dense(store, "mlp1.0.weight")?.weight;
        let mlp1_l1_w = dense(store, "mlp1.1.weight")?.weight;
        let mlp1_l2_w = dense(store, "mlp1.3.weight")?.weight;

        // Scratch sized at the checkpoint's `max_num_patches` (+ prefix).
        let rows = p_max + prefix;
        let buf_tokens = gpu.alloc(rows * hid * 2)?;
        let buf_norm = gpu.alloc(rows * hid * 2)?;
        let buf_qkv = gpu.alloc(rows * 3 * hid * 2)?;
        let buf_o = gpu.alloc(rows * hid * 2)?;
        // buf_mlp doubles as the mlp1 l1 output: rows/4 × mlp1_hidden
        // (== rows × inter for mlp1_hidden = 4*inter).
        let buf_mlp = gpu.alloc(rows * inter.max(mlp1_hidden / 4) * 2)?;
        let buf_shuffled = gpu.alloc(rows * inter * 2)?;
        let buf_pixels = gpu.alloc(p_max * 3 * patch * patch * 2)?;
        let buf_rho_cos = gpu.alloc(rows * head_dim * 2)?;
        let buf_rho_sin = gpu.alloc(rows * head_dim * 2)?;
        let buf_out = gpu.alloc((rows / 4) * out_h * 2)?;
        // identity ropes: BF16 1.0/0.0 over [rows × head_dim]
        let mut cosv = vec![0u8; rows * head_dim * 2];
        for ch in cosv.chunks_exact_mut(2) {
            ch.copy_from_slice(&[0x80, 0x3F]);
        }
        gpu.copy_h2d(&cosv, buf_rho_cos)?;
        gpu.copy_h2d(&vec![0u8; rows * head_dim * 2], buf_rho_sin)?;
        let buf_pos = gpu.alloc(rows * hid * 2)?;

        let embedder = RadioEmbedder {
            patch_w,
            pos_embed_dev,
            pos_embed_host,
            cls_register,
            max_rows: side,
            max_cols: side,
            hidden: hid,
            patch,
            prefix_tokens: prefix,
            k_gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            k_add: gpu.kernel("vision_encoder", "vision_add_inplace")?,
            buf_pos,
            buf_pos_cap: rows * hid,
        };

        Ok(Self {
            embedder,
            blocks,
            final_ln_w,
            final_ln_b,
            mlp1_norm_w,
            mlp1_l1_w,
            mlp1_l2_w,
            kk_norm: gpu.kernel("vision_encoder", "vision_layer_norm")?,
            kk_gemm_bias: gpu.kernel("vision_encoder", "vision_gemm_bias")?,
            kk_gemm_pipelined: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            kk_add_bias: gpu.kernel("vision_encoder", "vision_add_bias")?,
            kk_add: gpu.kernel("vision_encoder", "vision_add_inplace")?,
            kk_gelu: gpu.kernel("vision_encoder", "vision_gelu_erf")?,
            kk_attn: gpu.kernel("vision_encoder", "vision_attention_rope")?,
            kk_rms: gpu.kernel("vision_encoder", "radio_rms_norm")?,
            kk_relu2: gpu.kernel("relu2", "relu_squared_inplace")?,
            kk_copy: gpu.kernel("vision_encoder", "vision_bf16_copy")?,
            buf_tokens,
            buf_norm,
            buf_qkv,
            buf_o,
            buf_mlp,
            buf_shuffled,
            buf_pixels,
            buf_rho_cos,
            buf_rho_sin,
            buf_out,
            out_hidden_size: out_h,
            p_max,
            head_dim,
            intermediate: inter,
            shuffled_w: inter,
            mlp1_hidden,
            cfg: omni,
        })
    }

    /// `forward_batched` parity with `VisionEncoder`: per item
    /// `(pixels, grid_h, grid_w)` → encode, pack rows into `buf_out`.
    /// Returns `(h/2, w/2, merged_p)` per item.
    pub fn forward_batched(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let mut out = Vec::with_capacity(images.len());
        let mut row_off = 0usize;
        for (px, gh, gw) in images {
            let merged = gh * gw / 4;
            self.encode_image_into(gpu, px, *gh, *gw, row_off, stream)?;
            out.push((gh / 2, gw / 2, merged));
            row_off += merged;
        }
        Ok(out)
    }
}
