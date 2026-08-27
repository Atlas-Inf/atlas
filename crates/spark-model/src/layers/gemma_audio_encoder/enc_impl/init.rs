// SPDX-License-Identifier: AGPL-3.0-only

//! `GemmaAudioEncoder::new` constructor: geometry validation (fail fast),
//! one-clip-cap scratch allocation, kernel-handle resolution (reused vs
//! Wave-4C stubs) and the host-side relative-position-key precompute.

use anyhow::{Result, ensure};
use atlas_core::config::GemmaAudioConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{GemmaAudioEncoder, GemmaAudioWeights, OUT_HIDDEN_SIZE};
use super::f32_to_bf16_bits;
use crate::media::gemma_audio::{rel_pos_embeddings, validate_audio_geometry};

impl GemmaAudioEncoder {
    /// Build the audio tower from loaded weights + config.
    ///
    /// Weights come as already-loaded BF16 device pointers
    /// ([`GemmaAudioWeights`]); the loader (Wave 4B) slices the checkpoint's
    /// `model.audio_tower.*` / `model.embed_audio.*` tensors into them. The
    /// constructor validates the geometry against [`GemmaAudioConfig`]
    /// (heads divide hidden, layer count, two conv stages, mel/flatten
    /// relations, `output_proj_dims == OUT_HIDDEN_SIZE`, silu activation),
    /// allocates the one-clip scratch set, resolves kernel handles (shared
    /// kernels hard/soft, gemma-specific ones as documented Wave-4C stubs),
    /// and precomputes the per-layer relative position keys on the host
    /// (the `pos_emb` table is pure config math; the `relative_k_proj`
    /// weight is downloaded once per layer).
    pub fn new(
        w: &GemmaAudioWeights,
        cfg: &GemmaAudioConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let head_dim = hidden / heads;
        let inter = 4 * hidden; // HF hardcodes the FFN width at 4×hidden.
        let t_max = cfg.token_cap;
        let chunk = cfg.attention_chunk_size;
        let max_past = cfg.attention_context_left.saturating_sub(1);
        let max_future = cfg.attention_context_right;
        let context = chunk + max_past + max_future;
        let mel = cfg.mel_bins;
        let c1 = cfg.subsampling_conv_channels[1];
        // Flatten width per time row after the convs: (mel/4) × last channels
        // (32×32 = 1024 on the shipped tower — VERIFIED: the checkpoint's
        // `input_proj_linear.weight` is [1024, 1024]).
        let flatten = (mel / 4) * c1;

        // ── Geometry validation (PCND: refuse, don't silently stretch) ──
        validate_audio_geometry(cfg)?;
        ensure!(
            w.layers.len() == cfg.num_hidden_layers,
            "gemma audio: {num} layer weight sets for {cfg} layers",
            num = w.layers.len(),
            cfg = cfg.num_hidden_layers
        );
        ensure!(
            cfg.output_proj_dims == OUT_HIDDEN_SIZE,
            "gemma audio: output_proj_dims {} != OUT_HIDDEN_SIZE {OUT_HIDDEN_SIZE} — the \
             text splice copies buf_out rows verbatim",
            cfg.output_proj_dims
        );
        ensure!(
            cfg.activation == "silu",
            "gemma audio: activation {:?} — the encoder implements silu only",
            cfg.activation
        );
        ensure!(
            chunk >= 1 && cfg.attention_context_left >= 1,
            "gemma audio: chunk {chunk} / context_left {} must be >= 1",
            cfg.attention_context_left
        );
        ensure!(
            cfg.conv_kernel_size >= 1,
            "gemma audio: conv_kernel_size must be >= 1"
        );
        ensure!(head_dim >= 1, "gemma audio: head_dim must be >= 1");

        // ── Scratch: one clip's caps (see module docs) ──
        // buf_mel stages the 4× mel budget (t_max token rows ← 4×t_max mel
        // frames); every row buffer is t_max rows wide.
        let bf16 = |n: usize| -> Result<DevicePtr> { gpu.alloc(n * 2) };
        let buf_mel = bf16(4 * t_max * mel)?;
        let buf_mask_mel = gpu.alloc(4 * t_max)?; // u8 per mel frame
        let buf_mask_attn = gpu.alloc(t_max * chunk * context)?; // u8 blocked masks
        let buf_conv = bf16(t_max * flatten)?;
        // Subsample conv1 intermediate: [2×t_max rows × mel/2 freq × c0 ch]
        // (conv1 rows = ceil(4×t_max/2) = 2×t_max at the mel budget cap).
        let c0 = cfg.subsampling_conv_channels[0];
        let buf_conv1 = bf16(2 * t_max * (mel / 2) * c0)?;
        let buf_h1 = bf16(t_max * hidden)?;
        let buf_h2 = bf16(t_max * hidden)?;
        let buf_qkv = bf16(t_max * 3 * hidden)?;
        let buf_mlp = bf16(t_max * hidden)?;
        let buf_wide = bf16(t_max * 2 * hidden)?; // light-conv GLU staging
        let buf_ffn = bf16(t_max * inter)?;
        let buf_proj = bf16(t_max * OUT_HIDDEN_SIZE)?;
        let buf_out = bf16(t_max * OUT_HIDDEN_SIZE)?;
        let norm_unit_w = bf16(OUT_HIDDEN_SIZE)?;

        // ── Kernel handles ──
        // Reused (shape-compatible) kernels: hard where the tree always
        // ships them, soft otherwise (try_kernel → null → launch_optional).
        let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16")?;
        let k_rms_norm = gpu.kernel("norm", "rms_norm")?;
        let k_sigmoid_gate = gpu.kernel("residual_add", "sigmoid_gate_mul")?;
        let k_scaled_add = gpu.kernel("residual_add", "bf16_scaled_add")?;
        let k_add = crate::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace");
        // ClippableLinear clamp: REUSED from the vision tree (Wave 3) — same
        // class, same `(buf, lo, hi, n)` contract. Null today → no-op.
        let k_clamp = crate::layers::try_kernel(gpu, "gemma_vision", "gemma_vision_clamp");
        // Gemma-specific kernels — Wave 4C adds them to the gemma-4-e2b tree
        // under the `gemma_audio` module. Null today → launch_optional no-ops.
        let k_silu = crate::layers::try_kernel(gpu, "gemma_audio", "gemma_audio_silu");
        let k_subsample_conv1 =
            crate::layers::try_kernel(gpu, "gemma_audio", "gemma_audio_subsample_conv1");
        let k_subsample_conv2 =
            crate::layers::try_kernel(gpu, "gemma_audio", "gemma_audio_subsample_conv2");
        let k_chunked_attn =
            crate::layers::try_kernel(gpu, "gemma_audio", "gemma_audio_chunked_attn");
        let k_conv1d = crate::layers::try_kernel(gpu, "gemma_audio", "gemma_audio_conv1d");
        let k_bias_add = crate::layers::try_kernel(gpu, "gemma_audio", "gemma_audio_bias_add");

        // ── Host-side prep state ──
        // Relative position keys per layer: rel_k = relative_k_proj(pos_emb),
        // [context/2+1, hidden] — the fixed sinusoid table projected ONCE at
        // init (HF computes it every forward from non-learned buffers; the
        // projection weight is the only learned part). Host f32 math → BF16
        // upload, same idiom as the vision encoder's host position table.
        let n_pos = context / 2 + 1;
        let pos_emb = rel_pos_embeddings(hidden, context);
        let mut relative_k = Vec::with_capacity(w.layers.len());
        for lyr in &w.layers {
            let w_n = hidden * hidden;
            let mut bytes = vec![0u8; w_n * 2];
            gpu.copy_d2h(lyr.self_attn.relative_k_proj.weight, &mut bytes)?;
            let wf: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            // rel_k[i][j] = Σ_k pos_emb[i][k] × W[j][k] (row-major [n_pos, hidden]).
            let mut rk = vec![0.0f32; n_pos * hidden];
            for i in 0..n_pos {
                for j in 0..hidden {
                    let mut acc = 0.0f64;
                    for k in 0..hidden {
                        acc += pos_emb[i * hidden + k] as f64 * wf[j * hidden + k] as f64;
                    }
                    rk[i * hidden + j] = acc as f32;
                }
            }
            let b16: Vec<u16> = rk.iter().map(|&v| f32_to_bf16_bits(v)).collect();
            let ptr = bf16(n_pos * hidden)?;
            // SAFETY: `b16` is a live `vec![u16; n_pos*hidden]`; byte length
            // derived from the same Vec; u16/u8 have no invalid bit patterns.
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(b16.as_ptr() as *const u8, b16.len() * 2) };
            gpu.copy_h2d(bytes, ptr)?;
            relative_k.push(ptr);
        }

        // ── Per-layer spd = softplus(per_dim_scale) ──
        // The chunked-attention kernel's `spd` argument: HF applies
        // `F.softplus(per_dim_scale)` once and the kernel reads the result
        // as a per-dim query pre-scale. Host-side f32 softplus (matches the
        // CUDA kernel's threshold: x > 20 → identity), BF16 upload.
        let mut spd_bufs = Vec::with_capacity(w.layers.len());
        for lyr in &w.layers {
            let mut bytes = vec![0u8; head_dim * 2];
            gpu.copy_d2h(lyr.self_attn.per_dim_scale.weight, &mut bytes)?;
            let spd: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    let x = f32::from_bits((bits as u32) << 16);
                    let v = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                    f32_to_bf16_bits(v)
                })
                .collect();
            let ptr = bf16(head_dim)?;
            // SAFETY: `spd` is a live `vec![u16; head_dim]`; byte length
            // derived from the same Vec; u16/u8 have no invalid bit patterns.
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(spd.as_ptr() as *const u8, spd.len() * 2) };
            gpu.copy_h2d(bytes, ptr)?;
            spd_bufs.push(ptr);
        }

        // embed_audio RMSNorm(with_scale=False): ones weight (the checkpoint
        // ships no norm tensor for it — verified — pure `x·rms`).
        let ones: Vec<u16> = std::iter::repeat_n(f32_to_bf16_bits(1.0), OUT_HIDDEN_SIZE).collect();
        // SAFETY: `ones` is a live `vec![u16; OUT_HIDDEN_SIZE]`; byte length
        // derived from that same Vec; u16/u8 have no invalid bit patterns.
        let ones_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 2) };
        gpu.copy_h2d(ones_bytes, norm_unit_w)?;

        Ok(GemmaAudioEncoder {
            subsample_conv0_w: w.subsample.conv0.weight,
            subsample_ln0_w: w.subsample.ln0.weight,
            subsample_conv1_w: w.subsample.conv1.weight,
            subsample_ln1_w: w.subsample.ln1.weight,
            subsample_proj_w: w.subsample.input_proj_linear.weight,
            layers: w.layers.clone(),
            output_proj_w: w.output_proj.weight.weight,
            output_proj_b: w.output_proj.bias.weight,
            embed_audio_proj_w: w.embed_audio_projection.weight,
            k_gemm,
            k_rms_norm,
            k_add,
            k_sigmoid_gate,
            k_scaled_add,
            k_clamp,
            k_silu,
            k_subsample_conv1,
            k_subsample_conv2,
            k_chunked_attn,
            k_conv1d,
            k_bias_add,
            relative_k,
            spd_bufs,
            buf_mel,
            buf_mask_mel,
            buf_mask_attn,
            buf_conv,
            buf_conv1,
            buf_h1,
            buf_h2,
            buf_qkv,
            buf_mlp,
            buf_wide,
            buf_ffn,
            buf_proj,
            buf_out,
            norm_unit_w,
            hidden_size: hidden,
            num_heads: heads,
            head_dim,
            intermediate_size: inter,
            chunk_size: chunk,
            max_past,
            max_future,
            context_size: context,
            conv_kernel: cfg.conv_kernel_size,
            mel_bins: mel,
            flatten_dim: flatten,
            t_max,
            norm_eps: cfg.norm_eps,
            residual_weight: cfg.residual_weight as f32,
            // HF Gemma4AudioAttention scales: q_scale = head_dim^-0.5 / ln 2,
            // k_scale = ln(1+e) / ln 2.
            q_scale: (head_dim as f32).powf(-0.5) / std::f32::consts::LN_2,
            k_scale: (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2,
            total_soft_tokens: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}
