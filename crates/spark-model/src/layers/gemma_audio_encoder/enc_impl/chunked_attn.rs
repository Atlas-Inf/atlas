// SPDX-License-Identifier: AGPL-3.0-only

//! Chunked local attention sub-block: `norm_pre_attn` → clipped q/k/v GEMMs
//! → q/k scale step (Wave-4C `gemma_audio_softplus`) → chunked attention
//! (Wave-4C `gemma_audio_chunked_attn`) → clipped `post` GEMM →
//! `norm_post_attn` → residual add. The M-agnostic GEMMs/norms run once over
//! the packed batch; the attention launches per clip over its disjoint
//! q/k/v slices.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{
    ATTENTION_INVALID_LOGITS, ATTENTION_LOGIT_CAP, GemmaAudioEncoder, GemmaAudioLayerWeights,
};
use super::launch_optional;
use crate::layers::ops;

impl GemmaAudioEncoder {
    /// Run one clip's chunked attention over its disjoint q/k/v slices, with
    /// its precomputed `rel_k`, `spd` (softplus(per_dim_scale)) and blocked
    /// attention mask.
    ///
    /// Contract (gemma_audio_encoder.cu, verified against the PTX): the
    /// chunked-attn kernel takes RAW q/k/v — it applies q_scale, k_scale and
    /// the `spd` query pre-scale INTERNALLY (HF `Gemma4AudioAttention` math),
    /// so no host-side q/k scaling step runs here. Grid is one block per
    /// 12-token chunk (NOT per head — each warp handles one query, 4 lanes
    /// per head cover head_dim in 4-stride dims).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn chunked_attn_stage(
        &self,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        o: DevicePtr,
        spd: DevicePtr,
        rel_k: DevicePtr,
        mask: DevicePtr,
        seq: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let s = seq as u32;
        let heads = self.num_heads as u32;
        let hd = self.head_dim as u32;
        // Chunked attention: one block per (query-block), each warp handling
        // one query-in-chunk (4 lanes per head × heads = 32 lanes).
        let nblocks = seq.div_ceil(self.chunk_size);
        launch_optional(
            gpu,
            self.k_chunked_attn,
            "gemma_audio_chunked_attn",
            stream,
            |kl| {
                kl.grid([nblocks as u32, 1, 1])
                    .block([(self.chunk_size * 32) as u32, 1, 1])
                    .arg_ptr(q)
                    .arg_ptr(k)
                    .arg_ptr(v)
                    .arg_ptr(spd)
                    .arg_ptr(rel_k)
                    .arg_ptr(mask)
                    .arg_ptr(o)
                    .arg_u32(s)
                    .arg_u32(heads)
                    .arg_u32(hd)
                    .arg_f32(ATTENTION_LOGIT_CAP)
                    .arg_f32(ATTENTION_INVALID_LOGITS)
                    .arg_u32(self.chunk_size as u32)
                    .arg_u32((self.max_past + 1) as u32)
                    .arg_u32(self.max_future as u32)
            },
        )
    }

    /// Attention sub-block over the packed batch (`rows` rows): residual save
    /// → `norm_pre_attn` → clipped q/k/v GEMMs into `buf_qkv` → per-clip
    /// chunked attention → clipped `post` GEMM → `norm_post_attn` →
    /// residual add. `buf_h1` holds the hidden states on entry and exit.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_sub_block(
        &self,
        blk: &GemmaAudioLayerWeights,
        rows: usize,
        t_i: &[usize],
        t_off: &[usize],
        a_off: &[usize],
        rel_k: DevicePtr,
        spd: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let rows32 = rows as u32;
        let n_h = rows * self.hidden_size;
        // Residual save for the attention sub-block (the layer's post-FFN1
        // state).
        gpu.copy_d2d(self.buf_h1, self.buf_h2, n_h * 2)?;
        // norm_pre_attn, in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.norm_pre_attn,
            self.buf_h1,
            rows32,
            h,
            self.norm_eps,
            stream,
        )?;
        // Clipped q/k/v GEMMs → buf_qkv [Σ, 3×hidden]: q, k, v planes.
        let q = self.buf_qkv;
        let k = self.buf_qkv.offset(n_h * 2);
        let v = self.buf_qkv.offset(2 * n_h * 2);
        self.clipped_gemm(
            gpu,
            self.buf_h1,
            &blk.self_attn.q_proj,
            q,
            rows32,
            h,
            h,
            stream,
        )?;
        self.clipped_gemm(
            gpu,
            self.buf_h1,
            &blk.self_attn.k_proj,
            k,
            rows32,
            h,
            h,
            stream,
        )?;
        self.clipped_gemm(
            gpu,
            self.buf_h1,
            &blk.self_attn.v_proj,
            v,
            rows32,
            h,
            h,
            stream,
        )?;
        // Attention per clip over its disjoint slices.
        for (i, &t) in t_i.iter().enumerate() {
            let base = t_off[i] * self.hidden_size * 2;
            let q_i = self.buf_qkv.offset(base);
            let k_i = self.buf_qkv.offset(base + n_h * 2);
            let v_i = self.buf_qkv.offset(base + 2 * n_h * 2);
            let o_i = self.buf_mlp.offset(base);
            let mask = self.buf_mask_attn.offset(a_off[i]);
            self.chunked_attn_stage(q_i, k_i, v_i, o_i, spd, rel_k, mask, t, gpu, stream)?;
        }
        // Clipped post GEMM: buf_mlp → buf_h1 (normed input fully consumed
        // by q/k/v).
        self.clipped_gemm(
            gpu,
            self.buf_mlp,
            &blk.self_attn.post,
            self.buf_h1,
            rows32,
            h,
            h,
            stream,
        )?;
        // norm_post_attn + residual add, both in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.norm_post_attn,
            self.buf_h1,
            rows32,
            h,
            self.norm_eps,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)
    }
}
