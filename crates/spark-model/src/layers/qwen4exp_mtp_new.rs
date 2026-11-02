// SPDX-License-Identifier: AGPL-3.0-only
//!
//! Ported from Enntity/atlas (AGPL-3.0-only, sparkqwen drafter series).

//! Construction of the qwen4_exp MTP proposer and its per-sequence state.
//! Split out of `qwen4exp_mtp.rs` for the 500-LoC cap.

use super::*;

impl Qwen4ExpMtpHead {
    pub fn new(
        module: Qwen4ExpMtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        // The config the body was built with (the TP=1 view), so the cache
        // holds every KV head the unsharded body writes.
        let config = &module.config;
        let h = config.hidden_size;
        let hc = config.hc_mult.max(1);

        // The drafter's attention writes its OWN cache, never the target's.
        // Shape matches a target full-attention layer so the reused body's
        // `write_kv_cache` / paged decode land at the right strides. The body
        // was built with `attn_idx = <number of target full-attention layers>`
        // and indexes the pool at THAT index, so the pool needs that many + 1
        // layer slots even though only the last is ever written.
        let target_attn_layers = config
            .layer_types
            .iter()
            .filter(|t| matches!(t, atlas_core::config::LayerType::FullAttention))
            .count();
        let kv_config = drafter_kv_config(
            target_attn_layers,
            config.num_key_value_heads,
            config.head_dim,
        );
        // `ATLAS_QWEN4EXP_MTP_PREFILL`: every speculating sequence
        // (`mtp_max_seqs`) may hold `window` prompt rows for its lifetime, on
        // top of the decode rows the one sequence's worth below covers.
        let prompt_rows = qwen4exp_mtp_prompt::window() * crate::speculative::mtp_max_seqs();
        let num_blocks = (max_seq_len + prompt_rows) / kv_config.block_size + 1;
        if prompt_rows > 0 {
            tracing::info!(
                "qwen4_exp MTP drafter KV pool: {num_blocks} blocks ({:.0} MiB), \
                 {prompt_rows} of its rows for prompts",
                (num_blocks * kv_config.block_bytes_kv_all_layers()) as f64 / (1 << 20) as f64
            );
        }
        let block_size = kv_config.block_size;
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;

        let rows = qwen4exp_mtp_batch::PROPOSE_BATCH_MAX;
        let bf16 = |n: usize| -> Result<DevicePtr> { gpu.alloc(rows * n * 2) };
        // Every block a `max_seq_len` drafter sequence can reference, + 1 for
        // the row being written (the allocator's `seq_len / bs + 1`).
        let batch_slab_blocks = max_seq_len / block_size + 1;
        let draft = crate::layers::qwen4exp_draft_head::DraftHead::build(
            &lm_head,
            config.vocab_size,
            h,
            mtp_vocab_size,
            gpu,
        )?;
        let conf_stop = crate::layers::qwen4exp_draft_head::conf_stop_from(
            std::env::var("ATLAS_QWEN4EXP_MTP_CONFIDENCE").ok(),
        )?;

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            draft,
            conf_stop,
            kv_cache: Mutex::new(kv_cache),
            embed_buf: bf16(h)?,
            normed_e: bf16(h)?,
            e_branch: bf16(h)?,
            normed_h: bf16(hc * h)?,
            h_streams: bf16(hc * h)?,
            h_out: bf16(h)?,
            argmax_out: gpu.alloc(4)?,
            // Offset-from-1 RMSNorm, matching `hc_norm` in the shadowed mHC
            // kernel. NOT `rms_norm_vanilla` — that would apply `w` instead of
            // `1 + w` and silently shift every drafted logit.
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_f32_k: gpu.kernel("norm", "rms_norm_f32")?,
            rms_norm_f32_grouped_k: crate::layers::try_kernel_gated(
                std::env::var("ATLAS_QWEN4EXP_MTP_GROUPED_NORM").as_deref() == Ok("1"),
                gpu,
                "norm",
                "rms_norm_f32_grouped",
            ),
            f32_residual_add_k: gpu.kernel("norm", "f32_residual_add")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            dense_gemv_batchm_k: crate::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            wide_rows: super::ops::Qwen4ExpWideRows::resolve(gpu, "qwen4_exp"),
            batched_embed_k: crate::layers::try_kernel(gpu, "embed_from_argmax", "batched_embed"),
            argmax_batch_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_batch"),
            argmax_batch_lp_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_batch_lp"),
            batch_slab: gpu.alloc(qwen4exp_mtp_batch::slab_bytes(batch_slab_blocks))?,
            batch_slab_blocks,
            tp: None,
            absorb: qwen4exp_mtp_absorb::AbsorbBufs::new(gpu, hc, h)?,
        })
    }

    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Qwen4ExpMtpProposerState> {
        Ok(Qwen4ExpMtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            awaiting_verdict: false,
            body_state: self.module.body.alloc_state(gpu)?,
            end_floor: 0,
            absorb: Default::default(),
        })
    }

    /// `residual[i] += bf16(src[i])` over `n` elements — the BF16 branch
    /// accumulating onto the FP32 stream highway.
    pub(super) fn f32_add_bf16(
        &self,
        gpu: &dyn GpuBackend,
        residual: DevicePtr,
        src: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        let block = 256u32;
        KernelLaunch::new(gpu, self.f32_residual_add_k)
            .grid([n.div_ceil(block), 1, 1])
            .block([block, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(src)
            .arg_u32(n)
            .launch(stream)
    }
}
