// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next Multi-Token-Prediction (MTP) draft proposer.
//!
//! Implements [`DraftProposer`] over the `Qwen4ExpMtpModule` loaded by
//! `load_qwen4exp_mtp_module`. Like DeepSeek-V4's proposer (and unlike the
//! Qwen-shaped [`crate::layers::MtpHead`], a hand-rolled attention + MoE
//! block), the body here is a REUSED full model layer, so the proposer only
//! wraps it with the MTP-specific ends.
//!
//! ## Forward (`propose`, one draft position)
//!
//! ```text
//!   n_h[s] = grouped_rms_norm(streams[s], pre_fc_norm_hidden[s])   // s < hc_mult
//!   n_e    = rms_norm(embed[token], pre_fc_norm_embedding)
//!   e      = fc_embedding · n_e                                    // shared across streams
//!   streams[s] = e + fc_hidden · n_h[s]                            // per-stream, shared weight
//!   body.decode(streams, …, mtp_kv_cache)                          // MIDDLE mHC + attn + MoE
//!   h_out  = hc_head(streams)                                      // mtp.hyper_connection_mixer
//!   logits = lm_head(h_out)                                        // SHARED head
//! ```
//!
//! ## Why the input is the stream highway, not `target_hidden`
//!
//! `target_hidden` is the target's post-mixer, `hidden`-wide state. This
//! drafter cannot use it: `mtp.pre_fc_norm_hidden` is `[hc_mult * hidden]`, so
//! the block consumes the residual BEFORE the model-level mixer collapses it.
//! That state is exactly what `ctx.buffers.hc_streams()` still holds when the
//! proposer runs — the last trunk layer collapses INTO `hidden` and leaves the
//! streams intact — so no new plumbing is needed, and the `target_hidden`
//! argument is deliberately unused.
//!
//! Reconstructing the streams from the collapsed hidden (broadcast, or
//! `hc_expand`) is NOT equivalent: the mixer's collapse is lossy and the
//! checkpoint ships no MTP expand weights. It would run and draft badly.
//!
//! For draft positions after the first, the input is the drafter's OWN streams
//! from the previous position, which `body.decode` left in place — the same
//! recurrence, one buffer.
//!
//! ## Two conventions this file must match exactly
//!
//! Both are read off the shadowed `hyper_connection.cu`, not assumed:
//!
//!   1. `hc_norm` is **GROUPED**, `group_size = hidden`: each stream is
//!      normalised over its OWN slice, not once across the flattened
//!      `hc_mult * hidden` row.
//!   2. The scale is **offset from 1** (`x * rms * (1 + w)`), not `x * rms * w`.
//!
//! `rms_norm_f32` implements both (and reads the FP32 highway directly), so a
//! per-stream launch with the matching weight slice is exactly the grouped
//! form. Getting either wrong yields finite, plausible, wrong drafts — which
//! costs an acceptance run to notice.

use std::any::Any;

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::qwen4_exp::Qwen4ExpMtpModule;
use crate::weight_map::DenseWeight;

/// Per-sequence state for the qwen4_exp MTP proposer.
pub struct Qwen4ExpMtpProposerState {
    /// Block table for the drafter's OWN KV cache.
    pub block_table: Vec<u32>,
    /// Current sequence length in the drafter's KV cache.
    pub seq_len: usize,
    /// Drafts produced by the last `propose` (for `after_verify` trimming).
    pub last_num_drafted: usize,
    /// Per-layer state for the reused body.
    pub body_state: Box<dyn LayerState>,
}

impl ProposerState for Qwen4ExpMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Qwen3.8-Flash-Next MTP draft proposer.
pub struct Qwen4ExpMtpHead {
    module: Qwen4ExpMtpModule,
    /// Shared token embedding table (BF16), from the target.
    embed_tokens: DenseWeight,
    /// Shared LM head (BF16 on this checkpoint), from the target. Every draft
    /// is re-verified by the target's own head, so the draft head can only
    /// affect acceptance, never an emitted token.
    lm_head: DenseWeight,
    mtp_vocab_size: u32,
    /// The drafter's own single-layer KV cache.
    kv_cache: Mutex<PagedKvCache>,

    // Scratch owned by the proposer rather than borrowed from `ctx.buffers`.
    // 6 small buffers, ~50 KB total at hidden=2560/hc=4: cheap, and it removes
    // every aliasing question against the trunk's buffers, which the body is
    // simultaneously using.
    embed_buf: DevicePtr,
    normed_e: DevicePtr,
    e_branch: DevicePtr,
    normed_h: DevicePtr,
    h_streams: DevicePtr,
    h_out: DevicePtr,
    argmax_out: DevicePtr,

    // Kernel handles.
    rms_norm_k: KernelHandle,
    rms_norm_f32_k: KernelHandle,
    f32_residual_add_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_head_k: KernelHandle,
    argmax_k: KernelHandle,
}

impl Qwen4ExpMtpHead {
    pub fn new(
        module: Qwen4ExpMtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
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
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: target_attn_layers + 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let num_blocks = max_seq_len / kv_config.block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;

        let bf16 = |n: usize| -> Result<DevicePtr> { gpu.alloc(n * 2) };

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
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
            f32_residual_add_k: gpu.kernel("norm", "f32_residual_add")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Qwen4ExpMtpProposerState> {
        Ok(Qwen4ExpMtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state: self.module.body.alloc_state(gpu)?,
        })
    }

    /// `residual[i] += bf16(src[i])` over `n` elements — the BF16 branch
    /// accumulating onto the FP32 stream highway.
    fn f32_add_bf16(
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

#[path = "qwen4exp_mtp_forward.rs"]
mod qwen4exp_mtp_forward;

impl DraftProposer for Qwen4ExpMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    fn propose(
        &self,
        last_token: u32,
        _target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;

        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        for i in 0..num_drafts {
            // Each step reads the stream highway, which the previous step's
            // `body.decode` left holding the drafter's own residual — so
            // unlike the collapsed-hidden proposers there is nothing to thread
            // between iterations.
            let draft = self.forward_one(
                current_token,
                position + i,
                st,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            tracing::debug!(
                "qwen4_exp MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} -> draft={draft}",
                position + i,
                st.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
        }
        st.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;
        let num_drafted = st.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(num_accepted);
        if num_to_trim > 0 {
            st.seq_len = st.seq_len.saturating_sub(num_to_trim);
        }
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;
        if !st.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&st.block_table);
            st.block_table.clear();
        }
        st.seq_len = 0;
        Ok(())
    }
}
