// SPDX-License-Identifier: AGPL-3.0-only

//! Ling-3.2-air NEXTN (MTP) draft proposer.
//!
//! Implements [`DraftProposer`] over the [`LingMtpModule`] loaded by
//! `weight_loader::bailing::mtp::load_ling_mtp_module`. Like
//! DeepSeek-V4's ([`crate::layers::DeepseekV4MtpHead`]) the module's body
//! is a full reused family layer (here the bailing gated-MLA layer 42 with
//! its own 512-expert NVFP4 MoE), so the bulk of the draft forward is
//! delegated to `body.decode()` and this head supplies only the MTP-only
//! ends. The target-relative contract is the house one:
//! `save_hidden_for_mtp` stages the RAW final-layer hidden (Ling models
//! `hnorm` it itself) of the last committed decode row, and `propose`
//! receives it as `target_hidden`.
//!
//! Single draft per pass — `num_nextn_predict_layers = 1`, and the family's
//! layers cap verify at K = 2 (`TransformerLayer::verify_max_drafts`
//! `Some(1)`), so every `propose` walks at most one position and the
//! scheduler's K=2 lane serves the verify.
//!
//! Forward (`propose`, one position):
//!
//! ```text
//!   pair     = ling_mtp_eh_norm(embed_table, word(last_token), enorm,
//!                               target_hidden, hnorm)   // [2, hidden]
//!   h_in     = eh_proj  · pair                          // dense_gemv
//!   body.decode(h_in, drafter_kv, seq_len = st.rows)    // MLA + own MoE
//!   normed   = final_layernorm(h_in)
//!   logits   = lm_head(normed)                          // shared head
//!   draft    = ling_mtp_draft_pick(logits)              // on-device pick
//! ```
//!
//! The draft token and its input word live in device memory
//! (`ling_mtp_eh_norm` reads the token from a device word, PR #111's
//! graph-capturable-chain contract); only the final pick is read back to
//! the host (the eager scheduler hands drafts around as `Vec<u32>`).
//!
//! Drafter context: `prefill_drafter` builds the private single-layer KV
//! over the prompt pairs `(embed(t_{i+1}), hidden_i)` in chunks (the
//! `ling_mla_kv_a_norm_rope_cache` write path only — one autoregressive
//! layer's K/V depends only on its own normed input row), so the first
//! propose attends over the whole prompt. The RoPE positions are the
//! drafter's own row space (compact, append-only): every committed pair
//! appends exactly one row, so relative distances stay sequence-true.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{ForwardContext, LayerState, TransformerLayer};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::bailing::LingMtpModule;
use crate::weight_map::DenseWeight;

/// Rows per `prefill_drafter` chunk (also the eh-pair staging row cap).
const PREFILL_CHUNK: usize = 64;

/// Per-sequence drafter state.
pub struct LingMtpProposerState {
    /// Block table for the drafter's OWN single-layer KV cache.
    pub block_table: Vec<u32>,
    /// Drafter rows written (its compact KV row count).
    pub rows: usize,
    /// Drafts the last `propose` produced (for `after_verify` trimming).
    pub last_num_drafted: usize,
    /// Body sub-state (`alloc_state` of the reused family layer).
    pub body_state: Box<dyn LayerState>,
}

impl ProposerState for LingMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Ling-3.2-air NEXTN draft proposer.
pub struct LingMtpHead {
    module: LingMtpModule,
    /// Shared token embedding table (BF16), from the target model.
    embed_tokens: DenseWeight,
    /// Shared BF16 LM head, from the target model.
    lm_head: DenseWeight,
    vocab: u32,
    hidden: u32,
    eps: f32,
    /// The drafter's own single-layer absorbed-MLA K cache (1 x 576 BF16).
    kv_cache: Mutex<PagedKvCache>,
    // Kernel handles.
    eh_norm_k: KernelHandle,
    draft_pick_k: KernelHandle,
    gemv_k: KernelHandle,
    gemm_k: KernelHandle,
    rms_norm_k: KernelHandle,
    // Device scratch.
    /// The draft input token word (`ling_mtp_eh_norm` reads it here).
    draft_word: DevicePtr,
    /// `ling_mtp_draft_pick`'s pick for the single row + its full-vocab
    /// log-softmax probability.
    pick_word: DevicePtr,
    pick_logprob: DevicePtr,
    /// The `[2, hidden]` enorm/hnorm pair `eh_proj` consumes.
    eh_pair: DevicePtr,
    /// Prefill chunk staging: `eh_proj` output rows `[PREFILL_CHUNK, hidden]`.
    prefill_pairs: DevicePtr,
    prefill_eh_out: DevicePtr,
    /// The chunk's draft token words, `[PREFILL_CHUNK]` u32.
    prefill_tokens: DevicePtr,
}

impl LingMtpHead {
    /// Build the proposer from the loaded module and the SHARED embedding
    /// + LM head pointers (no copies — the weights are read in place).
    pub fn new(
        module: LingMtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
    ) -> Result<Self> {
        ensure_geometry(config)?;
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: mla_cache_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mtp_num_blocks = max_seq_len / kv_config.block_size + 2;
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;
        let h = config.hidden_size;
        let alloc_bf16 = |elems: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(elems * 2)?;
            gpu.memset(p, 0, elems * 2)?;
            Ok(p)
        };
        let head = Self {
            module,
            embed_tokens,
            lm_head,
            vocab: config.vocab_size as u32,
            hidden: h as u32,
            eps: config.rms_norm_eps as f32,
            kv_cache: Mutex::new(kv_cache),
            eh_norm_k: required(gpu, "ling_mtp_eh", "ling_mtp_eh_norm")?,
            draft_pick_k: required(gpu, "ling_mtp_eh", "ling_mtp_draft_pick")?,
            gemv_k: required(gpu, "gemv", "dense_gemv_bf16")?,
            gemm_k: required(gpu, "gemm", "dense_gemm_bf16")?,
            rms_norm_k: required(gpu, "rms_norm_vanilla", "rms_norm_vanilla")?,
            draft_word: alloc_bf16(2)?,
            pick_word: alloc_bf16(2)?,
            pick_logprob: alloc_bf16(2)?,
            eh_pair: alloc_bf16(2 * h)?,
            prefill_pairs: alloc_bf16(PREFILL_CHUNK * 2 * h)?,
            prefill_eh_out: alloc_bf16(PREFILL_CHUNK * h)?,
            prefill_tokens: alloc_bf16(PREFILL_CHUNK * 2)?,
        };
        tracing::info!(
            blocks = mtp_num_blocks,
            "ling NEXTN drafter: private 1x{} BF16 K pool + {} MB head scratch",
            mla_cache_dim,
            (PREFILL_CHUNK * 6 * h * 2) / 1024 / 1024
        );
        Ok(head)
    }

    /// Grow the drafter's block table to cover position `pos` (row space).
    fn ensure_table_blocks(&self, state: &mut LingMtpProposerState, pos: usize) -> Result<()> {
        let mut kv = self.kv_cache.lock();
        let bs = kv.block_size();
        let blocks_needed = (pos / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv.alloc_block()?);
        }
        Ok(())
    }

    /// One draft position: the eh pair + eh_proj + full body layer + final
    /// norm + shared LM head + on-device pick.
    fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        _position: usize,
        state: &mut LingMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        // The draft token word (ling_mtp_eh_norm reads it from device; the
        // house precedent stages these 4-byte uploads async the same way).
        gpu.copy_h2d_async(&token.to_le_bytes(), self.draft_word, stream)?;
        // [enorm(embed[token]) | hnorm(hidden)] — the [2, hidden] pair.
        KernelLaunch::new(gpu, self.eh_norm_k)
            .grid([2, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.embed_tokens.weight)
            .arg_ptr(self.draft_word)
            .arg_ptr(target_hidden)
            .arg_ptr(self.module.enorm.weight)
            .arg_ptr(self.module.hnorm.weight)
            .arg_ptr(self.eh_pair)
            .arg_u32(h)
            .arg_f32(self.eps)
            .launch(stream)?;
        // eh_proj: [hidden, 2 * hidden] over the pair.
        let h_in = ctx.buffers.hidden_states();
        ops::dense_gemv(
            gpu,
            self.gemv_k,
            self.eh_pair,
            &self.module.eh_proj,
            h_in,
            h,
            2 * h,
            stream,
        )?;
        // The body layer (gated-MLA + own MoE) decorates `h_in` in place at
        // the drafter's own next row.
        self.ensure_table_blocks(state, state.rows)?;
        let mtp_ctx = ForwardContext {
            ssm_batch: None,
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: None,
            profile: ctx.profile,
            comm: None, // the drafter runs on this rank only — no collectives
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: ctx.moe_lora_route,
        };
        {
            let mut kv_cache = self.kv_cache.lock();
            let mut disk_block_ids: Vec<u32> = Vec::new();
            let mut disk_last_offloaded: Vec<u32> = vec![0u32; 1];
            self.module.body.decode(
                h_in,
                ctx.buffers.residual(),
                state.body_state.as_mut(),
                &mut kv_cache,
                state.rows,
                &mut state.block_table,
                &mut disk_block_ids,
                &mut disk_last_offloaded,
                &mtp_ctx,
                stream,
            )?;
        }
        // Final layernorm + the shared LM head.
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            h_in,
            &self.module.final_norm,
            final_normed,
            1,
            h,
            self.eps,
            stream,
        )?;
        let logits = ctx.buffers.logits();
        ops::dense_gemv(
            gpu,
            self.gemv_k,
            final_normed,
            &self.lm_head,
            logits,
            self.vocab,
            h,
            stream,
        )?;
        // The pick (and its full-vocab log-softmax probability), on device.
        if let Some(bitmask) = grammar_bitmask {
            state.rows += 1;
            return crate::layers::argmax_grammar_masked(
                gpu,
                logits,
                self.vocab as usize,
                bitmask,
                0,
            );
        }
        KernelLaunch::new(gpu, self.draft_pick_k)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(logits)
            .arg_u32(self.vocab)
            .arg_ptr(self.pick_word)
            .arg_u32(1)
            .arg_ptr(self.pick_logprob)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(self.vocab)
            .arg_u32(0xFFFF_FFFFu32)
            .arg_u32(0xFFFF_FFFFu32)
            .arg_u32(0xFFFF_FFFFu32)
            .arg_u32(0xFFFF_FFFFu32)
            .launch(stream)?;
        let mut buf = [0u8; 4];
        gpu.copy_d2h(self.pick_word, &mut buf)?;
        state.rows += 1;
        Ok(u32::from_le_bytes(buf))
    }
}

fn required(gpu: &dyn GpuBackend, module: &str, name: &str) -> Result<KernelHandle> {
    let k = gpu.kernel(module, name)?;
    anyhow::ensure!(
        k.0 != 0,
        "ling NEXTN drafter: kernel {module}:{name} missing from the target"
    );
    Ok(k)
}

fn ensure_geometry(config: &ModelConfig) -> Result<()> {
    anyhow::ensure!(
        config.model_type == "bailing_moe_v3_vl",
        "ling NEXTN drafter: non-bailing model_type {:?}",
        config.model_type
    );
    Ok(())
}

fn downcast<'a>(state: &'a mut dyn ProposerState) -> Result<&'a mut LingMtpProposerState> {
    state
        .as_any_mut()
        .downcast_mut::<LingMtpProposerState>()
        .ok_or_else(|| anyhow::anyhow!("ling NEXTN: invalid proposer state"))
}

impl DraftProposer for LingMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(LingMtpProposerState {
            block_table: Vec::new(),
            rows: 0,
            last_num_drafted: 0,
            body_state: self.module.body.alloc_state(gpu)?,
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
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
        let st = downcast(state)?;
        if num_drafts > 1 {
            tracing::debug!(
                "ling NEXTN propose: {num_drafts} requested, serving 1 (single-draft wiring)"
            );
        }
        let draft = self.forward_one(
            last_token,
            target_hidden,
            position,
            st,
            ctx,
            stream,
            grammar_bitmask,
        )?;
        st.last_num_drafted = 1;
        Ok(vec![draft])
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let st = downcast(state)?;
        if st.rows != 0 || prompt_tokens.len() < 2 {
            // Only the cold, whole-prompt context: a warm turn's carry and
            // the per-popose appends keep rows current on their own.
            return Ok(0);
        }
        let gpu = ctx.gpu;
        let h = self.hidden as usize;
        let rows = prompt_tokens.len() - 1;
        self.ensure_table_blocks(st, rows.saturating_sub(1))?;
        let mut written = 0usize;
        while written < rows {
            let len = (rows - written).min(PREFILL_CHUNK);
            self.ensure_table_blocks(st, written + len - 1)?;
            // The chunk's draft token ids land in device memory once
            // (`ling_mtp_eh_norm` reads its token from a device word);
            // row r's word is `prefill_tokens + r`.
            let chunk_tokens: Vec<u8> = prompt_tokens[written + 1..written + len + 1]
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect();
            gpu.copy_h2d_async(&chunk_tokens, self.prefill_tokens, stream)?;
            // The eh pairs: row r of the chunk is pair key
            // `written + r` = (embed(prompt[k+1]), hidden_k at k = written+r).
            for r in 0..len {
                let k0 = written + r;
                KernelLaunch::new(gpu, self.eh_norm_k)
                    .grid([2, 1, 1])
                    .block([self.hidden.min(1024), 1, 1])
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(self.prefill_tokens.offset(r * 4))
                    .arg_ptr(hiddens.offset(k0 * h * 2))
                    .arg_ptr(self.module.enorm.weight)
                    .arg_ptr(self.module.hnorm.weight)
                    .arg_ptr(self.prefill_pairs.offset(r * 2 * h * 2))
                    .arg_u32(self.hidden)
                    .arg_f32(self.eps)
                    .launch(stream)?;
            }
            // eh_proj over the chunk rows, then the private KV rows.
            ops::dense_gemm(
                gpu,
                self.gemm_k,
                self.prefill_pairs,
                &self.module.eh_proj,
                self.prefill_eh_out,
                len as u32,
                self.hidden,
                2 * self.hidden,
                stream,
            )?;
            {
                let mut kv_cache = self.kv_cache.lock();
                self.module.body.kv_only_rows(
                    gpu,
                    self.prefill_eh_out,
                    &mut kv_cache,
                    len,
                    written,
                    &st.block_table,
                    stream,
                )?;
            }
            st.rows += len;
            written += len;
        }
        Ok(rows)
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        downcast(state).map(|s| s.rows).unwrap_or(0)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let st = downcast(state)?;
        let num_to_trim = crate::layers::mtp_head::mtp_rows_to_trim(
            st.last_num_drafted.max(1),
            num_accepted,
            false,
        );
        if num_to_trim > 0 {
            st.rows = st.rows.saturating_sub(num_to_trim);
        }
        Ok(())
    }

    fn free_state(
        &self,
        _gpu: &dyn GpuBackend,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        state: &mut dyn ProposerState,
    ) -> Result<()> {
        let st = downcast(state)?;
        if !st.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&st.block_table);
            st.block_table.clear();
        }
        st.rows = 0;
        Ok(())
    }
}
