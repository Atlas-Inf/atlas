// SPDX-License-Identifier: AGPL-3.0-only
//! MiMo-V2 multi-module MTP draft proposer.
//!
//! The checkpoint ships three independent `model.mtp.layers.*` decoder
//! bodies. Each body is an MiMo sliding-window layer (fused QK192/V128,
//! sink bias, dense FFN), followed by the shared LM head. This differs from
//! the Qwen-shaped `MtpHead`: `eh_proj` consumes `enorm(embed) ||
//! hnorm(hidden)`, while each draft slot uses a different body and KV pool.

use std::any::Any;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState, TransformerLayer};
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::layers::{Qwen3AttentionLayer, argmax_grammar_masked};
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::DenseWeight;

/// One checkpoint MTP module: the reused decoder body plus its MTP ends.
pub struct MimoMtpModule {
    pub body: Qwen3AttentionLayer,
    /// `enorm`: RMSNorm on the token embedding before concat.
    pub pre_fc_norm_embedding: DenseWeight,
    /// `hnorm`: RMSNorm on the incoming hidden state before concat.
    pub pre_fc_norm_hidden: DenseWeight,
    /// `eh_proj`: `[hidden, 2 * hidden]` concat projection.
    pub fc: DenseWeight,
    /// `final_layernorm` before the shared LM head.
    pub norm: DenseWeight,
}

/// Per-module drafter state (each module has its own KV history).
struct ModuleState {
    block_table: Vec<u32>,
    seq_len: usize,
    body_state: Box<dyn LayerState>,
}

/// Per-sequence state for all three MiMo MTP modules.
pub struct MimoMtpProposerState {
    modules: Vec<ModuleState>,
    last_num_drafted: usize,
}

/// Module `i`'s row is committed once its shifted input token survives verify:
/// module 0 consumes the already-committed `x_{t+1}`, so it survives even a
/// full reject; module `i > 0` consumes draft `i` and survives only when at
/// least `i` drafts were accepted.
fn module_row_is_committed(module_index: usize, num_accepted: usize) -> bool {
    module_index <= num_accepted
}

/// `propose(position)` is called with the pending token's sequence index
/// (`t + 1`). MiMo MTP rows attend at the anchor hidden's position `t`.
fn mimo_anchor_position(position: usize) -> usize {
    position.saturating_sub(1)
}

impl ProposerState for MimoMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// MiMo-V2 MTP draft proposer, one body per draft slot.
pub struct MimoMtpHead {
    modules: Vec<MimoMtpModule>,
    kv_caches: Vec<Mutex<PagedKvCache>>,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    vocab_size: u32,
    embed_buf: DevicePtr,
    normed_embed: DevicePtr,
    normed_hidden: DevicePtr,
    concat_buf: DevicePtr,
    normed_out: DevicePtr,
    argmax_out: DevicePtr,
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    argmax_k: KernelHandle,
}

impl MimoMtpHead {
    pub fn num_modules(&self) -> usize {
        self.modules.len()
    }

    pub fn new(
        modules: Vec<MimoMtpModule>,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        gpu: &dyn GpuBackend,
        config: &ModelConfig,
        max_seq_len: usize,
        kv_block_size: usize,
        mtp_vocab_size: usize,
        max_batch_size: usize,
        main_kv_blocks: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            !modules.is_empty(),
            "MimoMtpHead requires at least one module"
        );
        let per_seq_blocks = max_seq_len / kv_block_size + 1;
        let spec_seqs = max_batch_size
            .max(1)
            .min(crate::speculative::mtp_max_seqs());
        let mtp_num_blocks = per_seq_blocks
            .saturating_mul(spec_seqs)
            .min(main_kv_blocks.max(per_seq_blocks));
        let mut kv_caches = Vec::with_capacity(modules.len());
        for _ in &modules {
            let cfg = KvCacheConfig {
                block_size: kv_block_size,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![(config.num_key_value_heads, config.head_dim)],
                cache_blocks_per_seq: None,
            };
            kv_caches.push(Mutex::new(PagedKvCache::new(cfg, mtp_num_blocks, gpu)?));
        }
        let h = config.hidden_size;
        let vocab_size = if mtp_vocab_size > 0 {
            mtp_vocab_size.min(config.vocab_size) as u32
        } else {
            config.vocab_size as u32
        };
        Ok(Self {
            modules,
            kv_caches,
            embed_tokens,
            lm_head,
            vocab_size,
            embed_buf: gpu.alloc(h * 2)?,
            normed_embed: gpu.alloc(h * 2)?,
            normed_hidden: gpu.alloc(h * 2)?,
            concat_buf: gpu.alloc(h * 4)?,
            normed_out: gpu.alloc(h * 2)?,
            argmax_out: gpu.alloc(4)?,
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    /// One draft through module `idx`. Every MiMo MTP layer consumes the
    /// TARGET hidden (`h_t`), never the previous draft layer's output; only
    /// the input token rotates (`x_{t+idx+1}`).
    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        idx: usize,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut ModuleState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;
        let module = &self.modules[idx];

        let embed_src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu
            .copy_d2d_async(embed_src, self.embed_buf, row_bytes, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            self.embed_buf,
            &module.pre_fc_norm_embedding,
            self.normed_embed,
            1,
            h as u32,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &module.pre_fc_norm_hidden,
            self.normed_hidden,
            1,
            h as u32,
            eps,
            stream,
        )?;
        ctx.gpu
            .copy_d2d_async(self.normed_embed, self.concat_buf, row_bytes, stream)?;
        ctx.gpu.copy_d2d_async(
            self.normed_hidden,
            self.concat_buf.offset(row_bytes),
            row_bytes,
            stream,
        )?;
        let hidden = ctx.buffers.hidden_states();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            self.concat_buf,
            &module.fc,
            hidden,
            h as u32,
            (h * 2) as u32,
            stream,
        )?;

        let mut kv = self.kv_caches[idx].lock();
        let bs = kv.block_size();
        let blocks_needed = state.seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv.alloc_block()?);
        }
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + (state.seq_len % bs) as i64;
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            global_slot,
            (state.seq_len + 1) as i32,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;
        let meta = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: state.block_table.len() as u32,
            num_seqs: 1,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };
        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(meta),
            ssm_batch: None,
            profile: ctx.profile,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let residual = ctx.buffers.residual();
        let mut disk_block_ids = Vec::new();
        let mut disk_last_offloaded = Vec::new();
        module.body.decode(
            hidden,
            residual,
            state.body_state.as_mut(),
            &mut kv,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        drop(kv);

        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &module.norm,
            self.normed_out,
            1,
            h as u32,
            eps,
            stream,
        )?;
        let logits = ctx.buffers.logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            self.normed_out,
            &self.lm_head,
            logits,
            self.vocab_size,
            h as u32,
            stream,
        )?;
        let draft = if let Some(bitmask) = grammar_bitmask {
            argmax_grammar_masked(ctx.gpu, logits, self.vocab_size as usize, bitmask, position)?
        } else {
            ops::argmax_bf16(
                ctx.gpu,
                self.argmax_k,
                logits,
                self.argmax_out,
                self.vocab_size,
                stream,
            )?;
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(self.argmax_out, &mut buf)?;
            u32::from_le_bytes(buf)
        };
        state.seq_len += 1;
        Ok(draft)
    }
}

impl DraftProposer for MimoMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        let mut modules = Vec::with_capacity(self.modules.len());
        for module in &self.modules {
            modules.push(ModuleState {
                block_table: Vec::new(),
                seq_len: 0,
                body_state: module.body.alloc_state(gpu)?,
            });
        }
        Ok(Box::new(MimoMtpProposerState {
            modules,
            last_num_drafted: 0,
        }))
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any()
            .downcast_ref::<MimoMtpProposerState>()
            .map(|s| s.modules[0].seq_len)
            .unwrap_or(0)
    }

    #[allow(clippy::too_many_arguments)]
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
        let st = state
            .as_any_mut()
            .downcast_mut::<MimoMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid mimo_v2 MTP proposer state"))?;
        let k = num_drafts.min(self.modules.len());
        let mut drafts = Vec::with_capacity(k);
        let mut current_token = last_token;
        let anchor_position = mimo_anchor_position(position);
        for i in 0..k {
            let draft = self.forward_one(
                i,
                current_token,
                target_hidden,
                anchor_position,
                &mut st.modules[i],
                ctx,
                stream,
                grammar_bitmask,
            )?;
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
            .downcast_mut::<MimoMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid mimo_v2 MTP proposer state"))?;
        let k = st.last_num_drafted;
        for (i, module) in st.modules.iter_mut().take(k).enumerate() {
            if !module_row_is_committed(i, num_accepted) {
                module.seq_len = module.seq_len.saturating_sub(1);
            }
        }
        st.last_num_drafted = 0;
        Ok(())
    }

    fn free_state(
        &self,
        _gpu: &dyn GpuBackend,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        state: &mut dyn ProposerState,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<MimoMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid mimo_v2 MTP proposer state"))?;
        for (i, module) in st.modules.iter_mut().enumerate() {
            if !module.block_table.is_empty() {
                self.kv_caches[i].lock().free_blocks(&module.block_table);
                module.block_table.clear();
            }
            module.seq_len = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{mimo_anchor_position, module_row_is_committed};

    #[test]
    fn mimo_anchor_is_the_hidden_position_not_the_pending_token() {
        assert_eq!(mimo_anchor_position(41), 40);
        assert_eq!(mimo_anchor_position(0), 0);
    }

    #[test]
    fn shifted_module_rows_commit_only_after_their_draft_survives() {
        assert!(module_row_is_committed(0, 0));
        assert!(!module_row_is_committed(1, 0));
        assert!(module_row_is_committed(1, 1));
        assert!(!module_row_is_committed(2, 1));
        assert!(module_row_is_committed(2, 2));
        assert!(module_row_is_committed(2, 3));
    }
}
