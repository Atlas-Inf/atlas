// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM-5 MTP proposer using publisher weights and target verification.
//! Private ordinary-residual MLA state is primed and repaired from target hiddens.
use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::glm5_mtp_cursor::{
    append_cursor, bounded_draft_width, required_blocks, rewind_len_native_gamma,
};
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::glm5::Glm5MtpModule;
use crate::weight_map::DenseWeight;
use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache, SparseIndexCacheConfig};
use std::any::Any;

#[path = "glm5_mtp_capacity.rs"]
mod capacity;
#[path = "glm5_mtp_confidence.rs"]
mod confidence;
#[path = "glm5_mtp_context.rs"]
mod context;
#[path = "glm5_mtp_prefill.rs"]
mod prefill;
#[path = "glm5_mtp_prefix.rs"]
mod prefix;
#[path = "glm5_mtp_proposer.rs"]
mod proposer;
#[path = "glm5_mtp_serial.rs"]
mod serial;
use confidence::{ConfidencePlan, ConfidenceState};

pub struct Glm5MtpProposerState {
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub last_num_drafted: usize,
    pub body_state: Box<dyn LayerState>,
    confidence: ConfidenceState,
    target_hidden: DevicePtr,
    target_end: usize,
}
impl ProposerState for Glm5MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
pub struct Glm5MtpHead {
    module: Glm5MtpModule,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    mtp_vocab_size: u32,
    draft_config: ModelConfig,
    arena: BufferArena,
    max_seq_len: usize,
    kv_cache: Mutex<Option<PagedKvCache>>,
    kv_config: KvCacheConfig,
    sparse_index: SparseIndexCacheConfig,
    tail_blocks: usize,
    max_blocks: usize,
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    bf16_concat_k: KernelHandle,
    argmax_k: KernelHandle,
    zero_first_embed: bool,
    confidence: ConfidencePlan,
}
impl Glm5MtpHead {
    pub fn new(
        module: Glm5MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        target_config: &ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
        max_sequences: usize,
    ) -> Result<Self> {
        anyhow::ensure!(max_seq_len > 0, "GLM-5 MTP: max_seq_len must be positive");
        let draft_config = crate::weight_loader::glm5::glm5_mtp_draft_config(target_config)
            .context("GLM-5 MTP: invalid target parallel geometry")?;
        anyhow::ensure!(
            !module.body.has_aux_state(),
            "GLM-5 MTP requires a stateless ordinary MLA body"
        );
        // Publisher head values are expanded to BF16 by the loader. Native MTP
        // does not create a quantized twin of either embeddings or output head.
        anyhow::ensure!(
            std::env::var_os("ATLAS_GLM_NATIVE_MTP_HEAD").is_none_or(|v| v == "bf16"),
            "native GLM MTP supports only the publisher head expanded to BF16"
        );
        anyhow::ensure!(
            std::env::var_os("ATLAS_GLM_NATIVE_MTP_FINAL_NORM").is_none_or(|v| v == "bf16_sum"),
            "native GLM MTP supports only bf16_sum final norm"
        );
        let zero_first_embed = prefill::parse_zero_first_embedding_env()?;
        let confidence = ConfidencePlan::from_env(gpu)?;
        let mla_shape = crate::model::glm_cache_plan::GlmMlaShape::for_model(
            &draft_config.model_type,
            draft_config.kv_lora_rank,
            draft_config.qk_rope_head_dim,
        )?
        .context("GLM-5 MTP: unsupported MLA cache geometry")?;
        let mla_cache_dim = mla_shape.head_dim();
        anyhow::ensure!(mla_cache_dim > 0, "GLM-5 MTP: requires a nonzero MLA cache");
        let sparse_index =
            mla_shape.bf16_index(draft_config.index_kpool, draft_config.index_head_dim)?;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: mla_shape.num_kv_heads(),
            head_dim: mla_cache_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let block_size = kv_config.block_size;
        let (max_blocks, _) = capacity::block_capacity(max_seq_len, max_sequences, block_size)?;
        let allocation_config = capacity::workspace_config(&draft_config);
        let arena = BufferArena::new(&allocation_config, 128, max_seq_len, block_size, 1, gpu)
            .context("GLM-5 MTP: failed to allocate private draft workspace")?;
        capacity::validate_workspace(&draft_config, arena.sizes(), max_seq_len, block_size)?;
        tracing::info!(
            max_seq_len,
            max_sequences,
            private_workspace_bytes = arena.sizes().total_bytes(),
            "native GLM MTP workspace reserved; private KV awaits joint target budget"
        );
        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            draft_config,
            arena,
            max_seq_len,
            kv_cache: Mutex::new(None),
            kv_config,
            sparse_index,
            tail_blocks: max_sequences,
            max_blocks,
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            bf16_concat_k: gpu.kernel("residual_add", "bf16_concat")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            zero_first_embed,
            confidence,
        })
    }
    /// Rank-local incremental and fixed bytes charged to the target pool budget.
    pub(crate) fn private_kv_budget(&self, target_block_size: usize) -> Result<(usize, usize)> {
        anyhow::ensure!(
            target_block_size == self.kv_config.block_size,
            "native and target KV block sizes must match"
        );
        let per_block = capacity::private_pool_bytes(&self.kv_config, self.sparse_index, 1)?;
        let fixed =
            capacity::private_pool_bytes(&self.kv_config, self.sparse_index, self.tail_blocks)?;
        Ok((per_block, fixed))
    }

    /// Target admission bounds aggregate live rows; prefix copies need one
    /// extra open tail per active sequence, not a full context per sequence.
    pub(crate) fn allocate_private_kv(
        &self,
        target_blocks: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let blocks = capacity::aggregate_blocks(target_blocks, self.tail_blocks)?;
        let bytes = capacity::private_pool_bytes(&self.kv_config, self.sparse_index, blocks)?;
        let mut slot = self.kv_cache.lock();
        anyhow::ensure!(slot.is_none(), "native MTP private KV already allocated");
        let mut cache = PagedKvCache::new_with_v_alias(self.kv_config.clone(), blocks, gpu, true)?;
        cache.attach_sparse_index(self.sparse_index, gpu)?;
        *slot = Some(cache);
        tracing::info!(
            target_blocks,
            private_blocks = blocks,
            private_pool_bytes = bytes,
            "native GLM MTP private KV allocated from joint aggregate budget"
        );
        Ok(())
    }

    fn cache(&self) -> Result<MappedMutexGuard<'_, PagedKvCache>> {
        MutexGuard::try_map(self.kv_cache.lock(), Option::as_mut)
            .map_err(|_| anyhow::anyhow!("native MTP private KV has not been allocated"))
    }
}
