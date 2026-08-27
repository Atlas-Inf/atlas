// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next (`qwen4_exp`).
//!
//! Weight loading only so far — `load_layers` is not implemented, because the
//! layers it would return do not exist yet: low-rank hyper-connections, the PLE
//! tower and the sparse-attention indexer are all new, and the linear-attention
//! and 512-expert MoE paths need adapting. See `docs/porting/QWEN4_EXP.md`.
//!
//! Registering the loader anyway is deliberate: it turns "unsupported model
//! type" — which says nothing — into a message that names what is missing, and
//! it lets the weight side be exercised against a real checkpoint before any of
//! the forward pass exists.

pub mod weights;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense_auto};

pub use weights::Qwen4ExpWeights;

pub struct Qwen4ExpWeightLoader;

impl ModelWeightLoader for Qwen4ExpWeightLoader {
    fn supports_tp(&self) -> bool {
        // Nothing is sharded yet. Declaring false is what makes `--tp-size > 1`
        // fail at startup rather than silently replicate.
        false
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, &format!("{}.embed_tokens.weight", weights::LM), gpu)
    }

    /// There is NO separate final norm in this architecture — the published
    /// checkpoints carry no `model.language_model.norm.weight`. The trunk's
    /// `hyper_connection_mixer.hc_norm` is what normalises the residual before
    /// the LM head, and it is `hc_count * hidden` wide rather than `hidden`,
    /// because the residual is that many streams concatenated.
    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(
            store,
            &format!("{}.hyper_connection_mixer.hc_norm.weight", weights::LM),
            gpu,
        )
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if config.tie_word_embeddings {
            return self.load_embedding(store, config, gpu);
        }
        dense_auto(store, "lm_head.weight", gpu)
    }

    /// The checkpoint ships one MTP block, but its shape is not the
    /// `MtpWeights` any existing family uses: it carries its own 512-expert
    /// stack, its own sparse-attention indexer, its own hyper-connections, and
    /// two input norms of DIFFERENT widths (`pre_fc_norm_embedding` is hidden,
    /// `pre_fc_norm_hidden` is `hc_count * hidden`). Returning None means
    /// `--speculative` is refused at pre-flight rather than half-wired.
    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        // Load what exists, so a checkpoint problem surfaces as a named tensor
        // rather than hiding behind the unimplemented forward pass.
        let qctx = crate::weight_map::QuantizeCtx {
            absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
            quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
            stream: gpu.default_stream(),
        };
        let variant = crate::weight_map::detect_nvfp4_variant(store, config);
        let loaded = Qwen4ExpWeights::load(store, config, gpu, variant, qctx)?;
        anyhow::bail!(
            "qwen4_exp weights load ({} layers, {} PLE towers, {} routed experts/layer), but the \
             forward pass is not implemented: no Qwen4ExpLayer sequences the verified blocks \
             against Atlas's KV paging and buffer arena yet. The per-block kernels DO exist and \
             are checked against the CPU oracles (grouped norm, hyper-connections, PLE tower, \
             gated-delta-net, gated-Q attention, trunk expand). \
             See docs/porting/QWEN4_EXP.md.",
            loaded.layers.len(),
            loaded.layers.iter().filter(|l| l.ple.is_some()).count(),
            loaded.layers.first().map_or(0, |l| l.moe.experts.len()),
        )
    }
}
