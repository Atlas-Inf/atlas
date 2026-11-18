// SPDX-License-Identifier: AGPL-3.0-only

//! Loading policy for the dedicated, eager BF16-highway Hy4 model.
use super::*;
use crate::weight_map::DenseWeight;

pub(crate) struct Hy4WeightLoader;
impl ModelWeightLoader for Hy4WeightLoader {
    fn supports_tp(&self) -> bool {
        true
    }
    fn dead_namespaces(&self, _c: &ModelConfig) -> Vec<String> {
        ["vit.", "visual.", "model.visual.", "model.dfly_layers."]
            .map(String::from)
            .to_vec()
    }
    fn load_layers(
        &self,
        _s: &WeightStore,
        _c: &ModelConfig,
        _g: &dyn GpuBackend,
        _d: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        anyhow::bail!(
            "Hy4 uses its dedicated eager Model backend, not generic FP32 mHC TransformerLayers"
        )
    }
    fn load_embedding(
        &self,
        s: &WeightStore,
        _c: &ModelConfig,
        _g: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: s.get("model.embed_tokens.weight")?.ptr,
        })
    }
    fn load_final_norm(
        &self,
        s: &WeightStore,
        _c: &ModelConfig,
        _g: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: s.get("model.norm.weight")?.ptr,
        })
    }
    fn load_lm_head(
        &self,
        s: &WeightStore,
        _c: &ModelConfig,
        _g: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: s.get("lm_head.weight")?.ptr,
        })
    }
    fn load_mtp_weights(
        &self,
        _s: &WeightStore,
        _c: &ModelConfig,
        _g: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
