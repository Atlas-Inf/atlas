// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit experiment for four BF16 projections in GLM's paged prefill.
use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::{layer::ForwardContext, layers::ops, weight_map::DenseWeight};

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_PAGED_PREFILL_BF16_GEMM must be 0 or 1, got {other:?}"),
    }
}

pub(super) fn enabled(model: &str) -> Result<bool> {
    if model != "glm5_next" {
        return Ok(false);
    }
    match std::env::var("ATLAS_GLM_PAGED_PREFILL_BF16_GEMM") {
        Ok(value) => parse(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse(None),
        Err(error) => Err(error.into()),
    }
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paged_glm_projection(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
        accelerated: bool,
    ) -> Result<()> {
        // The existing helper selects BF16 cuBLAS when enabled, otherwise its
        // established TC/scalar fallback. No weight or activation conversion.
        if accelerated && ctx.config.model_type == "glm5_next" {
            return self.mla_prefill_dense(input, weight, output, m, n, k, ctx, stream);
        }
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            input,
            weight,
            output,
            m,
            n,
            k,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn paged_projection_flag_is_explicit() {
        assert!(!parse(None).unwrap());
        assert!(!parse(Some("0")).unwrap());
        assert!(parse(Some("1")).unwrap());
        for value in ["", "true", "2", " 1"] {
            assert!(parse(Some(value)).is_err());
        }
    }
}
