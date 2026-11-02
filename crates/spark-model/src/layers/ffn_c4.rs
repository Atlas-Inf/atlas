// SPDX-License-Identifier: AGPL-3.0-only

use super::FfnComponent;
use crate::layer::ForwardContext;
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

impl FfnComponent {
    pub fn forward_independent(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            crate::model::glm_independent::ffn_rows_selected(ctx, rows)?,
            "independent FFN requires independent rows or a GLM DFlash verify block"
        );
        match self {
            Self::Moe(m) => m.forward_independent(input, rows, ctx, stream),
            Self::Dense(d) if d.can_forward_km(rows as u32) => {
                d.forward_km(input, rows as u32, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::Dense(d) => {
                d.forward_prefill(input, rows, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::None => Ok(input),
        }
    }
    /// None leaves the caller's exact scalar path intact, including dense layers.
    pub(crate) fn try_forward_c2_compact(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        if let Self::Moe(m) = self
            && ctx.config.model_type == "glm5_next"
            && m.c2_compact_enabled()
        {
            return m.forward_c2_compact(input, ctx, stream).map(Some);
        }
        Ok(None)
    }
    /// Independent C4 FFN. Never aliases the temporal verifier dispatch.
    pub fn forward_c4(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            crate::model::glm_c4::enabled(&ctx.config.model_type),
            "GLM independent C4 FFN requires its explicit opt-in"
        );
        match self {
            Self::Moe(m) => m.forward_c4(input, ctx, stream),
            Self::Dense(d) if d.can_forward_km(4) => {
                d.forward_km(input, 4, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            // Native NVFP4 dense layers have no W4A16 weight placeholders.
            // Retain their existing four-row dense prefill implementation.
            Self::Dense(d) => {
                d.forward_prefill(input, 4, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::None => Ok(input),
        }
    }
}
