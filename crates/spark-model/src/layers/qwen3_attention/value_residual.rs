// SPDX-License-Identifier: AGPL-3.0-only

//! Ripple value residuals (`config.value_residual`).
//!
//! Reference (`modeling_ricursive.py`, SGLang `inference/local/model.py`):
//! every attention layer projects V from
//! `torch.lerp(vres, normed, sigmoid(value_res_lambda))`, where `vres` is the
//! scaled layer-0 embedding of the same token and `normed` the layer's
//! input-norm output (Q, K and the gate still use `normed`).
//!
//! Where `vres` comes from: `hidden` IS the residual stream
//! (`rms_norm_residual` copies it raw into `residual` and normalizes it), and
//! every forward entry hands layer 0 the scaled embeddings in `hidden`. So
//! layer 0 snapshots its own `hidden` rows into a model-owned shadow of the
//! arena's `hidden_states` region, at the same byte offset, and every later
//! layer reads its rows back at its own `hidden`'s offset. No embedding call
//! site needs a hook — the GPU-side embeds (graph-captured decode, n-gram
//! fused embeds) are covered by construction.
//!
//! A `hidden` that does not lie inside the arena's `hidden_states` region is
//! refused rather than mapped: an unmapped row would silently read another
//! token's embedding.

use std::sync::atomic::Ordering;

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;

/// Model-owned buffers shared by every attention layer of a value-residual model.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ValresBuffers {
    /// Shadow of the arena `hidden_states` region holding the scaled
    /// embeddings, written by the source layer (layer 0).
    pub shadow: DevicePtr,
    /// Packed `[rows, hidden]` scratch for the lerped V input.
    pub vin: DevicePtr,
    /// Byte size of both buffers (== the arena's `hidden_states` size).
    pub bytes: usize,
    /// This layer writes the shadow (the first attention layer).
    pub is_source: bool,
}

impl Qwen3AttentionLayer {
    pub(crate) fn set_value_residual_blend(&mut self, blend: f32) {
        self.valres_blend = Some(blend);
    }

    pub(crate) fn set_value_residual_buffers(&mut self, bufs: ValresBuffers) {
        self.valres = Some(bufs);
    }

    pub(crate) fn has_value_residual(&self) -> bool {
        self.valres_blend.is_some()
    }

    /// Compute this forward's V input for `rows` tokens whose residual rows
    /// start at `hidden` and whose input-norm rows start at `normed`.
    pub(crate) fn prepare_value_input(
        &self,
        ctx: &ForwardContext,
        hidden: DevicePtr,
        normed: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(blend) = self.valres_blend else {
            return Ok(());
        };
        let Some(bufs) = self.valres else {
            anyhow::bail!(
                "L{}: value residuals enabled but the model never installed their buffers",
                self.attn_layer_idx
            );
        };
        let h = ctx.config.hidden_size;
        let row_bytes = h * 2;
        let base = ctx.buffers.hidden_states();
        ensure!(
            hidden.0 >= base.0,
            "L{}: value-residual hidden {:#x} is below the arena hidden_states {:#x}",
            self.attn_layer_idx,
            hidden.0,
            base.0
        );
        let off = (hidden.0 - base.0) as usize;
        ensure!(
            off.is_multiple_of(row_bytes) && off + rows * row_bytes <= bufs.bytes,
            "L{}: value-residual rows [{off}, +{}) outside the {}-byte shadow",
            self.attn_layer_idx,
            rows * row_bytes,
            bufs.bytes
        );
        let vres = bufs.shadow.offset(off);
        if bufs.is_source {
            ctx.gpu
                .copy_d2d_async(hidden, vres, rows * row_bytes, stream)?;
        }
        let n = (rows * h) as u32;
        KernelLaunch::new(ctx.gpu, self.valres_lerp_k)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(vres)
            .arg_ptr(normed)
            .arg_ptr(bufs.vin)
            .arg_u32(rows as u32)
            .arg_u32(h as u32)
            .arg_u32(h as u32)
            .arg_f32(blend)
            .launch(stream)?;
        self.valres_vin.store(bufs.vin.0, Ordering::Relaxed);
        Ok(())
    }

    /// The input the V projection reads: the lerped value input on a
    /// value-residual model (set by `prepare_value_input` for this forward),
    /// else `normed`.
    pub(crate) fn v_input(&self, normed: DevicePtr) -> DevicePtr {
        if self.valres_blend.is_none() {
            return normed;
        }
        DevicePtr(self.valres_vin.load(Ordering::Relaxed))
    }
}

/// Allocate the value-residual shadow + V-input scratch (each the size of the
/// arena's `hidden_states` region) and hand them to every attention layer;
/// the first layer writes the shadow. Called once at model construction.
pub(crate) fn install_value_residual(
    layers: &mut [Box<dyn crate::layer::TransformerLayer>],
    hidden_states_bytes: usize,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<()> {
    ensure!(hidden_states_bytes > 0, "value residuals: empty hidden_states region");
    let shadow = gpu.alloc(hidden_states_bytes)?;
    let vin = gpu.alloc(hidden_states_bytes)?;
    for (i, layer) in layers.iter_mut().enumerate() {
        let Some(attn) = layer
            .as_any_mut()
            .and_then(|a| a.downcast_mut::<Qwen3AttentionLayer>())
        else {
            anyhow::bail!("value residuals: layer {i} is not a Qwen3AttentionLayer");
        };
        ensure!(
            attn.has_value_residual(),
            "value residuals: layer {i} has no value_res_lambda blend"
        );
        attn.set_value_residual_buffers(ValresBuffers {
            shadow,
            vin,
            bytes: hidden_states_bytes,
            is_source: i == 0,
        });
    }
    tracing::info!(
        "value residuals: {} layers, shadow + V-input scratch {:.1} MB each",
        layers.len(),
        hidden_states_bytes as f64 / 1e6
    );
    Ok(())
}
