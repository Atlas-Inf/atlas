// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer state for `qwen4_exp` (Qwen3.8-Flash-Next).
//!
//! Three shapes of layer, and they need different state:
//!
//! * **linear attention** (36 of 48) — gated-delta-net recurrent state plus a
//!   conv sliding window, the same pair Atlas's GDN already uses.
//! * **full attention** (12 of 48) — nothing persistent here; K/V lives in the
//!   `PagedKvCache`.
//! * **the PLE layer** (exactly one, at the one-indexed `ple_layer_ids`) — a
//!   short-conv window on top of whichever of the above it is.
//!
//! That last one is the trap. The PLE conv is **dilated by `ngram_size`**, so
//! its window is `(kernel - 1) * ngram_size` — 9 positions on the published
//! model, not `kernel - 1` = 3. Sizing it as an undilated conv silently
//! truncates the receptive field: the layer still runs, still produces text,
//! and has quietly stopped seeing two thirds of its context.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::any::Any;

use crate::layer::LayerState;

/// State for one `qwen4_exp` decoder layer.
pub struct Qwen4ExpLayerState {
    /// GDN recurrent state `[num_v_heads, key_head_dim, value_head_dim]` f32.
    /// `None` on full-attention layers.
    pub h_state: Option<DevicePtr>,
    /// GDN conv sliding window `[conv_dim, conv_kernel]` f32. `None` on
    /// full-attention layers.
    pub conv_state: Option<DevicePtr>,
    /// PLE short-conv window `[hc_count * hidden, (kernel-1) * ngram_size]`
    /// BF16. Only the PLE layer has one.
    pub ple_conv_state: Option<DevicePtr>,
    /// The last `ngram_size - 1` token ids, which is all the n-gram hash needs
    /// to continue — the ids are the only history it reads, never hidden
    /// state, which is what makes a decode step cheap and speculation-safe.
    pub ngram_context: Vec<u32>,
}

impl LayerState for Qwen4ExpLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Byte sizes for one layer's state, derived from the config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4ExpStateSizes {
    pub h_state_bytes: usize,
    pub conv_state_bytes: usize,
    pub ple_conv_bytes: usize,
    pub ngram_carry: usize,
}

impl Qwen4ExpStateSizes {
    /// `is_linear` selects the GDN pair; `has_ple` adds the PLE window.
    pub fn new(config: &ModelConfig, is_linear: bool, has_ple: bool) -> Self {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        // q, k and v share the conv: two key-width slices plus one value-width.
        let conv_dim = key_dim * 2 + value_dim;

        let h_state_bytes = if is_linear {
            config.linear_num_value_heads
                * config.linear_key_head_dim
                * config.linear_value_head_dim
                * 4
        } else {
            0
        };
        let conv_state_bytes = if is_linear {
            conv_dim * config.linear_conv_kernel_dim * 4
        } else {
            0
        };
        // DILATED: (kernel - 1) * ngram_size, not kernel - 1.
        let ple_conv_bytes = if has_ple {
            config.hc_count
                * config.hidden_size
                * (config.ple_conv_kernel_size.saturating_sub(1) * config.ngram_size)
                * 2
        } else {
            0
        };
        Self {
            h_state_bytes,
            conv_state_bytes,
            ple_conv_bytes,
            ngram_carry: if has_ple {
                config.ngram_size.saturating_sub(1)
            } else {
                0
            },
        }
    }

    /// Allocate and zero the state this layer needs.
    pub fn alloc(&self, gpu: &dyn GpuBackend, eos: u32) -> Result<Box<dyn LayerState>> {
        let zeroed = |bytes: usize| -> Result<Option<DevicePtr>> {
            if bytes == 0 {
                return Ok(None);
            }
            let ptr = gpu.alloc(bytes)?;
            gpu.memset(ptr, 0, bytes)?;
            Ok(Some(ptr))
        };
        Ok(Box::new(Qwen4ExpLayerState {
            h_state: zeroed(self.h_state_bytes)?,
            conv_state: zeroed(self.conv_state_bytes)?,
            ple_conv_state: zeroed(self.ple_conv_bytes)?,
            // Seeded with EOS, not zero: the reference starts a sequence with
            // `ngram_size - 1` EOS tokens of carried context, and a zero seed
            // hashes the first tokens of every sequence to the wrong rows.
            ngram_context: vec![eos; self.ngram_carry],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn published() -> ModelConfig {
        atlas_core::config::parse_config(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_data/qwen4_exp_flash_next_config.json"
        )))
        .expect("published config parses")
    }

    /// The PLE conv window is dilated. `(4 - 1) * 3 = 9` positions of
    /// `hc_count * hidden` BF16 — three times what an undilated reading gives,
    /// and the undilated version fails silently.
    #[test]
    fn the_ple_conv_window_is_dilated() {
        let cfg = published();
        let sizes = Qwen4ExpStateSizes::new(&cfg, true, true);
        let wide = cfg.hc_count * cfg.hidden_size;
        assert_eq!(wide, 10_240);
        assert_eq!(sizes.ple_conv_bytes, wide * 9 * 2);
        // What an undilated conv would have allocated, for contrast.
        assert_ne!(sizes.ple_conv_bytes, wide * 3 * 2);
        assert_eq!(sizes.ngram_carry, 2);
    }

    /// Full-attention layers keep nothing here — K/V is in the paged cache.
    #[test]
    fn attention_layers_hold_no_recurrent_state() {
        let cfg = published();
        let sizes = Qwen4ExpStateSizes::new(&cfg, false, false);
        assert_eq!(sizes.h_state_bytes, 0);
        assert_eq!(sizes.conv_state_bytes, 0);
        assert_eq!(sizes.ple_conv_bytes, 0);
    }

    /// GDN state is per VALUE head over key x value dims: q and k are shared
    /// across `num_v_heads / num_k_heads` value heads, so sizing the state by
    /// key heads under-allocates by that factor (3 on this model).
    #[test]
    fn gdn_state_is_sized_by_value_heads() {
        let cfg = published();
        let sizes = Qwen4ExpStateSizes::new(&cfg, true, false);
        let expect =
            cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * 4;
        assert_eq!(sizes.h_state_bytes, expect);
        assert_eq!(cfg.linear_num_value_heads / cfg.linear_num_key_heads, 3);

        // The conv covers q, k and v together.
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        assert_eq!(
            sizes.conv_state_bytes,
            (key_dim * 2 + value_dim) * cfg.linear_conv_kernel_dim * 4
        );
        assert_eq!(key_dim * 2 + value_dim, 10_240);
    }
}

// ── The decoder layer ──────────────────────────────────────────────────────

use crate::layer::ForwardContext;
use crate::layers::moe::MoeLayer;
use crate::layers::ops::qwen4exp as q4e;
use crate::weight_map::DenseWeight;

/// Device-side weights for one hyper-connection block.
///
/// Owned rather than borrowed because the layer outlives the loader's
/// `Qwen4ExpWeights`, and the collapse needs all four together.
pub struct HcBlock {
    pub hc_norm: DevicePtr,
    pub mix_down: DenseWeight,
    pub mix_up: DenseWeight,
    pub block_inject: Option<DenseWeight>,
}

impl HcBlock {
    fn as_ops(&self) -> q4e::HyperConnectionWeights<'_> {
        q4e::HyperConnectionWeights {
            hc_norm: self.hc_norm,
            mix_down: &self.mix_down,
            mix_up: &self.mix_up,
            block_inject: self.block_inject.as_ref(),
        }
    }
}

/// Scratch one layer needs for the two hyper-connection collapses.
///
/// Allocated per layer rather than taken from the shared arena: `norm_output`
/// is `hidden`-wide, and every buffer here except `mixed` is `hc_count *
/// hidden`. Borrowing a narrower buffer would overrun it silently.
pub struct HcScratch {
    inner: q4e::HyperConnectionScratch,
}

impl HcScratch {
    pub fn alloc(gpu: &dyn GpuBackend, config: &ModelConfig) -> Result<Self> {
        let wide = config.hc_count * config.hidden_size;
        let bf16 = 2;
        Ok(Self {
            inner: q4e::HyperConnectionScratch {
                normed: gpu.alloc(wide * bf16)?,
                lowrank: gpu.alloc(config.hc_lowrank * bf16)?,
                gate: gpu.alloc(wide * bf16)?,
                mixed: gpu.alloc(config.hidden_size * bf16)?,
                raw_injection: gpu.alloc(config.hc_count * bf16)?,
                injection: gpu.alloc(config.hc_count * bf16)?,
            },
        })
    }
}

/// One `qwen4_exp` decoder layer.
///
/// The shape that matters: this layer does NOT read or write the `hidden`
/// pointer the trait hands it. The trunk state is the `hc_count`-stream
/// residual in `ctx.buffers.hc_streams()`, which is `hc_count * hidden` wide
/// and lives for the whole depth. `hidden` is `hidden_size`, so writing the
/// residual through it would overrun by 4x on the published config.
///
/// DeepSeek-V4's mHC layers take the same shape, for the same reason.
pub struct Qwen4ExpLayer {
    pub kernels: q4e::Qwen4ExpKernels,
    pub dims: q4e::Qwen4ExpDims,
    /// Collapse before the mixer (attention or gated-delta-net).
    pub attn_hc: HcBlock,
    /// Collapse before the MoE.
    pub mlp_hc: HcBlock,
    /// The 512-expert MoE, loaded through the SHARED path — this model's
    /// tensor naming is what `load_moe_inner` already expects.
    pub ffn: MoeLayer,
    pub scratch: HcScratch,
}

impl Qwen4ExpLayer {
    /// Collapse the residual, run `block` on the `hidden`-wide result, and
    /// scatter its output back per stream.
    ///
    /// This is the whole layer's shape, twice: once around the mixer and once
    /// around the MoE. Checked end-to-end against the oracle in
    /// `qwen4exp_grouped_norm_microtest` (`hc_sandwich_roundtrip`) — the
    /// per-kernel checks cannot see an error BETWEEN the two steps.
    ///
    /// `block` receives the collapsed input and returns the block output; it
    /// must not write `residual`, which is still live.
    fn sandwich<F>(
        &self,
        residual: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        hc: &HcBlock,
        block: F,
    ) -> Result<()>
    where
        F: FnOnce(DevicePtr, &ForwardContext, u64) -> Result<DevicePtr>,
    {
        q4e::hyper_connection_collapse(
            ctx.gpu,
            &self.kernels,
            &self.dims,
            &hc.as_ops(),
            residual,
            &self.scratch.inner,
            stream,
        )?;
        let out = block(self.scratch.inner.mixed, ctx, stream)?;
        q4e::scatter_add(
            ctx.gpu,
            &self.kernels,
            &self.dims,
            out,
            self.scratch.inner.injection,
            residual,
            1,
            stream,
        )
    }

    /// The MoE half. Identical on all 48 layers, and built entirely from
    /// pieces that are already checked: the sandwich against the oracle, and
    /// `MoeLayer::forward` as the shared 512-expert path.
    pub fn mlp_block(
        &self,
        residual: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.sandwich(residual, ctx, stream, &self.mlp_hc, |input, ctx, stream| {
            self.ffn.forward(input, ctx, stream)
        })
    }
}
