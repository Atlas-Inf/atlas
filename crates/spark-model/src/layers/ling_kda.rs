// SPDX-License-Identifier: AGPL-3.0-only

//! Ling's one-token KDA attention branch.
//!
//! The implementation deliberately stops at the attention output.  A caller
//! owns the residual stream and decides when to add this branch, which keeps
//! the layer usable by both the ordinary decode path and the speculative
//! verifier.  Projection format selection is delegated to [`LingProjection`]
//! so BF16 trunk weights and checkpoint-native E4M3 weights share one path.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::ling_projection::{LingProjection, LingProjectionScratch};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Pinned Ling-3.2-air KDA geometry.  These values are also the geometry
/// assumed by Atlas's `kda_recurrent_bf16` and gated-norm kernels.
pub const LING_KDA_HIDDEN_SIZE: usize = 2560;
pub const LING_KDA_HEADS: usize = 32;
pub const LING_KDA_HEAD_DIM: usize = 128;
pub const LING_KDA_PROJECTION_SIZE: usize = LING_KDA_HEADS * LING_KDA_HEAD_DIM;
pub const LING_KDA_CONV_WIDTH: usize = 4;
pub const LING_KDA_LOWER_BOUND: f32 = -5.0;
pub const LING_KDA_RMS_EPS: f32 = 1.0e-6;

/// Runtime knobs which are part of the checkpoint's KDA contract.
#[derive(Debug, Clone, Copy)]
pub struct LingKdaConfig {
    pub lower_bound: f32,
    pub rms_norm_eps: f32,
}

impl LingKdaConfig {
    pub const fn pinned() -> Self {
        Self {
            lower_bound: LING_KDA_LOWER_BOUND,
            rms_norm_eps: LING_KDA_RMS_EPS,
        }
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.lower_bound == LING_KDA_LOWER_BOUND,
            "Ling KDA requires lower_bound={LING_KDA_LOWER_BOUND}, got {}",
            self.lower_bound
        );
        ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "Ling KDA RMS epsilon must be finite and positive, got {}",
            self.rms_norm_eps
        );
        Ok(())
    }
}

/// Weights consumed by the KDA branch.  `load` only validates and borrows
/// already-uploaded store entries; it does not create a second weight loader.
pub struct LingKdaWeights {
    pub q_proj: LingProjection,
    pub k_proj: LingProjection,
    pub v_proj: LingProjection,
    pub f_proj: LingProjection,
    pub b_proj: LingProjection,
    pub g_proj: LingProjection,
    pub o_proj: LingProjection,
    pub q_conv: DenseWeight,
    pub k_conv: DenseWeight,
    pub v_conv: DenseWeight,
    pub a_log: DenseWeight,
    pub dt_bias: DenseWeight,
    pub o_norm: DenseWeight,
}

impl LingKdaWeights {
    /// Load the KDA names below `prefix` from an existing [`WeightStore`].
    /// Projection tensors use `<name>.weight`; the remaining names match the
    /// Ling checkpoint exactly (`A_log`, `dt_bias`, and `o_norm.weight`).
    pub fn load(store: &WeightStore, prefix: &str) -> Result<Self> {
        let name = |local: &str| {
            if prefix.is_empty() {
                local.to_owned()
            } else {
                format!("{prefix}.{local}")
            }
        };
        let dense = |local: &str, shape: &[usize], dtype: WeightDtype| -> Result<DenseWeight> {
            let full = name(local);
            let tensor = store.get(&full)?;
            ensure!(
                tensor.shape == shape,
                "Ling tensor {full} has shape {:?}; expected {shape:?}",
                tensor.shape
            );
            ensure!(
                tensor.dtype == dtype,
                "Ling tensor {full} has dtype {:?}; expected {dtype:?}",
                tensor.dtype
            );
            Ok(DenseWeight { weight: tensor.ptr })
        };
        let conv = |local: &str| -> Result<DenseWeight> {
            let full = name(local);
            let tensor = store.get(&full)?;
            ensure!(
                tensor.shape == [LING_KDA_PROJECTION_SIZE, LING_KDA_CONV_WIDTH]
                    || tensor.shape == [LING_KDA_PROJECTION_SIZE, 1, LING_KDA_CONV_WIDTH],
                "Ling tensor {full} has shape {:?}; expected [{}, {}] or [{}, 1, {}]",
                tensor.shape,
                LING_KDA_PROJECTION_SIZE,
                LING_KDA_CONV_WIDTH,
                LING_KDA_PROJECTION_SIZE,
                LING_KDA_CONV_WIDTH
            );
            ensure!(
                tensor.dtype == WeightDtype::BF16,
                "Ling tensor {full} has dtype {:?}; expected BF16",
                tensor.dtype
            );
            Ok(DenseWeight { weight: tensor.ptr })
        };

        Ok(Self {
            q_proj: LingProjection::load(
                store,
                &name("q_proj"),
                LING_KDA_PROJECTION_SIZE,
                LING_KDA_HIDDEN_SIZE,
            )?,
            k_proj: LingProjection::load(
                store,
                &name("k_proj"),
                LING_KDA_PROJECTION_SIZE,
                LING_KDA_HIDDEN_SIZE,
            )?,
            v_proj: LingProjection::load(
                store,
                &name("v_proj"),
                LING_KDA_PROJECTION_SIZE,
                LING_KDA_HIDDEN_SIZE,
            )?,
            f_proj: LingProjection::load(
                store,
                &name("f_proj"),
                LING_KDA_PROJECTION_SIZE,
                LING_KDA_HIDDEN_SIZE,
            )?,
            b_proj: LingProjection::load(
                store,
                &name("b_proj"),
                LING_KDA_HEADS,
                LING_KDA_HIDDEN_SIZE,
            )?,
            g_proj: LingProjection::load(
                store,
                &name("g_proj"),
                LING_KDA_PROJECTION_SIZE,
                LING_KDA_HIDDEN_SIZE,
            )?,
            o_proj: LingProjection::load(
                store,
                &name("o_proj"),
                LING_KDA_HIDDEN_SIZE,
                LING_KDA_PROJECTION_SIZE,
            )?,
            q_conv: conv("q_conv1d.weight")?,
            k_conv: conv("k_conv1d.weight")?,
            v_conv: conv("v_conv1d.weight")?,
            a_log: dense("A_log", &[LING_KDA_HEADS], WeightDtype::FP32)?,
            dt_bias: dense(
                "dt_bias",
                &[LING_KDA_HEADS, LING_KDA_HEAD_DIM],
                WeightDtype::FP32,
            )?,
            o_norm: dense("o_norm.weight", &[LING_KDA_HEAD_DIM], WeightDtype::BF16)?,
        })
    }
}

const CONV_STATE_BYTES: usize =
    LING_KDA_PROJECTION_SIZE * LING_KDA_CONV_WIDTH * std::mem::size_of::<f32>();
const H_STATE_BYTES: usize =
    LING_KDA_HEADS * LING_KDA_HEAD_DIM * LING_KDA_HEAD_DIM * std::mem::size_of::<f32>();

/// Persistent one-token KDA state.  The pointers refer to FP32 allocations;
/// the caller owns their lifetime and should zero them before the first token.
#[derive(Debug, Clone, Copy)]
pub struct LingKdaState {
    pub q_conv: DevicePtr,
    pub k_conv: DevicePtr,
    pub v_conv: DevicePtr,
    pub h: DevicePtr,
}

impl LingKdaState {
    pub const fn conv_state_bytes() -> usize {
        CONV_STATE_BYTES
    }

    pub const fn h_state_bytes() -> usize {
        H_STATE_BYTES
    }

    fn validate(self) -> Result<()> {
        for (label, ptr) in [
            ("q_conv", self.q_conv),
            ("k_conv", self.k_conv),
            ("v_conv", self.v_conv),
            ("h", self.h),
        ] {
            ensure!(!ptr.is_null(), "Ling KDA state {label} is null");
        }
        Ok(())
    }
}

/// Device scratch for one token.  `qkv_raw` receives projection planes and
/// `qkv` receives the three convolved planes in the same contiguous q|k|v
/// layout expected by the recurrent kernel.
#[derive(Debug, Clone, Copy)]
pub struct LingKdaScratch {
    pub qkv_raw: DevicePtr,
    pub qkv: DevicePtr,
    pub raw_gate: DevicePtr,
    pub raw_beta: DevicePtr,
    pub gate: DevicePtr,
    pub recurrent: DevicePtr,
    pub gated: DevicePtr,
    pub output: DevicePtr,
}

impl LingKdaScratch {
    pub const fn new(
        qkv_raw: DevicePtr,
        qkv: DevicePtr,
        raw_gate: DevicePtr,
        raw_beta: DevicePtr,
        gate: DevicePtr,
        recurrent: DevicePtr,
        gated: DevicePtr,
        output: DevicePtr,
    ) -> Self {
        Self {
            qkv_raw,
            qkv,
            raw_gate,
            raw_beta,
            gate,
            recurrent,
            gated,
            output,
        }
    }

    fn validate(self) -> Result<()> {
        for (label, ptr) in [
            ("qkv_raw", self.qkv_raw),
            ("qkv", self.qkv),
            ("raw_gate", self.raw_gate),
            ("raw_beta", self.raw_beta),
            ("gate", self.gate),
            ("recurrent", self.recurrent),
            ("gated", self.gated),
            ("output", self.output),
        ] {
            ensure!(!ptr.is_null(), "Ling KDA scratch {label} is null");
        }
        Ok(())
    }
}

/// One-token KDA attention branch for the pinned Ling geometry.
pub struct LingKdaLayer {
    weights: LingKdaWeights,
    config: LingKdaConfig,
    conv_k: KernelHandle,
    conv_prefill_k: KernelHandle,
    conv_prefill_tp_k: KernelHandle,
    pack_k: KernelHandle,
    recurrent_k: KernelHandle,
    gated_norm_k: KernelHandle,
    residual_add_k: KernelHandle,
}

impl LingKdaLayer {
    pub fn new(
        weights: LingKdaWeights,
        config: LingKdaConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            weights,
            config,
            conv_k: gpu.kernel("causal_conv1d", "causal_conv1d_update")?,
            conv_prefill_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            conv_prefill_tp_k: crate::layers::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_tp",
            ),
            pack_k: gpu.kernel("kda", "kda_pack_qkv")?,
            recurrent_k: gpu.kernel("kda", "kda_recurrent_bf16")?,
            gated_norm_k: gpu.kernel("kda", "kda_sigmoid_gated_rms_norm")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
        })
    }

    /// Run Q/K/V + f/b/g projections, depthwise causal convolution, KDA
    /// recurrence, gated RMSNorm, and o_proj.  `input_norm` is the BF16
    /// pre-attention-normalized hidden row.  The returned pointer is a branch
    /// result; this function never mutates or adds to the residual stream.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_attention(
        &self,
        gpu: &dyn GpuBackend,
        input_norm: DevicePtr,
        state: LingKdaState,
        scratch: LingKdaScratch,
        input_projection_scratch: LingProjectionScratch,
        output_projection_scratch: LingProjectionScratch,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(!input_norm.is_null(), "Ling KDA input is null");
        state.validate()?;
        scratch.validate()?;

        let plane_bytes = LING_KDA_PROJECTION_SIZE * std::mem::size_of::<u16>();
        self.weights.q_proj.forward(
            gpu,
            input_norm,
            scratch.qkv_raw,
            1,
            input_projection_scratch,
            stream,
        )?;
        self.weights.k_proj.forward(
            gpu,
            input_norm,
            scratch.qkv_raw.offset(plane_bytes),
            1,
            input_projection_scratch,
            stream,
        )?;
        self.weights.v_proj.forward(
            gpu,
            input_norm,
            scratch.qkv_raw.offset(2 * plane_bytes),
            1,
            input_projection_scratch,
            stream,
        )?;
        self.weights.f_proj.forward(
            gpu,
            input_norm,
            scratch.raw_gate,
            1,
            input_projection_scratch,
            stream,
        )?;
        self.weights.b_proj.forward(
            gpu,
            input_norm,
            scratch.raw_beta,
            1,
            input_projection_scratch,
            stream,
        )?;
        self.weights.g_proj.forward(
            gpu,
            input_norm,
            scratch.gate,
            1,
            input_projection_scratch,
            stream,
        )?;

        for (state_ptr, input, output, weight) in [
            (
                state.q_conv,
                scratch.qkv_raw,
                scratch.qkv,
                &self.weights.q_conv,
            ),
            (
                state.k_conv,
                scratch.qkv_raw.offset(plane_bytes),
                scratch.qkv.offset(plane_bytes),
                &self.weights.k_conv,
            ),
            (
                state.v_conv,
                scratch.qkv_raw.offset(2 * plane_bytes),
                scratch.qkv.offset(2 * plane_bytes),
                &self.weights.v_conv,
            ),
        ] {
            ops::conv1d_update(
                gpu,
                self.conv_k,
                state_ptr,
                input,
                weight,
                output,
                LING_KDA_PROJECTION_SIZE as u32,
                LING_KDA_CONV_WIDTH as u32,
                1,
                stream,
            )?;
        }

        ops::kda_recurrent(
            gpu,
            self.recurrent_k,
            scratch.qkv,
            scratch.raw_gate,
            scratch.raw_beta,
            self.weights.a_log.weight,
            self.weights.dt_bias.weight,
            state.h,
            scratch.recurrent,
            1,
            LING_KDA_HEADS as u32,
            LING_KDA_HEAD_DIM as u32,
            self.config.lower_bound,
            stream,
        )?;
        ops::kda_sigmoid_gated_norm(
            gpu,
            self.gated_norm_k,
            scratch.recurrent,
            scratch.gate,
            self.weights.o_norm.weight,
            scratch.gated,
            1,
            LING_KDA_HEADS as u32,
            LING_KDA_HEAD_DIM as u32,
            self.config.rms_norm_eps,
            stream,
        )?;
        self.weights.o_proj.forward(
            gpu,
            scratch.gated,
            scratch.output,
            1,
            output_projection_scratch,
            stream,
        )?;
        Ok(scratch.output)
    }

    /// Add the attention branch to the caller-owned residual row in place.
    /// Keeping this separate prevents speculative or multi-branch callers
    /// from accidentally adding the residual twice.
    pub fn add_residual(
        &self,
        gpu: &dyn GpuBackend,
        residual: DevicePtr,
        attention: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ensure!(!residual.is_null(), "Ling KDA residual is null");
        ensure!(!attention.is_null(), "Ling KDA attention output is null");
        ops::residual_add(
            gpu,
            self.residual_add_k,
            residual,
            attention,
            LING_KDA_HIDDEN_SIZE as u32,
            stream,
        )
    }
}

#[path = "ling_kda_prefill.rs"]
mod prefill;
pub use prefill::LingKdaPrefillScratch;

#[cfg(test)]
#[path = "ling_kda_tests.rs"]
mod tests;
