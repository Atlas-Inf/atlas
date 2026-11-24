// SPDX-License-Identifier: AGPL-3.0-only

//! Token-parallel projection/convolution front-end for Ling KDA prefill.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ling_projection::LingProjectionScratch;
use crate::layers::ops;

use super::{
    LING_KDA_CONV_WIDTH, LING_KDA_HEAD_DIM, LING_KDA_HEADS, LING_KDA_PROJECTION_SIZE, LingKdaLayer,
    LingKdaState,
};

/// Arena-backed row-major storage for one KDA prefill. `qkv_planes` and
/// `conv_planes` are plane-major (`Q[rows] | K[rows] | V[rows]`); `qkv` is
/// the packed row-major form consumed by the recurrent kernel.
#[derive(Clone, Copy)]
pub struct LingKdaPrefillScratch {
    pub qkv_planes: DevicePtr,
    pub qkv: DevicePtr,
    pub conv_planes: DevicePtr,
    pub raw_gate: DevicePtr,
    pub raw_beta: DevicePtr,
    pub gate: DevicePtr,
    pub recurrent: DevicePtr,
    pub gated: DevicePtr,
    pub output: DevicePtr,
}

impl LingKdaPrefillScratch {
    fn validate(self) -> Result<()> {
        for (name, ptr) in [
            ("qkv_planes", self.qkv_planes),
            ("qkv", self.qkv),
            ("conv_planes", self.conv_planes),
            ("raw_gate", self.raw_gate),
            ("raw_beta", self.raw_beta),
            ("gate", self.gate),
            ("recurrent", self.recurrent),
            ("gated", self.gated),
            ("output", self.output),
        ] {
            ensure!(!ptr.is_null(), "Ling KDA prefill scratch {name} is null");
        }
        Ok(())
    }
}

impl LingKdaLayer {
    /// Process all prefill rows with one projection sweep per matrix and one
    /// token-parallel causal-convolution launch per Q/K/V plane. The KDA
    /// recurrence itself is one launch over the sequence and advances the
    /// caller's slot state exactly as decode does.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill(
        &self,
        gpu: &dyn GpuBackend,
        input_norm: DevicePtr,
        tokens: usize,
        states: &[LingKdaState],
        fp32_workspace: Option<DevicePtr>,
        scratch: LingKdaPrefillScratch,
        projection: LingProjectionScratch,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(tokens > 0, "Ling KDA prefill needs at least one token");
        ensure!(tokens <= u32::MAX as usize, "Ling KDA prefill is too long");
        ensure!(!input_norm.is_null(), "Ling KDA prefill input is null");
        ensure!(
            states.len() == 1 || states.len() == tokens,
            "expected a prompt state or one state per decode row"
        );
        for state in states {
            state.validate()?;
        }
        let rows_per_state = tokens / states.len();
        let row_bytes = LING_KDA_PROJECTION_SIZE * 2;
        scratch.validate()?;
        let rows = tokens as u32;
        let plane_bytes = tokens * LING_KDA_PROJECTION_SIZE * 2;
        let projection_rows = [
            (&self.weights.q_proj, scratch.qkv_planes),
            (&self.weights.k_proj, scratch.qkv_planes.offset(plane_bytes)),
            (
                &self.weights.v_proj,
                scratch.qkv_planes.offset(2 * plane_bytes),
            ),
        ];
        for (weight, output) in projection_rows {
            weight.forward(gpu, input_norm, output, rows, projection, stream)?;
        }
        self.weights
            .f_proj
            .forward(gpu, input_norm, scratch.raw_gate, rows, projection, stream)?;
        self.weights
            .b_proj
            .forward(gpu, input_norm, scratch.raw_beta, rows, projection, stream)?;
        self.weights
            .g_proj
            .forward(gpu, input_norm, scratch.gate, rows, projection, stream)?;

        let conv_bytes = LingKdaState::conv_state_bytes();
        let input_stride = LING_KDA_PROJECTION_SIZE as u32;
        for (index, state) in states.iter().enumerate() {
            let first = index * rows_per_state;
            for (plane, (state_ptr, weight)) in [
                (0usize, (state.q_conv, &self.weights.q_conv)),
                (1usize, (state.k_conv, &self.weights.k_conv)),
                (2usize, (state.v_conv, &self.weights.v_conv)),
            ] {
                ops::conv1d_update_prefill(
                    gpu,
                    self.conv_prefill_k,
                    self.conv_prefill_tp_k,
                    state_ptr,
                    scratch
                        .qkv_planes
                        .offset(plane * plane_bytes + first * row_bytes),
                    weight,
                    DevicePtr::NULL,
                    scratch
                        .conv_planes
                        .offset(plane * plane_bytes + first * row_bytes),
                    LING_KDA_PROJECTION_SIZE as u32,
                    LING_KDA_CONV_WIDTH as u32,
                    rows_per_state as u32,
                    input_stride,
                    input_stride,
                    stream,
                )?;
                debug_assert_eq!(state_ptr.0, state.q_conv.offset(plane * conv_bytes).0);
            }
        }
        ops::kda_pack_qkv(
            gpu,
            self.pack_k,
            scratch.conv_planes,
            scratch.qkv,
            rows,
            LING_KDA_PROJECTION_SIZE as u32,
            stream,
        )?;
        if let Some(workspace) = fp32_workspace {
            ensure!(
                states.len() == 1,
                "register-resident KDA needs a single prompt"
            );
            let fp32_plane = plane_bytes * 2;
            let q_norm = workspace;
            let k_norm = workspace.offset(fp32_plane);
            let beta = workspace.offset(2 * fp32_plane);
            // The packed QKV has consumed all three convolution planes. Leave
            // the first BF16 plane for recurrent output and the fourth for g.
            let decay = scratch.conv_planes.offset(plane_bytes);
            ops::kda_preprocess_regresident(
                gpu,
                self.preprocess_regresident_k,
                scratch.qkv,
                scratch.raw_gate,
                scratch.raw_beta,
                self.weights.a_log.weight,
                self.weights.dt_bias.weight,
                q_norm,
                k_norm,
                decay,
                beta,
                rows,
                LING_KDA_HEADS as u32,
                LING_KDA_HEAD_DIM as u32,
                self.config.lower_bound,
                stream,
            )?;
            ops::kda_recurrent_regresident(
                gpu,
                self.subwarp_k.unwrap_or(self.recurrent_regresident_k),
                if self.subwarp_k.is_some() { 16 } else { 4 },
                scratch.qkv,
                q_norm,
                k_norm,
                decay,
                beta,
                states[0].h,
                scratch.recurrent,
                rows,
                LING_KDA_HEADS as u32,
                LING_KDA_HEAD_DIM as u32,
                stream,
            )?;
        } else if rows_per_state == 1 && (2..=8).contains(&states.len()) {
            let mut launch = spark_runtime::kernel_args::KernelLaunch::new(gpu, self.batch_k)
                .grid([LING_KDA_HEADS as u32, rows, 1])
                .block([128, 1, 1])
                .arg_ptr(scratch.qkv)
                .arg_ptr(scratch.raw_gate)
                .arg_ptr(scratch.raw_beta)
                .arg_ptr(self.weights.a_log.weight)
                .arg_ptr(self.weights.dt_bias.weight)
                .arg_ptr(scratch.recurrent);
            for i in 0..8 {
                launch = launch.arg_ptr(states.get(i).map_or(DevicePtr::NULL, |state| state.h));
            }
            launch
                .arg_u32(LING_KDA_HEADS as u32)
                .arg_u32(LING_KDA_HEAD_DIM as u32)
                .arg_f32(self.config.lower_bound)
                .launch(stream)?;
        } else {
            for (index, state) in states.iter().enumerate() {
                let first = index * rows_per_state;
                ops::kda_recurrent(
                    gpu,
                    self.recurrent_k,
                    scratch.qkv.offset(first * 3 * row_bytes),
                    scratch.raw_gate.offset(first * row_bytes),
                    scratch.raw_beta.offset(first * (LING_KDA_HEADS * 2)),
                    self.weights.a_log.weight,
                    self.weights.dt_bias.weight,
                    state.h,
                    scratch.recurrent.offset(first * row_bytes),
                    rows_per_state as u32,
                    LING_KDA_HEADS as u32,
                    LING_KDA_HEAD_DIM as u32,
                    self.config.lower_bound,
                    stream,
                )?;
            }
        }
        ops::kda_sigmoid_gated_norm(
            gpu,
            self.gated_norm_k,
            scratch.recurrent,
            scratch.gate,
            self.weights.o_norm.weight,
            scratch.gated,
            rows,
            LING_KDA_HEADS as u32,
            LING_KDA_HEAD_DIM as u32,
            self.config.rms_norm_eps,
            stream,
        )?;
        self.weights.o_proj.forward(
            gpu,
            scratch.gated,
            scratch.output,
            rows,
            projection,
            stream,
        )?;
        Ok(scratch.output)
    }
}
