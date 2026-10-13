// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::Glm5KdaLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum KdaRecurrentPath {
    Reference,
    RegisterResident,
}

impl KdaRecurrentPath {
    fn select(decode: bool, register_resident_prefill: bool) -> Self {
        if !decode && register_resident_prefill {
            Self::RegisterResident
        } else {
            Self::Reference
        }
    }
}

pub(super) fn parse_register_resident_prefill(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_KDA_REGRESIDENT_PREFILL must be 0 or 1, got {other:?}"),
    }
}

pub(super) fn scratch_is_sufficient(
    heads: usize,
    dim: usize,
    top_k: usize,
    moe_intermediate: usize,
    hidden: usize,
) -> bool {
    let plane_bytes = heads * dim * size_of::<f32>();
    let routed_intermediate_bytes = top_k * moe_intermediate * size_of::<u16>();
    let routed_output_bytes = top_k * hidden * size_of::<u16>();
    routed_intermediate_bytes >= plane_bytes
        && routed_output_bytes >= plane_bytes + heads * size_of::<f32>()
}

impl Glm5KdaLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_recurrent(
        &self,
        qkv: DevicePtr,
        raw_gate: DevicePtr,
        raw_beta: DevicePtr,
        state: DevicePtr,
        output: DevicePtr,
        tokens: u32,
        decode: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match KdaRecurrentPath::select(decode, self.register_resident_prefill) {
            KdaRecurrentPath::Reference => ops::kda_recurrent(
                ctx.gpu,
                self.recurrent_k,
                qkv,
                raw_gate,
                raw_beta,
                self.weights.a_log.weight,
                self.weights.dt_bias.weight,
                state,
                output,
                tokens,
                self.heads as u32,
                self.dim as u32,
                self.lower_bound,
                stream,
            ),
            KdaRecurrentPath::RegisterResident => {
                // Attention has finished with the expert buffers; MoE will
                // overwrite them after KDA. Reusing them keeps the fast path
                // allocation-free while preserving FP32 normalization/decay.
                let plane_bytes = tokens as usize * self.heads * self.dim * size_of::<f32>();
                let q_norm = ctx.buffers.expert_gate_out();
                let k_norm = ctx.buffers.expert_up_out();
                let decay = ctx.buffers.expert_down_out();
                let beta = decay.offset(plane_bytes);
                ops::kda_preprocess_regresident(
                    ctx.gpu,
                    self.preprocess_regresident_k,
                    qkv,
                    raw_gate,
                    raw_beta,
                    self.weights.a_log.weight,
                    self.weights.dt_bias.weight,
                    q_norm,
                    k_norm,
                    decay,
                    beta,
                    tokens,
                    self.heads as u32,
                    self.dim as u32,
                    self.lower_bound,
                    stream,
                )?;
                ops::kda_recurrent_regresident(
                    ctx.gpu,
                    self.recurrent_regresident_k,
                    qkv,
                    q_norm,
                    k_norm,
                    decay,
                    beta,
                    state,
                    output,
                    tokens,
                    self.heads as u32,
                    self.dim as u32,
                    stream,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_resident_path_is_prefill_only_and_explicit() {
        assert_eq!(
            KdaRecurrentPath::select(false, true),
            KdaRecurrentPath::RegisterResident
        );
        assert_eq!(
            KdaRecurrentPath::select(true, true),
            KdaRecurrentPath::Reference
        );
        assert_eq!(
            KdaRecurrentPath::select(false, false),
            KdaRecurrentPath::Reference
        );
    }

    #[test]
    fn register_resident_flag_rejects_ambiguous_values() {
        assert!(!parse_register_resident_prefill(None).unwrap());
        assert!(!parse_register_resident_prefill(Some("0")).unwrap());
        assert!(parse_register_resident_prefill(Some("1")).unwrap());
        assert!(parse_register_resident_prefill(Some("true")).is_err());
    }

    #[test]
    fn glm_scratch_planes_fit_existing_expert_buffers() {
        assert!(scratch_is_sufficient(64, 128, 8, 2048, 6144));
        assert!(!scratch_is_sufficient(64, 128, 1, 2048, 6144));
    }
}
