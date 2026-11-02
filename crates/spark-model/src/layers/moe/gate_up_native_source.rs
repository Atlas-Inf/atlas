// SPDX-License-Identifier: AGPL-3.0-only
//! Construction-only standard-NVFP4 provenance; deliberately no serving caller.
//! Dead-code allowance is temporary staging, not an enabled loader capability.
#![allow(dead_code)]
use crate::weight_loader::glm5::retirement::RetirementLog;
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};
use std::collections::HashSet;

pub(super) const PACKED_BYTES: usize = 4_194_304;
pub(super) const SCALE_BYTES: usize = 524_288;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Span {
    pub ptr: DevicePtr,
    pub bytes: usize,
}
impl Span {
    pub(super) fn new(ptr: DevicePtr, bytes: usize, alignment: u64) -> Result<Self> {
        ensure!(
            ptr.0 != 0 && ptr.0.is_multiple_of(alignment) && bytes > 0,
            "invalid source span"
        );
        ensure!(
            ptr.0.checked_add(bytes as u64).is_some(),
            "source span overflow"
        );
        Ok(Self { ptr, bytes })
    }
    pub(super) fn disjoint(self, other: Self) -> bool {
        self.ptr.0 + self.bytes as u64 <= other.ptr.0
            || other.ptr.0 + other.bytes as u64 <= self.ptr.0
    }
}
pub(super) struct NativeProjection {
    pub expert: usize,
    pub is_up: bool,
    pub packed: Span,
    pub scales: Span,
    pub scalar: Span,
    pub scalar_bits: u32,
    pub input: Option<Span>,
    pub input_bits: Option<u32>,
}
/// Exact metadata captured only while validating the immutable native owner.
#[derive(Clone)]
pub(super) struct CheckpointIdentity {
    name: String,
    dtype: WeightDtype,
    shape: Vec<usize>,
    pub span: Span,
}
impl CheckpointIdentity {
    pub(super) fn validate(&self, store: &WeightStore) -> Result<()> {
        let tensor = store.get(&self.name)?;
        ensure!(
            tensor.ptr == self.span.ptr
                && tensor.dtype == self.dtype
                && tensor.shape == self.shape
                && tensor
                    .shape
                    .iter()
                    .try_fold(tensor.dtype.byte_size(), |n, &v| n.checked_mul(v))
                    == Some(self.span.bytes),
            "retained GU checkpoint identity changed: {}",
            self.name
        );
        Ok(())
    }
}
pub(super) struct NativeGateUpLayer<'s, 'g> {
    projections: Vec<NativeProjection>,
    retained: Vec<CheckpointIdentity>,
    stream: u64,
    owner: &'s WeightStore,
    gpu: &'g dyn GpuBackend,
    retirement: Option<&'s RetirementLog<'s>>,
}
impl<'s, 'g> NativeGateUpLayer<'s, 'g> {
    pub(super) fn from_store(
        store: &'s WeightStore,
        config: &ModelConfig,
        layer: usize,
        local: &[bool],
        gpu: &'g dyn GpuBackend,
        stream: u64,
    ) -> Result<Self> {
        Self::from_owner(store, config, layer, local, gpu, stream, None)
    }
    pub(super) fn from_live(
        retirement: &'s RetirementLog<'s>,
        config: &ModelConfig,
        layer: usize,
        local: &[bool],
        gpu: &'g dyn GpuBackend,
        stream: u64,
    ) -> Result<Self> {
        Self::from_owner(
            retirement.store(),
            config,
            layer,
            local,
            gpu,
            stream,
            Some(retirement),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn from_owner(
        store: &'s WeightStore,
        config: &ModelConfig,
        layer: usize,
        local: &[bool],
        gpu: &'g dyn GpuBackend,
        stream: u64,
        retirement: Option<&'s RetirementLog<'s>>,
    ) -> Result<Self> {
        ensure!(
            !gpu.stream_is_capturing(stream),
            "native provenance during capture"
        );
        ensure!(
            config.model_type == "glm5_next"
                && config.hidden_size == 4096
                && config.moe_intermediate_size == 2048
                && config.num_experts == 288
                && config.num_experts_per_tok == 8
                && layer < config.num_hidden_layers,
            "unsupported GLM target geometry/layer"
        );
        ensure!(
            config.ep_world_size == 2
                && config.tp_world_size == 2
                && config.ep_rank < 2
                && config.tp_rank == config.ep_rank,
            "native gate/up requires matching TP=EP=2 ranks"
        );
        ensure!(
            config.adapter_max_rank == 0
                && (config.weight_prefix.is_empty() || config.weight_prefix == "model"),
            "unsupported adapter or non-target weight prefix"
        );
        ensure!(
            local.len() == 288
                && local
                    .iter()
                    .enumerate()
                    .all(|(e, &v)| v == config.is_local_expert(e)),
            "rank-local ownership map mismatch"
        );
        let mut projections = Vec::with_capacity(288);
        let mut retained = Vec::with_capacity(288 * 4);
        let mut protected = Vec::with_capacity(1152);
        let mut names = HashSet::with_capacity(1152);
        for (expert, &owned) in local.iter().enumerate() {
            if !owned {
                continue;
            }
            for (is_up, projection) in [(false, "gate_proj"), (true, "up_proj")] {
                let prefix = format!(
                    "{}.mlp.experts.{expert}.{projection}",
                    config.layer_prefix(layer)
                );
                for marker in [
                    "weight_packed",
                    "weight_global_scale",
                    "scale",
                    "weight_scale_inv",
                ] {
                    ensure!(
                        !store.contains(&format!("{prefix}.{marker}")),
                        "nonstandard native NVFP4 marker {marker}"
                    );
                }
                let mut tensor = |suffix: &str, dtype, shape: Option<&[usize]>| -> Result<Span> {
                    let name = format!("{prefix}.{suffix}");
                    if let Some(log) = retirement {
                        ensure!(
                            log.is_live(&name, gpu)?,
                            "native checkpoint already retired: {name}"
                        );
                    }
                    let w = store.get(&name)?;
                    ensure!(w.dtype == dtype, "native dtype mismatch: {name}");
                    let elements = w
                        .shape
                        .iter()
                        .try_fold(1usize, |n, &v| n.checked_mul(v))
                        .ok_or_else(|| anyhow::anyhow!("native shape overflow: {name}"))?;
                    ensure!(
                        shape.map_or(elements == 1, |s| w.shape == s),
                        "native shape mismatch: {name}"
                    );
                    let bytes = elements
                        .checked_mul(w.dtype.byte_size())
                        .ok_or_else(|| anyhow::anyhow!("native byte extent overflow: {name}"))?;
                    let span = Span::new(w.ptr, bytes, if shape.is_some() { 16 } else { 4 })?;
                    retained.push(CheckpointIdentity {
                        name: name.clone(),
                        dtype: w.dtype,
                        shape: w.shape.clone(),
                        span,
                    });
                    names.insert(name);
                    protected.push(span);
                    Ok(span)
                };
                let packed = tensor("weight", WeightDtype::UInt8, Some(&[2048, 2048]))?;
                let scales = tensor("weight_scale", WeightDtype::FP8E4M3, Some(&[2048, 256]))?;
                let scalar = tensor("weight_scale_2", WeightDtype::FP32, None)?;
                let input = if store.contains(&format!("{prefix}.input_scale")) {
                    Some(tensor("input_scale", WeightDtype::FP32, None)?)
                } else {
                    None
                };
                projections.push(NativeProjection {
                    expert,
                    is_up,
                    packed,
                    scales,
                    scalar,
                    scalar_bits: 0,
                    input,
                    input_bits: None,
                });
            }
        }
        ensure!(projections.len() == 288, "incomplete local projection set");
        protected.sort_unstable_by_key(|s| s.ptr.0);
        ensure!(
            protected.windows(2).all(|s| s[0].disjoint(s[1])),
            "native projection/metadata alias"
        );
        // A foreign/shared/down/MTP store entry must not alias a destination.
        // Scan without retaining the model's tensors or copying their bytes.
        for name in store.names().filter(|name| !names.contains(*name)) {
            if let Some(log) = retirement
                && !log.is_live(name, gpu)?
            {
                continue;
            }
            let w = store.get(name)?;
            let bytes = w
                .shape
                .iter()
                .try_fold(w.dtype.byte_size(), |n, &v| n.checked_mul(v))
                .ok_or_else(|| anyhow::anyhow!("foreign store extent overflow: {name}"))?;
            let span = Span::new(w.ptr, bytes, 1)?;
            let index = protected.partition_point(|s| s.ptr.0 < span.ptr.0 + span.bytes as u64);
            ensure!(
                index == 0 || protected[index - 1].disjoint(span),
                "native/foreign allocation alias: {name}"
            );
        }
        // All source extents/aliases have passed before the first scalar read.
        let read_scalar = |span: Span| -> Result<u32> {
            let mut bytes = [0u8; 4];
            gpu.copy_d2h_on_stream(span.ptr, &mut bytes, stream)?;
            let bits = u32::from_le_bytes(bytes);
            ensure!(f32::from_bits(bits).is_finite(), "nonfinite native scalar");
            Ok(bits)
        };
        for projection in &mut projections {
            projection.scalar_bits = read_scalar(projection.scalar)?;
            projection.input_bits = projection.input.map(read_scalar).transpose()?;
        }
        Ok(Self {
            projections,
            retained,
            stream,
            owner: store,
            gpu,
            retirement,
        })
    }
    pub(super) fn projections(&self) -> &[NativeProjection] {
        &self.projections
    }
    pub(super) fn retained(&self) -> &[CheckpointIdentity] {
        &self.retained
    }
    pub(super) fn scratch_is_disjoint(&self, scratch: Span) -> bool {
        if let Some(log) = self.retirement {
            return log
                .disjoint_live(scratch.ptr, scratch.bytes, self.gpu)
                .is_ok();
        }
        // Recheck every borrowed store owner, not only selected projections.
        // No model-wide metadata retention, device access or allocation.
        self.owner.names().all(|name| {
            self.owner
                .get(name)
                .ok()
                .and_then(|w| {
                    let bytes = w
                        .shape
                        .iter()
                        .try_fold(w.dtype.byte_size(), |n, &v| n.checked_mul(v))?;
                    Span::new(w.ptr, bytes, 1).ok()
                })
                .is_some_and(|span| span.disjoint(scratch))
        })
    }
    pub(super) fn stream(&self) -> u64 {
        self.stream
    }
    pub(super) fn gpu(&self) -> &dyn GpuBackend {
        self.gpu
    }
}
#[cfg(test)]
#[path = "gate_up_native_source_tests.rs"]
mod tests;
