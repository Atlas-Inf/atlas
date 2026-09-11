// SPDX-License-Identifier: AGPL-3.0-only

use crate::gpu::DevicePtr;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

pub const GRAPH_KEY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphPhase {
    Prefill,
    Decode,
    Verify,
    Propose,
    Fused,
}

impl GraphPhase {
    pub(crate) const COUNT: usize = 5;

    pub(crate) const ALL: [Self; Self::COUNT] = [
        Self::Prefill,
        Self::Decode,
        Self::Verify,
        Self::Propose,
        Self::Fused,
    ];

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Prefill => 0,
            Self::Decode => 1,
            Self::Verify => 2,
            Self::Propose => 3,
            Self::Fused => 4,
        }
    }
}

impl fmt::Display for GraphPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Verify => "verify",
            Self::Propose => "propose",
            Self::Fused => "fused",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphMode {
    Disabled,
    Full,
    Breakable,
    Piecewise,
}

impl fmt::Display for GraphMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Disabled => "disabled",
            Self::Full => "full",
            Self::Breakable => "breakable",
            Self::Piecewise => "piecewise",
        })
    }
}

impl FromStr for GraphMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "disabled" => Ok(Self::Disabled),
            "full" => Ok(Self::Full),
            "breakable" => Ok(Self::Breakable),
            "piecewise" => Ok(Self::Piecewise),
            _ => Err(format!(
                "unknown CUDA graph mode {value:?}; expected disabled, full, breakable, or piecewise"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SpeculativeAlgorithm {
    None = 0,
    Mtp = 1,
    Dflash = 2,
    Dspark = 3,
    SelfSpeculative = 4,
    Ngram = 5,
}

impl SpeculativeAlgorithm {
    /// Reverse of the `#[repr(u8)]` discriminant, for the metrics label.
    pub(crate) const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Mtp,
            2 => Self::Dflash,
            3 => Self::Dspark,
            4 => Self::SelfSpeculative,
            5 => Self::Ngram,
            _ => Self::None,
        }
    }
}

impl fmt::Display for SpeculativeAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Mtp => "mtp",
            Self::Dflash => "dflash",
            Self::Dspark => "dspark",
            Self::SelfSpeculative => "self",
            Self::Ngram => "ngram",
        })
    }
}

impl FromStr for SpeculativeAlgorithm {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "mtp" => Ok(Self::Mtp),
            "dflash" => Ok(Self::Dflash),
            "dspark" => Ok(Self::Dspark),
            "self" => Ok(Self::SelfSpeculative),
            "ngram" => Ok(Self::Ngram),
            _ => Err(format!(
                "unknown speculative algorithm {value:?}; expected none, mtp, dflash, dspark, self, or ngram"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphSegment {
    Whole,
    PrefillCompute,
    PrefillLayers,
    PrefillFinalize,
    DecodeBody,
    VerifyBody,
    ProposeBody,
    FusedBody,
    Conditional,
    LoopBody,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeBucket {
    pub token_limit: u32,
    pub request_limit: u32,
}

impl ShapeBucket {
    pub fn contains(self, token_count: u32, request_count: u32) -> bool {
        token_count <= self.token_limit && request_count <= self.request_limit
    }
}

pub fn classify_shape_bucket(
    buckets: &[ShapeBucket],
    token_count: u32,
    request_count: u32,
) -> Option<ShapeBucket> {
    buckets
        .iter()
        .copied()
        .filter(|bucket| bucket.contains(token_count, request_count))
        .min_by_key(|bucket| (bucket.token_limit, bucket.request_limit))
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphPayload {
    Prefill {
        request_count: u32,
        token_count: u32,
        sequence_lengths: Vec<u32>,
        slots: Vec<u32>,
        block_counts: Vec<u32>,
        chunk_start: u32,
        last_chunk: bool,
        paged: bool,
        mrope: bool,
    },
    Decode {
        request_count: u32,
        padded_request_count: u32,
        slots: Vec<u32>,
    },
    DecodeLoop {
        request_count: u32,
        max_iterations: u32,
        output_capacity: u32,
        slots: Vec<u32>,
    },
    Verify {
        request_count: u32,
        row_count: u32,
        slots: Vec<u32>,
        depths: Vec<u32>,
        layout: Vec<u32>,
    },
    Propose {
        request_count: u32,
        draft_depth: u32,
        segment_index: u32,
        /// Opaque, model-specific identity words that must be part of the key
        /// or a captured propose subgraph would be replayed against state it
        /// was not captured for. The DFlash drafter fills this with its
        /// `DflashGraphIdentity` (graph owner, block-table address, ctx
        /// accumulator address, scratch pointer, lane); an empty vector means
        /// the model's propose key needs no extra identity.
        #[serde(default)]
        key_words: Vec<u64>,
    },
    Fused {
        request_count: u32,
        row_count: u32,
        draft_depth: u32,
        slots: Vec<u32>,
    },
}

impl GraphPayload {
    pub fn phase(&self) -> GraphPhase {
        match self {
            Self::Prefill { .. } => GraphPhase::Prefill,
            Self::Decode { .. } | Self::DecodeLoop { .. } => GraphPhase::Decode,
            Self::Verify { .. } => GraphPhase::Verify,
            Self::Propose { .. } => GraphPhase::Propose,
            Self::Fused { .. } => GraphPhase::Fused,
        }
    }

    pub fn slots(&self) -> &[u32] {
        match self {
            Self::Prefill { slots, .. }
            | Self::Decode { slots, .. }
            | Self::DecodeLoop { slots, .. }
            | Self::Verify { slots, .. }
            | Self::Fused { slots, .. } => slots,
            Self::Propose { .. } => &[],
        }
    }

    pub fn shape_counts(&self) -> (u32, u32) {
        match self {
            Self::Prefill {
                request_count,
                token_count,
                ..
            } => (*token_count, *request_count),
            Self::Decode { request_count, .. } => (*request_count, *request_count),
            Self::DecodeLoop {
                request_count,
                max_iterations,
                ..
            } => (
                request_count.saturating_mul(*max_iterations),
                *request_count,
            ),
            Self::Verify {
                request_count,
                row_count,
                ..
            }
            | Self::Fused {
                request_count,
                row_count,
                ..
            } => (*row_count, *request_count),
            Self::Propose {
                request_count,
                draft_depth,
                ..
            } => (request_count.saturating_mul(*draft_depth), *request_count),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphFingerprint {
    pub runtime: String,
    pub model: String,
    pub kernel_build: String,
    pub device: String,
    pub cuda: String,
    pub driver: String,
    pub memory_layout: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphKey {
    pub schema_version: u32,
    pub phase: GraphPhase,
    pub mode: GraphMode,
    pub segment: GraphSegment,
    pub speculative_algorithm: SpeculativeAlgorithm,
    pub fingerprint: GraphFingerprint,
    pub resource_generation: u64,
    pub payload: GraphPayload,
}

impl GraphKey {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != GRAPH_KEY_SCHEMA_VERSION {
            return Err(format!(
                "graph key schema {} is not supported by runtime schema {}",
                self.schema_version, GRAPH_KEY_SCHEMA_VERSION
            ));
        }
        if self.phase != self.payload.phase() {
            return Err(format!(
                "graph key phase {} does not match payload phase {}",
                self.phase,
                self.payload.phase()
            ));
        }
        if self.mode == GraphMode::Disabled {
            return Err("a disabled graph mode cannot identify an executable graph".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphIdentity {
    pub bucket: ShapeBucket,
    pub key: GraphKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCost {
    pub estimated_bytes: u64,
    pub node_count: u64,
    pub child_count: u32,
    pub staging_bytes: u64,
}

impl GraphCost {
    pub fn total_bytes(self) -> u64 {
        self.estimated_bytes.saturating_add(self.staging_bytes)
    }

    pub fn include(self, child: Self) -> Self {
        Self {
            estimated_bytes: self.estimated_bytes.saturating_add(child.estimated_bytes),
            node_count: self.node_count.saturating_add(child.node_count),
            child_count: self
                .child_count
                .saturating_add(child.child_count)
                .saturating_add(1),
            staging_bytes: self.staging_bytes.saturating_add(child.staging_bytes),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEnvironment {
    pub device: String,
    pub cuda: String,
    pub driver: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCapabilities {
    pub basic_graphs: bool,
    pub debug_dot: bool,
    pub graph_upload: bool,
    pub conditional_nodes: bool,
    pub while_nodes: bool,
    pub native_serialization: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedLoopState {
    pub continuation: DevicePtr,
    pub iteration: DevicePtr,
    pub cancellation: DevicePtr,
    pub exit_reason: DevicePtr,
    pub output: DevicePtr,
    pub output_capacity: u32,
    pub max_iterations: u32,
}

impl BoundedLoopState {
    pub fn validate(self) -> Result<(), String> {
        if [
            self.continuation,
            self.iteration,
            self.cancellation,
            self.exit_reason,
            self.output,
        ]
        .into_iter()
        .any(DevicePtr::is_null)
        {
            return Err("bounded CUDA graph loop state contains a null device pointer".to_string());
        }
        if self.max_iterations == 0 || self.output_capacity < self.max_iterations {
            return Err(
                "bounded CUDA graph loop output capacity must cover its hard cap".to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u32)]
pub enum BoundedLoopExitReason {
    Running = 0,
    Stop = 1,
    Cancelled = 2,
    IterationCap = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum GraphFallbackReason {
    RuntimeDisabled,
    PhaseDisabled,
    ModeUnsupported,
    ShapeUnsupported,
    ModelUnsupported,
    FeatureUnsupported,
    PointerUnstable,
    AllocationDuringCapture,
    HostSynchronization,
    DynamicTopology,
    DriverUnsupported,
    HardwareUnsupported,
    DiagnosticActive,
    HighSpeedSwap,
    Profiling,
    ConfidentialComputingRestriction,
    CacheQuota,
    NegativeCached,
    CaptureFailed,
    CaptureBodyFailed,
    InstantiateFailed,
    ReplayLaunchFailed,
    ReplayAsyncFault,
    StaleKey,
    ManifestIncompatible,
    MemoryPressure,
    StreamMismatch,
    Retired,
    Internal,
}

impl GraphFallbackReason {
    pub const ALL: [Self; 29] = [
        Self::RuntimeDisabled,
        Self::PhaseDisabled,
        Self::ModeUnsupported,
        Self::ShapeUnsupported,
        Self::ModelUnsupported,
        Self::FeatureUnsupported,
        Self::PointerUnstable,
        Self::AllocationDuringCapture,
        Self::HostSynchronization,
        Self::DynamicTopology,
        Self::DriverUnsupported,
        Self::HardwareUnsupported,
        Self::DiagnosticActive,
        Self::HighSpeedSwap,
        Self::Profiling,
        Self::ConfidentialComputingRestriction,
        Self::CacheQuota,
        Self::NegativeCached,
        Self::CaptureFailed,
        Self::CaptureBodyFailed,
        Self::InstantiateFailed,
        Self::ReplayLaunchFailed,
        Self::ReplayAsyncFault,
        Self::StaleKey,
        Self::ManifestIncompatible,
        Self::MemoryPressure,
        Self::StreamMismatch,
        Self::Retired,
        Self::Internal,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::RuntimeDisabled => "runtime_disabled",
            Self::PhaseDisabled => "phase_disabled",
            Self::ModeUnsupported => "mode_unsupported",
            Self::ShapeUnsupported => "shape_unsupported",
            Self::ModelUnsupported => "model_unsupported",
            Self::FeatureUnsupported => "feature_unsupported",
            Self::PointerUnstable => "pointer_unstable",
            Self::AllocationDuringCapture => "allocation_during_capture",
            Self::HostSynchronization => "host_synchronization",
            Self::DynamicTopology => "dynamic_topology",
            Self::DriverUnsupported => "driver_unsupported",
            Self::HardwareUnsupported => "hardware_unsupported",
            Self::DiagnosticActive => "diagnostic_active",
            Self::HighSpeedSwap => "high_speed_swap",
            Self::Profiling => "profiling",
            Self::ConfidentialComputingRestriction => "confidential_computing_restriction",
            Self::CacheQuota => "cache_quota",
            Self::NegativeCached => "negative_cached",
            Self::CaptureFailed => "capture_failed",
            Self::CaptureBodyFailed => "capture_body_failed",
            Self::InstantiateFailed => "instantiate_failed",
            Self::ReplayLaunchFailed => "replay_launch_failed",
            Self::ReplayAsyncFault => "replay_async_fault",
            Self::StaleKey => "stale_key",
            Self::ManifestIncompatible => "manifest_incompatible",
            Self::MemoryPressure => "memory_pressure",
            Self::StreamMismatch => "stream_mismatch",
            Self::Retired => "retired",
            Self::Internal => "internal",
        }
    }

    pub(crate) const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureFailurePolicy {
    Retry,
    NegativeCache,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphRuntimeError {
    pub reason: GraphFallbackReason,
    pub detail: String,
}

impl GraphRuntimeError {
    pub fn new(reason: GraphFallbackReason, detail: impl Into<String>) -> Self {
        Self {
            reason,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for GraphRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.reason.label(), self.detail)
    }
}

impl std::error::Error for GraphRuntimeError {}
