// SPDX-License-Identifier: AGPL-3.0-only

//! The eager-fallback taxonomy: every reason a graph phase may decline to
//! capture or replay. Split out of `types.rs` for the 500-LoC cap; re-exported
//! from `types`.

use serde::{Deserialize, Serialize};

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
