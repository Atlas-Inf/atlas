// SPDX-License-Identifier: AGPL-3.0-only

//! The operation receipt: `format 2`, `coldsnap-operation-receipt`.
//!
//! Receipt parsing is strict on purpose. The controller's own decoder uses
//! `DisallowUnknownFields` and rejects trailing JSON, so a receipt that would
//! not survive its own round trip is not evidence. [`UnknownFieldPolicy`]
//! offers a bounded escape hatch — unknown **top-level** fields become
//! warnings — but strict is what the crate recommends.
//!
//! [`UnknownFieldPolicy`]: crate::config::UnknownFieldPolicy

use serde::Deserialize as _;

use crate::config::UnknownFieldPolicy;
use crate::error::{ColdsnapError, ProtocolViolation, ViolationKind};
use crate::id::OperationId;
use crate::protocol::constants::{RECEIPT_FORMAT, RECEIPT_KIND};
use crate::protocol::driver::SnapshotDriver;
use crate::protocol::engine::ObservedEngine;
use crate::protocol::operation::Operation;
use crate::protocol::timestamp::Rfc3339Nano;
use crate::sha::RequestSha256;

/// The top-level fields this crate knows about. Anything else is an unknown
/// field, and its treatment depends on the configured policy.
pub const KNOWN_TOP_LEVEL_FIELDS: [&str; 15] = [
    "format",
    "kind",
    "operation_id",
    "operation",
    "state",
    "request_sha256",
    "engine",
    "snapshot_driver",
    "replay_semantics",
    "started_at",
    "completed_at",
    "duration_seconds",
    "timing",
    "result",
    "error",
];

/// The receipt's verdict.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptState {
    /// The operation completed.
    Succeeded,
    /// The operation ran and definitively failed.
    Failed,
}

impl ReceiptState {
    /// Parse the wire state. `None` for anything else — never treated as
    /// success.
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "succeeded" => Some(Self::Succeeded),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }

    /// The wire form.
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }
}

/// The receipt's `result` object.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationResult {
    /// The artifact the operation read or produced.
    #[serde(default)]
    pub artifact: Option<String>,
    /// The artifact the operation wrote.
    #[serde(default)]
    pub output: Option<String>,
    /// Whether the run was `--prepare-only`.
    #[serde(default)]
    pub prepared: bool,
}

/// The receipt as it arrives. Field names and optionality mirror
/// `snapshot.OperationReceipt` exactly.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptWire {
    /// Always [`RECEIPT_FORMAT`].
    pub format: u32,
    /// Always [`RECEIPT_KIND`].
    pub kind: String,
    /// The request's `id`.
    pub operation_id: String,
    /// The request's `operation`.
    pub operation: String,
    /// `succeeded` or `failed`.
    pub state: String,
    /// The controller's digest of the request it decoded.
    pub request_sha256: String,
    /// The engine that ran the operation.
    pub engine: String,
    /// The snapshot driver that ran the operation.
    pub snapshot_driver: String,
    /// The operation's replay semantics.
    pub replay_semantics: String,
    /// RFC 3339 start instant.
    pub started_at: String,
    /// RFC 3339 completion instant.
    pub completed_at: String,
    /// Wall-clock duration.
    pub duration_seconds: f64,
    /// The timing envelope. Retained verbatim: this crate does not interpret
    /// the span tree, and modelling a diagnostic tree it never reads would
    /// only add a way to fail.
    pub timing: serde_json::Value,
    /// Artifact/output/prepared.
    pub result: OperationResult,
    /// Present exactly when `state` is `failed`.
    #[serde(default)]
    pub error: Option<String>,
}

/// A validated receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct Receipt {
    /// The request's id, echoed.
    pub operation_id: OperationId,
    /// The operation, echoed.
    pub operation: Operation,
    /// The verdict.
    pub state: ReceiptState,
    /// The controller's request digest.
    pub request_sha256: RequestSha256,
    /// The engine that ran it, tolerantly classified.
    pub engine: ObservedEngine,
    /// The snapshot driver that ran it.
    pub snapshot_driver: SnapshotDriver,
    /// The replay semantics the controller reported.
    pub replay_semantics: String,
    /// Start instant.
    pub started_at: Rfc3339Nano,
    /// Completion instant.
    pub completed_at: Rfc3339Nano,
    /// Wall-clock duration.
    pub duration_seconds: f64,
    /// Artifact/output/prepared.
    pub result: OperationResult,
    /// The failure text, when the operation failed.
    pub error: Option<String>,
    /// Unknown top-level fields, when the policy tolerates them.
    pub warnings: Vec<String>,
    /// The timing envelope, verbatim.
    pub timing: serde_json::Value,
}

impl Receipt {
    /// Whether the controller reported success.
    pub fn is_success(&self) -> bool {
        self.state == ReceiptState::Succeeded
    }

    /// Whether the receipt claims a `--prepare-only` run.
    pub fn is_prepared(&self) -> bool {
        self.result.prepared
    }
}

fn field_violation(detail: impl Into<String>) -> ColdsnapError {
    ColdsnapError::Protocol(ProtocolViolation::new(
        ViolationKind::InvalidReceiptField,
        detail,
    ))
}

impl Receipt {
    /// Convert a wire receipt into a validated one.
    ///
    /// Every check the controller's `OperationReceipt.Validate` performs is
    /// repeated here, because a receipt that would fail the controller's own
    /// validation is not something to act on.
    pub fn from_wire(wire: ReceiptWire, warnings: Vec<String>) -> Result<Self, ColdsnapError> {
        if wire.format != RECEIPT_FORMAT || wire.kind != RECEIPT_KIND {
            return Err(ColdsnapError::Protocol(ProtocolViolation::new(
                ViolationKind::WrongEnvelope,
                format!(
                    "receipt envelope is format {} kind {:?}, expected format {} kind {:?}",
                    wire.format, wire.kind, RECEIPT_FORMAT, RECEIPT_KIND
                ),
            )));
        }

        let operation_id = OperationId::parse(&wire.operation_id)
            .map_err(|e| field_violation(format!("receipt operation_id is invalid: {e}")))?;
        let operation = Operation::from_wire(&wire.operation).ok_or_else(|| {
            field_violation(format!(
                "receipt operation {:?} is not one of the seven",
                wire.operation
            ))
        })?;
        let state = ReceiptState::from_wire(&wire.state).ok_or_else(|| {
            field_violation(format!(
                "receipt state {:?} is neither 'succeeded' nor 'failed'",
                wire.state
            ))
        })?;
        let request_sha256 = RequestSha256::parse_wire(&wire.request_sha256)
            .map_err(|e| field_violation(format!("receipt request_sha256 is invalid: {e}")))?;
        let snapshot_driver =
            SnapshotDriver::from_wire(&wire.snapshot_driver).ok_or_else(|| {
                field_violation(format!(
                    "receipt snapshot_driver {:?} is not registered",
                    wire.snapshot_driver
                ))
            })?;
        let started_at = Rfc3339Nano::parse(&wire.started_at)
            .map_err(|e| field_violation(format!("receipt started_at is invalid: {e}")))?;
        let completed_at = Rfc3339Nano::parse(&wire.completed_at)
            .map_err(|e| field_violation(format!("receipt completed_at is invalid: {e}")))?;
        if completed_at.instant() < started_at.instant() {
            return Err(field_violation("receipt completed_at is before started_at"));
        }
        if !wire.duration_seconds.is_finite() || wire.duration_seconds < 0.0 {
            return Err(field_violation(
                "receipt duration_seconds is negative or not finite",
            ));
        }

        // The controller recomputes this from the operation and the prepared
        // flag; a mismatch means the receipt does not describe the operation
        // it names.
        let expected = operation.replay_semantics(wire.result.prepared);
        if wire.replay_semantics != expected {
            return Err(field_violation(format!(
                "receipt replay_semantics {:?} disagrees with {operation} (expected {expected:?})",
                wire.replay_semantics
            )));
        }

        match (state, wire.error.as_deref()) {
            (ReceiptState::Succeeded, Some(text)) if !text.is_empty() => {
                return Err(field_violation(
                    "a succeeded receipt must not carry an error",
                ));
            }
            (ReceiptState::Failed, None) => {
                return Err(field_violation("a failed receipt must carry an error"));
            }
            (ReceiptState::Failed, Some("")) => {
                return Err(field_violation(
                    "a failed receipt must carry a non-empty error",
                ));
            }
            _ => {}
        }

        Ok(Self {
            operation_id,
            operation,
            state,
            request_sha256,
            engine: ObservedEngine::from_wire(&wire.engine),
            snapshot_driver,
            replay_semantics: wire.replay_semantics,
            started_at,
            completed_at,
            duration_seconds: wire.duration_seconds,
            result: wire.result,
            error: wire.error,
            warnings,
            timing: wire.timing,
        })
    }
}

/// Parse exactly one JSON receipt from `bytes`.
///
/// Rejects empty output, trailing content, and any second document. The
/// controller reserves stdout for the receipt and sends progress to stderr,
/// so a receipt that does not stand alone is a protocol violation rather than
/// something to scrape around.
pub fn parse(bytes: &[u8], policy: UnknownFieldPolicy) -> Result<Receipt, ColdsnapError> {
    // A UTF-8 BOM is a wrapper artefact, not part of the document. Rejecting a
    // receipt for it would report a protocol violation where the controller
    // behaved correctly and something in between added three bytes.
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);

    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Err(ColdsnapError::Protocol(ProtocolViolation::new(
            ViolationKind::EmptyStdout,
            "the controller produced no receipt on stdout",
        )));
    }

    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let mut value = serde_json::Value::deserialize(&mut deserializer).map_err(|error| {
        ColdsnapError::Protocol(ProtocolViolation::new(
            ViolationKind::MalformedReceipt,
            format!("stdout is not a single JSON document: {error}"),
        ))
    })?;
    deserializer.end().map_err(|error| {
        ColdsnapError::Protocol(ProtocolViolation::new(
            ViolationKind::TrailingJson,
            format!("trailing content followed the receipt: {error}"),
        ))
    })?;

    let mut warnings = Vec::new();
    let object = value.as_object_mut().ok_or_else(|| {
        ColdsnapError::Protocol(ProtocolViolation::new(
            ViolationKind::MalformedReceipt,
            "the receipt is not a JSON object",
        ))
    })?;

    let unknown: Vec<String> = object
        .keys()
        .filter(|key| !KNOWN_TOP_LEVEL_FIELDS.contains(&key.as_str()))
        .cloned()
        .collect();
    match policy {
        UnknownFieldPolicy::Strict => {
            if let Some(first) = unknown.first() {
                return Err(ColdsnapError::Protocol(ProtocolViolation::new(
                    ViolationKind::InvalidReceiptField,
                    format!(
                        "receipt carries unknown field {first:?} (and {} more); \
                         the controller's own decoder rejects these",
                        unknown.len().saturating_sub(1)
                    ),
                )));
            }
        }
        UnknownFieldPolicy::IgnoreTopLevelWithWarning => {
            for key in &unknown {
                warnings.push(format!("ignored unknown receipt field {key:?}"));
                object.remove(key);
            }
        }
    }

    let wire: ReceiptWire = serde_json::from_value(value).map_err(|error| {
        ColdsnapError::Protocol(ProtocolViolation::new(
            ViolationKind::InvalidReceiptField,
            format!("receipt does not match the contract: {error}"),
        ))
    })?;

    Receipt::from_wire(wire, warnings)
}

// Split out for the 500-LoC cap, per the Atlas idiom (see
// `crates/spark-server/src/main_modules/model_swap.rs`).
#[cfg(test)]
#[path = "receipt_tests.rs"]
mod tests;
