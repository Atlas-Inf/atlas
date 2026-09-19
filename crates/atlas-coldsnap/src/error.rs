// SPDX-License-Identifier: AGPL-3.0-only

//! The crate's error taxonomy.
//!
//! The distinction that matters is between *the controller said no* and *we
//! cannot tell what the controller did*. [`ColdsnapError::Protocol`] is the
//! latter's sharp edge: the CLI contradicted itself, so nothing it emitted may
//! be believed. [`crate::ColdsnapOutcome`] carries the softer "unknown" case,
//! which is a valid result rather than an error.

use crate::id::InvalidOperationId;
use crate::runner::RunnerError;
use crate::sha::InvalidRequestSha256;

/// The specific way the CLI contradicted its own contract.
///
/// Enumerated rather than free-text because callers branch on these: a
/// digest mismatch means the receipt belongs to a different request, while
/// empty stdout usually means the process died before it could report.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationKind {
    /// Stdout carried no bytes at all.
    EmptyStdout,
    /// Stdout carried bytes that are not one well-formed JSON document.
    MalformedReceipt,
    /// A second JSON document (or trailing non-whitespace) followed the receipt.
    TrailingJson,
    /// The receipt's `format` / `kind` pair is not the one this crate speaks.
    WrongEnvelope,
    /// A receipt field failed validation (bad id, bad timestamp, bad state …).
    InvalidReceiptField,
    /// The receipt's `request_sha256` is not the digest of the bytes we sent.
    DigestMismatch,
    /// The receipt's `operation_id` / `operation` is not the one we requested.
    RequestMismatch,
    /// Exit status and receipt `state` disagree.
    ExitStateDisagreement,
    /// Stdout exceeded the configured cap and was truncated before parsing.
    StdoutTooLarge,
}

/// A contradiction between what the CLI did and what its contract says it
/// must do. Never treat this as success, and never silently retry it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("ColdSnap protocol violation ({kind:?}): {detail}")]
pub struct ProtocolViolation {
    /// Which invariant was broken.
    pub kind: ViolationKind,
    /// Human-readable specifics, safe to log (never full request bytes).
    pub detail: String,
}

impl ProtocolViolation {
    pub(crate) fn new(kind: ViolationKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

/// The top-level failure type.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ColdsnapError {
    /// The process could not be run, or its I/O failed. Distinct from a
    /// process that ran and reported failure.
    #[error(transparent)]
    Runner(#[from] RunnerError),

    /// The CLI contradicted its own contract. Never success.
    #[error(transparent)]
    Protocol(#[from] ProtocolViolation),

    /// This crate (or the controller) cannot express what was asked: an engine
    /// ColdSnap has no adapter for, an operation a controller lacks, a request
    /// format it does not accept.
    #[error("ColdSnap does not support {subject}: {detail}")]
    Unsupported {
        /// What was asked for, e.g. `"engine 'atlas'"`.
        subject: String,
        /// Why it cannot be honoured.
        detail: String,
    },

    /// The controller did not produce an answer at all — it timed out, was
    /// killed, or exited without a usable result. Distinct from a receipt that
    /// reports failure, which is an operation outcome rather than an error.
    #[error("the ColdSnap controller did not answer: {0}")]
    ControllerUnavailable(String),

    /// A request could not be serialized. Practically unreachable for these
    /// types, but a `Result` is cheaper than an `unwrap` in a library.
    #[error("failed to encode the ColdSnap request: {0}")]
    Encode(#[from] serde_json::Error),

    /// An operation id did not match ColdSnap's `id` pattern.
    #[error(transparent)]
    InvalidOperationId(#[from] InvalidOperationId),

    /// A request digest was not `sha256:` + 64 lowercase hex digits.
    #[error(transparent)]
    InvalidDigest(#[from] InvalidRequestSha256),
}

impl ColdsnapError {
    /// Build an [`ColdsnapError::Unsupported`] from its two parts.
    pub(crate) fn unsupported(subject: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Unsupported {
            subject: subject.into(),
            detail: detail.into(),
        }
    }

    /// True when retrying the same call could plausibly succeed. Only
    /// [`ColdsnapError::Runner`] qualifies; a protocol violation or an
    /// unsupported request is deterministic.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Runner(_))
    }
}
