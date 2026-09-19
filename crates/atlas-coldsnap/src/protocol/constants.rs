// SPDX-License-Identifier: AGPL-3.0-only

//! Literal envelopes. Every parse validates against these rather than
//! accepting whatever `format` number happens to arrive.

/// `snapshot.RequestFormat`.
pub const REQUEST_FORMAT: u32 = 4;
/// `snapshot.RequestKind`.
pub const REQUEST_KIND: &str = "coldsnap-operation-request";

/// `snapshot.OperationReceiptFormat`.
pub const RECEIPT_FORMAT: u32 = 2;
/// `snapshot.OperationReceiptKind`.
pub const RECEIPT_KIND: &str = "coldsnap-operation-receipt";

/// `cli.capabilitiesFormat`.
pub const CAPABILITIES_FORMAT: u32 = 1;
/// The capabilities document's `kind`.
pub const CAPABILITIES_KIND: &str = "coldsnap-controller-capabilities";

/// `snapshot.ArtifactFormat`. Format 9 is the only accepted artifact format;
/// recorded here because a request's `artifact` must point at one.
pub const ARTIFACT_FORMAT: u32 = 9;

/// `cli.maximumRequestBytes` — the controller refuses a larger request.
pub const MAXIMUM_REQUEST_BYTES: usize = 16 << 20;
