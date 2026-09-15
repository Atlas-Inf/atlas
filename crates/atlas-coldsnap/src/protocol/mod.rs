// SPDX-License-Identifier: AGPL-3.0-only

//! The ColdSnap wire contract.
//!
//! Transcribed from the controller's Go source rather than from prose, so the
//! field names, literal envelopes, and validators match what the controller
//! actually enforces:
//!
//! | Artifact | Source | Envelope |
//! |---|---|---|
//! | Request | `internal/snapshot/schema.go` | format 4, `coldsnap-operation-request` |
//! | Receipt | `internal/snapshot/receipt.go` | format 2, `coldsnap-operation-receipt` |
//! | Capabilities | `internal/cli/capabilities.go` | format 1, `coldsnap-controller-capabilities` |
//!
//! Wire DTOs and domain types are kept apart on purpose. A DTO mirrors the
//! JSON; a domain type carries the validation the controller promises. When
//! the upstream contract moves, only the DTO and its conversion change.

pub mod capabilities;
pub mod constants;
pub mod driver;
pub mod engine;
pub mod operation;
pub mod receipt;
pub mod request;
pub mod timestamp;
