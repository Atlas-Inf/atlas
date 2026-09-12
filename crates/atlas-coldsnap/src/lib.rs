// SPDX-License-Identifier: AGPL-3.0-only
#![deny(warnings)]
#![deny(clippy::all)]

//! A strict, CUDA-free client for the [ColdSnap] controller protocol.
//!
//! ColdSnap is a manager-driven snapshot and memory-lifecycle system for
//! distributed vLLM / SGLang inference. Its public CLI is a *controller
//! primitive*: a placement manager sends one immutable JSON request on stdin
//! and reads exactly one JSON receipt from stdout. This crate is Atlas's
//! anti-corruption layer over that boundary — it knows how to talk to
//! ColdSnap safely and faithfully, and nothing else.
//!
//! [ColdSnap]: https://github.com/sparksq/coldsnap
//!
//! # What this crate owns
//!
//! * The wire contract: [`protocol::request`], [`protocol::receipt`],
//!   [`protocol::capabilities`], and the literal `format` / `kind` constants
//!   each one is validated against.
//! * Argv construction and the single process-I/O seam, [`runner::CommandRunner`].
//! * [`template::PreparedRequest`]: bytes that were serialized once, hashed,
//!   and written verbatim — so the digest the controller echoes back is a
//!   statement about the bytes it actually received.
//! * [`template::RestoreShapedTemplate`]: the mechanical derivation of a
//!   `sleep` / `wake` / `status` request from a restore request.
//!
//! # What this crate deliberately does NOT own
//!
//! * **Atlas lifecycle policy.** What "sleep" means for an Atlas CUDA
//!   context — release weights and KV, retain the process — is decided by
//!   `spark-server`, which calls this crate. A foreign-protocol adapter that
//!   also owns Atlas domain semantics would invert the dependency and make
//!   every lifecycle-policy change a change to the ColdSnap adapter.
//! * **Retry, locking, and reconciliation policy.** See the outcome taxonomy
//!   below; the caller decides.
//! * **Paths.** `artifact` / `output` are opaque strings. This crate never
//!   creates, resolves, normalizes, or deletes them.
//!
//! # The engine gap, stated plainly
//!
//! ColdSnap validates `launch.engine` against exactly `vllm` and `sglang`
//! (`internal/engineadapter.Lookup`, and again in receipt validation). **There
//! is no Atlas engine adapter upstream.** This crate therefore refuses to emit
//! an operation request for any other engine — see
//! [`protocol::engine::ColdsnapEngine`] — rather than papering over the gap by
//! claiming `vllm`. A receipt that says `engine: "vllm"` for a workload that
//! is not vLLM is worse than no receipt: it makes lifecycle decisions look
//! evidenced when they are not.
//!
//! # Outcomes are four-way, not two-way
//!
//! A process boundary has more disagreement modes than success and failure,
//! and a lifecycle caller has to tell them apart. [`ColdsnapOutcome`] keeps
//! them separate:
//!
//! * [`ColdsnapOutcome::Success`] — exit 0 **and** a valid succeeded receipt
//!   that cross-checks against the request we sent.
//! * [`ColdsnapOutcome::OperationFailed`] — the controller ran the operation
//!   and reported a definitive failure.
//! * [`ColdsnapOutcome::Unknown`] — the operation may or may not have
//!   happened (timeout, killed, no receipt). The caller must reconcile via
//!   `status`; it must not blindly retry a mutating operation, because
//!   `sleep` / `wake` take a fresh `id` every time and are therefore not
//!   deduplicated by the controller.
//! * [`ColdsnapError::ProtocolViolation`] — the CLI behaved inconsistently
//!   (nonzero exit with a succeeded receipt, wrong digest, trailing JSON).
//!   Never success, never silently retried.
//!
//! # Testing
//!
//! Everything here is testable without a GPU or the `coldsnap` binary:
//! [`runner::fake::ScriptedRunner`] replays canned process results, and
//! [`runner::fake::FakeRunner`] records invocations for assertion.

pub mod client;
pub mod config;
pub mod error;
pub mod id;
pub mod protocol;
pub mod runner;
pub mod sha;
pub mod template;

pub use client::{ColdsnapClient, ColdsnapOutcome, OperationFailure, UnknownOperation};
pub use config::ColdsnapConfig;
pub use error::ColdsnapError;
pub use id::OperationId;
pub use protocol::capabilities::Capabilities;
pub use protocol::engine::{ColdsnapEngine, ObservedEngine};
pub use protocol::operation::Operation;
pub use protocol::receipt::{Receipt, ReceiptState};
pub use protocol::request::OperationRequest;
pub use runner::{CommandRunner, Invocation, RunnerOutput};
pub use sha::RequestSha256;
pub use template::{PreparedRequest, RestoreShapedTemplate};
