// SPDX-License-Identifier: AGPL-3.0-only

//! Request-local failures.
//!
//! On a multi-rank world any worker error used to end the process: after a
//! failed collective the communicator's state is unknown, and a worker that
//! kept going would let the head issue its next collective against a peer that
//! is out of step. Some failures are not of that kind. A malformed vision
//! layout (placeholders the encoded rows do not cover, positions the grids do
//! not explain) is decided from data every rank holds alike, at the same point
//! of the same command, before the chunk's layers launch a collective; every
//! rank raises it together, so the ranks are still in step. Such an error
//! fails its request only: the head frees the sequence and resets the worker's
//! slot (`0xFFFFFFF1`), and the pair keeps serving.
//!
//! Raise one only where that holds. A failure one rank alone can hit (a head-
//! only encode) must first be agreed across the ranks
//! (`TransformerModel::ep_min_u32`) at a point both reach.

use std::fmt;

/// A failure every rank raised at the same point of the same request.
#[derive(Debug)]
pub struct RequestLocalError(pub String);

impl fmt::Display for RequestLocalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RequestLocalError {}

/// A request-local error carrying `msg`.
pub fn request_local(msg: impl fmt::Display) -> anyhow::Error {
    anyhow::Error::new(RequestLocalError(msg.to_string()))
}

/// Whether `e`, under any context added since, is request-local.
pub fn is_request_local(e: &anyhow::Error) -> bool {
    e.downcast_ref::<RequestLocalError>().is_some()
}

#[cfg(test)]
mod tests {
    use super::{is_request_local, request_local};
    use anyhow::Context;

    #[test]
    fn context_layers_keep_the_classification() {
        let e = Err::<(), _>(request_local("vision rows short"))
            .context("prefill chunk")
            .context("worker command")
            .unwrap_err();
        assert!(is_request_local(&e));
        assert!(format!("{e:#}").contains("vision rows short"));
    }

    #[test]
    fn other_errors_stay_fatal() {
        let e = anyhow::anyhow!("NCCL broadcast failed").context("prefill chunk");
        assert!(!is_request_local(&e));
    }
}
