// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit configuration for talking to the controller.
//!
//! There is deliberately no `Default` impl. A timeout that "seems reasonable"
//! is wrong for `capture` (the vLLM adapter's own budget is 45 minutes) and
//! wrong for `status` (seconds). The caller — which knows which operation it
//! is about to run — owns the number.

use std::path::PathBuf;
use std::time::Duration;

/// What to do when a receipt carries a top-level field this crate does not
/// know about.
///
/// Receipts are validated with `deny_unknown_fields`, mirroring the
/// controller's own `DisallowUnknownFields` decoder. That is the right
/// default: a receipt that grew a field is a contract change, and quietly
/// ignoring it is how a caller ends up acting on a field it never read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnknownFieldPolicy {
    /// Any unknown field is a protocol violation. The default the crate
    /// recommends, and the one that matches the controller's own decoder.
    Strict,
    /// Unknown **top-level** fields are recorded as warnings on the receipt
    /// and otherwise ignored, so a newer controller can add fields without
    /// breaking a caller that does not read them. Unknown fields nested
    /// inside a known object remain a violation — a bounded tolerance, not a
    /// blanket one.
    IgnoreTopLevelWithWarning,
}

/// Where the controller binary is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryLocation {
    /// A path the caller resolved. Not looked up on `PATH`.
    Explicit(PathBuf),
    /// A bare program name, resolved through `PATH` by the runner.
    OnPath(String),
}

impl BinaryLocation {
    /// The program to hand to the OS.
    pub fn program(&self) -> &std::ffi::OsStr {
        match self {
            Self::Explicit(path) => path.as_os_str(),
            Self::OnPath(name) => std::ffi::OsStr::new(name),
        }
    }
}

/// Everything the client needs that is not part of a single request.
#[derive(Debug, Clone)]
pub struct ColdsnapConfig {
    /// The controller binary.
    pub binary: BinaryLocation,
    /// Wall-clock budget for one invocation. On expiry the child is killed
    /// and the operation is classified **unknown**, never failed — the
    /// manager-side work may continue after the CLI dies.
    pub timeout: Duration,
    /// Hard cap on stdout. The receipt is small; anything larger is a
    /// protocol violation or a runaway, and is killed rather than buffered.
    pub max_stdout_bytes: usize,
    /// Hard cap on stderr. Exceeding it truncates with a flag; stderr is
    /// diagnostic and must never fail an otherwise valid operation.
    pub max_stderr_bytes: usize,
    /// Unknown-receipt-field handling.
    pub unknown_field_policy: UnknownFieldPolicy,
    /// Extra environment for every invocation.
    ///
    /// This is where an orchestrator puts the operation-scoped host-provider
    /// credentials (`COLDSNAP_HOST_PROVIDER_SOCKET`,
    /// `COLDSNAP_HOST_PROVIDER_TOKEN`) that ColdSnap requires for anything
    /// other than `capabilities`. Empty means "none", which is explicit
    /// rather than a default.
    pub environment: Vec<(String, String)>,
}

impl ColdsnapConfig {
    /// Build a config. Every knob is required; see the module docs.
    pub fn new(
        binary: BinaryLocation,
        timeout: Duration,
        max_stdout_bytes: usize,
        max_stderr_bytes: usize,
        unknown_field_policy: UnknownFieldPolicy,
    ) -> Self {
        Self {
            binary,
            timeout,
            max_stdout_bytes,
            max_stderr_bytes,
            unknown_field_policy,
            environment: Vec::new(),
        }
    }

    /// Attach the environment every invocation should receive.
    pub fn with_environment(mut self, environment: Vec<(String, String)>) -> Self {
        self.environment = environment;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_binary_is_not_resolved_through_path() {
        let location = BinaryLocation::Explicit(PathBuf::from("/opt/coldsnap/bin/coldsnap"));
        assert_eq!(
            location.program(),
            std::ffi::OsStr::new("/opt/coldsnap/bin/coldsnap")
        );
    }

    #[test]
    fn on_path_binary_keeps_the_bare_name() {
        let location = BinaryLocation::OnPath("coldsnap".to_owned());
        assert_eq!(location.program(), std::ffi::OsStr::new("coldsnap"));
    }

    #[test]
    fn a_new_config_has_no_ambient_environment() {
        let config = ColdsnapConfig::new(
            BinaryLocation::OnPath("coldsnap".to_owned()),
            Duration::from_secs(5),
            1024,
            4096,
            UnknownFieldPolicy::Strict,
        );
        assert!(config.environment.is_empty());
    }
}
