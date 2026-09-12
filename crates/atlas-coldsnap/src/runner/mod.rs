// SPDX-License-Identifier: AGPL-3.0-only

//! The process-I/O seam.
//!
//! Everything this crate does to the outside world passes through
//! [`CommandRunner`]. That is what makes the protocol logic — digest
//! cross-checks, the exit-status/receipt-state matrix, restore-shaped
//! templates — testable on a machine with no `coldsnap` binary and no GPU.

pub mod fake;
pub mod process;

use std::time::Duration;

/// One fully-specified child process invocation.
///
/// There is no shell: `program` and `args` go straight to the OS. The
/// controller's CLI takes an immutable JSON request on stdin, so `stdin` is
/// normally `Some(bytes)` that were serialized and hashed exactly once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    /// Program to execute.
    pub program: std::ffi::OsString,
    /// Arguments, in order.
    pub args: Vec<String>,
    /// Bytes to write to the child's stdin, then close it.
    pub stdin: Option<Vec<u8>>,
    /// Wall-clock budget. On expiry the child (and, on Unix, its process
    /// group) is killed and the caller must classify the result as unknown.
    pub timeout: Duration,
    /// Hard cap on captured stdout.
    pub max_stdout_bytes: usize,
    /// Hard cap on captured stderr.
    pub max_stderr_bytes: usize,
    /// Extra environment for the child.
    pub environment: Vec<(String, String)>,
}

/// What a finished (or killed) child produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunnerOutput {
    /// Exit code, or `None` when the process was signalled or killed.
    pub exit_code: Option<i32>,
    /// Terminating signal, when the platform reports one.
    pub signal: Option<i32>,
    /// Captured stdout, bounded by the invocation's cap.
    pub stdout: Vec<u8>,
    /// Captured stderr, bounded by the invocation's cap.
    pub stderr: Vec<u8>,
    /// Whether stdout hit its cap and was cut short.
    pub stdout_truncated: bool,
    /// Whether stderr hit its cap and was cut short.
    pub stderr_truncated: bool,
    /// The child was killed because the timeout expired.
    pub timed_out: bool,
    /// The child closed stdin before it was fully written — usually because
    /// it exited early. Not an error in itself: its stdout/stderr and exit
    /// status are still evidence.
    pub stdin_broken_pipe: bool,
}

impl RunnerOutput {
    /// A process that ran to completion with `exit_code` and captured streams.
    ///
    /// Convenience for custom [`CommandRunner`] implementations and tests;
    /// the real runner builds its own with truncation and signal detail.
    pub fn exited(exit_code: i32, stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        Self {
            exit_code: Some(exit_code),
            signal: None,
            stdout,
            stderr,
            stdout_truncated: false,
            stderr_truncated: false,
            timed_out: false,
            stdin_broken_pipe: false,
        }
    }

    /// A process killed because its budget expired: no exit status, no output.
    pub fn killed_by_timeout() -> Self {
        Self {
            exit_code: None,
            signal: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            timed_out: true,
            stdin_broken_pipe: false,
        }
    }

    /// Whether the process exited zero.
    ///
    /// Necessary but never sufficient for success: the caller must also hold
    /// a valid succeeded receipt that matches the request it sent.
    pub fn exited_zero(&self) -> bool {
        self.exit_code == Some(0)
    }

    /// A short, safe description of how the process ended, for error text.
    pub fn termination(&self) -> String {
        if self.timed_out {
            return "timed out and was killed".to_owned();
        }
        match (self.exit_code, self.signal) {
            (Some(code), _) => format!("exited with status {code}"),
            (None, Some(sig)) => format!("terminated by signal {sig}"),
            (None, None) => "ended without an exit status".to_owned(),
        }
    }

    /// Stderr as lossy UTF-8, truncated for display. Request and receipt
    /// payloads are never logged wholesale; this is for diagnostics only.
    pub fn stderr_lossy(&self) -> String {
        const CAP: usize = 2048;
        let text = String::from_utf8_lossy(&self.stderr);
        if text.len() <= CAP {
            return text.into_owned();
        }
        let mut end = CAP;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}… (truncated)", &text[..end])
    }
}

/// Why the process could not be run or its I/O failed.
///
/// Distinct from a process that ran and reported failure — that is an
/// operation outcome, not a runner error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RunnerError {
    /// The child could not be started: missing binary, permission denied,
    /// wrong executable format.
    #[error("failed to start {program}: {source}")]
    Spawn {
        /// The program that could not be started.
        program: String,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// The child started but its I/O could not be collected.
    #[error("runner I/O failure while {context}: {source}")]
    Io {
        /// What was being attempted.
        context: &'static str,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// The child started but could not be waited on.
    #[error("failed to wait for the child process: {0}")]
    Wait(#[source] std::io::Error),
}

/// Runs a child process to completion and collects its bounded output.
pub trait CommandRunner: Send + Sync {
    /// Run `invocation`. Implementations must not use a shell, must bound
    /// both output streams, and must kill the child on timeout.
    fn run(&self, invocation: &Invocation) -> Result<RunnerOutput, RunnerError>;
}
