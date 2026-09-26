// SPDX-License-Identifier: AGPL-3.0-only

//! The client: argv, one request in, one receipt out, and the classification
//! of every way those two can disagree.
//!
//! # The exit-status / receipt-state matrix
//!
//! The controller says the process exit status is authoritative, and that a
//! failed operation exits nonzero with `state: "failed"`. Those two signals
//! are cross-checked rather than either being trusted alone:
//!
//! | Exit | Receipt | Result |
//! |---|---|---|
//! | zero | valid `succeeded` | [`ColdsnapOutcome::Success`] |
//! | nonzero | valid `failed` | [`ColdsnapOutcome::OperationFailed`] |
//! | zero | valid `failed` | protocol violation |
//! | nonzero | valid `succeeded` | protocol violation |
//! | zero | absent/invalid | protocol violation |
//! | nonzero | absent/invalid | [`ColdsnapOutcome::Unknown`] |
//! | timeout / signalled | any | [`ColdsnapOutcome::Unknown`] |
//!
//! The last two rows are the ones worth stating twice. A process that died
//! without a receipt may still have changed something, so its outcome is
//! *unknown* — not failed. And a timeout is only abortive: killing the CLI
//! does not cancel manager-side work, which is why a caller must reconcile
//! with `status` rather than retry.

use crate::config::ColdsnapConfig;
use crate::error::{ColdsnapError, ProtocolViolation, ViolationKind};
use crate::protocol::capabilities::Capabilities;
use crate::protocol::operation::Operation;
use crate::protocol::receipt::{self, Receipt, ReceiptState};
use crate::runner::{CommandRunner, Invocation, RunnerOutput};
use crate::template::PreparedRequest;

/// A controller call that ran and definitively failed.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationFailure {
    /// The failed receipt, with the controller's own error text.
    pub receipt: Receipt,
    /// The process exit code.
    pub exit_code: Option<i32>,
}

/// Why an operation's result is indeterminate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownReason {
    /// The budget expired and the child was killed. The operation may have
    /// completed, or may still be running on the manager side.
    TimedOut,
    /// The child ended without an exit status — typically a signal.
    NoExitStatus,
    /// The child exited nonzero without producing a usable receipt.
    ProcessFailedWithoutReceipt,
}

/// An operation whose side effects cannot be determined.
#[derive(Debug, Clone, PartialEq)]
pub struct UnknownOperation {
    /// Why it is indeterminate.
    pub reason: UnknownReason,
    /// The process exit code, when there was one.
    pub exit_code: Option<i32>,
    /// The terminating signal, when there was one.
    pub signal: Option<i32>,
    /// Whether the child was killed for exceeding its budget.
    pub timed_out: bool,
    /// Bounded stderr, for diagnostics. Never request or receipt payloads.
    pub stderr: String,
    /// Whether stderr hit its cap. A truncated diagnostic can hide the real
    /// reason the process died, so the caller is told rather than misled.
    pub stderr_truncated: bool,
    /// The child closed stdin before the request was fully written, so
    /// delivery is unconfirmed.
    pub stdin_broken_pipe: bool,
}

impl UnknownOperation {
    /// Whether retrying the same operation is safe.
    ///
    /// Always `false`. `sleep`, `wake` and `restore` each take a fresh id, so
    /// the controller does not deduplicate them; a retry is a second
    /// operation, not a repeat of the first. Reconcile with `status` first.
    pub const fn retry_is_safe(&self) -> bool {
        false
    }
}

/// The result of a controller operation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ColdsnapOutcome {
    /// Exit zero and a valid succeeded receipt that matches the request.
    Success(Receipt),
    /// The controller ran the operation and reported failure.
    OperationFailed(OperationFailure),
    /// The operation may or may not have happened. Reconcile, do not retry.
    Unknown(UnknownOperation),
}

impl ColdsnapOutcome {
    /// The receipt, when the controller produced one.
    pub fn receipt(&self) -> Option<&Receipt> {
        match self {
            Self::Success(receipt) => Some(receipt),
            Self::OperationFailed(failure) => Some(&failure.receipt),
            Self::Unknown(_) => None,
        }
    }

    /// Whether the controller reported success.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }
}

/// Talks to a ColdSnap controller over its CLI.
#[derive(Debug, Clone)]
pub struct ColdsnapClient<R: CommandRunner> {
    runner: R,
    config: ColdsnapConfig,
}

impl<R: CommandRunner> ColdsnapClient<R> {
    /// Build a client from an explicit runner and config.
    pub fn new(runner: R, config: ColdsnapConfig) -> Self {
        Self { runner, config }
    }

    /// The configuration in use.
    pub fn config(&self) -> &ColdsnapConfig {
        &self.config
    }

    /// The runner in use, for test assertions.
    pub fn runner(&self) -> &R {
        &self.runner
    }

    /// Read the controller's machine-readable contract.
    ///
    /// `coldsnap capabilities` takes no request and emits no receipt, so a
    /// timeout or a nonzero exit is an error rather than an outcome. Callers
    /// should run this before any operation and use
    /// [`Capabilities::admit`] to check support.
    pub fn capabilities(&self) -> Result<Capabilities, ColdsnapError> {
        let output = self
            .runner
            .run(&self.invocation("capabilities", Vec::new(), None, false))?;
        if output.stdout_truncated {
            return Err(ColdsnapError::Protocol(ProtocolViolation::new(
                ViolationKind::StdoutTooLarge,
                "the capabilities document exceeded the stdout cap",
            )));
        }
        if output.timed_out {
            return Err(ColdsnapError::ControllerUnavailable(
                "capabilities timed out".to_owned(),
            ));
        }
        if !output.exited_zero() {
            return Err(ColdsnapError::ControllerUnavailable(format!(
                "capabilities {}: {}",
                output.termination(),
                output.stderr_lossy().trim()
            )));
        }
        Capabilities::parse(&output.stdout)
    }

    /// Run a prepared operation and classify the result.
    pub fn run(&self, prepared: &PreparedRequest) -> Result<ColdsnapOutcome, ColdsnapError> {
        let output = self.runner.run(&self.invocation(
            prepared.operation.as_wire_str(),
            vec!["--request-json".to_owned(), "-".to_owned()],
            Some(prepared.bytes.clone()),
            true,
        ))?;
        self.classify(output, prepared)
    }

    /// Run a `restore --prepare-only`, which validates the contract and stages
    /// artifacts without starting a workload.
    pub fn restore_prepare_only(
        &self,
        prepared: &PreparedRequest,
    ) -> Result<ColdsnapOutcome, ColdsnapError> {
        if prepared.operation != Operation::Restore {
            return Err(ColdsnapError::unsupported(
                format!("--prepare-only for {}", prepared.operation),
                "prepare-only exists only for restore",
            ));
        }
        let output = self.runner.run(&self.invocation(
            "restore",
            vec![
                "--request-json".to_owned(),
                "-".to_owned(),
                "--prepare-only".to_owned(),
            ],
            Some(prepared.bytes.clone()),
            true,
        ))?;
        self.classify(output, prepared)
    }

    /// Run `sleep`, refusing a prepared request for any other operation.
    pub fn sleep(&self, prepared: &PreparedRequest) -> Result<ColdsnapOutcome, ColdsnapError> {
        self.run_expecting(prepared, Operation::Sleep)
    }

    /// Run `wake`, refusing a prepared request for any other operation.
    pub fn wake(&self, prepared: &PreparedRequest) -> Result<ColdsnapOutcome, ColdsnapError> {
        self.run_expecting(prepared, Operation::Wake)
    }

    /// Run `status`, refusing a prepared request for any other operation.
    pub fn status(&self, prepared: &PreparedRequest) -> Result<ColdsnapOutcome, ColdsnapError> {
        self.run_expecting(prepared, Operation::Status)
    }

    fn run_expecting(
        &self,
        prepared: &PreparedRequest,
        expected: Operation,
    ) -> Result<ColdsnapOutcome, ColdsnapError> {
        if prepared.operation != expected {
            return Err(ColdsnapError::unsupported(
                format!("{expected} with a {} request", prepared.operation),
                "the prepared request names a different operation",
            ));
        }
        self.run(prepared)
    }

    /// Build an invocation from the config.
    ///
    /// `receipt` is a separate parameter from `stdin` on purpose: the two are
    /// logically independent, and coupling them would silently drop the
    /// receipt for any future operation that takes no request body.
    fn invocation(
        &self,
        operation: &str,
        mut args: Vec<String>,
        stdin: Option<Vec<u8>>,
        receipt: bool,
    ) -> Invocation {
        let mut full = Vec::with_capacity(args.len() + 5);
        full.push(operation.to_owned());
        full.append(&mut args);
        if receipt {
            full.push("--receipt-json".to_owned());
            full.push("-".to_owned());
        }
        Invocation {
            program: self.config.binary.program().to_os_string(),
            args: full,
            stdin,
            timeout: self.config.timeout,
            max_stdout_bytes: self.config.max_stdout_bytes,
            max_stderr_bytes: self.config.max_stderr_bytes,
            environment: self.config.environment.clone(),
        }
    }

    /// Apply the classification matrix to a finished process.
    fn classify(
        &self,
        output: RunnerOutput,
        prepared: &PreparedRequest,
    ) -> Result<ColdsnapOutcome, ColdsnapError> {
        // A timeout wins over every other signal. If the process was killed
        // for exceeding its budget, nothing it printed is authoritative — so
        // reporting a stdout-cap violation here would be technically true and
        // operationally wrong.
        if output.timed_out {
            return Ok(ColdsnapOutcome::Unknown(UnknownOperation {
                reason: UnknownReason::TimedOut,
                exit_code: output.exit_code,
                signal: output.signal,
                timed_out: true,
                stderr: output.stderr_lossy(),
                stderr_truncated: output.stderr_truncated,
                stdin_broken_pipe: output.stdin_broken_pipe,
            }));
        }

        if output.stdout_truncated {
            return Err(ColdsnapError::Protocol(ProtocolViolation::new(
                ViolationKind::StdoutTooLarge,
                format!(
                    "the receipt exceeded the {} byte stdout cap and was cut short",
                    self.config.max_stdout_bytes
                ),
            )));
        }

        // Ended without an exit status — typically a signal. Indeterminate,
        // never "failed".
        if output.exit_code.is_none() {
            return Ok(ColdsnapOutcome::Unknown(UnknownOperation {
                reason: UnknownReason::NoExitStatus,
                exit_code: None,
                signal: output.signal,
                timed_out: false,
                stderr: output.stderr_lossy(),
                stderr_truncated: output.stderr_truncated,
                stdin_broken_pipe: output.stdin_broken_pipe,
            }));
        }

        let mut receipt = match receipt::parse(&output.stdout, self.config.unknown_field_policy) {
            Ok(receipt) => receipt,
            Err(error) => {
                if output.exited_zero() {
                    // Exit zero with no trustworthy receipt is a contradiction,
                    // and must never be reported as success.
                    return Err(error);
                }
                // Nonzero with no usable receipt: the process failed, but
                // without a receipt we cannot say whether it changed anything.
                return Ok(ColdsnapOutcome::Unknown(UnknownOperation {
                    reason: UnknownReason::ProcessFailedWithoutReceipt,
                    exit_code: output.exit_code,
                    signal: output.signal,
                    timed_out: false,
                    stderr: output.stderr_lossy(),
                    stderr_truncated: output.stderr_truncated,
                    stdin_broken_pipe: output.stdin_broken_pipe,
                }));
            }
        };

        // The child closing stdin early means we cannot prove the controller
        // read the whole request. That is usually benign (it may simply stop
        // reading once the JSON value is complete) but it is never silent.
        if output.stdin_broken_pipe {
            receipt.warnings.push(
                "the controller closed stdin before the request was fully written; \
                       delivery is unconfirmed"
                    .to_owned(),
            );
        }
        if output.stderr_truncated {
            receipt
                .warnings
                .push("stderr exceeded its cap and was truncated".to_owned());
        }

        // Identity cross-checks. The receipt's own `request_sha256` is not
        // comparable — the controller hashes a canonicalized, default-resolved
        // re-encoding rather than the bytes it received — so identity rests on
        // the echoed id and operation, which are exact.
        if receipt.operation_id != prepared.id {
            return Err(ColdsnapError::Protocol(ProtocolViolation::new(
                ViolationKind::RequestMismatch,
                format!(
                    "receipt is for operation id {:?}, this request is {:?}",
                    receipt.operation_id, prepared.id
                ),
            )));
        }
        if receipt.operation != prepared.operation {
            return Err(ColdsnapError::Protocol(ProtocolViolation::new(
                ViolationKind::RequestMismatch,
                format!(
                    "receipt is for a {} operation, this request is {}",
                    receipt.operation, prepared.operation
                ),
            )));
        }

        match (output.exited_zero(), receipt.state) {
            (true, ReceiptState::Succeeded) => Ok(ColdsnapOutcome::Success(receipt)),
            (false, ReceiptState::Failed) => {
                Ok(ColdsnapOutcome::OperationFailed(OperationFailure {
                    receipt,
                    exit_code: output.exit_code,
                }))
            }
            (true, ReceiptState::Failed) => Err(ColdsnapError::Protocol(ProtocolViolation::new(
                ViolationKind::ExitStateDisagreement,
                "the controller exited zero but reported a failed receipt",
            ))),
            (false, ReceiptState::Succeeded) => {
                Err(ColdsnapError::Protocol(ProtocolViolation::new(
                    ViolationKind::ExitStateDisagreement,
                    "the controller reported a succeeded receipt but exited nonzero",
                )))
            }
        }
    }
}
