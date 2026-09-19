// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end tests of the client's classification matrix, driven entirely
//! through the runner seam.
//!
//! No `coldsnap` binary and no GPU are involved: a [`ScriptedRunner`] replays
//! a process result, and the assertions are about what the *client* concludes.
//! That is the point of the seam — every branch below is a real failure mode
//! of a process boundary, and all of them are reachable on a laptop.

use std::time::Duration;

use atlas_coldsnap::client::{ColdsnapOutcome, UnknownReason};
use atlas_coldsnap::config::{BinaryLocation, ColdsnapConfig, UnknownFieldPolicy};
use atlas_coldsnap::protocol::driver::SnapshotDriver;
use atlas_coldsnap::protocol::engine::ColdsnapEngine;
use atlas_coldsnap::protocol::operation::Operation;
use atlas_coldsnap::runner::fake::ScriptedRunner;
use atlas_coldsnap::runner::{RunnerError, RunnerOutput};
use atlas_coldsnap::template::{PreparedRequest, RestoreShapedTemplate};
use atlas_coldsnap::{ColdsnapClient, ColdsnapError, OperationId};

const CAPABILITIES: &str = include_str!("fixtures/capabilities.json");
const SLEEP_SUCCEEDED: &str = include_str!("fixtures/receipt_sleep_succeeded.json");
const WAKE_FAILED: &str = include_str!("fixtures/receipt_wake_failed.json");

fn config() -> ColdsnapConfig {
    ColdsnapConfig::new(
        BinaryLocation::OnPath("coldsnap".to_owned()),
        Duration::from_secs(30),
        1 << 20,
        1 << 20,
        UnknownFieldPolicy::Strict,
    )
}

fn sleep_request() -> PreparedRequest {
    template()
        .for_operation(Operation::Sleep, OperationId::parse("sleep-v1").unwrap())
        .unwrap()
}

fn wake_request() -> PreparedRequest {
    template()
        .for_operation(Operation::Wake, OperationId::parse("wake-v1").unwrap())
        .unwrap()
}

/// A restore request to derive live-control operations from.
fn template() -> RestoreShapedTemplate {
    RestoreShapedTemplate::from_bytes(
        serde_json::json!({
            "format": 4,
            "kind": "coldsnap-operation-request",
            "operation": "restore",
            "id": "qwen08b-tp1-restore-v1",
            "snapshot_driver": {"id": "n610"},
            "artifact": "./artifacts/qwen08b-tp1-published.json",
            "launch": {"engine": "vllm", "model": {"id": "m", "revision": "r"}},
            "policy": {"process": {"backend": "cuda-criu"}},
            "validation": {"health_path": "/health", "prompt": "p", "expected": "e"},
            "workload": {"cluster_id": "two-node"}
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap()
}

/// The receipt fixture, re-pointed at whichever id and operation the request
/// under test actually used.
fn receipt_for(fixture: &str, id: &str, operation: &str) -> Vec<u8> {
    let mut value: serde_json::Value = serde_json::from_str(fixture).unwrap();
    value["operation_id"] = serde_json::json!(id);
    value["operation"] = serde_json::json!(operation);
    serde_json::to_vec(&value).unwrap()
}

// ── The happy path ────────────────────────────────────────────────────────

#[test]
fn exit_zero_with_a_matching_succeeded_receipt_is_success() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        receipt_for(SLEEP_SUCCEEDED, "sleep-v1", "sleep"),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());

    let outcome = client.run(&sleep_request()).unwrap();
    let ColdsnapOutcome::Success(receipt) = outcome else {
        panic!("expected success, got {outcome:?}");
    };
    assert_eq!(receipt.operation, Operation::Sleep);
    assert_eq!(receipt.operation_id.as_str(), "sleep-v1");
    assert_eq!(receipt.snapshot_driver, SnapshotDriver::N610);
    assert_eq!(receipt.duration_seconds, 7.5);
}

#[test]
fn the_invocation_is_the_documented_shape() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        receipt_for(SLEEP_SUCCEEDED, "sleep-v1", "sleep"),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());
    client.sleep(&sleep_request()).unwrap();

    let invocation = client.runner().last_invocation().unwrap();
    assert_eq!(
        invocation.args,
        ["sleep", "--request-json", "-", "--receipt-json", "-"]
    );
    assert_eq!(invocation.program, std::ffi::OsStr::new("coldsnap"));
    // The exact bytes that were hashed are the bytes that were sent.
    assert_eq!(invocation.stdin.unwrap(), sleep_request().as_bytes());
}

#[test]
fn a_nonzero_exit_with_a_matching_failed_receipt_is_a_definitive_failure() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        1,
        receipt_for(WAKE_FAILED, "wake-v1", "wake"),
        b"adapter exited".to_vec(),
    ));
    let client = ColdsnapClient::new(runner, config());

    let outcome = client.wake(&wake_request()).unwrap();
    let ColdsnapOutcome::OperationFailed(failure) = outcome else {
        panic!("expected a definitive failure, got {outcome:?}");
    };
    assert_eq!(failure.exit_code, Some(1));
    assert!(
        failure
            .receipt
            .error
            .as_deref()
            .unwrap()
            .contains("hydration boundary")
    );
}

// ── Exit status and receipt state must agree ──────────────────────────────

#[test]
fn exit_zero_with_a_failed_receipt_is_a_protocol_violation() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        receipt_for(WAKE_FAILED, "wake-v1", "wake"),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());
    assert!(client.wake(&wake_request()).is_err());
}

#[test]
fn a_nonzero_exit_with_a_succeeded_receipt_is_a_protocol_violation() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        7,
        receipt_for(SLEEP_SUCCEEDED, "sleep-v1", "sleep"),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());
    assert!(client.run(&sleep_request()).is_err());
}

// ── Stdout hygiene ────────────────────────────────────────────────────────

#[test]
fn exit_zero_with_empty_stdout_is_a_protocol_violation() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(0, Vec::new(), Vec::new()));
    let client = ColdsnapClient::new(runner, config());
    assert!(client.run(&sleep_request()).is_err());
}

#[test]
fn exit_zero_with_partial_json_is_a_protocol_violation() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        b"{\"format\": 2, \"kind\":".to_vec(),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());
    assert!(client.run(&sleep_request()).is_err());
}

#[test]
fn exit_zero_with_trailing_json_is_a_protocol_violation() {
    let mut stdout = receipt_for(SLEEP_SUCCEEDED, "sleep-v1", "sleep");
    stdout.extend_from_slice(b"\n{\"second\": true}");
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(0, stdout, Vec::new()));
    let client = ColdsnapClient::new(runner, config());
    assert!(client.run(&sleep_request()).is_err());
}

#[test]
fn a_truncated_stdout_is_a_protocol_violation_rather_than_a_parse_attempt() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput {
        stdout_truncated: true,
        ..RunnerOutput::exited(0, b"{".to_vec(), Vec::new())
    });
    let client = ColdsnapClient::new(runner, config());
    assert!(client.run(&sleep_request()).is_err());
}

// ── Identity cross-checks ─────────────────────────────────────────────────

#[test]
fn a_receipt_for_another_operation_id_is_rejected() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        receipt_for(SLEEP_SUCCEEDED, "some-other-op-v1", "sleep"),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());
    assert!(client.run(&sleep_request()).is_err());
}

#[test]
fn a_receipt_for_another_operation_is_rejected() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        receipt_for(SLEEP_SUCCEEDED, "sleep-v1", "wake"),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());
    assert!(client.run(&sleep_request()).is_err());
}

// ── Indeterminate outcomes ────────────────────────────────────────────────

#[test]
fn a_timeout_is_unknown_not_failed() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::killed_by_timeout());
    let client = ColdsnapClient::new(runner, config());

    let outcome = client.run(&sleep_request()).unwrap();
    let ColdsnapOutcome::Unknown(unknown) = outcome else {
        panic!("a timeout must be unknown, got {outcome:?}");
    };
    assert_eq!(unknown.reason, UnknownReason::TimedOut);
    assert!(unknown.timed_out);
    // Retrying would issue a second sleep: the id is fresh every time.
    assert!(!unknown.retry_is_safe());
}

#[test]
fn a_timeout_outranks_a_stdout_cap_violation() {
    // Both are true when a runaway child is killed. The timeout is the one
    // that matters: nothing the process printed is authoritative.
    let runner = ScriptedRunner::new().then_output(RunnerOutput {
        stdout_truncated: true,
        ..RunnerOutput::killed_by_timeout()
    });
    let client = ColdsnapClient::new(runner, config());
    let outcome = client.run(&sleep_request()).unwrap();
    assert!(matches!(
        outcome,
        ColdsnapOutcome::Unknown(unknown) if unknown.reason == UnknownReason::TimedOut
    ));
}

#[test]
fn an_early_closed_stdin_is_reported_as_a_warning_on_success() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput {
        stdin_broken_pipe: true,
        ..RunnerOutput::exited(
            0,
            receipt_for(SLEEP_SUCCEEDED, "sleep-v1", "sleep"),
            Vec::new(),
        )
    });
    let client = ColdsnapClient::new(runner, config());

    let outcome = client.run(&sleep_request()).unwrap();
    let ColdsnapOutcome::Success(receipt) = outcome else {
        panic!("expected success with a warning, got {outcome:?}");
    };
    assert!(
        receipt
            .warnings
            .iter()
            .any(|w| w.contains("delivery is unconfirmed")),
        "warnings were {:?}",
        receipt.warnings
    );
}

#[test]
fn truncated_stderr_is_reported_rather_than_hidden() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput {
        stderr_truncated: true,
        ..RunnerOutput::exited(2, Vec::new(), b"partial".to_vec())
    });
    let client = ColdsnapClient::new(runner, config());

    let outcome = client.run(&sleep_request()).unwrap();
    let ColdsnapOutcome::Unknown(unknown) = outcome else {
        panic!("expected unknown, got {outcome:?}");
    };
    assert!(unknown.stderr_truncated);
}

#[test]
fn a_signalled_process_is_unknown() {
    // A signal death reports no exit code, which is what makes it
    // indeterminate rather than failed.
    let runner = ScriptedRunner::new().then_output(RunnerOutput {
        exit_code: None,
        signal: Some(9),
        ..RunnerOutput::exited(0, Vec::new(), Vec::new())
    });
    let client = ColdsnapClient::new(runner, config());

    let outcome = client.run(&sleep_request()).unwrap();
    let ColdsnapOutcome::Unknown(unknown) = outcome else {
        panic!("a signalled process must be unknown, got {outcome:?}");
    };
    assert_eq!(unknown.reason, UnknownReason::NoExitStatus);
    assert_eq!(unknown.signal, Some(9));
}

#[test]
fn a_nonzero_exit_without_a_receipt_is_unknown_rather_than_failed() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        2,
        Vec::new(),
        b"panic: nil map".to_vec(),
    ));
    let client = ColdsnapClient::new(runner, config());

    let outcome = client.run(&sleep_request()).unwrap();
    let ColdsnapOutcome::Unknown(unknown) = outcome else {
        panic!("a receiptless failure must be unknown, got {outcome:?}");
    };
    assert_eq!(unknown.reason, UnknownReason::ProcessFailedWithoutReceipt);
    assert!(unknown.stderr.contains("panic"));
}

// ── Runner failures ───────────────────────────────────────────────────────

#[test]
fn a_spawn_failure_surfaces_as_a_runner_error() {
    let runner = ScriptedRunner::new().then_error(RunnerError::Spawn {
        program: "coldsnap".to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
    });
    let client = ColdsnapClient::new(runner, config());
    let err = client.run(&sleep_request()).unwrap_err();
    assert!(matches!(err, ColdsnapError::Runner(_)));
    assert!(err.is_retryable());
}

// ── Capabilities ──────────────────────────────────────────────────────────

#[test]
fn capabilities_parses_the_contract_and_admits_a_supported_combination() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        CAPABILITIES.as_bytes().to_vec(),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());

    let capabilities = client.capabilities().unwrap();
    assert_eq!(capabilities.controller.version, "0.3.23");
    capabilities
        .admit(
            Operation::Sleep,
            ColdsnapEngine::Vllm,
            SnapshotDriver::N610,
            &["operation-receipts"],
        )
        .unwrap();
    // The capabilities command takes no request and no receipt flag.
    assert_eq!(
        client.runner().last_invocation().unwrap().args,
        ["capabilities"]
    );
}

#[test]
fn capabilities_refuses_a_combination_the_controller_lacks() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        CAPABILITIES.as_bytes().to_vec(),
        Vec::new(),
    ));
    let client = ColdsnapClient::new(runner, config());
    let capabilities = client.capabilities().unwrap();
    let err = capabilities
        .admit(
            Operation::Sleep,
            ColdsnapEngine::Vllm,
            SnapshotDriver::N610,
            &["nope"],
        )
        .unwrap_err();
    assert!(matches!(err, ColdsnapError::Unsupported { .. }));
}

#[test]
fn a_capabilities_timeout_is_an_error_not_an_outcome() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::killed_by_timeout());
    let client = ColdsnapClient::new(runner, config());
    let err = client.capabilities().unwrap_err();
    assert!(matches!(err, ColdsnapError::ControllerUnavailable(_)));
}

#[test]
fn a_capabilities_nonzero_exit_is_an_error() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        1,
        Vec::new(),
        b"coldsnap: unknown flag".to_vec(),
    ));
    let client = ColdsnapClient::new(runner, config());
    let err = client.capabilities().unwrap_err();
    assert!(matches!(err, ColdsnapError::ControllerUnavailable(_)));
    assert!(err.to_string().contains("unknown flag"));
}

// ── Operation-specific entry points ───────────────────────────────────────

#[test]
fn sleep_refuses_a_request_prepared_for_wake() {
    let runner = ScriptedRunner::new();
    let client = ColdsnapClient::new(runner, config());
    let err = client.sleep(&wake_request()).unwrap_err();
    assert!(matches!(err, ColdsnapError::Unsupported { .. }));
    // Refused before any process was spawned.
    assert!(client.runner().invocations().is_empty());
}

#[test]
fn prepare_only_is_restore_only() {
    let runner = ScriptedRunner::new();
    let client = ColdsnapClient::new(runner, config());
    assert!(client.restore_prepare_only(&sleep_request()).is_err());
    assert!(client.runner().invocations().is_empty());
}

#[test]
fn prepare_only_appends_the_flag() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(0, Vec::new(), Vec::new()));
    let client = ColdsnapClient::new(runner, config());
    let restore = template()
        .for_operation(
            Operation::Restore,
            OperationId::parse("restore-v2").unwrap(),
        )
        .unwrap();
    // Exit zero with no receipt is a protocol violation, but the invocation
    // is recorded before the classification runs.
    let _ = client.restore_prepare_only(&restore);
    assert_eq!(
        client.runner().last_invocation().unwrap().args,
        [
            "restore",
            "--request-json",
            "-",
            "--prepare-only",
            "--receipt-json",
            "-"
        ]
    );
}

#[test]
fn the_host_provider_environment_is_passed_through() {
    let runner = ScriptedRunner::new().then_output(RunnerOutput::exited(
        0,
        receipt_for(SLEEP_SUCCEEDED, "sleep-v1", "sleep"),
        Vec::new(),
    ));
    let config = config().with_environment(vec![
        (
            "COLDSNAP_HOST_PROVIDER_SOCKET".to_owned(),
            "/run/x.sock".to_owned(),
        ),
        (
            "COLDSNAP_HOST_PROVIDER_TOKEN".to_owned(),
            "secret".to_owned(),
        ),
    ]);
    let client = ColdsnapClient::new(runner, config);
    client.run(&sleep_request()).unwrap();

    let invocation = client.runner().last_invocation().unwrap();
    assert!(
        invocation
            .environment
            .iter()
            .any(|(key, value)| key == "COLDSNAP_HOST_PROVIDER_SOCKET" && value == "/run/x.sock")
    );
}
