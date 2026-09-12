// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn succeeded_receipt() -> String {
    serde_json::json!({
        "format": 2,
        "kind": "coldsnap-operation-receipt",
        "operation_id": "sleep-v1",
        "operation": "sleep",
        "state": "succeeded",
        "request_sha256": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "engine": "vllm",
        "snapshot_driver": "n610",
        "replay_semantics": "convergent",
        "started_at": "2026-08-15T12:00:00.000000000Z",
        "completed_at": "2026-08-15T12:00:07.500000000Z",
        "duration_seconds": 7.5,
        "timing": {"format": 2, "clocks": [], "spans": []},
        "result": {"artifact": "./a.json", "prepared": false}
    })
    .to_string()
}

#[test]
fn parses_a_well_formed_succeeded_receipt() {
    let receipt = parse(succeeded_receipt().as_bytes(), UnknownFieldPolicy::Strict).unwrap();
    assert!(receipt.is_success());
    assert_eq!(receipt.operation, Operation::Sleep);
    assert_eq!(receipt.engine, ObservedEngine::Vllm);
    assert_eq!(receipt.snapshot_driver, SnapshotDriver::N610);
    assert_eq!(receipt.duration_seconds, 7.5);
    assert!(receipt.warnings.is_empty());
}

#[test]
fn parses_a_failed_receipt_and_keeps_the_error() {
    let json = serde_json::json!({
        "format": 2,
        "kind": "coldsnap-operation-receipt",
        "operation_id": "wake-v2",
        "operation": "wake",
        "state": "failed",
        "request_sha256": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "engine": "sglang",
        "snapshot_driver": "n580",
        "replay_semantics": "convergent",
        "started_at": "2026-08-15T12:00:00Z",
        "completed_at": "2026-08-15T12:00:01Z",
        "duration_seconds": 1.0,
        "timing": {},
        "result": {},
        "error": "hydration boundary not reached"
    })
    .to_string();
    let receipt = parse(json.as_bytes(), UnknownFieldPolicy::Strict).unwrap();
    assert!(!receipt.is_success());
    assert_eq!(
        receipt.error.as_deref(),
        Some("hydration boundary not reached")
    );
}

#[test]
fn a_utf8_bom_is_tolerated_rather_than_reported_as_a_violation() {
    let mut bytes = b"\xef\xbb\xbf".to_vec();
    bytes.extend_from_slice(succeeded_receipt().as_bytes());
    let receipt = parse(&bytes, UnknownFieldPolicy::Strict).unwrap();
    assert!(receipt.is_success());
}

#[test]
fn empty_stdout_is_a_protocol_violation() {
    let err = parse(b"   \n", UnknownFieldPolicy::Strict).unwrap_err();
    assert!(matches!(
        err,
        ColdsnapError::Protocol(v) if v.kind == ViolationKind::EmptyStdout
    ));
}

#[test]
fn trailing_json_is_rejected() {
    let mut json = succeeded_receipt();
    json.push_str("{\"extra\":true}");
    let err = parse(json.as_bytes(), UnknownFieldPolicy::Strict).unwrap_err();
    assert!(matches!(
        err,
        ColdsnapError::Protocol(v) if v.kind == ViolationKind::TrailingJson
    ));
}

#[test]
fn two_receipts_are_rejected() {
    let json = format!("{}{}", succeeded_receipt(), succeeded_receipt());
    let err = parse(json.as_bytes(), UnknownFieldPolicy::Strict).unwrap_err();
    assert!(matches!(
        err,
        ColdsnapError::Protocol(v) if v.kind == ViolationKind::TrailingJson
    ));
}

#[test]
fn unknown_top_level_fields_are_a_violation_under_the_strict_policy() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["future_field"] = serde_json::json!(true);
    let err = parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).unwrap_err();
    assert!(matches!(
        err,
        ColdsnapError::Protocol(v) if v.kind == ViolationKind::InvalidReceiptField
    ));
}

#[test]
fn unknown_top_level_fields_warn_under_the_tolerant_policy() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["future_field"] = serde_json::json!(true);
    let receipt = parse(
        value.to_string().as_bytes(),
        UnknownFieldPolicy::IgnoreTopLevelWithWarning,
    )
    .unwrap();
    assert_eq!(receipt.warnings.len(), 1);
    assert!(receipt.warnings[0].contains("future_field"));
    assert!(receipt.is_success());
}

#[test]
fn unknown_nested_fields_remain_a_violation_even_when_tolerant() {
    // The tolerance is deliberately bounded to the top level.
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["result"]["surprise"] = serde_json::json!(1);
    assert!(
        parse(
            value.to_string().as_bytes(),
            UnknownFieldPolicy::IgnoreTopLevelWithWarning
        )
        .is_err()
    );
}

#[test]
fn a_wrong_envelope_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["format"] = serde_json::json!(1);
    let err = parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).unwrap_err();
    assert!(matches!(
        err,
        ColdsnapError::Protocol(v) if v.kind == ViolationKind::WrongEnvelope
    ));
}

#[test]
fn a_mismatched_replay_semantics_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["replay_semantics"] = serde_json::json!("reconcile-replace");
    assert!(parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).is_err());
}

#[test]
fn succeeded_with_an_error_field_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["error"] = serde_json::json!("but it worked?");
    assert!(parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).is_err());
}

#[test]
fn failed_without_an_error_field_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["state"] = serde_json::json!("failed");
    assert!(parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).is_err());
}

#[test]
fn an_unknown_state_is_never_success() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["state"] = serde_json::json!("partially-succeeded");
    assert!(parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).is_err());
}

#[test]
fn completed_before_started_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["completed_at"] = serde_json::json!("2026-08-15T11:59:59Z");
    assert!(parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).is_err());
}

#[test]
fn a_negative_duration_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["duration_seconds"] = serde_json::json!(-1.0);
    assert!(parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).is_err());
}

#[test]
fn an_unregistered_driver_is_rejected() {
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["snapshot_driver"] = serde_json::json!("n999");
    assert!(parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).is_err());
}

#[test]
fn an_unrecognised_engine_is_preserved_not_rejected() {
    // Receipts are evidence, not requests: a future engine must not make
    // an otherwise valid receipt unparseable.
    let mut value: serde_json::Value = serde_json::from_str(&succeeded_receipt()).unwrap();
    value["engine"] = serde_json::json!("atlas");
    let receipt = parse(value.to_string().as_bytes(), UnknownFieldPolicy::Strict).unwrap();
    assert_eq!(receipt.engine, ObservedEngine::Other("atlas".to_owned()));
}
