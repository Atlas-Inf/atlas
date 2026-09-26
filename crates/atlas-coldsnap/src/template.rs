// SPDX-License-Identifier: AGPL-3.0-only

//! Prepared requests, and the restore-shaped derivation of `sleep` / `wake` /
//! `status`.
//!
//! ColdSnap's rule is specific: to control a live workload you copy the
//! *restore* request, retain its exact `artifact`, `launch`, `validation`, and
//! `workload.cluster_id`, give it a **fresh id**, and change `operation`.
//!
//! Doing that through the typed [`OperationRequest`](crate::protocol::request::OperationRequest)
//! would be a mistake. A typed round trip can drop a field the caller sent,
//! reorder nested objects, coerce a number, or turn an absent field into a
//! present one — and every one of those changes the committed request digest
//! that the controller records and echoes back. So the derivation works on the
//! raw JSON [`serde_json::Value`] and mutates exactly two keys.

use crate::error::ColdsnapError;
use crate::id::OperationId;
use crate::protocol::constants::{MAXIMUM_REQUEST_BYTES, REQUEST_FORMAT, REQUEST_KIND};
use crate::protocol::operation::Operation;
use crate::protocol::request::OperationRequest;
use crate::sha::RequestSha256;

/// A request serialized once and ready to write verbatim.
///
/// The bytes here are the bytes that must go to the child's stdin, exactly as
/// they were hashed.
///
/// # Why [`bytes_sha256`](Self::bytes_sha256) is not compared to the receipt
///
/// It is tempting to check this against the receipt's `request_sha256`. Doing
/// so would reject every valid receipt. The controller does not hash the bytes
/// it received: `snapshot.RequestSHA256` is
/// `sha256(CanonicalJSON(json.Marshal(request)))` — a re-encoding of the
/// *decoded, default-resolved* struct with recursively sorted keys and
/// Python-style escaping (`internal/canonicaljson/canonical.go`). Two requests
/// with identical meaning can therefore hash differently, and a request that
/// omits a field hashes as the default the controller substituted.
///
/// So this digest identifies *our* bytes for logging and correlation. The
/// receipt's digest is recorded for audit; identity is established by the
/// echoed `operation_id` and `operation`, which are exactly comparable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedRequest {
    /// The operation being requested.
    pub operation: Operation,
    /// The request's fresh id.
    pub id: OperationId,
    /// The exact bytes to send.
    pub bytes: Vec<u8>,
    /// The digest of `bytes` as sent. See the type docs.
    pub bytes_sha256: RequestSha256,
}

impl PreparedRequest {
    /// Prepare an already-validated typed request.
    pub fn from_request(request: &OperationRequest) -> Result<Self, ColdsnapError> {
        request.validate()?;
        let bytes = serde_json::to_vec(request)?;
        Self::from_bytes(request.operation, request.id.clone(), bytes)
    }

    /// Prepare bytes the caller serialized, checking the size cap and
    /// computing the digest.
    pub fn from_bytes(
        operation: Operation,
        id: OperationId,
        bytes: Vec<u8>,
    ) -> Result<Self, ColdsnapError> {
        if bytes.len() > MAXIMUM_REQUEST_BYTES {
            return Err(ColdsnapError::unsupported(
                format!("a {} request of {} bytes", operation, bytes.len()),
                format!(
                    "the controller refuses requests larger than {MAXIMUM_REQUEST_BYTES} bytes"
                ),
            ));
        }

        // The receipt is correlated back by the echoed `operation` and `id`,
        // which are compared against THESE fields. If the bytes said something
        // else, that check would be vacuous — it would confirm that the
        // controller echoed what it was told, not that it ran what we meant.
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            ColdsnapError::unsupported(
                format!("a {operation} request"),
                format!("the bytes are not a JSON document: {error}"),
            )
        })?;
        let object = parsed.as_object().ok_or_else(|| {
            ColdsnapError::unsupported(
                format!("a {operation} request"),
                "the bytes are not a JSON object",
            )
        })?;
        let wire_operation = object.get("operation").and_then(serde_json::Value::as_str);
        let wire_id = object.get("id").and_then(serde_json::Value::as_str);
        if wire_operation != Some(operation.as_wire_str()) || wire_id != Some(id.as_str()) {
            return Err(ColdsnapError::unsupported(
                format!("a {operation} request with id {:?}", id.as_str()),
                format!(
                    "the serialized bytes describe operation {wire_operation:?} with id {wire_id:?}; \
                     the receipt cross-check compares against those, so they must agree"
                ),
            ));
        }

        let bytes_sha256 = RequestSha256::from_bytes(&bytes);
        Ok(Self {
            operation,
            id,
            bytes,
            bytes_sha256,
        })
    }

    /// The serialized request.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The request's length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the request is zero bytes. Always false in practice — a valid
    /// request is at least a JSON object — but clippy asks for it alongside
    /// `len`.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// A restore request retained as JSON, for deriving live-control operations.
#[derive(Debug, Clone, PartialEq)]
pub struct RestoreShapedTemplate {
    value: serde_json::Value,
}

impl RestoreShapedTemplate {
    /// Adopt an existing restore request.
    ///
    /// Validated once here, then reused mechanically — so a malformed template
    /// fails before a controller is ever spawned, and cannot fail differently
    /// on the second use.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ColdsnapError> {
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
            ColdsnapError::unsupported(
                "a restore template",
                format!("the document is not valid JSON: {error}"),
            )
        })?;
        let object = value.as_object().ok_or_else(|| {
            ColdsnapError::unsupported("a restore template", "the document is not a JSON object")
        })?;

        let format = object.get("format").and_then(serde_json::Value::as_u64);
        let kind = object.get("kind").and_then(serde_json::Value::as_str);
        if format != Some(u64::from(REQUEST_FORMAT)) || kind != Some(REQUEST_KIND) {
            return Err(ColdsnapError::unsupported(
                "a restore template",
                format!("envelope is format {format:?} kind {kind:?}"),
            ));
        }

        let operation = object
            .get("operation")
            .and_then(serde_json::Value::as_str)
            .and_then(Operation::from_wire)
            .ok_or_else(|| {
                ColdsnapError::unsupported(
                    "a restore template",
                    "it does not name a known operation",
                )
            })?;
        if !is_restore_shaped(operation) {
            return Err(ColdsnapError::unsupported(
                format!("a {operation} request as a live-control template"),
                "only restore-shaped requests (restore/sleep/wake/status) carry the \
                 artifact, launch, validation and workload identity that live control needs",
            ));
        }

        if object
            .get("artifact")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(ColdsnapError::unsupported(
                "a restore template",
                "it carries no artifact, so there is nothing to sleep, wake or inspect",
            ));
        }
        for required in ["launch", "validation"] {
            if !object.contains_key(required) {
                return Err(ColdsnapError::unsupported(
                    "a restore template",
                    format!("it is missing the {required:?} object"),
                ));
            }
        }

        Ok(Self { value })
    }

    /// Adopt a typed restore request.
    pub fn from_request(request: &OperationRequest) -> Result<Self, ColdsnapError> {
        request.validate()?;
        Self::from_bytes(&serde_json::to_vec(request)?)
    }

    /// The template's JSON, for inspection.
    pub fn value(&self) -> &serde_json::Value {
        &self.value
    }

    /// The id currently in the template.
    pub fn current_id(&self) -> Option<&str> {
        self.value.get("id").and_then(serde_json::Value::as_str)
    }

    /// Derive a live-control request, changing exactly `operation` and `id`.
    ///
    /// The template itself is never mutated: deriving a `wake` from the same
    /// template that produced a `sleep` must not inherit the sleep's id.
    pub fn for_operation(
        &self,
        operation: Operation,
        id: OperationId,
    ) -> Result<PreparedRequest, ColdsnapError> {
        if !is_restore_shaped(operation) {
            return Err(ColdsnapError::unsupported(
                format!("{operation} from a restore template"),
                "only restore/sleep/wake/status are derived this way; capture and \
                 publish are their own requests",
            ));
        }
        // The controller requires a fresh id per operation, and `sleep`/`wake`
        // are not deduplicated by id — so reusing the template's id would
        // produce a second operation that looks like the first.
        if self.current_id() == Some(id.as_str()) {
            return Err(ColdsnapError::unsupported(
                format!("a {operation} request reusing id {:?}", id.as_str()),
                "live-control operations need a fresh id; the template's own id \
                 belongs to the restore it came from",
            ));
        }

        let mut derived = self.value.clone();
        let object = derived
            .as_object_mut()
            .expect("validated as an object at construction");
        object.insert(
            "operation".to_owned(),
            serde_json::Value::String(operation.as_wire_str().to_owned()),
        );
        object.insert(
            "id".to_owned(),
            serde_json::Value::String(id.as_str().to_owned()),
        );

        let bytes = serde_json::to_vec(&derived)?;
        PreparedRequest::from_bytes(operation, id, bytes)
    }
}

/// Whether an operation is derived from a restore request.
fn is_restore_shaped(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::Restore | Operation::Sleep | Operation::Wake | Operation::Status
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template_json() -> serde_json::Value {
        serde_json::json!({
            "format": 4,
            "kind": "coldsnap-operation-request",
            "operation": "restore",
            "id": "qwen08b-tp1-restore-v1",
            "snapshot_driver": {"id": "n610"},
            "artifact": "./artifacts/qwen08b-tp1-published.json",
            "launch": {"engine": "vllm", "model": {"id": "m", "revision": "r"}},
            "policy": {"process": {"backend": "cuda-criu", "kv_discard": true}},
            "validation": {"health_path": "/health", "prompt": "p", "expected": "e"},
            "workload": {"cluster_id": "two-node"},
            "an_unknown_future_field": {"keep": "me"}
        })
    }

    fn template() -> RestoreShapedTemplate {
        RestoreShapedTemplate::from_bytes(template_json().to_string().as_bytes()).unwrap()
    }

    fn id(value: &str) -> OperationId {
        OperationId::parse(value).unwrap()
    }

    #[test]
    fn sleep_changes_only_operation_and_id() {
        let template = template();
        let prepared = template
            .for_operation(Operation::Sleep, id("qwen08b-tp1-sleep-v1"))
            .unwrap();

        let derived: serde_json::Value = serde_json::from_slice(prepared.as_bytes()).unwrap();
        let mut expected = template_json();
        expected["operation"] = serde_json::json!("sleep");
        expected["id"] = serde_json::json!("qwen08b-tp1-sleep-v1");
        assert_eq!(derived, expected);
    }

    #[test]
    fn retained_fields_survive_verbatim() {
        let prepared = template()
            .for_operation(Operation::Wake, id("wake-v1"))
            .unwrap();
        let derived: serde_json::Value = serde_json::from_slice(prepared.as_bytes()).unwrap();

        assert_eq!(
            derived["artifact"],
            serde_json::json!("./artifacts/qwen08b-tp1-published.json")
        );
        assert_eq!(derived["launch"], template_json()["launch"]);
        assert_eq!(derived["validation"], template_json()["validation"]);
        assert_eq!(
            derived["workload"]["cluster_id"],
            serde_json::json!("two-node")
        );
        assert_eq!(derived["format"], serde_json::json!(4));
        assert_eq!(
            derived["kind"],
            serde_json::json!("coldsnap-operation-request")
        );
        // An unknown field is preserved rather than dropped by a typed model.
        assert_eq!(
            derived["an_unknown_future_field"],
            serde_json::json!({"keep": "me"})
        );
    }

    #[test]
    fn the_template_is_not_mutated_by_deriving() {
        let template = template();
        let _ = template
            .for_operation(Operation::Sleep, id("sleep-v1"))
            .unwrap();
        assert_eq!(template.current_id(), Some("qwen08b-tp1-restore-v1"));
        let again = template
            .for_operation(Operation::Wake, id("wake-v1"))
            .unwrap();
        let derived: serde_json::Value = serde_json::from_slice(again.as_bytes()).unwrap();
        assert_eq!(derived["operation"], serde_json::json!("wake"));
    }

    #[test]
    fn the_digest_is_over_the_derived_bytes() {
        let prepared = template()
            .for_operation(Operation::Status, id("status-v1"))
            .unwrap();
        assert_eq!(
            prepared.bytes_sha256,
            RequestSha256::from_bytes(prepared.as_bytes())
        );
        // And differs from the template's own digest.
        assert_ne!(
            prepared.bytes_sha256,
            RequestSha256::from_bytes(template_json().to_string().as_bytes())
        );
    }

    #[test]
    fn reusing_the_templates_id_is_refused() {
        let err = template()
            .for_operation(Operation::Sleep, id("qwen08b-tp1-restore-v1"))
            .unwrap_err();
        assert!(matches!(err, ColdsnapError::Unsupported { .. }));
    }

    #[test]
    fn capture_and_publish_are_not_derivable() {
        for operation in [
            Operation::Capture,
            Operation::Publish,
            Operation::PublishNative,
        ] {
            assert!(
                template().for_operation(operation, id("x-v1")).is_err(),
                "{operation} should not be derivable"
            );
        }
    }

    #[test]
    fn a_template_without_an_artifact_is_refused() {
        let mut value = template_json();
        value.as_object_mut().unwrap().remove("artifact");
        assert!(RestoreShapedTemplate::from_bytes(value.to_string().as_bytes()).is_err());
    }

    #[test]
    fn a_non_restore_shaped_template_is_refused() {
        let mut value = template_json();
        value["operation"] = serde_json::json!("capture");
        assert!(RestoreShapedTemplate::from_bytes(value.to_string().as_bytes()).is_err());
    }

    #[test]
    fn a_template_with_a_wrong_envelope_is_refused() {
        let mut value = template_json();
        value["format"] = serde_json::json!(3);
        assert!(RestoreShapedTemplate::from_bytes(value.to_string().as_bytes()).is_err());
    }

    #[test]
    fn bytes_that_disagree_with_the_metadata_are_refused() {
        // The receipt cross-check compares the echoed operation/id against the
        // PreparedRequest's fields. If the bytes said something else, that
        // check would confirm nothing.
        let bytes = serde_json::to_vec(&serde_json::json!({
            "format": 4,
            "kind": "coldsnap-operation-request",
            "operation": "sleep",
            "id": "sleep-v1",
            "artifact": "./a.json"
        }))
        .unwrap();

        // Same id, different operation.
        let err = PreparedRequest::from_bytes(Operation::Wake, id("sleep-v1"), bytes.clone())
            .unwrap_err();
        assert!(matches!(err, ColdsnapError::Unsupported { .. }));

        // Same operation, different id.
        assert!(PreparedRequest::from_bytes(Operation::Sleep, id("wake-v1"), bytes).is_err());
    }

    #[test]
    fn non_json_bytes_are_refused() {
        let err =
            PreparedRequest::from_bytes(Operation::Sleep, id("sleep-v1"), b"not json".to_vec())
                .unwrap_err();
        assert!(matches!(err, ColdsnapError::Unsupported { .. }));
    }

    #[test]
    fn an_oversized_request_is_refused_before_any_spawn() {
        let prepared = PreparedRequest::from_bytes(
            Operation::Sleep,
            id("big-v1"),
            vec![b'x'; MAXIMUM_REQUEST_BYTES + 1],
        );
        assert!(matches!(prepared, Err(ColdsnapError::Unsupported { .. })));
    }

    #[test]
    fn a_typed_restore_request_can_seed_a_template() {
        let typed =
            crate::protocol::request::fixtures::request(Operation::Restore, Some("./a.json"));
        let template = RestoreShapedTemplate::from_request(&typed).unwrap();
        assert_eq!(template.current_id(), Some("op-v1"));

        // And a live-control request derived from it carries the typed
        // request's launch topology through unchanged.
        let prepared = template
            .for_operation(Operation::Sleep, id("sleep-v9"))
            .unwrap();
        let derived: serde_json::Value = serde_json::from_slice(prepared.as_bytes()).unwrap();
        assert_eq!(
            derived["launch"],
            serde_json::to_value(&typed.launch).unwrap()
        );
    }
}
