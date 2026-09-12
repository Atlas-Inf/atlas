// SPDX-License-Identifier: AGPL-3.0-only

//! The seven operations the controller accepts, and what each one's replay
//! semantics are.
//!
//! Replay semantics are not decoration: they are the controller's own answer
//! to "what happens if I run this twice?", and they are what decides whether a
//! caller may retry. `sleep` / `wake` / `status` are **convergent** — running
//! one twice is meant to land in the same state — while `capture` **conflicts**
//! with an existing output. Getting this wrong is how a retry silently
//! produces a second artifact or a second sleep.

use std::fmt;

/// A ColdSnap operation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operation {
    /// Build a snapshot artifact from a running workload.
    Capture,
    /// Make a capture artifact portable.
    Publish,
    /// Publish a native model payload.
    PublishNative,
    /// Activate a workload from an artifact.
    Restore,
    /// Release managed weights and graphs, retaining the process.
    Sleep,
    /// Hydrate a slept workload.
    Wake,
    /// Report node-local lifecycle state.
    Status,
}

impl Operation {
    /// Every operation, in the controller's own registration order.
    pub const ALL: [Self; 7] = [
        Self::Capture,
        Self::Publish,
        Self::PublishNative,
        Self::Restore,
        Self::Sleep,
        Self::Wake,
        Self::Status,
    ];

    /// The operation as it appears on the wire and on the command line.
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Capture => "capture",
            Self::Publish => "publish",
            Self::PublishNative => "publish-native",
            Self::Restore => "restore",
            Self::Sleep => "sleep",
            Self::Wake => "wake",
            Self::Status => "status",
        }
    }

    /// Parse a wire operation. `None` for anything the controller would reject.
    pub fn from_wire(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|operation| operation.as_wire_str() == value)
    }

    /// Whether the operation changes a workload's state.
    ///
    /// `status` is the only read-only operation, which is why it is the one a
    /// caller may use to reconcile after an unknown outcome.
    pub const fn is_mutating(self) -> bool {
        !matches!(self, Self::Status)
    }

    /// Whether the operation requires an `artifact` field.
    pub const fn requires_artifact(self) -> bool {
        matches!(
            self,
            Self::Publish
                | Self::PublishNative
                | Self::Restore
                | Self::Sleep
                | Self::Wake
                | Self::Status
        )
    }

    /// Whether the operation requires an `output` field.
    pub const fn requires_output(self) -> bool {
        matches!(self, Self::Capture | Self::Publish | Self::PublishNative)
    }

    /// The controller's `ReplaySemantics` for this operation.
    ///
    /// Mirrors `snapshot.ReplaySemantics`. `prepare_only` only applies to
    /// `restore`, where it makes the call a safe repeat.
    pub const fn replay_semantics(self, prepare_only: bool) -> &'static str {
        if prepare_only {
            return "safe-repeat";
        }
        match self {
            Self::Capture => "conflict-on-existing-output",
            Self::Publish | Self::PublishNative => "content-idempotent",
            Self::Restore => "reconcile-replace",
            Self::Sleep | Self::Wake | Self::Status => "convergent",
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire_str())
    }
}

impl serde::Serialize for Operation {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_wire_str())
    }
}

impl<'de> serde::Deserialize<'de> for Operation {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::from_wire(&raw).ok_or_else(|| {
            serde::de::Error::custom(format!("unsupported ColdSnap operation {raw:?}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_names_round_trip_for_every_operation() {
        for operation in Operation::ALL {
            assert_eq!(
                Operation::from_wire(operation.as_wire_str()),
                Some(operation)
            );
        }
    }

    #[test]
    fn an_unknown_operation_is_rejected_on_the_wire() {
        assert_eq!(Operation::from_wire("hibernate"), None);
        assert!(serde_json::from_str::<Operation>("\"hibernate\"").is_err());
    }

    #[test]
    fn replay_semantics_match_the_controller() {
        // Copied from snapshot.ReplaySemantics; a drift here would let a
        // caller retry an operation the controller considers conflicting.
        assert_eq!(
            Operation::Capture.replay_semantics(false),
            "conflict-on-existing-output"
        );
        assert_eq!(
            Operation::Publish.replay_semantics(false),
            "content-idempotent"
        );
        assert_eq!(
            Operation::PublishNative.replay_semantics(false),
            "content-idempotent"
        );
        assert_eq!(
            Operation::Restore.replay_semantics(false),
            "reconcile-replace"
        );
        for convergent in [Operation::Sleep, Operation::Wake, Operation::Status] {
            assert_eq!(convergent.replay_semantics(false), "convergent");
        }
        assert_eq!(Operation::Restore.replay_semantics(true), "safe-repeat");
        assert_eq!(Operation::Sleep.replay_semantics(true), "safe-repeat");
    }

    #[test]
    fn only_status_is_read_only() {
        for operation in Operation::ALL {
            assert_eq!(operation.is_mutating(), operation != Operation::Status);
        }
    }

    #[test]
    fn artifact_and_output_requirements_match_the_validator() {
        assert!(Operation::Sleep.requires_artifact());
        assert!(Operation::Wake.requires_artifact());
        assert!(Operation::Status.requires_artifact());
        assert!(!Operation::Capture.requires_artifact());
        assert!(Operation::Capture.requires_output());
        assert!(!Operation::Sleep.requires_output());
    }
}
