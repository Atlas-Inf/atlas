// SPDX-License-Identifier: AGPL-3.0-only

//! ColdSnap operation identifiers.
//!
//! The controller validates every request `id` against
//! `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. Rejecting a bad id here rather than
//! letting the controller reject it is not ceremony: `sleep` / `wake` /
//! `status` each take a **fresh** id, so a malformed one is a request that
//! will fail after a process spawn and a manager-side round trip.
//!
//! The pattern is small and closed, so it is implemented directly rather than
//! pulling in a regex engine for one expression.

use std::fmt;

/// A validated ColdSnap operation id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationId(String);

/// Why an operation id was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid ColdSnap operation id {value:?}: {reason}")]
pub struct InvalidOperationId {
    /// The offending value.
    pub value: String,
    /// A fixed explanation, safe to surface.
    pub reason: &'static str,
}

impl InvalidOperationId {
    fn new(value: impl Into<String>, reason: &'static str) -> Self {
        Self {
            value: value.into(),
            reason,
        }
    }
}

/// `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, transcribed from
/// `internal/snapshot/schema.go`.
fn is_valid(value: &str) -> Result<(), &'static str> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err("must not be empty");
    };
    if !first.is_ascii_alphanumeric() {
        return Err("must start with an ASCII letter or digit");
    }
    if value.len() > 128 {
        return Err("must be at most 128 bytes");
    }
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-') {
            return Err("may contain only ASCII letters, digits, '.', '_' and '-'");
        }
    }
    Ok(())
}

impl OperationId {
    /// Validate `value` as an operation id.
    pub fn parse(value: &str) -> Result<Self, InvalidOperationId> {
        match is_valid(value) {
            Ok(()) => Ok(Self(value.to_owned())),
            Err(reason) => Err(InvalidOperationId::new(value, reason)),
        }
    }

    /// The id as it appears on the wire.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for OperationId {
    type Error = InvalidOperationId;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match is_valid(&value) {
            Ok(()) => Ok(Self(value)),
            Err(reason) => Err(InvalidOperationId::new(value, reason)),
        }
    }
}

impl TryFrom<&str> for OperationId {
    type Error = InvalidOperationId;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl fmt::Display for OperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for OperationId {
    type Err = InvalidOperationId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl serde::Serialize for OperationId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for OperationId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_documented_shape() {
        for good in [
            "a",
            "0",
            "qwen08b-tp1-sleep-v1",
            "A.B_C-9",
            &"x".repeat(128),
        ] {
            assert!(OperationId::parse(good).is_ok(), "{good:?} should parse");
        }
    }

    #[test]
    fn rejects_leading_punctuation_and_overlong_values() {
        // A leading '-', '_' or '.' would collide with CLI flag parsing on the
        // controller side, which is exactly what the pattern exists to stop.
        for bad in ["", "-x", "_x", ".x", "x y", "x/y", "é"] {
            assert!(OperationId::parse(bad).is_err(), "{bad:?} should fail");
        }
        assert!(OperationId::parse(&"x".repeat(129)).is_err());
    }

    #[test]
    fn rejects_a_multibyte_value_whose_byte_length_exceeds_the_cap() {
        // 128 chars but 256 bytes: the controller's cap is on bytes.
        let value = "é".repeat(64);
        assert!(OperationId::parse(&value).is_err());
    }

    #[test]
    fn round_trips_through_serde() {
        let id = OperationId::parse("sleep-v1").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"sleep-v1\"");
        assert_eq!(serde_json::from_str::<OperationId>(&json).unwrap(), id);
    }

    #[test]
    fn deserializing_an_invalid_id_fails_rather_than_constructing_one() {
        assert!(serde_json::from_str::<OperationId>("\"!bad\"").is_err());
    }
}
