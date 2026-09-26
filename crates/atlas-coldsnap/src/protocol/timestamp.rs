// SPDX-License-Identifier: AGPL-3.0-only

//! RFC 3339 timestamps, as the receipt carries them.
//!
//! Kept as both the raw string and a parsed value. The raw string is what the
//! receipt actually said — useful in an error, and stable across parser
//! versions — while the parsed value is what lets `completed_at` be compared
//! against `started_at` instead of merely eyeballed.

use std::fmt;

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// A validated RFC 3339 timestamp with the raw text retained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rfc3339Nano {
    raw: String,
    parsed: OffsetDateTime,
}

/// Why a timestamp was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid RFC 3339 timestamp {value:?}: {reason}")]
pub struct InvalidTimestamp {
    /// The offending value.
    pub value: String,
    /// Why it failed.
    pub reason: String,
}

impl Rfc3339Nano {
    /// Parse `raw` as RFC 3339.
    pub fn parse(raw: &str) -> Result<Self, InvalidTimestamp> {
        let parsed = OffsetDateTime::parse(raw, &Rfc3339).map_err(|error| InvalidTimestamp {
            value: raw.to_owned(),
            reason: error.to_string(),
        })?;
        Ok(Self {
            raw: raw.to_owned(),
            parsed,
        })
    }

    /// The timestamp exactly as it appeared on the wire.
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// The parsed instant, for comparisons.
    pub fn instant(&self) -> OffsetDateTime {
        self.parsed
    }
}

impl fmt::Display for Rfc3339Nano {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

impl serde::Serialize for Rfc3339Nano {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.raw)
    }
}

impl<'de> serde::Deserialize<'de> for Rfc3339Nano {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nanosecond_precision_with_zulu() {
        let ts = Rfc3339Nano::parse("2026-08-15T12:00:00.123456789Z").unwrap();
        assert_eq!(ts.raw(), "2026-08-15T12:00:00.123456789Z");
    }

    #[test]
    fn parses_an_explicit_offset() {
        let ts = Rfc3339Nano::parse("2026-08-15T12:00:00+02:00").unwrap();
        assert_eq!(ts.raw(), "2026-08-15T12:00:00+02:00");
    }

    #[test]
    fn offsets_are_compared_as_instants_not_text() {
        // Same instant, different offsets: lexicographic comparison would
        // get this backwards.
        let earlier = Rfc3339Nano::parse("2026-08-15T12:00:00+02:00").unwrap();
        let later = Rfc3339Nano::parse("2026-08-15T11:00:01Z").unwrap();
        assert!(later.instant() > earlier.instant());
        assert!(later.raw() < earlier.raw());
    }

    #[test]
    fn rejects_a_non_rfc3339_value() {
        assert!(Rfc3339Nano::parse("2026-08-15 12:00:00").is_err());
        assert!(Rfc3339Nano::parse("not a timestamp").is_err());
        assert!(Rfc3339Nano::parse("").is_err());
    }
}
