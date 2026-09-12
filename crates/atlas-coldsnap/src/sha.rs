// SPDX-License-Identifier: AGPL-3.0-only

//! The request digest the controller echoes back in every receipt.
//!
//! ColdSnap computes `request_sha256` over the canonical form of the request
//! it *decoded*. This crate computes it over the exact bytes it *sent*. When
//! those agree, the receipt provably belongs to this request — which is what
//! makes a stale or crossed receipt detectable instead of merely unlikely.

use sha2::{Digest, Sha256};
use std::fmt;

/// A `sha256:<64 lowercase hex>` digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestSha256([u8; 32]);

/// Why a digest string was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid ColdSnap request digest {value:?}: {reason}")]
pub struct InvalidRequestSha256 {
    /// The offending value.
    pub value: String,
    /// A fixed explanation, safe to surface.
    pub reason: &'static str,
}

impl InvalidRequestSha256 {
    fn new(value: impl Into<String>, reason: &'static str) -> Self {
        Self {
            value: value.into(),
            reason,
        }
    }
}

impl RequestSha256 {
    /// The SHA-256 of `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Parse the wire form: the literal prefix `sha256:` followed by exactly
    /// 64 lowercase hex digits.
    pub fn parse_wire(value: &str) -> Result<Self, InvalidRequestSha256> {
        let Some(hex) = value.strip_prefix("sha256:") else {
            return Err(InvalidRequestSha256::new(
                value,
                "must start with 'sha256:'",
            ));
        };
        if hex.len() != 64 {
            return Err(InvalidRequestSha256::new(
                value,
                "must carry exactly 64 hex digits",
            ));
        }
        let mut out = [0u8; 32];
        for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
            let hi = hex_val(chunk[0])
                .ok_or_else(|| InvalidRequestSha256::new(value, "must be lowercase hex"))?;
            let lo = hex_val(chunk[1])
                .ok_or_else(|| InvalidRequestSha256::new(value, "must be lowercase hex"))?;
            out[i] = (hi << 4) | lo;
        }
        Ok(Self(out))
    }

    /// The digest as it appears on the wire.
    pub fn to_wire(self) -> String {
        let mut s = String::with_capacity(7 + 64);
        s.push_str("sha256:");
        for byte in self.0 {
            s.push(HEX[(byte >> 4) as usize] as char);
            s.push(HEX[(byte & 0x0f) as usize] as char);
        }
        s
    }

    /// The raw digest bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Uppercase is deliberately rejected: ColdSnap's `digestPattern` is
/// `^sha256:[0-9a-f]{64}$`, so accepting `A-F` here would let a value through
/// that the controller itself would never emit.
fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

impl fmt::Display for RequestSha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_wire())
    }
}

impl serde::Serialize for RequestSha256 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_wire())
    }
}

impl<'de> serde::Deserialize<'de> for RequestSha256 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse_wire(&raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The empty-string digest, as published in FIPS 180-4 / every SHA-256
    /// test vector. Anchors the implementation to a value we did not compute.
    #[test]
    fn matches_the_published_empty_input_vector() {
        let digest = RequestSha256::from_bytes(b"");
        assert_eq!(
            digest.to_wire(),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn round_trips_wire_form() {
        let digest = RequestSha256::from_bytes(b"{\"format\":4}");
        let wire = digest.to_wire();
        assert_eq!(RequestSha256::parse_wire(&wire).unwrap(), digest);
    }

    #[test]
    fn rejects_missing_prefix_wrong_length_and_uppercase() {
        let valid = RequestSha256::from_bytes(b"x").to_wire();
        let hex = valid.strip_prefix("sha256:").unwrap();

        assert!(RequestSha256::parse_wire(hex).is_err());
        assert!(RequestSha256::parse_wire(&valid[..valid.len() - 1]).is_err());
        assert!(RequestSha256::parse_wire(&valid.to_uppercase()).is_err());
        assert!(RequestSha256::parse_wire("sha256:zzzz").is_err());
    }
}
