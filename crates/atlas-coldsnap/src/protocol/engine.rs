// SPDX-License-Identifier: AGPL-3.0-only

//! Engines, and the gap this crate refuses to paper over.
//!
//! Two types, because the two directions have different obligations.
//!
//! [`ColdsnapEngine`] is **closed**: it is what this crate will put in a
//! request, and ColdSnap's `engineadapter.Lookup` admits exactly these two
//! values. An open enum here would make it easy to emit a request the
//! controller is guaranteed to reject, and would obscure the fact that Atlas
//! has no adapter upstream.
//!
//! [`ObservedEngine`] is **tolerant**: it is what arrived in a receipt or a
//! capabilities document. A newer controller may name an engine this crate
//! does not know, and reporting that faithfully is better than failing to
//! parse the document at all. It is never used to build a request.

use std::fmt;

/// An engine ColdSnap has a registered adapter for.
///
/// If you are looking for `Atlas`: it is deliberately absent. ColdSnap's
/// `engineadapter` registry contains `vllm` and `sglang` only, and receipt
/// validation rejects anything else, so an `atlas` engine request would fail
/// after a spawn and a manager-side round trip. See the crate docs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColdsnapEngine {
    /// vLLM.
    Vllm,
    /// SGLang.
    Sglang,
}

impl ColdsnapEngine {
    /// Both engines, in the registry's sorted order.
    pub const ALL: [Self; 2] = [Self::Sglang, Self::Vllm];

    /// The engine as it appears on the wire.
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }

    /// Parse a wire engine, rejecting anything not registered upstream.
    pub fn from_wire(value: &str) -> Result<Self, UnsupportedEngine> {
        match value {
            "vllm" => Ok(Self::Vllm),
            "sglang" => Ok(Self::Sglang),
            other => Err(UnsupportedEngine(other.to_owned())),
        }
    }
}

/// An engine ColdSnap has no adapter for, so no request may name it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("ColdSnap has no engine adapter for {0:?}; its registry admits only 'vllm' and 'sglang'")]
pub struct UnsupportedEngine(pub String);

impl fmt::Display for ColdsnapEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire_str())
    }
}

impl TryFrom<&str> for ColdsnapEngine {
    type Error = UnsupportedEngine;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::from_wire(value)
    }
}

impl serde::Serialize for ColdsnapEngine {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_wire_str())
    }
}

impl<'de> serde::Deserialize<'de> for ColdsnapEngine {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::from_wire(&raw).map_err(serde::de::Error::custom)
    }
}

/// An engine name as *observed* in wire data. Never used to build a request.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObservedEngine {
    /// vLLM.
    Vllm,
    /// SGLang.
    Sglang,
    /// Something this crate does not know. Preserved verbatim so a caller can
    /// log it rather than silently discarding evidence.
    Other(String),
}

impl ObservedEngine {
    /// Classify a wire engine name without rejecting the unknown.
    pub fn from_wire(value: &str) -> Self {
        match value {
            "vllm" => Self::Vllm,
            "sglang" => Self::Sglang,
            other => Self::Other(other.to_owned()),
        }
    }

    /// The wire form, whether or not this crate recognises it.
    pub fn as_wire_str(&self) -> &str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
            Self::Other(name) => name,
        }
    }

    /// Whether this crate can build a request naming this engine.
    pub fn as_supported(&self) -> Option<ColdsnapEngine> {
        match self {
            Self::Vllm => Some(ColdsnapEngine::Vllm),
            Self::Sglang => Some(ColdsnapEngine::Sglang),
            Self::Other(_) => None,
        }
    }
}

impl fmt::Display for ObservedEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire_str())
    }
}

impl serde::Serialize for ObservedEngine {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_wire_str())
    }
}

impl<'de> serde::Deserialize<'de> for ObservedEngine {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::from_wire(&String::deserialize(deserializer)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_engines_round_trip() {
        for engine in ColdsnapEngine::ALL {
            assert_eq!(ColdsnapEngine::from_wire(engine.as_wire_str()), Ok(engine));
        }
    }

    #[test]
    fn atlas_is_refused_rather_than_accepted_as_an_open_string() {
        let err = ColdsnapEngine::from_wire("atlas").unwrap_err();
        assert_eq!(err.0, "atlas");
        assert!(err.to_string().contains("vllm"));
    }

    #[test]
    fn observed_engines_preserve_the_unknown() {
        let observed = ObservedEngine::from_wire("atlas");
        assert_eq!(observed, ObservedEngine::Other("atlas".to_owned()));
        assert_eq!(observed.as_wire_str(), "atlas");
        assert_eq!(observed.as_supported(), None);
    }

    #[test]
    fn an_observed_known_engine_can_be_promoted() {
        assert_eq!(
            ObservedEngine::from_wire("vllm").as_supported(),
            Some(ColdsnapEngine::Vllm)
        );
    }

    #[test]
    fn serde_refuses_an_unknown_engine_for_request_side_types() {
        assert!(serde_json::from_str::<ColdsnapEngine>("\"atlas\"").is_err());
        assert!(serde_json::from_str::<ObservedEngine>("\"atlas\"").is_ok());
    }
}
