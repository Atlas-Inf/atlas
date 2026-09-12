// SPDX-License-Identifier: AGPL-3.0-only

//! Snapshot drivers.
//!
//! Closed, because `snapshotdriver.Lookup` is closed: core ColdSnap
//! deliberately has no `auto` value, so an orchestrator must resolve its
//! hardware policy before it constructs a request. Modelling that as an enum
//! makes the resolution a compile-time obligation rather than a runtime hope.

use std::fmt;

/// A snapshot implementation the controller knows.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SnapshotDriver {
    /// Pre-CUDA process template, then fresh CUDA reconstruction.
    /// Requires host driver 580+.
    N580,
    /// Initialized-CUDA process state with CUDA checkpoint restoration.
    /// Requires host driver 610+.
    N610,
}

impl SnapshotDriver {
    /// Both drivers, in the registry's sorted order.
    pub const ALL: [Self; 2] = [Self::N580, Self::N610];

    /// The driver id as it appears on the wire.
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::N580 => "n580",
            Self::N610 => "n610",
        }
    }

    /// The minimum NVIDIA host driver major version this driver needs.
    pub const fn minimum_nvidia_driver_major(self) -> u32 {
        match self {
            Self::N580 => 580,
            Self::N610 => 610,
        }
    }

    /// Parse a wire driver id. `None` for anything unregistered.
    pub fn from_wire(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|driver| driver.as_wire_str() == value)
    }

    /// The `{"id": …}` selection object a request carries.
    pub fn selection(self) -> SnapshotDriverSelection {
        SnapshotDriverSelection { id: self }
    }
}

impl fmt::Display for SnapshotDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire_str())
    }
}

impl serde::Serialize for SnapshotDriver {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_wire_str())
    }
}

impl<'de> serde::Deserialize<'de> for SnapshotDriver {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::from_wire(&raw)
            .ok_or_else(|| serde::de::Error::custom(format!("unsupported snapshot driver {raw:?}")))
    }
}

/// The request's `snapshot_driver` object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotDriverSelection {
    /// The selected driver.
    pub id: SnapshotDriver,
}

impl From<SnapshotDriver> for SnapshotDriverSelection {
    fn from(id: SnapshotDriver) -> Self {
        Self { id }
    }
}

impl serde::Serialize for SnapshotDriverSelection {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let mut state = serializer.serialize_struct("SnapshotDriverSelection", 1)?;
        state.serialize_field("id", &self.id)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for SnapshotDriverSelection {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            id: SnapshotDriver,
        }
        let wire = Wire::deserialize(deserializer)?;
        Ok(Self { id: wire.id })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drivers_round_trip_and_carry_their_driver_floor() {
        assert_eq!(SnapshotDriver::N580.minimum_nvidia_driver_major(), 580);
        assert_eq!(SnapshotDriver::N610.minimum_nvidia_driver_major(), 610);
        for driver in SnapshotDriver::ALL {
            assert_eq!(
                SnapshotDriver::from_wire(driver.as_wire_str()),
                Some(driver)
            );
        }
    }

    #[test]
    fn auto_is_not_a_value() {
        // Core ColdSnap has no "auto"; an orchestrator must decide.
        assert_eq!(SnapshotDriver::from_wire("auto"), None);
    }

    #[test]
    fn selection_serializes_as_an_id_object() {
        let json = serde_json::to_string(&SnapshotDriver::N610.selection()).unwrap();
        assert_eq!(json, r#"{"id":"n610"}"#);
    }
}
