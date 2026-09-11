// SPDX-License-Identifier: AGPL-3.0-only

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GraphFaultPoint {
    Capture,
    Instantiate,
    Replay,
    Event,
}

pub(super) struct GraphFaults {
    capture: bool,
    instantiate: bool,
    replay: bool,
    event: bool,
}

impl GraphFaults {
    pub fn from_env() -> Self {
        Self::parse(&std::env::var("ATLAS_GRAPH_FAULT").unwrap_or_default())
    }

    fn parse(value: &str) -> Self {
        let has = |name| value.split(',').any(|item| item.trim() == name);
        Self {
            capture: has("capture"),
            instantiate: has("instantiate"),
            replay: has("replay"),
            event: has("event"),
        }
    }

    pub fn enabled(&self, point: GraphFaultPoint) -> bool {
        match point {
            GraphFaultPoint::Capture => self.capture,
            GraphFaultPoint::Instantiate => self.instantiate,
            GraphFaultPoint::Replay => self.replay,
            GraphFaultPoint::Event => self.event,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphFaultPoint, GraphFaults};

    #[test]
    fn parser_enables_only_named_graph_faults() {
        let faults = GraphFaults::parse("capture,replay");
        assert!(faults.enabled(GraphFaultPoint::Capture));
        assert!(faults.enabled(GraphFaultPoint::Replay));
        assert!(!faults.enabled(GraphFaultPoint::Instantiate));
        assert!(!faults.enabled(GraphFaultPoint::Event));
    }
}
