// SPDX-License-Identifier: AGPL-3.0-only
//! Model-scoped preservation of detector-classified literal tool envelopes.
use super::LeakMarkers;

impl LeakMarkers {
    /// A model that encodes incoming tool markup as prose must also retain
    /// unparseable envelopes the detector returns as content. Envelope markers
    /// take precedence over orphan markers in the shared sanitizer. Existing
    /// envelope formats and legacy models keep their marker policy unchanged.
    pub(crate) fn for_literal_xml(mut self, preserve: bool) -> Self {
        if preserve && self.envelope_open.is_empty() && self.orphan_open.contains(&"<tool_call>") {
            self.envelope_open = &["<tool_call>"];
            self.envelope_close = &["</tool_call>"];
        }
        self
    }
}
