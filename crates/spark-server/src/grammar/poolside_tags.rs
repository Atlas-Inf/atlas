// SPDX-License-Identifier: AGPL-3.0-only

//! Tag names of the Poolside v1 call layout,
//! `<CALL>NAME<KEY>K</KEY><VALUE>V</VALUE></CALL>`, which GLM/Laguna emit
//! bare and Hy4 emits namespaced inside an envelope.

pub struct PoolsideTags {
    pub call: &'static str,
    pub key: &'static str,
    pub value: &'static str,
    /// The block wrapping a turn's calls (`<ENVELOPE>…</ENVELOPE>`), which a
    /// forced call emits around itself.
    pub envelope: Option<&'static str>,
}

pub const POOLSIDE_V1_TAGS: PoolsideTags = PoolsideTags {
    call: "tool_call",
    key: "arg_key",
    value: "arg_value",
    envelope: None,
};

pub const HY4_TAGS: PoolsideTags = PoolsideTags {
    call: "tool_call:opensource",
    key: "arg_key:opensource",
    value: "arg_value:opensource",
    envelope: Some("tool_calls:opensource"),
};

impl PoolsideTags {
    /// `</KEY><VALUE>`, between an argument name and its value.
    pub(super) fn key_to_value(&self) -> String {
        format!("</{}><{}>", self.key, self.value)
    }

    /// The (opener, closer) wrapping a forced call: the call tag inside the
    /// envelope when the format has one.
    pub(super) fn forced_frame(&self) -> (String, String) {
        let (open, close) = (format!("<{}>", self.call), format!("</{}>", self.call));
        match self.envelope {
            Some(envelope) => (
                format!("<{envelope}>{open}"),
                format!("{close}</{envelope}>"),
            ),
            None => (open, close),
        }
    }
}
