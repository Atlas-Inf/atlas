// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Byte offset where the streaming sanitizer must start holding: the
/// earliest position `p` in the last `tag_max - 1` bytes such that
/// `buf[p..]` is a PROPER prefix of some marker (i.e. more bytes could
/// complete it). Returns `buf.len()` when no such suffix exists, so the
/// whole buffer is emitted. Always a UTF-8 char boundary.
pub(crate) fn marker_prefix_hold_start(
    buf: &str,
    tag_max: usize,
    markers: &tool_parser::LeakMarkers,
) -> usize {
    let window_start = buf.floor_char_boundary(buf.len().saturating_sub(tag_max.saturating_sub(1)));
    for (off, _) in buf[window_start..].char_indices() {
        let p = window_start + off;
        let tail = &buf[p..];
        let grows = markers
            .orphan_open
            .iter()
            .chain(markers.close.iter())
            .chain(markers.envelope_open.iter())
            .chain(markers.envelope_close.iter())
            .any(|m| m.len() > tail.len() && m.starts_with(tail));
        if grows {
            return p;
        }
    }
    buf.len()
}

#[cfg(test)]
mod tests {
    use super::super::{ToolKind, classify_tool, extract_bash_final_action, primary_arg_for_tool};

    #[test]
    fn bash_final_action_returns_last_segment() {
        let out =
            extract_bash_final_action("mkdir -p /tmp/x/src && cd /tmp/x && cargo init --name a");
        assert!(out.starts_with("cargo init"), "got: {out}");
    }

    #[test]
    fn bash_final_action_no_chain_returns_original() {
        let out = extract_bash_final_action("ls -la /tmp/x");
        assert!(out.starts_with("ls -la"));
    }

    #[test]
    fn bash_final_action_empty_returns_empty() {
        assert_eq!(extract_bash_final_action(""), "");
    }

    #[test]
    fn classify_tool_case_insensitive() {
        assert_eq!(classify_tool("Bash"), ToolKind::Bash);
        assert_eq!(classify_tool("bash"), ToolKind::Bash);
        assert_eq!(classify_tool("BASH"), ToolKind::Bash);
        assert_eq!(classify_tool("Write"), ToolKind::Write);
        assert_eq!(classify_tool("Edit"), ToolKind::Edit);
        assert_eq!(classify_tool("Read"), ToolKind::Read);
        assert_eq!(classify_tool("MultiEdit"), ToolKind::MultiEdit);
        assert_eq!(classify_tool("multiedit"), ToolKind::MultiEdit);
    }

    #[test]
    fn classify_tool_unknown_is_other() {
        assert_eq!(classify_tool("GetWeather"), ToolKind::Other);
        assert_eq!(classify_tool(""), ToolKind::Other);
        assert_eq!(classify_tool("Bashly"), ToolKind::Other);
    }

    #[test]
    fn primary_arg_write_snake_and_camel() {
        let out = primary_arg_for_tool("Write", r#"{"file_path":"/tmp/x.rs"}"#);
        assert_eq!(out.as_deref(), Some("/tmp/x.rs"));
        let out = primary_arg_for_tool("write", r#"{"filePath":"/tmp/y.rs"}"#);
        assert_eq!(out.as_deref(), Some("/tmp/y.rs"));
    }

    #[test]
    fn primary_arg_bash_collapses_chain() {
        let out = primary_arg_for_tool("Bash", r#"{"command":"cd /tmp && cargo build"}"#);
        assert!(out.as_ref().is_some_and(|s| s.starts_with("cargo build")));
    }

    #[test]
    fn primary_arg_unknown_tool_falls_back() {
        // ToolKind::Other has no well-known key but fallback path may
        // return the first non-empty string field.
        let out = primary_arg_for_tool("GetWeather", r#"{"location":"Paris"}"#);
        assert!(
            out.is_some(),
            "fallback path should return some(location=Paris)"
        );
    }

    #[test]
    fn primary_arg_malformed_json_returns_none() {
        assert_eq!(primary_arg_for_tool("Write", "not json"), None);
    }

    #[test]
    fn primary_arg_missing_key_returns_none() {
        let out = primary_arg_for_tool("Write", r#"{"content":"fn main(){}"}"#);
        assert_eq!(out, None);
    }
}
