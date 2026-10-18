// SPDX-License-Identifier: AGPL-3.0-only

//! Token ids of the qwen3_coder XML parameter markers (`<`, `=`, `>`, `</`,
//! `parameter`) for the tokenizer being served.
//!
//! `emit_step::update_tool_param_state` tracks whether generation is inside a
//! `<parameter=KEY>` VALUE, whose tokens are exempt from the 1024-token
//! envelope-stuck cap. It used to compare against Qwen3.6 BPE ids
//! (27/28/29/510/15704). Nemotron 3.5 Super VL serves the same XML tool format
//! with its own tokenizer (`<`=1060, `=`=1061, `>`=1062, `</`=1885,
//! `parameter`=31960, and `>\n` merged into one token, 1561), so the value
//! body was never recognised: every file-content token counted as envelope,
//! any argument past 1024 tokens tripped `tool_envelope_stuck`, and the
//! client got a salvaged fragment (`</parameter`) instead of the file.
//!
//! The ids are resolved from the tokenizer whenever one is loaded (startup and
//! model swap) and read here. A tokenizer that does not encode a marker as one
//! token resolves to `None`, and the guard then stays off rather than
//! misfiring: `--tool-max-tokens` still bounds a runaway envelope.

#[cfg(not(test))]
use std::sync::RwLock;

/// The marker ids for one tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolMarkers {
    pub lt: u32,
    pub eq: u32,
    /// `>` alone, plus `>\n` / `>\n\n` where the tokenizer merges them into
    /// one token (Nemotron does; Qwen does not).
    pub gt: [Option<u32>; 3],
    pub lt_slash: u32,
    pub parameter: u32,
}

impl ToolMarkers {
    pub fn is_gt(&self, tok: u32) -> bool {
        self.gt.contains(&Some(tok))
    }

    /// Resolve from an encoder (`text -> ids`, no special tokens). `None` when
    /// any required marker is not a single token.
    pub fn resolve(encode: impl Fn(&str) -> Option<Vec<u32>>) -> Option<Self> {
        let one = |s: &str| encode(s).filter(|ids| ids.len() == 1).map(|ids| ids[0]);
        Some(Self {
            lt: one("<")?,
            eq: one("=")?,
            gt: [one(">"), one(">\n"), one(">\n\n")],
            lt_slash: one("</")?,
            parameter: one("parameter")?,
        })
        .filter(|m| m.gt[0].is_some())
    }
}

/// Qwen3.6 / 3.8 byte-level BPE ids: the values the guard was hard-coded to,
/// kept as the default until a tokenizer installs its own (every serve path
/// does, in `tokenizer_runtime`), so nothing changes for a caller that never
/// loads one.
pub const QWEN: ToolMarkers = ToolMarkers {
    lt: 27,
    eq: 28,
    gt: [Some(29), None, None],
    lt_slash: 510,
    parameter: 15704,
};

#[cfg(not(test))]
static MARKERS: RwLock<Option<ToolMarkers>> = RwLock::new(Some(QWEN));

#[cfg(test)]
thread_local! {
    // Per test thread, so parallel tests installing different markers do not
    // observe each other's.
    static MARKERS: std::cell::Cell<Option<ToolMarkers>> = const { std::cell::Cell::new(Some(QWEN)) };
}

/// Install the markers of the tokenizer just loaded (`None` turns the
/// parameter-body tracking off).
pub fn set(markers: Option<ToolMarkers>) {
    #[cfg(not(test))]
    {
        *MARKERS.write().unwrap_or_else(|e| e.into_inner()) = markers;
    }
    #[cfg(test)]
    MARKERS.with(|m| m.set(markers));
}

/// The markers of the tokenizer being served.
pub fn get() -> Option<ToolMarkers> {
    #[cfg(not(test))]
    {
        *MARKERS.read().unwrap_or_else(|e| e.into_inner())
    }
    #[cfg(test)]
    MARKERS.with(|m| m.get())
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::ToolMarkers;

    /// Qwen3.6 / 3.8 byte-level BPE (verified with tokenizer.json 2026-10-18).
    pub use super::QWEN;
    /// Nemotron 3.5 Super VL (tokenizer.json of
    /// nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4, 2026-10-18).
    pub const NEMOTRON: ToolMarkers = ToolMarkers {
        lt: 1060,
        eq: 1061,
        gt: [Some(1062), Some(1561), None],
        lt_slash: 1885,
        parameter: 31960,
    };
}

#[cfg(test)]
mod tests {
    use super::fixtures::{NEMOTRON, QWEN};
    use super::*;

    fn enc(table: &'static [(&'static str, &'static [u32])]) -> impl Fn(&str) -> Option<Vec<u32>> {
        move |s| table.iter().find(|(k, _)| *k == s).map(|(_, v)| v.to_vec())
    }

    #[test]
    fn resolves_qwen_and_nemotron_markers() {
        let qwen = enc(&[
            ("<", &[27]),
            ("=", &[28]),
            (">", &[29]),
            (">\n", &[29, 198]),
            (">\n\n", &[29, 271]),
            ("</", &[510]),
            ("parameter", &[15704]),
        ]);
        assert_eq!(ToolMarkers::resolve(qwen), Some(QWEN));
        let nemo = enc(&[
            ("<", &[1060]),
            ("=", &[1061]),
            (">", &[1062]),
            (">\n", &[1561]),
            (">\n\n", &[1062, 1010, 1010]),
            ("</", &[1885]),
            ("parameter", &[31960]),
        ]);
        let m = ToolMarkers::resolve(nemo).unwrap();
        assert_eq!(m, NEMOTRON);
        assert!(m.is_gt(1062) && m.is_gt(1561) && !m.is_gt(29));
    }

    #[test]
    fn a_multi_token_marker_resolves_to_none() {
        let split = enc(&[
            ("<", &[1]),
            ("=", &[2]),
            (">", &[3]),
            ("</", &[1, 4]),
            ("parameter", &[5]),
        ]);
        assert_eq!(ToolMarkers::resolve(split), None);
    }
}

#[cfg(test)]
#[path = "tool_markers_emit_tests.rs"]
mod emit_tests;
