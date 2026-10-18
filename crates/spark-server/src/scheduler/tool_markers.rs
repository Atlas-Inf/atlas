// SPDX-License-Identifier: AGPL-3.0-only

//! Which tokens of the served tokenizer can make up the qwen3_coder XML
//! parameter markers (`<parameter=KEY>`, `</parameter>`).
//!
//! `emit_step::update_tool_param_state` tracks whether generation is inside a
//! `<parameter=KEY>` VALUE, whose tokens are exempt from the 1024-token
//! envelope-stuck cap. It used to compare against five exact Qwen3.6 BPE ids
//! (`<` 27, `=` 28, `>` 29, `</` 510, `parameter` 15704). That failed twice:
//!
//! * Nemotron 3.5 Super VL serves the same XML format with its own tokenizer
//!   (`<` 1060, `=` 1061, `>` 1062, `</` 1885, `parameter` 31960), so a value
//!   was never recognised and any argument past 1024 tokens was cut
//!   (`tool_envelope_stuck`; the client saw a salvaged `</parameter`).
//! * Even on Qwen, BPE merges the `=` into the key's first sub-word for common
//!   keys (`=file` `_path`, `=re` `placements`, `=new` `_str`), so those
//!   values were never recognised either. Nemotron also merges `>` with the
//!   newline (`>\n` 1561), and both merge punctuation into the close (`.</`).
//!
//! So the markers are token CLASSES, built once per tokenizer from the decoded
//! text of every vocabulary entry: an opening `<` (leading whitespace allowed),
//! `parameter`, a token starting with `=`, one starting with `>`, and one
//! ending with `</`. They are installed whenever a tokenizer is loaded
//! (`tokenizer_runtime`, startup and model swap) and read by the guard. A
//! tokenizer missing any class resolves to `None`, and the tracking and cap
//! stay off rather than misfire (`--tool-max-tokens` still bounds a runaway
//! envelope). Until a tokenizer installs its own, the slot holds the five ids
//! the guard was hard-coded to, so a caller that never loads one sees the old
//! behaviour.

use std::sync::Arc;

/// The marker token classes for one tokenizer (each a sorted id list).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolMarkers {
    lt: Vec<u32>,
    parameter: Vec<u32>,
    eq_start: Vec<u32>,
    gt_start: Vec<u32>,
    lt_slash_end: Vec<u32>,
}

fn has(set: &[u32], tok: u32) -> bool {
    set.binary_search(&tok).is_ok()
}

impl ToolMarkers {
    pub fn is_lt(&self, tok: u32) -> bool {
        has(&self.lt, tok)
    }
    pub fn is_parameter(&self, tok: u32) -> bool {
        has(&self.parameter, tok)
    }
    /// `=`, or `=` merged with the start of the key (`=new`, `=file`).
    pub fn is_eq_start(&self, tok: u32) -> bool {
        has(&self.eq_start, tok)
    }
    /// `>`, or `>` merged with what follows (`>\n`, `>\n\n`).
    pub fn is_gt(&self, tok: u32) -> bool {
        has(&self.gt_start, tok)
    }
    /// `</`, or `</` merged with what precedes it (`.</`, ` </`).
    pub fn is_lt_slash(&self, tok: u32) -> bool {
        has(&self.lt_slash_end, tok)
    }

    /// Build the classes from `(id, decoded text)` for every vocabulary entry.
    /// `None` when any class is empty.
    pub fn from_vocab<'a>(vocab: impl IntoIterator<Item = (u32, &'a str)>) -> Option<Self> {
        let mut m = Self {
            lt: Vec::new(),
            parameter: Vec::new(),
            eq_start: Vec::new(),
            gt_start: Vec::new(),
            lt_slash_end: Vec::new(),
        };
        for (id, text) in vocab {
            if text.trim_start() == "<" {
                m.lt.push(id);
            }
            if text == "parameter" {
                m.parameter.push(id);
            }
            if text.starts_with('=') {
                m.eq_start.push(id);
            }
            if text.starts_with('>') {
                m.gt_start.push(id);
            }
            if text.ends_with("</") {
                m.lt_slash_end.push(id);
            }
        }
        for v in [
            &mut m.lt,
            &mut m.parameter,
            &mut m.eq_start,
            &mut m.gt_start,
            &mut m.lt_slash_end,
        ] {
            v.sort_unstable();
            v.dedup();
        }
        let complete = [&m.lt, &m.parameter, &m.eq_start, &m.gt_start, &m.lt_slash_end]
            .iter()
            .all(|v| !v.is_empty());
        complete.then_some(m)
    }

    /// Build from a Hugging Face tokenizer: decode every id once.
    pub fn from_tokenizer(tok: &tokenizers::Tokenizer) -> Option<Self> {
        let n = tok.get_vocab_size(true) as u32;
        let texts: Vec<(u32, String)> = (0..n)
            .filter_map(|id| tok.decode(&[id], false).ok().map(|t| (id, t)))
            .collect();
        Self::from_vocab(texts.iter().map(|(id, t)| (*id, t.as_str())))
    }

    /// Class sizes, for the load log.
    pub fn summary(&self) -> String {
        format!(
            "< x{} parameter x{} =* x{} >* x{} *</ x{}",
            self.lt.len(),
            self.parameter.len(),
            self.eq_start.len(),
            self.gt_start.len(),
            self.lt_slash_end.len()
        )
    }

    /// The five ids the guard was hard-coded to (Qwen3.6 / 3.8 BPE): the
    /// default until a tokenizer installs its own classes.
    pub fn legacy_qwen() -> Self {
        Self {
            lt: vec![27],
            parameter: vec![15704],
            eq_start: vec![28],
            gt_start: vec![29],
            lt_slash_end: vec![510],
        }
    }
}

fn legacy() -> Arc<ToolMarkers> {
    static LEGACY: std::sync::OnceLock<Arc<ToolMarkers>> = std::sync::OnceLock::new();
    LEGACY
        .get_or_init(|| Arc::new(ToolMarkers::legacy_qwen()))
        .clone()
}

/// `None` = never installed (the legacy default applies);
/// `Some(None)` = installed, unresolved (tracking off).
type Slot = Option<Option<Arc<ToolMarkers>>>;

#[cfg(not(test))]
static MARKERS: std::sync::RwLock<Slot> = std::sync::RwLock::new(None);

#[cfg(test)]
thread_local! {
    // Per test thread, so parallel tests installing different markers do not
    // observe each other's.
    static MARKERS: std::cell::RefCell<Slot> = const { std::cell::RefCell::new(None) };
}

/// Install the classes of the tokenizer just loaded (`None` turns the
/// parameter-body tracking off).
pub fn set(markers: Option<ToolMarkers>) {
    let slot: Slot = Some(markers.map(Arc::new));
    #[cfg(not(test))]
    {
        *MARKERS.write().unwrap_or_else(|e| e.into_inner()) = slot;
    }
    #[cfg(test)]
    MARKERS.with(|m| *m.borrow_mut() = slot);
}

/// The classes of the tokenizer being served (the legacy Qwen ids until one is
/// installed). One uncontended read lock and an `Arc` clone per call.
pub fn get() -> Option<Arc<ToolMarkers>> {
    #[cfg(not(test))]
    let slot = MARKERS.read().unwrap_or_else(|e| e.into_inner()).clone();
    #[cfg(test)]
    let slot = MARKERS.with(|m| m.borrow().clone());
    slot.unwrap_or_else(|| Some(legacy()))
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::ToolMarkers;

    /// The parts of the Qwen3.6/3.8 vocabulary the markers depend on, with the
    /// merges seen in real tool calls (tokenizer.json, 2026-10-18): `=file`,
    /// `=re`, `=new`, punctuation-merged closes; `>` and `\n` separate.
    /// Merged-form ids >= 900_000 are stand-ins (only their text matters).
    pub fn qwen() -> ToolMarkers {
        ToolMarkers::from_vocab([
            (27, "<"),
            (366, " <"),
            (28, "="),
            (900_000, "=file"),
            (900_001, "=re"),
            (900_002, "=new"),
            (29, ">"),
            (510, "</"),
            (900_003, ".</"),
            (900_004, " </"),
            (15704, "parameter"),
            (198, "\n"),
        ])
        .unwrap()
    }

    /// Nemotron 3.5 Super VL (nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
    /// tokenizer.json, 2026-10-18): `>\n` merged (1561), `=new`, `.</`.
    pub fn nemotron() -> ToolMarkers {
        ToolMarkers::from_vocab([
            (1060, "<"),
            (1061, "="),
            (900_010, "=new"),
            (1062, ">"),
            (1561, ">\n"),
            (1885, "</"),
            (900_011, ".</"),
            (31960, "parameter"),
            (1010, "\n"),
        ])
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{nemotron, qwen};
    use super::*;

    #[test]
    fn classes_cover_the_merged_forms() {
        let q = qwen();
        assert!(q.is_lt(27) && q.is_lt(366) && !q.is_lt(510));
        assert!(q.is_eq_start(28) && q.is_eq_start(900_000) && q.is_eq_start(900_002));
        assert!(q.is_gt(29) && !q.is_gt(198));
        assert!(q.is_lt_slash(510) && q.is_lt_slash(900_003) && q.is_lt_slash(900_004));
        assert!(q.is_parameter(15704));
        let n = nemotron();
        assert!(n.is_gt(1062) && n.is_gt(1561) && !n.is_gt(29));
        assert!(n.is_eq_start(900_010) && n.is_lt_slash(900_011));
    }

    #[test]
    fn a_vocab_missing_a_class_resolves_to_none() {
        assert_eq!(
            ToolMarkers::from_vocab([(1, "<"), (2, "="), (3, ">"), (4, "</")]),
            None
        );
    }

    #[test]
    fn the_default_is_the_legacy_qwen_ids_until_installed() {
        assert_eq!(*get().unwrap(), ToolMarkers::legacy_qwen());
        set(None);
        assert!(get().is_none());
        set(Some(qwen()));
        assert!(get().unwrap().is_eq_start(900_000));
    }
}

#[cfg(test)]
#[path = "tool_markers_emit_tests.rs"]
mod emit_tests;
