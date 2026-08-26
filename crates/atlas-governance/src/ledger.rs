// SPDX-License-Identifier: AGPL-3.0-only

//! Reading, appending, and materialising a journey.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::event::Event;

/// Every event recorded for one pull request, in the order they were written.
#[derive(Debug, Clone, Default)]
pub struct Journey {
    pub events: Vec<Event>,
}

impl Journey {
    /// Deduplicate by [`Event::identity`], keeping the first occurrence.
    ///
    /// The file is a set, not a log: replaying a CI job appends the same
    /// records again, and a reader that counted them twice would report a gate
    /// as having been evaluated more often than it was.
    pub fn deduplicated(mut self) -> Self {
        let mut seen = BTreeSet::new();
        self.events.retain(|e| seen.insert(e.identity()));
        self
    }

    /// Events for one gate id, oldest first.
    pub fn gate_history<'a>(&'a self, gate: &'a str) -> impl Iterator<Item = &'a Event> {
        self.events.iter().filter(
            move |e| matches!(&e.kind, crate::event::EventKind::Gate { id, .. } if id == gate),
        )
    }
}

/// Append one event to a per-PR file, creating it if needed.
///
/// ★ One file per pull request, never a shared one. Two PRs cannot then touch
/// the same path, so the classic shared-state-file collision is designed out
/// rather than resolved. Within a file, records are only appended, so a textual
/// merge is union — declare `governance/*.jsonl merge=union` in
/// `.gitattributes` and concurrent appends stop conflicting entirely.
pub fn append(path: &Path, event: &Event) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let line = serde_json::to_string(event).context("encoding journey event")?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {} for append", path.display()))?;
    writeln!(file, "{line}").with_context(|| format!("appending to {}", path.display()))
}

/// Read a journey, skipping blank lines.
///
/// A malformed line is an error rather than a silent skip. The ledger's whole
/// value is that it is complete; a reader that quietly dropped what it could
/// not parse would report a partial history as a full one.
pub fn read_all(path: &Path) -> Result<Journey> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut events = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading {} line {}", path.display(), n + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        events.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parsing {} line {}", path.display(), n + 1))?,
        );
    }
    Ok(Journey { events })
}

/// The conventional path for a pull request's journey.
pub fn path_for(root: &Path, pr: u64) -> std::path::PathBuf {
    root.join("governance").join(format!("pr-{pr}.jsonl"))
}
