// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in eager-stream diagnostics. Nested intervals are reported exclusively.

use anyhow::{Result, ensure};
use spark_runtime::{gpu::GpuBackend, timing::Event};
use std::{cell::RefCell, collections::BTreeMap, sync::OnceLock};

static ENABLED: OnceLock<bool> = OnceLock::new();
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("ATLAS_HY4_TIMING").is_ok_and(|v| matches!(v.trim(), "1" | "true"))
    })
}

struct Record {
    name: &'static str,
    parent: Option<usize>,
    start: Event,
    end: Event,
}
#[derive(Default)]
struct State {
    records: Vec<Record>,
    stack: Vec<usize>,
    active: bool,
    failed: bool,
}
thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

pub struct Span {
    index: Option<usize>,
    stream: u64,
}
pub fn span(name: &'static str, stream: u64) -> Result<Span> {
    let index = STATE.with(|state| -> Result<_> {
        let mut s = state.borrow_mut();
        if !s.active {
            return Ok(None);
        }
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(stream)?;
        let parent = s.stack.last().copied();
        let index = s.records.len();
        s.records.push(Record {
            name,
            parent,
            start,
            end,
        });
        s.stack.push(index);
        Ok(Some(index))
    })?;
    Ok(Span { index, stream })
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(i) = self.index {
            STATE.with(|state| {
                let mut s = state.borrow_mut();
                s.failed |= s.records[i].end.record(self.stream).is_err();
                s.failed |= s.stack.pop() != Some(i);
            });
        }
    }
}

pub struct Token<'a> {
    gpu: &'a dyn GpuBackend,
    stream: u64,
    rank: usize,
    position: usize,
}
impl<'a> Token<'a> {
    pub fn begin(gpu: &'a dyn GpuBackend, stream: u64, rank: usize, position: usize) -> Self {
        if enabled() {
            STATE.with(|state| {
                *state.borrow_mut() = State {
                    active: true,
                    ..State::default()
                };
            });
        }
        Self {
            gpu,
            stream,
            rank,
            position,
        }
    }

    pub fn finish(self) -> Result<()> {
        if !enabled() {
            return Ok(());
        }
        self.gpu.synchronize(self.stream)?;
        STATE.with(|state| {
            let mut s = state.borrow_mut();
            ensure!(
                !s.failed && s.stack.is_empty(),
                "HY4 timing interval failed"
            );
            let mut rows = Vec::new();
            for r in &s.records {
                rows.push((r.name, r.parent, r.start.elapsed_ms(&r.end)?));
            }
            let ms = exclusive(&rows)?;
            let total: f64 = ms.values().sum();
            tracing::info!(
                "HY4_TIMING rank={} position={} total_ms={total:.6} gpu_ms={ms:?}",
                self.rank,
                self.position,
            );
            s.records.clear();
            s.active = false;
            Ok(())
        })
    }
}
impl Drop for Token<'_> {
    fn drop(&mut self) {
        STATE.with(|state| {
            let mut s = state.borrow_mut();
            s.active = false;
            s.records.clear();
            s.stack.clear();
        });
    }
}

fn exclusive(rows: &[(&'static str, Option<usize>, f64)]) -> Result<BTreeMap<&'static str, f64>> {
    let mut values: Vec<f64> = rows.iter().map(|r| r.2).collect();
    for (i, &(_, parent, ms)) in rows.iter().enumerate() {
        if let Some(p) = parent {
            ensure!(p < i, "HY4 timing invalid parent");
            values[p] -= ms;
        }
    }
    let mut result = BTreeMap::new();
    for ((name, _, _), ms) in rows.iter().zip(values) {
        *result.entry(*name).or_default() += ms;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_intervals_do_not_double_count() {
        let ms = exclusive(&[
            ("row", None, 10.),
            ("attention", Some(0), 7.),
            ("reduce", Some(1), 2.),
        ])
        .unwrap();
        assert_eq!(ms["row"], 3.);
        assert_eq!(ms["attention"], 5.);
        assert_eq!(ms["reduce"], 2.);
        assert_eq!(ms.values().sum::<f64>(), 10.);
    }
    #[test]
    fn empty_and_malformed_intervals() {
        assert!(exclusive(&[]).unwrap().is_empty());
        assert!(exclusive(&[("bad", Some(0), 1.)]).is_err());
    }
}
