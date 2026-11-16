// SPDX-License-Identifier: AGPL-3.0-only

//! Where a token's time goes, stage by stage (`ATLAS_DSV41_PROFILE=1`, debug
//! only). Each stage is bracketed by a stream synchronize, so the figures
//! are GPU time plus the host's, and the run is slower for it.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

struct Totals {
    stages: BTreeMap<&'static str, (Duration, u64)>,
    tokens: u64,
}

fn totals() -> Option<&'static Mutex<Totals>> {
    static ON: OnceLock<Option<Mutex<Totals>>> = OnceLock::new();
    ON.get_or_init(|| {
        std::env::var("ATLAS_DSV41_PROFILE")
            .is_ok_and(|v| v == "1")
            .then(|| {
                Mutex::new(Totals {
                    stages: BTreeMap::new(),
                    tokens: 0,
                })
            })
    })
    .as_ref()
}

/// Run `body` as stage `name`.
pub(super) fn stage<T>(
    gpu: &dyn GpuBackend,
    stream: u64,
    name: &'static str,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let Some(totals) = totals() else {
        return body();
    };
    gpu.synchronize(stream)?;
    let began = Instant::now();
    let out = body()?;
    gpu.synchronize(stream)?;
    let mut totals = totals.lock().expect("profile");
    let entry = totals.stages.entry(name).or_default();
    entry.0 += began.elapsed();
    entry.1 += 1;
    Ok(out)
}

/// The start of a stage that ends at [`end`].
pub(super) fn begin(gpu: &dyn GpuBackend, stream: u64) -> Result<Option<Instant>> {
    if totals().is_none() {
        return Ok(None);
    }
    gpu.synchronize(stream)?;
    Ok(Some(Instant::now()))
}

/// The end of stage `name`, begun at `began`.
pub(super) fn end(
    gpu: &dyn GpuBackend,
    stream: u64,
    name: &'static str,
    began: Option<Instant>,
) -> Result<()> {
    let (Some(totals), Some(began)) = (totals(), began) else {
        return Ok(());
    };
    gpu.synchronize(stream)?;
    let mut totals = totals.lock().expect("profile");
    let entry = totals.stages.entry(name).or_default();
    entry.0 += began.elapsed();
    entry.1 += 1;
    Ok(())
}

/// A token has left the last layer: every `EVERY` of them, log the stages.
pub(super) fn token_done() {
    const EVERY: u64 = 256;
    let Some(totals) = totals() else {
        return;
    };
    let mut totals = totals.lock().expect("profile");
    totals.tokens += 1;
    if totals.tokens % EVERY != 0 {
        return;
    }
    let all: Duration = totals.stages.values().map(|(t, _)| *t).sum();
    let mut lines: Vec<String> = totals
        .stages
        .iter()
        .map(|(name, (time, calls))| {
            format!(
                "{name} {:.2} ms/token ({:.0}%, {calls} calls)",
                time.as_secs_f64() * 1e3 / EVERY as f64,
                100.0 * time.as_secs_f64() / all.as_secs_f64().max(1e-9)
            )
        })
        .collect();
    lines.sort_by(|a, b| {
        let share = |s: &String| -> f64 {
            s.split('(')
                .nth(1)
                .and_then(|r| r.split('%').next())
                .and_then(|p| p.parse().ok())
                .unwrap_or(0.0)
        };
        share(b).total_cmp(&share(a))
    });
    tracing::info!(
        "deepseek_v41 profile, {EVERY} tokens, {:.1} ms/token in stages: {}",
        all.as_secs_f64() * 1e3 / EVERY as f64,
        lines.join("; ")
    );
    totals.stages.clear();
}
