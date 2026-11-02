// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_HOST_SYNC_TRACE=1`: where the host waits for the device. Every
//! stream/event/context synchronize and every blocking D2H copy of the CUDA
//! backend is timed and charged to the first `spark_model` frame that called
//! it (a backtrace a call: a diagnostic, it slows the host). [`take_report`]
//! hands the totals since the last call, the longest first.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;
#[cfg(feature = "cuda")]
use std::time::Instant;

/// Whether the trace is on. Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_HOST_SYNC_TRACE").as_deref() == Ok("1"))
}

/// `(what, site)` -> (calls, total wait).
type Stats = HashMap<(&'static str, String), (u64, Duration)>;

static STATS: Mutex<Option<Stats>> = Mutex::new(None);

/// The first `spark_model` frame of `bt`, as `function (file:line)`.
#[cfg(any(feature = "cuda", test))]
fn site(bt: &str) -> String {
    let mut lines = bt.lines();
    while let Some(line) = lines.next() {
        let frame = line.trim();
        if frame.contains("spark_model::") && !frame.contains("host_sync_trace") {
            let name = frame.split_once(": ").map_or(frame, |(_, n)| n);
            let at = lines
                .next()
                .map(|l| l.trim().trim_start_matches("at ").to_string())
                .unwrap_or_default();
            let file = at.rsplit('/').next().unwrap_or(&at).to_string();
            return format!("{name} ({file})");
        }
    }
    "outside spark_model".into()
}

/// Run `f`, a host wait of kind `what`, timing it when the trace is on.
#[cfg(feature = "cuda")]
pub(crate) fn traced<T>(what: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    let waited = t0.elapsed();
    let at = site(&std::backtrace::Backtrace::force_capture().to_string());
    let mut stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let entry = stats
        .get_or_insert_with(HashMap::new)
        .entry((what, at))
        .or_default();
    entry.0 += 1;
    entry.1 += waited;
    out
}

/// The waits since the last report, the longest total first (at most
/// `top`), one line each; empty when the trace is off or nothing waited.
pub fn take_report(top: usize) -> String {
    let stats = STATS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default();
    let mut rows: Vec<_> = stats.into_iter().collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1.1));
    let total: Duration = rows.iter().map(|r| r.1.1).sum();
    let calls: u64 = rows.iter().map(|r| r.1.0).sum();
    let mut out = if rows.is_empty() {
        String::new()
    } else {
        format!("{calls} host waits, {:.1} ms:", total.as_secs_f64() * 1e3)
    };
    for ((what, at), (n, t)) in rows.into_iter().take(top) {
        out += &format!("\n  {:8.1} ms {n:5}x {what} at {at}", t.as_secs_f64() * 1e3);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_site_is_the_first_model_frame() {
        let bt = "   0: std::backtrace::Backtrace::force_capture\n             at /rustc/x/library/std/src/backtrace.rs:1\n   1: spark_runtime::cuda_backend::sync\n             at ./crates/spark-runtime/src/a.rs:2\n   2: spark_model::layers::moe::forward_prefill_routed::run\n             at ./crates/spark-model/src/layers/moe/forward_prefill_routed.rs:118:18\n";
        assert_eq!(
            site(bt),
            "spark_model::layers::moe::forward_prefill_routed::run (forward_prefill_routed.rs:118:18)"
        );
        assert_eq!(site("   0: main\n"), "outside spark_model");
    }
}
