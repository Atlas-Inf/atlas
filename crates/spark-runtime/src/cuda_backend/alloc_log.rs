// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_ALLOC_LOG_MB=<n>`: log every device allocation of at least `n` MB
//! with the ledger's running total and the allocating call site, so startup
//! memory can be accounted item by item. Off (one cached read) by default.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

fn threshold() -> Option<usize> {
    static T: OnceLock<Option<usize>> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("ATLAS_ALLOC_LOG_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb << 20)
    })
}

/// Bytes of allocations below the threshold since the last logged one.
static SMALL: AtomicUsize = AtomicUsize::new(0);

/// The first frames outside the backend, in the model and server crates.
fn call_site() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    bt.lines()
        .map(str::trim)
        .filter(|l| {
            (l.contains("spark_model::") || l.contains("spark::") || l.contains("spark_runtime::"))
                && !l.contains("cuda_backend")
                && !l.contains("backtrace")
        })
        .map(|l| l.split_once(": ").map_or(l, |(_, f)| f))
        .map(|f| f.rsplit_once("::h").map_or(f, |(n, _)| n))
        .take(3)
        .collect::<Vec<_>>()
        .join(" <- ")
}

pub(super) fn note(bytes: usize, ledger_total: usize) {
    let Some(min) = threshold() else {
        return;
    };
    if bytes < min {
        SMALL.fetch_add(bytes, Ordering::Relaxed);
        return;
    }
    let small = SMALL.swap(0, Ordering::Relaxed);
    tracing::info!(
        "alloc {:.1} MB (ledger {:.2} GB, +{:.1} MB small since last) at {}",
        bytes as f64 / 1048576.0,
        ledger_total as f64 / 1e9,
        small as f64 / 1048576.0,
        call_site()
    );
}
