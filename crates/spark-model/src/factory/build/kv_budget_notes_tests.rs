// SPDX-License-Identifier: AGPL-3.0-only

//! What a KV sizing logs, which WARNs fire, and the measuring wrapper.

use super::tests::{driver, gib, ledger, start, start_at_070};
use super::*;

fn lines(i: &Inputs, pool_from_budget: bool, warn: bool) -> Vec<String> {
    notes(i, &size(i), pool_from_budget)
        .into_iter()
        .filter(|n| n.warn == warn)
        .map(|n| n.text)
        .collect()
}

fn warns(i: &Inputs) -> Vec<String> {
    lines(i, true, true)
}

fn infos(i: &Inputs) -> Vec<String> {
    lines(i, true, false)
}

#[test]
fn a_quiet_start_logs_both_measures_and_warns_about_nothing() {
    let input = start(117.1, 14.9, driver(102.0));
    assert_eq!(warns(&input), Vec::<String>::new());
    let infos = infos(&input);
    assert_eq!(infos.len(), 2, "{infos:?}");
    // The line operators already read; its wording is part of the runbook.
    assert_eq!(
        infos[0],
        "KV budget self-relative (auto): baseline-free 117.1 GB − free-now 14.9 GB \
         = Atlas-own 102.2 GB; co-tenants 4.6 GB excluded (set \
         ATLAS_KV_EXTERNAL_RESERVE_GB to override)"
    );
    assert_eq!(
        infos[1],
        "KV own-footprint cross-check: free-memory measure 102.20 GB, tracked \
         102.00 GB (device 101.10 GB by driver accounting + host 0.90 GB) → \
         102.20 GB pre-KV"
    );
}

#[test]
fn a_backend_without_an_account_logs_only_the_free_memory_measure() {
    let input = start(117.1, 14.9, None);
    assert_eq!(warns(&input), Vec::<String>::new());
    assert_eq!(infos(&input).len(), 1);
    // No baseline either (the mock backend): nothing to say.
    let mut raw = input;
    raw.baseline_free = None;
    assert!(notes(&raw, &size(&raw), true).is_empty());
}

#[test]
fn a_release_during_the_load_is_a_warn_that_names_it() {
    let warns = warns(&start(114.3, 14.4, driver(102.3)));
    assert_eq!(warns.len(), 1, "{warns:?}");
    assert!(
        warns[0].contains("Atlas allocated 102.30 GB but free memory fell by only 99.90 GB"),
        "{}",
        warns[0]
    );
    assert!(warns[0].contains("2.40 GB was released by something else"));
}

#[test]
fn an_over_large_manual_override_is_not_blamed_on_a_release() {
    let mut input = start(117.1, 14.9, driver(102.3));
    input.manual_external = Some(gib(20.0));
    let warns = warns(&input);
    assert_eq!(warns.len(), 1, "{warns:?}");
    assert!(warns[0].contains("ATLAS_KV_EXTERNAL_RESERVE_GB leaves 86.80 GB as Atlas-own"));
    assert!(warns[0].contains("discounts 15.50 GB more than other processes hold"));
    assert!(!warns[0].contains("released"), "{}", warns[0]);
}

#[test]
fn only_driver_accounting_can_say_something_else_took_memory() {
    // The delta is 3 GiB over tracked: more than the counters miss on a
    // quiet start. By the driver that is news.
    let by_driver = warns(&start(119.0, 15.8, driver(100.2)));
    assert_eq!(by_driver.len(), 1, "{by_driver:?}");
    assert!(by_driver[0].contains("3.00 GB was taken by something else"));
    // The 1.5 GiB a quiet start shows is not.
    assert_eq!(
        warns(&start(117.5, 15.8, driver(100.2))),
        Vec::<String>::new()
    );

    // By the ledger it is expected on every start: one WARN says the check
    // is weak, and nothing claims a co-tenant took memory.
    let by_ledger = warns(&start(117.5, 15.8, ledger(100.2)));
    assert_eq!(by_ledger.len(), 1, "{by_ledger:?}");
    assert!(by_ledger[0].contains("driver's per-process accounting is not available"));
    assert!(by_ledger[0].contains("only the 4.0 GB kept free"));
    assert!(!by_ledger[0].contains("taken by something else"));
    assert!(infos(&start(117.5, 15.8, ledger(100.2)))[1].contains("by allocation ledger"));
}

#[test]
fn an_over_count_is_reported_as_one() {
    let warns = warns(&start(117.1, 14.9, driver(107.0)));
    assert_eq!(warns.len(), 1, "{warns:?}");
    assert!(warns[0].contains("tracked 107.00 GB is 0.20 GB more than the 106.80 GB in use"));
    assert!(warns[0].contains("charging everything in use to Atlas"));
}

#[test]
fn the_headroom_warn_is_only_for_a_pool_the_budget_sizes() {
    let input = start(114.3, 12.0, driver(102.3));
    let warns = warns(&input);
    assert_eq!(warns.len(), 1, "{warns:?}");
    assert_eq!(
        warns[0],
        "KV pool limited by free memory: 12.00 GB free now; keeping 3.50 GB for \
         growth after sizing plus the 0.90 GB reserve unallocated → 7.60 GB for KV, \
         not the 9.98 GB the utilization budget allows. Other processes hold 7.40 GB."
    );
    // --high-speed-swap sizes the pool itself and ignores the budget.
    assert_eq!(lines(&input, false, true), Vec::<String>::new());
}

#[test]
fn the_no_room_error_says_what_ran_out() {
    // The ceiling is what ran out: the old advice holds.
    let mut input = start(117.1, 14.9, None);
    input.utilization = 0.5;
    let budget = size(&input);
    assert_eq!((budget.bytes, budget.headroom_limited), (0, false));
    assert_eq!(
        budget.no_room_advice(input.free_now),
        "Raise --gpu-memory-utilization or use a smaller model."
    );

    // Co-tenants are what ran out: 45 GiB of them at 0.70 leave 2.4 GiB
    // beyond the reserves, under the headroom.
    let crowded = start_at_070(45.0);
    let budget = size(&crowded);
    assert_eq!((budget.bytes, budget.headroom_limited), (0, true));
    let advice = budget.no_room_advice(crowded.free_now);
    assert!(advice.contains("still allows 10.9 GB for KV"), "{advice}");
    assert!(advice.contains("only 37.2 GB is free now"), "{advice}");
    assert!(advice.contains("other processes hold 45.0 GB"), "{advice}");
    assert!(advice.contains("will not help"), "{advice}");
}

#[test]
fn the_external_reserve_override_needs_a_number_above_zero() {
    for unset in [
        None,
        Some(""),
        Some("abc"),
        Some("0"),
        Some("-1"),
        Some("NaN"),
    ] {
        assert_eq!(external_reserve_bytes(unset), None, "{unset:?}");
    }
    assert_eq!(external_reserve_bytes(Some("4.6")), Some(gib(4.6)));
    assert_eq!(external_reserve_bytes(Some("20")), Some(gib(20.0)));
    // Absurd but parsable: saturates, and `size` saturates the discount.
    assert_eq!(external_reserve_bytes(Some("inf")), Some(usize::MAX));
}

#[test]
fn measure_reads_the_backend_and_sizes_what_size_sizes() {
    let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
    let (total, free) = (gpu.total_memory().unwrap(), gpu.free_memory().unwrap());
    let got = measure(&gpu, total, free, 0.93, gib(0.9), true);
    // The mock keeps no account of its own allocations.
    assert_eq!(got.footprint, None);
    let want = size(&Inputs {
        total,
        free_now: free,
        utilization: 0.93,
        reserve: gib(0.9),
        baseline_free: spark_runtime::gpu::baseline_free_bytes(),
        manual_external: external_reserve_bytes(
            std::env::var("ATLAS_KV_EXTERNAL_RESERVE_GB")
                .ok()
                .as_deref(),
        ),
        tracked: None,
    });
    assert_eq!(
        (got.basis, got.own, got.bytes),
        (want.basis, want.own, want.bytes)
    );
    assert_eq!(got.used_raw, total - free);
}
