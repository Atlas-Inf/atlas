// SPDX-License-Identifier: AGPL-3.0-only

//! KV budget arithmetic against the GB10 TP2 starts of 2026-09-30
//! (`--gpu-memory-utilization=0.93`, 121.7 GiB, 0.8 + 0.1 GiB reserves).
//! What is logged about a sizing is in `kv_budget_notes_tests.rs`.

use super::*;

pub(super) const GIB_F: f64 = (1u64 << 30) as f64;
/// Blocks the reference start sized from its 10.1 GiB budget.
const NORMAL_BLOCKS: usize = 102_066;
/// Bytes one block costs on the reference pair, from the three logged
/// sizings (10.1 GB → 102066, 12.2 GB → 123500, 14.1 GB → 143479 blocks),
/// which put it between 105.5 and 106.3 thousand. Independent of `size()`.
const BYTES_PER_BLOCK: usize = 106_000;

pub(super) fn gib(x: f64) -> usize {
    (x * GIB_F) as usize
}

/// The reference pair's inference reserve plus lazy-BF16 reserve.
fn reserve() -> usize {
    gib(0.8) + gib(0.1)
}

fn blocks(bytes: usize) -> usize {
    bytes / BYTES_PER_BLOCK
}

fn footprint(x: f64, device_source: DeviceSource) -> Option<OwnFootprint> {
    // Mostly device, some host, summing to exactly `gib(x)`.
    let host = gib(x).min(gib(0.9));
    Some(OwnFootprint {
        device: gib(x) - host,
        device_source,
        host,
    })
}

/// A tracked figure of `x` GiB backed by the driver's accounting.
pub(super) fn driver(x: f64) -> Option<OwnFootprint> {
    footprint(x, DeviceSource::Driver)
}

/// A tracked figure of `x` GiB that is only the allocation ledger.
pub(super) fn ledger(x: f64) -> Option<OwnFootprint> {
    footprint(x, DeviceSource::Ledger)
}

/// A start on the reference pair: `baseline`/`free_now` in GiB.
pub(super) fn start(baseline: f64, free_now: f64, tracked: Option<OwnFootprint>) -> Inputs {
    Inputs {
        total: gib(121.7),
        free_now: gib(free_now),
        utilization: 0.93,
        reserve: reserve(),
        baseline_free: Some(gib(baseline)),
        manual_external: None,
        tracked,
    }
}

/// The documented GB10 start of 2026-07-25 (`docs/campaigns/
/// gb10-decode-fold-2026-07/raw/slots192_alloc.txt`): 0.70, own 39.5 GiB,
/// 34.8 GiB of reserves, with `co_tenants` GiB held by others throughout.
pub(super) fn start_at_070(co_tenants: f64) -> Inputs {
    Inputs {
        total: gib(121.7),
        free_now: gib(121.7 - 39.5 - co_tenants),
        utilization: 0.70,
        reserve: gib(34.8),
        baseline_free: Some(gib(121.7 - co_tenants)),
        manual_external: None,
        tracked: None,
    }
}

/// The sizing this module replaced, kept verbatim as the reference a start
/// with quiet co-tenants must still reproduce to the byte.
fn legacy(i: &Inputs) -> usize {
    let mut used = i.total.saturating_sub(i.free_now);
    if let Some(ext) = i.manual_external {
        used = used.saturating_sub(ext);
    } else if let Some(baseline) = i.baseline_free {
        let own = baseline.saturating_sub(i.free_now);
        if own > 0 && own <= used {
            used = own;
        }
    }
    ((i.total as f64 * i.utilization) as usize)
        .saturating_sub(used)
        .saturating_sub(i.reserve)
        .min(i.free_now.saturating_sub(i.reserve))
}

/// What is free after the pool is allocated.
fn left(i: &Inputs) -> usize {
    i.free_now - size(i).bytes
}

#[test]
fn normal_start_keeps_the_legacy_pool_to_the_byte() {
    // Rank 0: baseline-free 117.1 − free-now 14.9 = Atlas-own 102.2 → 10.1 GiB.
    let want = legacy(&start(117.1, 14.9, None));
    assert!((want as f64 / GIB_F - 10.08).abs() < 0.01, "{want}");
    assert!(
        blocks(want).abs_diff(NORMAL_BLOCKS) < 600,
        "{}",
        blocks(want)
    );
    // Whatever a tracked measure at or below the delta says, nothing moves.
    for tracked in [
        None,
        driver(0.0),
        driver(60.0),
        driver(101.4),
        driver(102.2),
    ] {
        let got = size(&start(117.1, 14.9, tracked));
        assert_eq!(got.bytes, want, "tracked {tracked:?}");
        assert_eq!(got.basis, Basis::Delta);
        assert_eq!(got.own, gib(117.1) - gib(14.9));
        assert_eq!(got.headroom, HEADROOM_BYTES);
        assert!(!got.headroom_limited);
    }
}

#[test]
fn memory_released_by_another_process_is_not_credited_to_atlas() {
    // The bad start: a co-tenant freed 2.2 GiB (rank 0) and 3.4 GiB (rank 1)
    // during the load, so the delta read 100.1 and 97.9 for a process that
    // holds about 102.3. The delta alone sized 123500 and 143479 blocks.
    assert!(blocks(legacy(&start(114.3, 14.2, None))) > 120_000);
    assert!(blocks(legacy(&start(112.9, 15.0, None))) > 140_000);

    let rank0 = size(&start(114.3, 14.2, driver(102.3)));
    assert_eq!(rank0.own, gib(102.3));
    assert_eq!(rank0.basis_own, gib(114.3) - gib(14.2));
    assert_eq!(
        rank0.disagreement(),
        Some(Disagreement::TrackedAbove(gib(102.3) - rank0.basis_own))
    );
    // The ceiling would allow 9.98 GiB. These co-tenants hold 5.2 GiB, not
    // the normal 4.6, so the floor takes another 0.18: about 99K blocks.
    assert_eq!(rank0.by_budget, rank0.total_budget - gib(102.3) - reserve());
    assert!(rank0.headroom_limited);
    assert!(
        (98_500..NORMAL_BLOCKS).contains(&blocks(rank0.bytes)),
        "{}",
        blocks(rank0.bytes)
    );
    assert_eq!(
        left(&start(114.3, 14.2, driver(102.3))),
        reserve() + HEADROOM_BYTES
    );

    // Rank 1's tracked figure is an assumption (no log of it exists): a
    // worker without the drafter, 1 GiB under rank 0.
    let rank1 = size(&start(112.9, 15.0, driver(101.3)));
    assert_eq!(rank1.own, gib(101.3));
    assert!(blocks(rank1.bytes) < 108_000, "{}", blocks(rank1.bytes));
    // Both ranks then agree on the minimum, which is rank 0's.
    assert!(rank0.bytes < rank1.bytes);
}

#[test]
fn without_a_tracked_measure_the_headroom_floor_bounds_the_damage() {
    // Same bad start, no independent measure at all: the delta would size
    // 12.2 GiB, and the pool is capped by what is free now less the headroom.
    let input = start(114.3, 14.2, None);
    let got = size(&input);
    assert!(got.headroom_limited);
    assert_eq!(got.headroom, HEADROOM_BYTES);
    assert_eq!(got.bytes, gib(14.2) - reserve() - HEADROOM_BYTES);
    assert!(blocks(got.bytes) < 100_000, "{}", blocks(got.bytes));
    // 4.4 GiB stay free: 1.5 above empty once the process has grown by 2.9.
    assert_eq!(left(&input), reserve() + HEADROOM_BYTES);
}

#[test]
fn a_ledger_only_figure_buys_a_larger_headroom() {
    // No driver accounting: the ledger reads low (here by 1.5 GiB), so it
    // proves nothing against a release and the floor keeps 4 GiB instead.
    let quiet = size(&start(117.1, 14.9, ledger(100.7)));
    assert_eq!(quiet.headroom, UNVERIFIED_HEADROOM_BYTES);
    assert_eq!(quiet.own, gib(117.1) - gib(14.9));
    // On the reference start that costs under 0.1 GiB of pool.
    let normal = size(&start(117.1, 14.9, None)).bytes;
    assert!(quiet.bytes <= normal && normal - quiet.bytes < gib(0.1));

    // The bad start with a ledger too low to notice the release: the pool
    // leaves 4.9 GiB, what the reference start leaves (14.9 − 10.08 = 4.82).
    let bad = start(114.3, 14.2, ledger(99.0));
    assert_eq!(size(&bad).own, gib(114.3) - gib(14.2));
    assert_eq!(left(&bad), reserve() + UNVERIFIED_HEADROOM_BYTES);
    assert!(left(&bad) >= left(&start(117.1, 14.9, None)));
}

#[test]
fn a_co_tenant_that_allocates_during_the_load_shrinks_the_pool() {
    // 3 GiB taken by someone else mid-load: the delta over-measures (105.2),
    // the larger figure is kept, and the pool is 3 GiB smaller. Safe.
    let input = start(117.1, 11.9, driver(102.0));
    let got = size(&input);
    assert_eq!(got.own, gib(117.1) - gib(11.9));
    assert_eq!(got.bytes, legacy(&input));
    assert_eq!(
        got.bytes + (gib(14.9) - gib(11.9)),
        size(&start(117.1, 14.9, None)).bytes
    );
    assert_eq!(
        got.disagreement(),
        Some(Disagreement::TrackedBelow(got.own - gib(102.0)))
    );
}

#[test]
fn co_tenants_may_not_take_the_headroom() {
    // 7.4 GiB of co-tenant that stays: own is measured right (102.3), the
    // utilization budget would allow 9.98 GiB, but that would leave 2.0 GiB
    // for a process that still grows by 2.9.
    let input = start(114.3, 12.0, driver(102.3));
    let got = size(&input);
    assert!(got.headroom_limited);
    assert_eq!(got.bytes, gib(12.0) - reserve() - HEADROOM_BYTES);
    assert!(got.bytes < legacy(&input));
}

#[test]
fn the_reference_start_has_point_four_gib_of_co_tenant_slack() {
    // Co-tenants present before the load are in the baseline, so own stays
    // 102.2. The normal 4.6 GiB of them leaves 3.92 GiB beyond the reserves;
    // the pool starts to shrink, one for one, once they pass about 5.0.
    let with_co_tenants = |extra: f64| start(117.1 - extra, 14.9 - extra, None);
    for extra in [0.0, 0.2, 0.4] {
        let input = with_co_tenants(extra);
        assert!(!size(&input).headroom_limited, "{extra}");
        assert_eq!(size(&input).bytes, legacy(&input), "{extra}");
    }
    for extra in [0.5, 1.0, 2.0] {
        let input = with_co_tenants(extra);
        let got = size(&input);
        assert!(got.headroom_limited, "{extra}");
        let lost = legacy(&input) - got.bytes;
        assert!(lost.abs_diff(gib(extra - 0.42)) < gib(0.01), "{extra}");
        assert_eq!(left(&input), reserve() + HEADROOM_BYTES, "{extra}");
    }
}

#[test]
fn the_headroom_is_absolute_and_never_more_than_the_margin() {
    for (utilization, want) in [
        (0.15, HEADROOM_BYTES),
        (0.50, HEADROOM_BYTES),
        (0.70, HEADROOM_BYTES),
        (0.93, HEADROOM_BYTES),
        (0.97, HEADROOM_BYTES),
    ] {
        let mut input = start(117.1, 14.9, None);
        input.utilization = utilization;
        assert_eq!(size(&input).headroom, want, "{utilization}");
    }
    // Less than the headroom left outside the ceiling: all of it, no more.
    for utilization in [0.98, 0.99, 1.0] {
        let mut input = start(117.1, 14.9, ledger(100.0));
        input.utilization = utilization;
        let got = size(&input);
        assert_eq!(got.headroom, gib(121.7) - got.total_budget);
        assert!(got.headroom < HEADROOM_BYTES, "{utilization}");
    }
}

#[test]
fn a_device_atlas_has_to_itself_is_sized_as_before_at_every_utilization() {
    // No co-tenants: the floor leaves the ceiling alone, whatever it is.
    for utilization in [0.15, 0.5, 0.7, 0.9, 0.93, 0.97, 0.99, 1.0] {
        for own in [8.0, 40.0, 102.2] {
            let mut input = start(121.7, 121.7 - own, None);
            input.utilization = utilization;
            let got = size(&input);
            assert_eq!(got.bytes, legacy(&input), "{utilization} {own}");
            assert!(!got.headroom_limited, "{utilization} {own}");
        }
    }
}

#[test]
fn a_low_utilization_start_is_not_over_reserved() {
    // 2026-07-25: 121.7 × 0.70 = 85.2 budget, 39.5 own, 34.8 reserve, 9.1
    // co-tenants → 10.9 GiB, unchanged.
    let logged = start_at_070(9.1);
    assert!((size(&logged).bytes as f64 / GIB_F - 10.89).abs() < 0.01);
    assert_eq!(size(&logged).bytes, legacy(&logged));
    assert_eq!(size(&logged).headroom, HEADROOM_BYTES);
    // A 30 GiB co-tenant still costs nothing (a third of the 36.5 GiB margin
    // as headroom cut this pool to 5.2 GiB).
    let shared = start_at_070(30.0);
    assert_eq!(size(&shared).bytes, legacy(&shared));
    assert!(!size(&shared).headroom_limited);
    // At 36 GiB the ceiling would leave 0.5 GiB once the reserves are used;
    // the floor leaves 3.5 and the pool is 7.9 GiB, not zero.
    let crowded = start_at_070(36.0);
    assert!(size(&crowded).headroom_limited);
    assert!((size(&crowded).bytes as f64 / GIB_F - 7.9).abs() < 0.01);
    assert_eq!(left(&crowded), gib(34.8) + HEADROOM_BYTES);
}

#[test]
fn manual_external_reserve_still_wins_over_the_baseline() {
    // ATLAS_KV_EXTERNAL_RESERVE_GB=4.6 on the normal start: raw used 106.8
    // less 4.6 is the same 102.2, whatever the baseline says.
    let mut input = start(90.0, 14.9, None);
    input.manual_external = Some(gib(4.6));
    let got = size(&input);
    assert_eq!(got.basis, Basis::Manual);
    assert_eq!(got.own, gib(121.7) - gib(14.9) - gib(4.6));
    assert_eq!(got.bytes, legacy(&input));

    // An override that discounts more than the co-tenants hold cannot push
    // own below what the process provably allocated.
    input.manual_external = Some(gib(20.0));
    input.tracked = driver(102.3);
    let got = size(&input);
    assert_eq!(got.basis_own, gib(121.7) - gib(14.9) - gib(20.0));
    assert_eq!(got.own, gib(102.3));
    assert!(matches!(
        got.disagreement(),
        Some(Disagreement::TrackedAbove(_))
    ));

    // Larger than everything in use: saturates, never wraps.
    input.manual_external = Some(gib(500.0));
    input.tracked = None;
    assert_eq!(size(&input).own, 0);
}

#[test]
fn zero_and_absent_measures_fall_back_to_raw_used() {
    let raw_used = gib(121.7) - gib(14.9);

    // No baseline (mock backend) and no tracked measure: co-tenants count.
    let mut input = start(117.1, 14.9, None);
    input.baseline_free = None;
    let got = size(&input);
    assert_eq!((got.basis, got.own), (Basis::Raw, raw_used));
    assert_eq!(got.bytes, legacy(&input));
    assert_eq!(got.disagreement(), None);
    // With no baseline own is already everything in use; tracked adds nothing.
    input.tracked = driver(105.0);
    assert_eq!(size(&input).own, raw_used);

    // Baseline at or under free-now (delta zero) or above total: implausible.
    for baseline in [14.9, 10.0, 140.0] {
        let input = start(baseline, 14.9, driver(0.0));
        let got = size(&input);
        assert_eq!(
            (got.basis, got.own),
            (Basis::Implausible, raw_used),
            "{baseline}"
        );
        assert_eq!(got.bytes, legacy(&input));
    }
}

#[test]
fn a_tracked_figure_above_everything_in_use_is_charged_as_everything_in_use() {
    // More tracked than is in use system-wide (106.8): the counters
    // over-count. The delta alone is the measure known to be wrong, so the
    // charge is raw used, exactly as for an implausible delta.
    let raw_used = gib(121.7) - gib(14.9);
    let mut no_baseline = start(117.1, 14.9, None);
    no_baseline.baseline_free = None;
    for tracked in [107.0, 500.0] {
        let got = size(&start(117.1, 14.9, driver(tracked)));
        assert_eq!(got.own, raw_used, "{tracked}");
        assert_eq!(got.bytes, size(&no_baseline).bytes, "{tracked}");
        assert_eq!(
            got.disagreement(),
            Some(Disagreement::OverCount(gib(tracked) - raw_used))
        );
    }
    // Where own is already everything in use there is nothing to report.
    no_baseline.tracked = driver(500.0);
    assert_eq!(size(&no_baseline).disagreement(), None);
}

#[test]
fn nothing_left_is_zero_not_a_wraparound() {
    assert_eq!(size(&start(117.1, 0.5, driver(102.0))).bytes, 0);
    let mut input = start(117.1, 14.9, None);
    input.reserve = gib(200.0);
    assert_eq!(size(&input).bytes, 0);
    input = start(117.1, 14.9, None);
    input.utilization = 0.5;
    assert_eq!(size(&input).bytes, 0);
    // A utilization above one has no margin and no headroom.
    input.utilization = 1.5;
    assert_eq!(size(&input).headroom, 0);
}

#[test]
fn measures_disagree_only_past_their_thresholds() {
    // Tracked above the free-memory measure: half a GiB is the line.
    assert_eq!(
        size(&start(117.1, 14.9, driver(102.2 + 0.4))).disagreement(),
        None
    );
    assert!(
        size(&start(117.1, 14.9, driver(102.2 + 0.6)))
            .disagreement()
            .is_some()
    );
    // Tracked below it: the counters miss about 1.6 GiB on a quiet start, so
    // only a gap past 2.5 GiB says something else allocated during the load.
    for tracked in [102.2 - 0.6, 102.2 - 1.65, 102.2 - 2.4] {
        let got = size(&start(117.1, 14.9, driver(tracked)));
        assert_eq!(got.disagreement(), None, "{tracked}");
    }
    let got = size(&start(117.1, 14.9, driver(102.2 - 2.6)));
    assert!(matches!(
        got.disagreement(),
        Some(Disagreement::TrackedBelow(_))
    ));
    // Raw used includes co-tenants by design; a smaller tracked figure there
    // says nothing about either measure.
    let mut input = start(117.1, 14.9, driver(90.0));
    input.baseline_free = None;
    assert_eq!(size(&input).disagreement(), None);
}

#[test]
fn the_pool_is_never_larger_than_the_sizing_it_replaced_gave() {
    // Every change here charges more or leaves more; none can add pool.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = |below: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % below.max(1) as u64) as usize
    };
    for _ in 0..200_000 {
        let total = [gib(121.7), gib(60.0), gib(24.0), gib(16.0)][next(4)];
        let span = total + total / 4;
        let free_now = next(span);
        let utilization = [0.15, 0.5, 0.7, 0.9, 0.93, 0.97, 1.0, 1.2][next(8)];
        let reserve = next(gib(3.0));
        // Each measure is absent in some draws, as it can be at a real start.
        let baseline_free = (next(3) > 0).then(|| next(span));
        let manual_external = (next(4) == 0).then(|| next(gib(40.0)));
        let tracked = (next(3) > 0).then(|| OwnFootprint {
            device: next(span),
            device_source: [DeviceSource::Driver, DeviceSource::Ledger][next(2)],
            host: next(gib(2.0)),
        });
        let input = Inputs {
            total,
            free_now,
            utilization,
            reserve,
            baseline_free,
            manual_external,
            tracked,
        };
        let got = size(&input);
        assert!(got.bytes <= legacy(&input), "{input:?}");
        assert!(got.own <= got.used_raw, "{input:?}");
    }
}
