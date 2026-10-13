// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the one-shot layout, settings, stripes and flag words.

use super::*;

fn cfg(vars: &[(&str, &str)]) -> Option<Config> {
    Config::parse(|k| {
        vars.iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    })
}

#[test]
fn off_unless_requested_and_defaults_cover_decode_payloads() {
    assert_eq!(cfg(&[]), None);
    assert_eq!(cfg(&[("ATLAS_RDMA_ONESHOT", "0")]), None);
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]).unwrap();
    // The capture contract promises >= 512 KiB (k5 route: 8 rows x 5 x 8 KiB).
    assert_eq!(c.max, 1 << 20);
    assert!(c.max >= 512 << 10);
    assert_eq!(c.stripe_min, SPLIT_MIN);
    assert_eq!(c.timeout_ns, 30_000_000_000);
    assert!(!c.stage_fence);
    let c = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_MAX", "600000"),
        ("ATLAS_RDMA_ONESHOT_STRIPE_MIN", "16384"),
        ("ATLAS_RDMA_ONESHOT_TIMEOUT_MS", "0"),
        ("ATLAS_RDMA_ONESHOT_STAGE_FENCE", "1"),
    ])
    .unwrap();
    assert_eq!((c.max, c.stripe_min, c.timeout_ns), (600_000, 16384, 0));
    assert!(c.stage_fence);
    let big = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_MAX", "999999999"),
    ]);
    assert_eq!(big.unwrap().max, MAX_LIMIT);
    let small = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_MAX", "4096"),
    ]);
    assert_eq!(small.unwrap().max, MIN_MAX);
}

#[test]
fn eligibility_depends_only_on_size() {
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]).unwrap();
    for rows in 1..=8 {
        assert!(c.eligible(rows * 4096 * 2), "decode rows {rows}");
        assert!(c.eligible(rows * 4096 * 2 * 5), "k5 rows {rows}");
    }
    assert!(c.eligible(16) && c.eligible(2) && c.eligible(c.max));
    assert!(!c.eligible(0) && !c.eligible(3) && !c.eligible(c.max + 2));
}

#[test]
fn wire_distinguishes_configs() {
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]);
    assert_eq!(Config::wire(None), [0; 2]);
    assert_ne!(Config::wire(c), Config::wire(None));
    let d = cfg(&[
        ("ATLAS_RDMA_ONESHOT", "1"),
        ("ATLAS_RDMA_ONESHOT_STRIPE_MIN", "1"),
    ]);
    assert_ne!(Config::wire(c), Config::wire(d));
}

#[test]
fn region_layout_is_disjoint_and_aligned() {
    let max = 1 << 20;
    assert_eq!(recv_off(max, 0), max);
    assert_eq!(recv_off(max, 1), 2 * max);
    assert_eq!(ctrl_off(max), 3 * max);
    assert_eq!(region_bytes(max), ctrl_off(max) + CTRL_PAGE);
    // Every control word on its own 64-byte line inside the page.
    let mut words = vec![STAGE, POISON];
    words.extend((0..MAX_RAILS).flat_map(|r| [FLAGS + 64 * r, FLAG_SRC + 64 * r]));
    words.sort_unstable();
    assert!(words.windows(2).all(|w| w[1] >= w[0] + 64));
    assert!(words.iter().all(|w| w % 64 == 0 && w + 8 <= CTRL_PAGE));
    // Receive slots and the staging buffer are 16-byte aligned for the
    // kernel's vector loads whenever `max` is (it is a multiple of 64).
    assert!(recv_off(max, 1).is_multiple_of(16));
}

#[test]
fn stripes_cover_the_payload_in_aligned_parts() {
    assert_eq!(stripes(65536, 2, SPLIT_MIN), [(0, 65536)]);
    assert_eq!(stripes(65536, 2, 16384), [(0, 32768), (32768, 32768)]);
    assert_eq!(stripes(100, 2, 16), [(0, 64), (64, 36)]);
    // Too small to use every rail: fewer parts, and the kernel waits on
    // exactly as many flags as the proxy sends.
    assert_eq!(stripes(48, 4, 16), [(0, 48)]);
    for bytes in [2, 16, 8190, 65536, 327_680, 1 << 20] {
        for rails in 1..=MAX_RAILS {
            let parts = stripes(bytes, rails, 0);
            assert!(!parts.is_empty() && parts.len() <= rails);
            let mut next = 0;
            for &(off, len) in &parts {
                assert!(off == next && off % 64 == 0 && len > 0);
                next += len;
            }
            assert_eq!(next, bytes);
        }
    }
}

#[test]
fn flag_words_order_by_sequence_and_carry_the_size() {
    let (a, b) = (flag_word(7, 65536), flag_word(8, 16));
    assert!(a >> 24 == 7 && b >> 24 == 8 && a < b);
    assert_eq!(a & 0xff_ffff, 65536);
    assert_eq!(flag_word(1, MAX_LIMIT) & 0xff_ffff, MAX_LIMIT as u64);
    assert!(describe_poison(POISON_MISMATCH | 9).contains("op 9"));
    assert!(describe_poison(POISON_TIMEOUT | 3).contains("never arrived"));
    assert!(describe_poison(POISON_DESYNC | 5).contains("op 5: the peer's flag is more than one"));
    assert!(describe_poison(POISON_PROXY | 2).contains("op 2: the RDMA proxy failed"));
    assert!(describe_poison(POISON_LAUNCH).contains("launch failed"));
}

/// A zeroed stand-in for a pinned region of `bytes`, and its address.
fn region(bytes: usize) -> (Vec<u64>, usize) {
    let mut mem = vec![0u64; bytes / 8];
    let host = mem.as_mut_ptr() as usize;
    (mem, host)
}

#[test]
fn a_failed_proxy_poisons_the_channel_and_releases_the_stage_wait() {
    use super::super::{Peer, proxy::proxy_loop, region_bytes as legacy_bytes};
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]).unwrap();
    let ((_os, host), (_legacy, legacy)) = (region(region_bytes(c.max)), region(legacy_bytes(64)));
    let ctrl = host + ctrl_off(c.max);
    let stage = unsafe { &*((ctrl + STAGE) as *const AtomicU32) };
    // An ineligible staged size makes the proxy give up before it uses a rail.
    stage.store(3, Ordering::Release);
    let peer = Peer {
        base: 0,
        rkeys: Vec::new(),
    };
    let (jobs, stop) = Default::default();
    let ch = Channel::new(c, host, 0);
    let end = proxy_loop(
        Vec::new(),
        &[],
        &peer,
        legacy,
        64,
        &jobs,
        &stop,
        Some(ch),
        None,
    );
    assert!(end.unwrap_err().to_string().contains("not eligible"));
    // Nothing would clear `stage` again: the proxy leaves it clear, so the
    // stream reaches the kernel, and poisoned, so that kernel traps.
    assert_eq!(stage.load(Ordering::Acquire), 0);
    let why = poison_at(ctrl).expect("poisoned");
    assert!(why.contains("op 1: the RDMA proxy failed"), "{why}");
}

#[test]
fn host_poison_keeps_the_kernels_reason() {
    let c = cfg(&[("ATLAS_RDMA_ONESHOT", "1")]).unwrap();
    let (_os, host) = region(region_bytes(c.max));
    let ctrl = host + ctrl_off(c.max);
    assert_eq!(poison_at(ctrl), None);
    poison_host(ctrl, POISON_LAUNCH);
    assert!(poison_at(ctrl).unwrap().contains("launch failed"));
    unsafe { ((ctrl + POISON) as *mut u64).write(POISON_TIMEOUT | 4) };
    poison_host(ctrl, POISON_PROXY | 9);
    assert!(
        poison_at(ctrl)
            .unwrap()
            .contains("op 4: the peer never arrived")
    );
}
