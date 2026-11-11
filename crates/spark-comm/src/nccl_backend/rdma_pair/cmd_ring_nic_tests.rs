// SPDX-License-Identifier: AGPL-3.0-only

//! The command ring through the real NIC on one host (`#[ignore]`d: needs a
//! GPU for the pinned region and an RDMA device). Two [`RdmaPair`]s in one
//! process, bootstrapped over localhost, each rail's two QPs meeting inside
//! the HCA, served by the real proxy loops. One test sends the head's
//! step-shaped words while both ranks run legacy exchanges on their streams
//! (one proxy loop, one CQ) and checks both; the other reports the
//! send-to-receive latency of a verify step's words back to back, after a
//! drafter-sized pause, and after a pause long enough for the proxy and the
//! receiver to nap.
//!
//! ```text
//! ATLAS_GLM_CMD_RDMA=1 ATLAS_RDMA_RAILS=rocep1s0f0[,roceP2p1s0f0] \
//!   cargo test -p spark-comm cmd_ring_nic -- --ignored --nocapture
//! ```
//!
//! (in a container: --gpus all --device /dev/infiniband --cap-add IPC_LOCK
//! --ulimit memlock=-1:-1)

use super::*;
use std::time::{Duration, Instant};

unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuDevicePrimaryCtxRetain(ctx: *mut u64, dev: i32) -> i32;
    fn cuCtxSetCurrent(ctx: u64) -> i32;
    fn cuStreamCreate(stream: *mut u64, flags: u32) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> i32;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: u64) -> i32;
}

/// A driver call that must succeed.
macro_rules! ck {
    ($call:expr) => {
        cu(unsafe { $call }, stringify!($call)).unwrap()
    };
}

/// The CUDA context and both ranks' pairs, connected through the NIC, or
/// `None` (test skipped) without `ATLAS_GLM_CMD_RDMA=1`.
fn ranks() -> Option<(u64, Vec<RdmaPair>)> {
    if !cmd_ring::requested() {
        eprintln!("ATLAS_GLM_CMD_RDMA unset: skipped");
        return None;
    }
    let mut ctx = 0u64;
    ck!(cuInit(0));
    ck!(cuDevicePrimaryCtxRetain(&mut ctx, 0));
    ck!(cuCtxSetCurrent(ctx));
    let port = 20_000 + (std::process::id() % 20_000) as u16;
    let pairs = std::thread::scope(|s| {
        let ranks: Vec<_> = (0..2)
            .map(|rank| {
                s.spawn(move || {
                    ck!(cuCtxSetCurrent(ctx));
                    let link = bootstrap::Link {
                        at: (std::net::Ipv4Addr::LOCALHOST, port).into(),
                        worker: None,
                        lifeline: None,
                    };
                    RdmaPair::connect(rank, &link, 1 << 20).unwrap()
                })
            })
            .collect();
        ranks.into_iter().map(|r| r.join().unwrap()).collect()
    });
    Some((ctx, pairs))
}

fn mix(x: u64) -> u32 {
    (x.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 29) as u32
}

/// Word `i` of verify step `step`: slot, command, width, then the tokens.
fn step_words(step: u64) -> Vec<u32> {
    let k = 2 + (mix(step) % 7) as usize;
    let mut words = vec![step as u32 % 4, 0xffff_fff5, k as u32];
    words.extend((0..k as u64).map(|i| mix(step << 8 | i)));
    words
}

/// Send one step as the head does: slot, command, width, tokens, verdict.
fn send_step(head: &cmd_ring::CmdRing, step: u64) {
    let words = step_words(step);
    for word in &words[..3] {
        head.send(&[*word]).unwrap();
    }
    head.send(&words[3..]).unwrap();
    head.send(&[mix(!step) % 8]).unwrap();
}

/// Receive one step as the worker does and check every word.
fn recv_step(worker: &cmd_ring::CmdRing, step: u64) {
    let want = step_words(step);
    let mut head = [0u32; 3];
    for word in &mut head {
        worker.recv(std::slice::from_mut(word)).unwrap();
    }
    assert_eq!(head, want[..3], "step {step}");
    let mut tokens = vec![0u32; head[2] as usize];
    worker.recv(&mut tokens).unwrap();
    assert_eq!(tokens, want[3..], "step {step}");
    let mut verdict = [0u32];
    worker.recv(&mut verdict).unwrap();
    assert_eq!(verdict[0], mix(!step) % 8, "step {step}");
}

#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_GLM_CMD_RDMA=1"]
fn cmd_ring_nic_words_arrive_beside_legacy_exchanges() {
    let Some((ctx, pairs)) = ranks() else {
        return;
    };
    const BYTES: usize = 65_536;
    const STEPS_PER_ROUND: u64 = 40;
    let rounds: u64 = std::env::var("ATLAS_CMD_RING_NIC_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let (head, worker) = (pairs[0].cmd().unwrap(), pairs[1].cmd().unwrap());
    let bufs = [0, 1].map(|_| {
        let (mut src, mut dst, mut stream) = (0u64, 0u64, 0u64);
        ck!(cuMemAlloc_v2(&mut src, BYTES));
        ck!(cuMemAlloc_v2(&mut dst, BYTES));
        ck!(cuStreamCreate(&mut stream, 1));
        (src, dst, stream)
    });
    let payload = |rank: u64, round: u64| -> Vec<u32> {
        (0..BYTES as u64 / 4)
            .map(|i| mix(round << 20 | i << 1 | rank))
            .collect()
    };
    let mut wrong = 0usize;
    std::thread::scope(|s| {
        let receiver = s.spawn(|| {
            for step in 0..rounds * STEPS_PER_ROUND {
                recv_step(worker, step);
            }
        });
        for round in 0..rounds {
            // On the rank's own stream, and synced: a small upload from
            // pageable memory is otherwise still in flight when the call
            // returns, and nothing orders the exchange's staging copy on
            // another stream behind it.
            for (rank, &(src, _, stream)) in bufs.iter().enumerate() {
                let data = payload(rank as u64, round);
                ck!(cuMemcpyHtoDAsync_v2(
                    src,
                    data.as_ptr().cast(),
                    BYTES,
                    stream
                ));
                ck!(cuStreamSynchronize(stream));
            }
            // Words before, between and after the two ranks' exchange of
            // this round: the proxy serves them while the job waits on the
            // GPU and while its segment is on the wire.
            for step in 0..STEPS_PER_ROUND {
                if step == STEPS_PER_ROUND / 4 {
                    for (pair, &(src, dst, stream)) in pairs.iter().zip(&bufs) {
                        let land = |d, s, len| copy_async(d, s, len, stream);
                        assert!(pair.exchange(src, dst, BYTES, stream, land).unwrap());
                    }
                }
                send_step(head, round * STEPS_PER_ROUND + step);
            }
            for (rank, &(_, dst, stream)) in bufs.iter().enumerate() {
                let mut got = vec![0u32; BYTES / 4];
                ck!(cuMemcpyDtoHAsync_v2(
                    got.as_mut_ptr().cast(),
                    dst,
                    BYTES,
                    stream
                ));
                ck!(cuStreamSynchronize(stream));
                let want = payload(1 - rank as u64, round);
                wrong += got.iter().zip(&want).filter(|(a, b)| a != b).count();
            }
        }
        receiver.join().unwrap();
    });
    let _ = ctx;
    println!(
        "command ring NIC loopback: {} steps ({} words) in order beside {rounds} legacy exchanges per rank, {wrong} wrong exchange words",
        rounds * STEPS_PER_ROUND,
        (0..rounds * STEPS_PER_ROUND)
            .map(|s| step_words(s).len() + 1)
            .sum::<usize>()
    );
    assert_eq!(wrong, 0);
}

#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_GLM_CMD_RDMA=1"]
fn cmd_ring_nic_step_latency() {
    let Some((_, pairs)) = ranks() else {
        return;
    };
    let (head, worker) = (pairs[0].cmd().unwrap(), pairs[1].cmd().unwrap());
    let base = Instant::now();
    let now = || base.elapsed().as_nanos() as u64;
    // Per pause before a step: steps, and the bound on the median latency
    // from the first send to the last word read.
    let cases = [
        (Duration::ZERO, 5000u64, 100_000u64),
        (Duration::from_millis(7), 300, 100_000),
        (Duration::from_millis(80), 25, 1_000_000),
    ];
    let total: u64 = cases.iter().map(|c| c.1).sum();
    let (sent_at, done) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut report = String::new();
    std::thread::scope(|s| {
        let receiver = s.spawn(|| {
            let mut lat = Vec::with_capacity(total as usize);
            for step in 0..total {
                recv_step(worker, step);
                lat.push(now() - sent_at.load(Ordering::Acquire));
                done.store(step + 1, Ordering::Release);
            }
            lat
        });
        let mut step = 0u64;
        for &(pause, steps, _) in &cases {
            for _ in 0..steps {
                while done.load(Ordering::Acquire) < step {
                    std::hint::spin_loop();
                }
                let until = Instant::now() + pause;
                while Instant::now() < until {
                    std::hint::spin_loop();
                }
                sent_at.store(now(), Ordering::Release);
                send_step(head, step);
                step += 1;
            }
        }
        let lat = receiver.join().unwrap();
        let mut at = 0usize;
        for &(pause, steps, bound) in &cases {
            let mut part = lat[at..at + steps as usize].to_vec();
            at += steps as usize;
            part.sort_unstable();
            let pct = |p: usize| part[(part.len() - 1) * p / 100] as f64 / 1e3;
            report += &format!(
                "\n  pause {pause:>6?}: {steps} steps, send -> last word read p50 {:.1} us p90 {:.1} us p99 {:.1} us max {:.1} us",
                pct(50),
                pct(90),
                pct(99),
                pct(100)
            );
            assert!(
                part[part.len() / 2] < bound,
                "median {} ns after a {pause:?} pause exceeds {bound} ns",
                part[part.len() / 2]
            );
        }
    });
    println!("command ring NIC step latency (5 messages a step):{report}");
}
