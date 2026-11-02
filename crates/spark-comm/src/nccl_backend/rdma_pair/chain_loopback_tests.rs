// SPDX-License-Identifier: AGPL-3.0-only

//! A chain of four ranks through the real NIC on one host (GPU + RDMA
//! device; `#[ignore]`d): four [`RdmaChain`]s in one process, one stream each
//! on one GPU, bootstrapped through rank 0 over localhost, every link's QPs
//! meeting inside the HCA. It runs the real chain proxies (own sends,
//! forwards, completions reaped as they come) and the chain kernel. Every
//! all-reduce must equal the pairwise tree of `bf16_add_inplace` on the four
//! payloads, (x0 + x1) + (x2 + x3), bit for bit on every rank; every
//! broadcast the root's payload; every gather each rank's part in its place
//! (in sequence-parallel prefill's order: rank 0's part on top); every
//! reduce-scatter the all-reduce's words in each rank's own part (SP's
//! order), interleaved with the flood ops. Eager ops, graph replays, ops above the
//! one-shot maximum, and back-to-back ops with one rank held up.
//!
//!   ATLAS_ONESHOT_CUBIN_DIR=<cubins> ATLAS_RDMA_ONESHOT=1 ATLAS_RDMA_PAIR_CHAIN=1 \
//!   ATLAS_RDMA_LEFT_RAILS=rocep1s0f1,roceP2p1s0f1 ATLAS_RDMA_RIGHT_RAILS=rocep1s0f1,roceP2p1s0f1 \
//!   [ATLAS_RDMA_ONESHOT_SM_STAGE=1 ATLAS_RDMA_ONESHOT_PDL=0] \
//!     cargo test -p spark-comm chain_loopback -- --ignored --nocapture
//!
//! With SM staging keep PDL off here: four ranks share one GPU, and under
//! PDL each kernel lets its successor be scheduled before it waits, so a
//! replayed graph's later kernels can fill the GPU before every rank's
//! current one is resident (one rank per GPU, as in serving, cannot). For
//! the same reason the reduce-scatter's grid is 16 blocks under test
//! (`oneshot::RS_BLOCKS`): four ranks of 48 would not all be resident.

use super::*;
use crate::nccl_backend::rdma_pair::bootstrap::chain::Rendezvous;
use crate::nccl_backend::rdma_pair::chain::RdmaChain;

const RANKS: usize = 4;

/// Rank `rank`'s payload for op `k`: finite BF16 values (no NaN or inf, so
/// the comparison is by bits), every fifth op small integers.
fn payload(rank: usize, k: u64, n: usize) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let x = mix((k << 34) ^ ((rank as u64) << 32) ^ i);
            if k.is_multiple_of(5) {
                f32::to_bits((x % 61) as f32 - 30.0).wrapping_shr(16) as u16
            } else {
                // Sign, an exponent below 0xff, any mantissa.
                (x as u16 & 0x807f) | (((x >> 16) % 0xfe) as u16) << 7
            }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Sum,
    Broadcast(usize),
    /// `bytes / RANKS` per rank, rank `r`'s at part `RANKS - 1 - r`
    /// (`false`) or part `r` (`true`).
    Gather(bool),
    /// `bytes / RANKS` per rank, rank `r`'s at part `RANKS - 1 - r`.
    ReduceScatter,
}

/// One collective issued by every rank, in place on per-rank buffers.
struct Step {
    bytes: usize,
    kind: Kind,
    buf: [u64; RANKS],
    want: [u64; RANKS],
}

impl Step {
    fn new(bytes: usize, kind: Kind, misalign: u64) -> Self {
        Self {
            bytes,
            kind,
            buf: [0; RANKS].map(|_: u64| alloc(bytes + 256) + misalign),
            want: [0; RANKS].map(|_: u64| alloc(bytes)),
        }
    }

    fn prepare(&self, g: &Gpu, k: u64) {
        for rank in 0..RANKS {
            upload(g, self.buf[rank], &payload(rank, k, self.bytes / 2));
        }
        ck!(cuStreamSynchronize(g.stream));
    }

    fn issue(&self, chains: &[RdmaChain], streams: &[u64; RANKS]) {
        for (rank, chain) in chains.iter().enumerate() {
            self.issue_on(chain, rank, streams[rank]);
        }
    }

    fn issue_on(&self, chain: &RdmaChain, rank: usize, stream: u64) {
        let ptr = self.buf[rank];
        let ok = match self.kind {
            Kind::Sum => chain.all_reduce(ptr, self.bytes, stream),
            Kind::Broadcast(root) => chain.oneshot().broadcast(ptr, self.bytes, root, stream),
            Kind::Gather(up) => {
                let part = self.bytes / RANKS;
                if up {
                    chain.gather(ptr, part as i64, part, stream)
                } else {
                    let top = ptr + ((RANKS - 1) * part) as u64;
                    chain.gather(top, -(part as i64), part, stream)
                }
            }
            Kind::ReduceScatter => {
                let part = self.bytes / RANKS;
                let top = ptr + ((RANKS - 1) * part) as u64;
                chain.reduce_scatter(top, -(part as i64), part, false, stream)
            }
        };
        assert!(ok.unwrap());
    }

    /// Wrong values over all ranks against the reference for op `k`.
    fn check(&self, g: &Gpu, k: u64) -> usize {
        let n = self.bytes / 2;
        let want = match self.kind {
            Kind::Broadcast(root) => payload(root, k, n),
            Kind::Gather(up) => {
                let part = n / RANKS;
                let origin = |at: usize| if up { at } else { RANKS - 1 - at };
                (0..RANKS)
                    .flat_map(|at| payload(origin(at), k, n)[at * part..][..part].to_vec())
                    .collect()
            }
            Kind::Sum | Kind::ReduceScatter => {
                for (rank, &w) in self.want.iter().enumerate() {
                    upload(g, w, &payload(rank, k, n));
                }
                let w = self.want;
                launch_add(g, g.stream, w[0], w[1], n);
                launch_add(g, g.stream, w[2], w[3], n);
                launch_add(g, g.stream, w[0], w[2], n);
                download(g, w[0], n)
            }
        };
        (0..RANKS)
            .map(|rank| {
                let got = download(g, self.buf[rank], n);
                // A reduce-scatter leaves only this rank's own part.
                let (from, len) = if self.kind == Kind::ReduceScatter {
                    ((RANKS - 1 - rank) * n / RANKS, n / RANKS)
                } else {
                    (0, n)
                };
                got[from..][..len]
                    .iter()
                    .zip(&want[from..])
                    .filter(|(a, b)| a != b)
                    .count()
            })
            .sum()
    }
}

/// Synchronize every rank's stream, naming each rank's one-shot fault if
/// one failed.
fn sync(streams: &[u64; RANKS], chains: &[RdmaChain]) {
    for &s in streams {
        let status = unsafe { cuStreamSynchronize(s) };
        if status != 0 {
            let why: Vec<_> = chains.iter().map(|c| c.oneshot().poisoned()).collect();
            panic!("stream failed with CUDA status {status}; one-shot faults by rank: {why:?}");
        }
    }
}

/// Four chain ranks on this GPU through the NIC, or `None` (skipped).
fn ranks() -> Option<(Gpu, Vec<RdmaChain>)> {
    let g = gpu()?;
    if std::env::var("ATLAS_RDMA_ONESHOT").as_deref() != Ok("1") {
        eprintln!("ATLAS_RDMA_ONESHOT unset: skipped");
        return None;
    }
    let chain_kernel = load("rdma_oneshot.cubin", "rdma_chain_bf16");
    let (ctx, port) = (g.ctx, 20_000 + (std::process::id() % 20_000) as u16);
    let chains: Vec<RdmaChain> = std::thread::scope(|s| {
        let ranks: Vec<_> = (0..RANKS)
            .map(|rank| {
                s.spawn(move || {
                    ck!(cuCtxSetCurrent(ctx));
                    let at = Rendezvous {
                        at: (std::net::Ipv4Addr::LOCALHOST, port).into(),
                        workers: vec![std::net::Ipv4Addr::LOCALHOST.into()],
                        lifelines: Vec::new(),
                    };
                    RdmaChain::connect(rank, RANKS, &at).unwrap()
                })
            })
            .collect();
        ranks.into_iter().map(|r| r.join().unwrap()).collect()
    });
    for chain in &chains {
        for oneshot in chain.channels() {
            oneshot.set_kernel(chain_kernel);
        }
    }
    Some((g, chains))
}

#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR"]
fn chain_loopback_four_ranks() {
    let Some((g, chains)) = ranks() else {
        return;
    };
    let max = chains[0].oneshot().max_bytes();
    let streams = [g.stream, new_stream(), new_stream(), new_stream()];
    let steps = [
        Step::new(65_536, Kind::Sum, 0),
        Step::new(8192, Kind::Sum, 2),
        Step::new(131_072, Kind::Sum, 0),
        Step::new(18, Kind::Sum, 2),
        Step::new(4, Kind::Broadcast(0), 0),
        Step::new(327_680, Kind::Sum, 0),
        Step::new(max, Kind::Sum, 0),
        Step::new(4096, Kind::Broadcast(3), 0),
        Step::new(2 * max + 4096, Kind::Sum, 0),
        Step::new(24_576, Kind::Sum, 0),
        Step::new(12, Kind::Broadcast(2), 2),
        Step::new(32_768, Kind::Broadcast(1), 0),
        Step::new(4 * 8192, Kind::Gather(false), 0),
        Step::new(4 * 18, Kind::Gather(false), 2),
        Step::new(4 * (2 * max + 4096), Kind::Gather(false), 0),
        Step::new(4 * 8192, Kind::Gather(true), 2),
        Step::new(4 * (max + 64), Kind::Gather(true), 0),
        Step::new(4 * 8192, Kind::ReduceScatter, 0),
        Step::new(4 * 128, Kind::ReduceScatter, 0),
        Step::new(4 * (2 * max + 4096), Kind::ReduceScatter, 0),
        Step::new(65_536, Kind::Sum, 0),
        Step::new(4 * 65_536, Kind::ReduceScatter, 0),
    ];
    let rounds = std::env::var("ATLAS_ONESHOT_LOOPBACK_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let (mut k, mut wrong) = (0u64, 0usize);
    let eager = |step: &Step, k: &mut u64| {
        *k += 1;
        step.prepare(&g, *k);
        step.issue(&chains, &streams);
        sync(&streams, &chains);
        step.check(&g, *k)
    };
    for _ in 0..rounds {
        for step in &steps {
            wrong += eager(step, &mut k);
        }
    }
    let execs = [0, 1, 2, 3].map(|rank| {
        capture(
            &Gpu {
                stream: streams[rank],
                ..g
            },
            || {
                for step in &steps {
                    step.issue_on(&chains[rank], rank, streams[rank]);
                }
            },
        )
    });
    for replay in 0..rounds {
        let base = k;
        for (i, step) in steps.iter().enumerate() {
            step.prepare(&g, base + 1 + i as u64);
        }
        for rank in 0..RANKS {
            ck!(cuGraphLaunch(execs[rank], streams[rank]));
        }
        sync(&streams, &chains);
        k += steps.len() as u64;
        for (i, step) in steps.iter().enumerate() {
            wrong += step.check(&g, base + 1 + i as u64);
        }
        wrong += eager(&steps[replay % steps.len()], &mut k);
    }
    // Back to back, one sync per round, one rank's stream held up before
    // every step (the rank rotates, the length varies): the others run
    // ahead, their payloads wait at the proxies, slots are reused with no
    // host sync between.
    let drag = alloc(8 << 20);
    for round in 0..rounds {
        let base = k;
        for (i, step) in steps.iter().enumerate() {
            step.prepare(&g, base + 1 + i as u64);
        }
        for (i, step) in steps.iter().enumerate() {
            let n = (4 << 20) >> ((i + round) % 8);
            launch_add(&g, streams[(i + round) % RANKS], drag, drag, n);
            step.issue(&chains, &streams);
        }
        sync(&streams, &chains);
        k += steps.len() as u64;
        for (i, step) in steps.iter().enumerate() {
            wrong += step.check(&g, base + 1 + i as u64);
        }
    }
    // `ATLAS_RDMA_ONESHOT_CHANNELS=2`: every rank runs the steps on channel
    // 0 on its stream and, at once, on channel 1 on a second stream (the
    // order differs: channel 1 runs them backwards), with no sync between.
    if chains[0].channels().len() > 1 {
        let second = [0; RANKS].map(|_| new_stream());
        let other: Vec<Step> = steps
            .iter()
            .rev()
            .map(|s| Step::new(s.bytes, s.kind, 0))
            .collect();
        for _ in 0..rounds {
            let base = k;
            for (i, step) in steps.iter().chain(&other).enumerate() {
                step.prepare(&g, base + 1 + i as u64);
            }
            for rank in 0..RANKS {
                for (a, b) in steps.iter().zip(&other) {
                    a.issue_on(&chains[rank], rank, streams[rank]);
                    let _one = crate::chain_channel::enter(1);
                    b.issue_on(&chains[rank], rank, second[rank]);
                }
            }
            sync(&streams, &chains);
            sync(&second, &chains);
            for (i, step) in steps.iter().chain(&other).enumerate() {
                wrong += step.check(&g, base + 1 + i as u64);
            }
            k += 2 * steps.len() as u64;
        }
        // Back to back on both channels at once, no sync: the large ops
        // (split all-reduce, pipelined gather and reduce-scatter) of each
        // channel interleave with the other's. A gather in place repeats
        // exactly; the others only must not fault.
        let big: Vec<Step> = [Kind::Sum, Kind::Gather(false), Kind::ReduceScatter]
            .map(|kind| Step::new(4 << 20, kind, 0))
            .into_iter()
            .collect();
        let gathers = [
            Step::new(4 << 20, Kind::Gather(false), 0),
            Step::new(4 << 20, Kind::Gather(false), 0),
        ];
        k += 1;
        gathers.iter().for_each(|g2| g2.prepare(&g, k));
        for _ in 0..200 {
            for rank in 0..RANKS {
                for step in &big {
                    step.issue_on(&chains[rank], rank, streams[rank]);
                }
                gathers[0].issue_on(&chains[rank], rank, streams[rank]);
                let _one = crate::chain_channel::enter(1);
                for step in big.iter().rev() {
                    step.issue_on(&chains[rank], rank, second[rank]);
                }
                gathers[1].issue_on(&chains[rank], rank, second[rank]);
            }
        }
        sync(&streams, &chains);
        sync(&second, &chains);
        wrong += gathers.iter().map(|g2| g2.check(&g, k)).sum::<usize>();
        println!(
            "two channels: {rounds} rounds of both step lists at once, 200 back-to-back rounds"
        );
    }
    let mut timing = String::new();
    for (bytes, kind) in [
        (8192, Kind::Sum),
        (32_768, Kind::Sum),
        (65_536, Kind::Sum),
        (4 << 20, Kind::Sum),
        (4 << 20, Kind::Gather(false)),
        (4 << 20, Kind::ReduceScatter),
    ] {
        let step = Step::new(bytes, kind, 0);
        step.issue(&chains, &streams);
        sync(&streams, &chains);
        let t0 = Instant::now();
        for _ in 0..1000 {
            step.issue(&chains, &streams);
        }
        sync(&streams, &chains);
        let what = match kind {
            Kind::Gather(_) => "gather",
            Kind::ReduceScatter => "reduce-scatter",
            _ => "sum",
        };
        timing += &format!(" {what} {bytes}B {:.1}us", t0.elapsed().as_secs_f64() * 1e3);
    }
    let poison: Vec<_> = chains.iter().map(|c| c.oneshot().poisoned()).collect();
    println!(
        "chain NIC loopback: {k} steps, {wrong} wrong values, poison {poison:?}; back-to-back per op:{timing}"
    );
    assert_eq!(wrong, 0);
    assert!(poison.iter().all(Option::is_none));
}

#[path = "chain_loopback_ops_tests.rs"]
mod ops;
