// SPDX-License-Identifier: AGPL-3.0-only
//! Two-rank microbench + stress for the TP2 decode collectives through the
//! serving `CommBackend` (`NcclBackend`), one process per Spark (start rank 1
//! first). The transport comes from the environment, exactly as in serving:
//!
//! * NCCL:     `ATLAS_RDMA_ALLREDUCE` unset
//! * RDMA:     `ATLAS_RDMA_ALLREDUCE=1`
//! * one-shot: `ATLAS_RDMA_ALLREDUCE=1 ATLAS_RDMA_ONESHOT=1`
//!
//! plus `ATLAS_RDMA_PAIR_CHAIN`, `ATLAS_RDMA_ONESHOT_STRIPE_MIN`, ... Graph
//! rows need the one-shot channel (`all_reduce_capturable`).
//!
//! Per payload of `rows x 4096` BF16: synced latency p10/p50/p90 (one
//! `all_reduce_async` + stream sync; rank 1 first sleeps `--skew-us`, so rank
//! 0 reports the early rank and rank 1 the late one), back-to-back mean, and
//! the per-call mean of a replayed graph of `--calls-per-graph` capturable
//! all-reduces. Correctness always runs first: small-integer payloads whose
//! BF16 sums are exact, checked after eager calls and after graph replays
//! interleaved with eager calls; `--stress N` repeats that N times with a
//! fresh payload per call. A routing pass then drives every entry point the
//! model uses (`all_reduce`, `all_reduce_async`, `exchange_async` add and
//! copy, `peer_exchange_async`, and the capturable pair in one graph) back to
//! back with one sync, at a one-shot size next to one above the capturable
//! limit, so the channels interleave on the stream. Exit 1 on any wrong
//! value, 2 on an error (without NCCL teardown: the peer may be gone), naming
//! the one-shot fault if the channel stopped.
//!
//! `--cmd-steps N` adds the start of N verify steps as the engine issues it:
//! rank 0 pauses `--cmd-pause-us` (its drafter pass, rank 1 waiting), sends
//! slot, command, width and tokens, and both ranks run the step's first
//! all-reduce. Rank 0 reports first word to landed collective: what the head
//! waits when the worker learns of the step late. The words take the host
//! command channel under `ATLAS_GLM_CMD_RDMA=1`, else a broadcast each.
//!
//! Run (per rank):
//!   comm_pair_bench --rank R --master 10.100.192.2 [--port 29610]
//!     [--rows 1,2,4,5,7,8,16,32] [--iters 2000] [--calls-per-graph 87]
//!     [--skew-us 0] [--stress 0] [--peer-exchange-bytes 128]
//!     [--cmd-steps 0] [--cmd-pause-us 7000]
//! Build: ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash
//!   ATLAS_TARGET_QUANT=nvfp4 cargo build --release -p spark-model
//!   --features cuda,gpu-examples,nccl --example comm_pair_bench

use anyhow::{Context, Result, bail, ensure};
use spark_comm::nccl_backend::{ALL_REDUCE_DTYPE_BYTES, required_recv_bytes};
use spark_comm::{CommBackend, NcclBackend};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};
use std::time::{Duration, Instant};

const H: usize = 4096;

struct Args {
    rank: usize,
    master: String,
    port: u16,
    rows: Vec<usize>,
    iters: usize,
    calls_per_graph: usize,
    skew_us: u64,
    stress: usize,
    peer_exchange_bytes: usize,
    cmd_steps: usize,
    cmd_pause_us: u64,
}

fn parse() -> Result<Args> {
    let mut a = Args {
        rank: usize::MAX,
        master: String::new(),
        port: 29610,
        rows: vec![1, 2, 4, 5, 7, 8, 16, 32],
        iters: 2000,
        calls_per_graph: 87,
        skew_us: 0,
        stress: 0,
        peer_exchange_bytes: 128,
        cmd_steps: 0,
        cmd_pause_us: 7000,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it.next().with_context(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--rank" => a.rank = v.parse()?,
            "--master" => a.master = v,
            "--port" => a.port = v.parse()?,
            "--rows" => a.rows = v.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--iters" => a.iters = v.parse()?,
            "--calls-per-graph" => a.calls_per_graph = v.parse()?,
            "--skew-us" => a.skew_us = v.parse()?,
            "--stress" => a.stress = v.parse()?,
            "--peer-exchange-bytes" => a.peer_exchange_bytes = v.parse()?,
            "--cmd-steps" => a.cmd_steps = v.parse()?,
            "--cmd-pause-us" => a.cmd_pause_us = v.parse()?,
            _ => bail!("unknown flag {flag}"),
        }
    }
    ensure!(
        a.rank < 2 && !a.master.is_empty(),
        "need --rank 0|1 and --master"
    );
    ensure!(a.peer_exchange_bytes.is_multiple_of(ALL_REDUCE_DTYPE_BYTES));
    Ok(a)
}

/// Small integers: every rank's value and every two-rank sum is exact in BF16.
fn value(rank: usize, seq: u64, i: usize) -> f32 {
    ((rank * 7 + seq as usize * 13 + i * 3) % 61) as f32 - 30.0
}

fn payload(rank: usize, seq: u64, n: usize, sum: bool) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let v = if sum {
                value(0, seq, i) + value(1, seq, i)
            } else {
                value(rank, seq, i)
            };
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

struct Bench<'a> {
    g: &'a dyn GpuBackend,
    comm: &'a NcclBackend,
    stream: u64,
    rank: usize,
    seq: u64,
}

impl Bench<'_> {
    fn fetch(&self, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; bytes];
        self.g.copy_d2h(p, &mut v)?;
        Ok(v)
    }

    /// Load the next payload into each buffer; returns their sequence numbers.
    fn prepare(&mut self, bufs: &[DevicePtr], bytes: usize) -> Result<Vec<u64>> {
        bufs.iter()
            .map(|&b| {
                self.seq += 1;
                self.g
                    .copy_h2d(&payload(self.rank, self.seq, bytes / 2, false), b)?;
                Ok(self.seq)
            })
            .collect()
    }

    /// Wrong bytes across `bufs` after one all-reduce each.
    fn wrong(&self, bufs: &[DevicePtr], seqs: &[u64], bytes: usize) -> Result<usize> {
        self.g.synchronize(self.stream)?;
        let mut bad = 0;
        for (&b, &s) in bufs.iter().zip(seqs) {
            let want = payload(self.rank, s, bytes / 2, true);
            bad += self
                .fetch(b, bytes)?
                .iter()
                .zip(&want)
                .filter(|(x, y)| x != y)
                .count();
        }
        Ok(bad)
    }

    /// Capture one capturable all-reduce per buffer; `None` when unavailable.
    fn capture(&self, bufs: &[DevicePtr], bytes: usize) -> Result<Option<GraphHandle>> {
        if self.comm.capturable_all_reduce_max_bytes() < bytes {
            return Ok(None);
        }
        self.g.begin_capture(self.stream)?;
        for b in bufs {
            if !self.comm.all_reduce_capturable(b.0, bytes, self.stream)? {
                self.g.abort_capture_if_active(self.stream);
                return Ok(None);
            }
        }
        Ok(Some(self.g.end_capture(self.stream)?))
    }

    /// Eager all-reduces, then graph replays interleaved with eager calls.
    fn check(&mut self, bytes: usize, rounds: usize) -> Result<usize> {
        let one = [self.g.alloc(bytes)?];
        let bufs: Vec<DevicePtr> = (0..8).map(|_| self.g.alloc(bytes)).collect::<Result<_>>()?;
        let mut bad = 0;
        let graph = self.capture(&bufs, bytes)?;
        for _ in 0..rounds {
            let s = self.prepare(&one, bytes)?;
            self.comm.all_reduce_async(one[0].0, bytes, self.stream)?;
            bad += self.wrong(&one, &s, bytes)?;
            if let Some(h) = graph {
                let s = self.prepare(&bufs, bytes)?;
                self.g.launch_graph(h, self.stream)?;
                bad += self.wrong(&bufs, &s, bytes)?;
            }
        }
        if let Some(h) = graph {
            self.g.destroy_graph(h)?;
        }
        Ok(bad)
    }

    /// Wrong bytes in `dst` against the peer's payload `seq`.
    fn wrong_copy(&self, dst: DevicePtr, seq: u64, bytes: usize) -> Result<usize> {
        let want = payload(1 - self.rank, seq, bytes / 2, false);
        let got = self.fetch(dst, bytes)?;
        Ok(got.iter().zip(&want).filter(|(x, y)| x != y).count())
    }

    /// Every entry point, back to back with one sync per round, per size:
    /// buffers 0-2 are all-reduced, 3 is sent and lands in 4 and 5.
    fn routing(&mut self, sizes: [usize; 2], rounds: usize) -> Result<usize> {
        let (comm, st, mut bad) = (self.comm, self.stream, 0);
        let mut sets = Vec::new();
        for bytes in sizes {
            let b: Vec<DevicePtr> = (0..6).map(|_| self.g.alloc(bytes)).collect::<Result<_>>()?;
            sets.push((bytes, b));
        }
        let graph = if comm.capturable_all_reduce_max_bytes() >= sizes[0] {
            let (bytes, b) = &sets[0];
            self.g.begin_capture(st)?;
            ensure!(comm.all_reduce_capturable(b[0].0, *bytes, st)?);
            ensure!(comm.peer_exchange_capturable(b[3].0, b[5].0, *bytes, st)?);
            Some(self.g.end_capture(st)?)
        } else {
            None
        };
        for _ in 0..rounds {
            let mut seqs = Vec::new();
            for (bytes, b) in &sets {
                seqs.push(self.prepare(&b[..4], *bytes)?);
            }
            for (bytes, b) in &sets {
                let bytes = *bytes;
                comm.all_reduce(b[0].0, bytes)?;
                comm.all_reduce_async(b[1].0, bytes, st)?;
                if !comm.exchange_async(b[2].0, b[2].0, bytes, true, st)? {
                    comm.all_reduce_async(b[2].0, bytes, st)?;
                }
                if !comm.exchange_async(b[3].0, b[4].0, bytes, false, st)? {
                    comm.peer_exchange_async(b[3].0, b[4].0, bytes, st)?;
                }
                comm.peer_exchange_async(b[3].0, b[5].0, bytes, st)?;
            }
            for ((bytes, b), s) in sets.iter().zip(&seqs) {
                bad += self.wrong(&b[..3], &s[..3], *bytes)?;
                bad += self.wrong_copy(b[4], s[3], *bytes)?;
                bad += self.wrong_copy(b[5], s[3], *bytes)?;
            }
            if let Some(h) = graph {
                let (bytes, b) = &sets[0];
                let s = self.prepare(&[b[0], b[3]], *bytes)?;
                self.g.launch_graph(h, st)?;
                bad += self.wrong(&b[..1], &s[..1], *bytes)?;
                bad += self.wrong_copy(b[5], s[1], *bytes)?;
            }
        }
        Ok(bad)
    }

    fn peer_exchange_check(&mut self, bytes: usize) -> Result<usize> {
        let (send, recv) = (self.g.alloc(bytes)?, self.g.alloc(bytes)?);
        let mut bad = 0;
        for _ in 0..16 {
            let s = self.prepare(&[send], bytes)?[0];
            self.comm
                .peer_exchange_async(send.0, recv.0, bytes, self.stream)?;
            self.g.synchronize(self.stream)?;
            bad += self.wrong_copy(recv, s, bytes)?;
        }
        Ok(bad)
    }

    fn barrier(&self) -> Result<()> {
        self.comm.barrier()?;
        self.g.synchronize(self.stream)
    }

    /// Synced per-call latency percentiles (us) of `op`.
    fn latency(&self, iters: usize, skew_us: u64, op: &dyn Fn() -> Result<()>) -> Result<[f64; 3]> {
        self.barrier()?;
        let mut t = Vec::with_capacity(iters);
        for _ in 0..iters {
            if self.rank == 1 && skew_us > 0 {
                std::thread::sleep(Duration::from_micros(skew_us));
            }
            let t0 = Instant::now();
            op()?;
            self.g.synchronize(self.stream)?;
            t.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        t.sort_by(f64::total_cmp);
        Ok([t[iters / 10], t[iters / 2], t[iters * 9 / 10]])
    }

    /// One command message the way the engine moves it (`ep_broadcast_u32`,
    /// `ep_broadcast_tokens`, the worker's idle first word): rank 0 sends
    /// `words`, rank 1 receives as many. Returns the message on both.
    fn command(&self, buf: DevicePtr, words: &[u32], idle: bool) -> Result<Vec<u32>> {
        let (g, comm, n) = (self.g, self.comm, words.len());
        if n <= comm.command_words_max() {
            let mut got = words.to_vec();
            if self.rank == 0 {
                g.synchronize(self.stream)?;
                comm.send_command_words(words)?;
            } else {
                comm.recv_command_words(&mut got)?;
                g.synchronize(self.stream)?;
            }
            return Ok(got);
        }
        let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        if self.rank == 0 {
            g.copy_h2d(&bytes, buf)?;
            comm.broadcast(buf.0, 4 * n, 0)?;
            return Ok(words.to_vec());
        }
        if idle {
            comm.receive_idle_command_word(buf.0)?;
        } else {
            comm.broadcast(buf.0, 4 * n, 0)?;
        }
        g.synchronize(self.stream)?;
        g.copy_d2h(buf, &mut bytes)?;
        let word = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        Ok(bytes.chunks_exact(4).map(word).collect())
    }

    /// `steps` verify-step starts (see the module doc): us from rank 0's
    /// first word to its first collective landing (p10/p50/p90), and the
    /// wrong words rank 1 read.
    fn command_steps(
        &mut self,
        steps: usize,
        pause_us: u64,
        bytes: usize,
    ) -> Result<([f64; 3], usize)> {
        let (cmd, ar) = (self.g.alloc(64)?, self.g.alloc(bytes)?);
        let (mut t, mut bad) = (Vec::with_capacity(steps), 0);
        self.barrier()?;
        for step in 0..steps as u32 {
            let k = 2 + step % 7;
            let tokens: Vec<u32> = (0..k).map(|i| step * 31 + i).collect();
            if self.rank == 0 {
                let until = Instant::now() + Duration::from_micros(pause_us);
                while Instant::now() < until {
                    std::hint::spin_loop();
                }
            }
            let t0 = Instant::now();
            let mut got = Vec::new();
            for (i, word) in [step % 4, 0xffff_fff5, k].into_iter().enumerate() {
                got.extend(self.command(cmd, &[word], i == 0)?);
            }
            got.extend(self.command(cmd, &tokens, false)?);
            self.comm.all_reduce_async(ar.0, bytes, self.stream)?;
            self.g.synchronize(self.stream)?;
            t.push(t0.elapsed().as_secs_f64() * 1e6);
            bad += usize::from(got[..3] != [step % 4, 0xffff_fff5, k] || got[3..] != tokens[..]);
        }
        t.sort_by(f64::total_cmp);
        Ok(([t[steps / 10], t[steps / 2], t[steps * 9 / 10]], bad))
    }

    /// Mean us per call of `calls` back-to-back calls of `op`.
    fn throughput(&self, iters: usize, calls: usize, op: &dyn Fn() -> Result<()>) -> Result<f64> {
        self.barrier()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            op()?;
        }
        self.g.synchronize(self.stream)?;
        Ok(t0.elapsed().as_secs_f64() * 1e6 / (iters * calls) as f64)
    }
}

fn main() -> Result<()> {
    let a = parse()?;
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let stream = g.default_stream();
    let max_rows = a.rows.iter().copied().max().unwrap_or(1) * 5;
    let recv = required_recv_bytes(max_rows, H, ALL_REDUCE_DTYPE_BYTES)?;
    let comm = NcclBackend::new(a.rank, 2, &a.master, a.port, stream, recv)?;
    comm.set_add_kernel(g.kernel("bf16_add", "bf16_add_inplace")?.0);
    if let Ok(k) = g.kernel("rdma_oneshot", "rdma_oneshot_bf16") {
        comm.set_oneshot_kernel(k.0);
    }
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "-".into());
    println!(
        "rank {}: ATLAS_RDMA_ALLREDUCE={} ATLAS_RDMA_ONESHOT={} CHAIN={} STRIPE_MIN={}; capturable up to {} B",
        a.rank,
        env("ATLAS_RDMA_ALLREDUCE"),
        env("ATLAS_RDMA_ONESHOT"),
        env("ATLAS_RDMA_PAIR_CHAIN"),
        env("ATLAS_RDMA_ONESHOT_STRIPE_MIN"),
        comm.capturable_all_reduce_max_bytes()
    );
    // No NCCL teardown on either exit: after a fault the peer may be gone.
    match run(&a, g, &comm, stream, max_rows * H * 2) {
        Ok(bad) => std::process::exit(i32::from(bad != 0)),
        Err(e) => {
            let fault = comm.oneshot_poisoned();
            eprintln!("rank {}: FAILED: {e:#}; one-shot fault: {fault:?}", a.rank);
            std::process::exit(2)
        }
    }
}

/// Wrong bytes over every check; `large` is the biggest payload the backend
/// was sized for.
fn run(
    a: &Args,
    g: &dyn GpuBackend,
    comm: &NcclBackend,
    stream: u64,
    large: usize,
) -> Result<usize> {
    let mut b = Bench {
        g,
        comm,
        stream,
        rank: a.rank,
        seq: 0,
    };
    let mut bad = 0;
    for &rows in &a.rows {
        let bytes = rows * H * 2;
        let wrong = b.check(bytes, 8.max(a.stress))?;
        bad += wrong;
        let buf = g.alloc(bytes)?.0;
        let eager = || comm.all_reduce_async(buf, bytes, stream);
        let [p10, p50, p90] = b.latency(a.iters, a.skew_us, &eager)?;
        let mean = b.throughput(a.iters, 1, &eager)?;
        let graph = match b.capture(&vec![DevicePtr(buf); a.calls_per_graph], bytes)? {
            Some(h) => {
                let replay = || g.launch_graph(h, stream);
                let per_call = b.throughput(
                    a.iters / a.calls_per_graph.max(1) + 1,
                    a.calls_per_graph,
                    &replay,
                )?;
                g.destroy_graph(h)?;
                format!("{per_call:6.2}")
            }
            None => "   n/a".into(),
        };
        println!(
            "rank {} rows {rows:2} ({bytes:6} B): synced p10/p50/p90 {p10:6.1}/{p50:6.1}/{p90:6.1} us, \
             back-to-back {mean:6.2} us, graph {graph} us/call, wrong {wrong}",
            a.rank
        );
    }
    let pe = a.peer_exchange_bytes;
    let wrong = b.peer_exchange_check(pe)?;
    bad += wrong;
    let (send, recv) = (g.alloc(pe)?.0, g.alloc(pe)?.0);
    let [p10, p50, p90] = b.latency(a.iters, a.skew_us, &|| {
        comm.peer_exchange_async(send, recv, pe, stream)
    })?;
    println!(
        "rank {} peer exchange {pe} B: synced p10/p50/p90 {p10:.1}/{p50:.1}/{p90:.1} us, wrong {wrong}",
        a.rank
    );
    let small = a.rows[0] * H * 2;
    let wrong = b.routing([small, large], 8 + a.stress / 16)?;
    bad += wrong;
    println!(
        "rank {} routing ({small} B next to {large} B, every entry point): wrong {wrong}",
        a.rank
    );
    if a.cmd_steps > 0 {
        let (pause, words) = (a.cmd_pause_us, comm.command_words_max());
        let ([p10, p50, p90], wrong) = b.command_steps(a.cmd_steps, pause, small)?;
        bad += wrong;
        println!(
            "rank {} step words ({}) + first collective after a {pause} us pause: \
             p10/p50/p90 {p10:.1}/{p50:.1}/{p90:.1} us, wrong {wrong}",
            a.rank,
            if words > 0 {
                "host channel"
            } else {
                "broadcasts"
            }
        );
    }
    println!(
        "rank {}: {} ({bad} wrong bytes)",
        a.rank,
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    b.barrier()?;
    Ok(bad)
}
