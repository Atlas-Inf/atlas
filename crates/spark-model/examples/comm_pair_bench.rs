// SPDX-License-Identifier: AGPL-3.0-only
//! Microbench + stress for the TP decode collectives through the serving
//! `CommBackend` (`NcclBackend`): two ranks (`--world 2`, the default, one
//! process per Spark; start rank 1 first) take the RDMA pair, three or four
//! the RDMA chain. The transport comes from the environment, exactly as in
//! serving:
//!
//! * NCCL:     `ATLAS_RDMA_ALLREDUCE` unset
//! * RDMA:     `ATLAS_RDMA_ALLREDUCE=1` (`ATLAS_RDMA_RAILS` for a pair,
//!   `ATLAS_RDMA_LEFT_RAILS` / `ATLAS_RDMA_RIGHT_RAILS` per rank for a chain)
//! * one-shot: `ATLAS_RDMA_ALLREDUCE=1 ATLAS_RDMA_ONESHOT=1`
//!
//! plus `ATLAS_RDMA_PAIR_CHAIN`, `ATLAS_RDMA_ONESHOT_STRIPE_MIN`, ... Graph
//! rows need the one-shot channel (`all_reduce_capturable`).
//!
//! Per payload of `rows x 4096` BF16: synced latency p10/p50/p90 (one
//! `all_reduce_async` + stream sync; the last rank first sleeps `--skew-us`,
//! so the others report early ranks and it a late one), back-to-back mean,
//! and the per-call mean of a replayed graph of `--calls-per-graph`
//! capturable all-reduces. Correctness always runs first: small-integer
//! payloads whose BF16 sums are exact, checked after eager calls and after
//! graph replays interleaved with eager calls (`--stress N` repeats that N
//! times with a fresh payload per call), then pseudo-random payloads whose
//! result's hash every rank must share (the ranks of a TP group must agree
//! bit for bit), then a broadcast from every root. A pair then checks its
//! peer exchange and a routing pass that drives every entry point the model
//! uses (`all_reduce`, `all_reduce_async`, `exchange_async` add and copy,
//! `peer_exchange_async`, and the capturable pair in one graph) back to back
//! with one sync, at a one-shot size next to one above the capturable limit,
//! so the channels interleave on the stream. Exit 1 on any wrong value, 2 on
//! an error (without NCCL teardown: a peer may be gone), naming the one-shot
//! fault if the channel stopped.
//!
//! `--gather-rows R` (a chain) checks sequence-parallel prefill's gather of R
//! rows per rank (`CommBackend::gather_async`, rank 0's part on top) and times
//! it, the chain's reduce-scatter of R rows per rank
//! (`CommBackend::reduce_scatter_async`, four ranks) and the all-reduce of
//! all `R x world` rows: a prefill seam's collectives.
//!
//! `--cmd-steps N` adds the start of N verify steps as the engine issues it:
//! rank 0 pauses `--cmd-pause-us` (its drafter pass, the workers waiting),
//! sends slot, command, width and tokens, and every rank runs the step's
//! first all-reduce. Rank 0 reports first word to landed collective: what
//! the head waits when a worker learns of the step late. The words take the
//! host command channel under `ATLAS_GLM_CMD_RDMA=1`, else a broadcast each.
//!
//! `--rank 0,1` runs several ranks in one process, a thread and a stream
//! each, sharing its CUDA context: ranks in different processes on one GPU
//! are time-sliced (about 2 ms per switch), so their spinning kernels cannot
//! meet (NCCL also needs `NCCL_MULTI_RANK_GPU_ENABLE=1` for that).
//!
//! Run (per process):
//!   comm_pair_bench --rank R[,R...] [--world 2] --master 192.0.2.2
//!     [--port 29610] [--rows 1,2,4,5,7,8,16,32] [--iters 2000]
//!     [--calls-per-graph 87] [--skew-us 0] [--stress 0]
//!     [--peer-exchange-bytes 128] [--gather-rows 0] [--cmd-steps 0] [--cmd-pause-us 7000]
//! Build: ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash
//!   ATLAS_TARGET_QUANT=nvfp4 cargo build --release -p spark-model
//!   --features cuda,gpu-examples,nccl --example comm_pair_bench

#[path = "comm_pair_bench/bench.rs"]
mod bench;

use anyhow::{Context, Result, bail, ensure};
use bench::Bench;
use spark_comm::nccl_backend::{ALL_REDUCE_DTYPE_BYTES, required_recv_bytes};
use spark_comm::{CommBackend, NcclBackend};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const H: usize = 4096;

struct Args {
    ranks: Vec<usize>,
    world: usize,
    master: String,
    port: u16,
    rows: Vec<usize>,
    iters: usize,
    calls_per_graph: usize,
    skew_us: u64,
    stress: usize,
    peer_exchange_bytes: usize,
    gather_rows: usize,
    cmd_steps: usize,
    cmd_pause_us: u64,
}

fn list(v: &str) -> Result<Vec<usize>> {
    Ok(v.split(',').map(str::parse).collect::<Result<_, _>>()?)
}

fn parse() -> Result<Args> {
    let mut a = Args {
        ranks: Vec::new(),
        world: 2,
        master: String::new(),
        port: 29610,
        rows: vec![1, 2, 4, 5, 7, 8, 16, 32],
        iters: 2000,
        calls_per_graph: 87,
        skew_us: 0,
        stress: 0,
        peer_exchange_bytes: 128,
        gather_rows: 0,
        cmd_steps: 0,
        cmd_pause_us: 7000,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it.next().with_context(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--rank" => a.ranks = list(&v)?,
            "--world" => a.world = v.parse()?,
            "--master" => a.master = v,
            "--port" => a.port = v.parse()?,
            "--rows" => a.rows = list(&v)?,
            "--iters" => a.iters = v.parse()?,
            "--calls-per-graph" => a.calls_per_graph = v.parse()?,
            "--skew-us" => a.skew_us = v.parse()?,
            "--stress" => a.stress = v.parse()?,
            "--peer-exchange-bytes" => a.peer_exchange_bytes = v.parse()?,
            "--gather-rows" => a.gather_rows = v.parse()?,
            "--cmd-steps" => a.cmd_steps = v.parse()?,
            "--cmd-pause-us" => a.cmd_pause_us = v.parse()?,
            _ => bail!("unknown flag {flag}"),
        }
    }
    ensure!(
        (2..=4).contains(&a.world)
            && !a.ranks.is_empty()
            && a.ranks.iter().all(|&r| r < a.world)
            && !a.master.is_empty(),
        "need --rank R[,R...] below --world (2..=4) and --master"
    );
    ensure!(a.peer_exchange_bytes.is_multiple_of(ALL_REDUCE_DTYPE_BYTES));
    Ok(a)
}

fn main() -> Result<()> {
    let a = &parse()?;
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let codes: Vec<i32> = std::thread::scope(|s| {
        let ranks: Vec<_> = a
            .ranks
            .iter()
            .map(|&rank| s.spawn(move || rank_main(a, g, rank)))
            .collect();
        ranks.into_iter().map(|r| r.join().unwrap_or(2)).collect()
    });
    // No NCCL teardown on any exit: after a fault a peer may be gone.
    std::process::exit(codes.into_iter().max().unwrap_or(0))
}

/// One rank on its own thread: 0 when every check passed, 1 on a wrong
/// value, 2 on an error.
fn rank_main(a: &Args, g: &dyn GpuBackend, rank: usize) -> i32 {
    let (comm, stream) = match connect(a, g, rank) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("rank {rank}: FAILED to connect: {e:#}");
            return 2;
        }
    };
    match run(a, g, &comm, stream, rank) {
        Ok(bad) => i32::from(bad != 0),
        Err(e) => {
            let fault = comm.oneshot_poisoned();
            eprintln!("rank {rank}: FAILED: {e:#}; one-shot fault: {fault:?}");
            2
        }
    }
}

/// The rank's communicator over its stream (the default one when it is the
/// process's only rank), with the kernels set.
fn connect(a: &Args, g: &dyn GpuBackend, rank: usize) -> Result<(NcclBackend, u64)> {
    g.bind_to_thread()?;
    let stream = if a.ranks.len() == 1 {
        g.default_stream()
    } else {
        g.create_stream()?
    };
    let max_rows = a.rows.iter().copied().max().unwrap_or(1) * 5;
    let recv = required_recv_bytes(max_rows, H, ALL_REDUCE_DTYPE_BYTES)?;
    let comm = NcclBackend::new(rank, a.world, &a.master, a.port, stream, recv)?;
    comm.set_add_kernel(g.kernel("bf16_add", "bf16_add_inplace")?.0);
    if let Ok(k) = g.kernel("rdma_oneshot", comm.oneshot_kernel()) {
        comm.set_oneshot_kernel(k.0);
    }
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "-".into());
    println!(
        "rank {rank}/{}: ATLAS_RDMA_ALLREDUCE={} ATLAS_RDMA_ONESHOT={} CHAIN={} STRIPE_MIN={} \
         SM_STAGE={} CMD_RDMA={}; kernel {}; capturable up to {} B",
        a.world,
        env("ATLAS_RDMA_ALLREDUCE"),
        env("ATLAS_RDMA_ONESHOT"),
        env("ATLAS_RDMA_PAIR_CHAIN"),
        env("ATLAS_RDMA_ONESHOT_STRIPE_MIN"),
        env("ATLAS_RDMA_ONESHOT_SM_STAGE"),
        env("ATLAS_GLM_CMD_RDMA"),
        comm.oneshot_kernel(),
        comm.capturable_all_reduce_max_bytes()
    );
    Ok((comm, stream))
}

/// Wrong values over every check.
fn run(
    a: &Args,
    g: &dyn GpuBackend,
    comm: &NcclBackend,
    stream: u64,
    rank: usize,
) -> Result<usize> {
    let mut b = Bench {
        g,
        comm,
        stream,
        rank,
        world: a.world,
        seq: 0,
    };
    let mut bad = 0;
    for &rows in &a.rows {
        let bytes = rows * H * 2;
        let wrong = b.check(bytes, 8.max(a.stress))?;
        let differ = b.identity(bytes, 8.max(a.stress / 4))?;
        bad += wrong + differ;
        let buf = g.alloc(bytes)?.0;
        let eager = || comm.all_reduce_async(buf, bytes, stream);
        let [p10, p50, p90] = b.latency(a.iters, a.skew_us, &eager)?;
        let mean = b.throughput(a.iters, 1, &eager)?;
        let graph = match b.capture(&vec![DevicePtr(buf); a.calls_per_graph], bytes)? {
            Some(h) => {
                let replay = || g.launch_graph(h, stream);
                let iters = a.iters / a.calls_per_graph.max(1) + 1;
                let per_call = b.throughput(iters, a.calls_per_graph, &replay)?;
                g.destroy_graph(h)?;
                format!("{per_call:6.2}")
            }
            None => "   n/a".into(),
        };
        println!(
            "rank {rank} rows {rows:2} ({bytes:6} B): synced p10/p50/p90 {p10:6.1}/{p50:6.1}/{p90:6.1} us, \
             back-to-back {mean:6.2} us, graph {graph} us/call, wrong {wrong}, hash mismatches {differ}"
        );
    }
    for bytes in [4, 4096] {
        let wrong = b.broadcast_check(bytes)?;
        bad += wrong;
        let buf = g.alloc(bytes)?.0;
        let root = a.world - 1;
        let [p10, p50, p90] =
            b.latency(a.iters / 4, a.skew_us, &|| comm.broadcast(buf, bytes, root))?;
        println!(
            "rank {rank} broadcast {bytes} B from rank {root}: synced p10/p50/p90 \
             {p10:.1}/{p50:.1}/{p90:.1} us, wrong {wrong} (every root checked)"
        );
    }
    if a.world == 2 {
        let pe = a.peer_exchange_bytes;
        let wrong = b.peer_exchange_check(pe)?;
        bad += wrong;
        let (send, recv) = (g.alloc(pe)?.0, g.alloc(pe)?.0);
        let [p10, p50, p90] = b.latency(a.iters, a.skew_us, &|| {
            comm.peer_exchange_async(send, recv, pe, stream)
        })?;
        println!(
            "rank {rank} peer exchange {pe} B: synced p10/p50/p90 {p10:.1}/{p50:.1}/{p90:.1} us, wrong {wrong}"
        );
        let small = a.rows[0] * H * 2;
        let large = a.rows.iter().copied().max().unwrap_or(1) * 5 * H * 2;
        let wrong = b.routing([small, large], 8 + a.stress / 16)?;
        bad += wrong;
        println!(
            "rank {rank} routing ({small} B next to {large} B, every entry point): wrong {wrong}"
        );
    }
    if a.world > 2 && a.gather_rows > 0 {
        let part = a.gather_rows * H * 2;
        let wrong = b.gather_check(part, 4 + a.stress / 4)?;
        bad += wrong;
        let all = g.alloc(part * a.world)?;
        let top = all.offset((a.world - 1) * part).0;
        let gather = || {
            ensure!(comm.gather_async(top, -(part as i64), part, stream)?);
            Ok(())
        };
        let reduce = || comm.all_reduce_async(all.0, part * a.world, stream);
        let scatter = || {
            ensure!(comm.reduce_scatter_async(top, -(part as i64), part, stream)?);
            Ok(())
        };
        let iters = (a.iters / 50).max(5);
        let [_, g50, _] = b.latency(iters, a.skew_us, &gather)?;
        let [_, r50, _] = b.latency(iters, a.skew_us, &reduce)?;
        let scatter8 = || {
            ensure!(comm.reduce_scatter_fp8_async(top, -(part as i64), part, stream)?);
            Ok(())
        };
        // The fused op (ATLAS_RDMA_RS_FUSED_SEG), its rows staged by the copy
        // producer; an all-reduce between ops (a fused op right after
        // another is refused).
        let copy = g.kernel("rdma_oneshot", "rdma_rs_fused_copy").ok();
        let fused = || {
            let f = comm
                .rs_fused_begin(top, -(part as i64), part)?
                .context("fused reduce-scatter refused")?;
            spark_runtime::kernel_args::KernelLaunch::new(g, copy.context("no copy producer")?)
                .grid([32, 1, 1])
                .block([256, 1, 1])
                .arg_words(f.plan.words())
                .launch(stream)?;
            ensure!(comm.reduce_scatter_fused_async(&f, stream)?);
            comm.all_reduce_async(all.0, 8192, stream)
        };
        let small = || comm.all_reduce_async(all.0, 8192, stream);
        let rs = if comm.supports_reduce_scatter_async(part) {
            let [_, s50, _] = b.latency(iters, a.skew_us, &scatter)?;
            let [_, f50, _] = b.latency(iters, a.skew_us, &scatter8)?;
            let fused = if copy.is_some() && std::env::var("ATLAS_RDMA_RS_FUSED_SEG").is_ok() {
                let [_, u50, _] = b.latency(iters, a.skew_us, &fused)?;
                let [_, a50, _] = b.latency(iters, a.skew_us, &small)?;
                format!(", fused {:.0} us", u50 - a50)
            } else {
                String::new()
            };
            format!("{s50:.0} us (FP8 {f50:.0} us{fused})")
        } else {
            "n/a".into()
        };
        println!(
            "rank {rank} SP gather {} rows/rank ({part} B parts): p50 {g50:.0} us; \
             reduce-scatter p50 {rs}; all-reduce of all {} rows: p50 {r50:.0} us; wrong {wrong}",
            a.gather_rows,
            a.gather_rows * a.world
        );
        g.free(all)?;
        // The KDA o_proj all-to-all of this chunk (`gather_rows * world`
        // rows of 2048 E4M3 columns a rank, plus 1x32 scale tiles): its time.
        if a.world == 4 {
            let (kl, rows) = (2048usize, a.gather_rows);
            let (q_part, sf_part, band) = (rows * kl, rows * kl / 32, 16 * 512);
            let (q, sf) = (g.alloc(4 * q_part)?, g.alloc(4 * sf_part)?);
            let (aq, asf) = (g.alloc(4 * q_part)?, g.alloc(4 * sf_part)?);
            let planes = [
                spark_comm::AllToAllPlane {
                    src: q.offset(3 * q_part).0,
                    src_stride: -(q_part as i64),
                    dst: aq.0,
                    dst_stride: kl as i64,
                    rows,
                    row_bytes: kl,
                    pitch: 4 * kl,
                },
                spark_comm::AllToAllPlane {
                    src: sf.offset(3 * sf_part).0,
                    src_stride: -(sf_part as i64),
                    dst: asf.0,
                    dst_stride: band as i64,
                    rows: rows / 128,
                    row_bytes: band,
                    pitch: 4 * band,
                },
            ];
            let a2a = || {
                ensure!(
                    comm.all_to_all_async(&planes, stream)?,
                    "all-to-all refused"
                );
                Ok(())
            };
            if rows.is_multiple_of(128) && comm.all_to_all_async(&planes, stream)? {
                let [_, x50, _] = b.latency(iters, a.skew_us, &a2a)?;
                println!(
                    "rank {rank} KDA o_proj all-to-all {rows} rows x {kl} E4M3 + scales a part \
                     ({} B a pair): p50 {x50:.0} us",
                    q_part + sf_part
                );
            }
            for p in [q, sf, aq, asf] {
                g.free(p)?;
            }
        }
    }
    if a.cmd_steps > 0 {
        let (pause, words) = (a.cmd_pause_us, comm.command_words_max());
        let small = a.rows[0] * H * 2;
        let ([p10, p50, p90], wrong) = b.command_steps(a.cmd_steps, pause, small)?;
        bad += wrong;
        println!(
            "rank {rank} step words ({}) + first collective after a {pause} us pause: \
             p10/p50/p90 {p10:.1}/{p50:.1}/{p90:.1} us, wrong {wrong}",
            if words > 0 {
                "host channel"
            } else {
                "broadcasts"
            }
        );
    }
    println!(
        "rank {rank}: {} ({bad} wrong)",
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    b.barrier()?;
    Ok(bad)
}
