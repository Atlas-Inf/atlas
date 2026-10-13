// SPDX-License-Identifier: AGPL-3.0-only
//
// PLE n-gram fault latency at a decode step's cadence, with and without the
// NVMe keepalive (`ATLAS_PLE_NVME_KEEPALIVE_MS`). Each "step" resolves
// `--misses` fresh random rows (compulsory misses, as novel n-grams are),
// releases the batch, then idles `--gap-ms` (the rest of the step). Prints
// the resolve-time distribution per arm, and checks that both arms faulted
// byte-identical rows (the ticker never touches the cache).
//
//   ngram-keepalive-bench --file <table.safetensors> [--misses 32]
//       [--gap-ms 140] [--steps 40] [--period-ms 10] [--stride 160]
//
// `ATLAS_PLE_FAULT_POOL=1` runs both arms on the persistent fault workers;
// the printed fnv1a64 of the faulted rows must match the run without it.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use spark_storage::NgramRowCache;
use spark_storage::cuda_min::CudaCtx;

struct Args {
    file: PathBuf,
    misses: usize,
    gap_ms: u64,
    steps: usize,
    period_ms: u64,
    stride: usize,
}

fn parse() -> Args {
    let mut a = Args {
        file: PathBuf::new(),
        misses: 32,
        gap_ms: 140,
        steps: 40,
        period_ms: 10,
        stride: 160,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().expect("missing value");
        match k.as_str() {
            "--file" => a.file = PathBuf::from(v()),
            "--misses" => a.misses = v().parse().unwrap(),
            "--gap-ms" => a.gap_ms = v().parse().unwrap(),
            "--steps" => a.steps = v().parse().unwrap(),
            "--period-ms" => a.period_ms = v().parse().unwrap(),
            "--stride" => a.stride = v().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(!a.file.as_os_str().is_empty(), "--file required");
    a
}

/// One arm: returns sorted resolve times (us) and the faulted rows' bytes.
fn arm(a: &Args, rows: u64, keepalive: bool, seed: u64) -> Result<(Vec<u128>, Vec<u8>)> {
    let slots = (a.steps * a.misses).next_power_of_two().max(65536);
    let mut c = NgramRowCache::open(&a.file, None, rows, a.stride, slots)?;
    if keepalive {
        c.start_keepalive(
            Duration::from_millis(a.period_ms),
            Duration::from_millis(2000),
        )?;
    }
    let mut x = seed;
    let mut us = Vec::with_capacity(a.steps);
    let mut bytes = Vec::new();
    let mut slots_out = Vec::new();
    for _ in 0..a.steps {
        let ids: Vec<u64> = (0..a.misses)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x % rows
            })
            .collect();
        let t = Instant::now();
        c.resolve(&ids, &mut slots_out)?;
        us.push(t.elapsed().as_micros());
        for &s in &slots_out {
            bytes.extend(c.copy_slot(s)?);
        }
        c.end_batch();
        std::thread::sleep(Duration::from_millis(a.gap_ms));
    }
    println!(
        "    ({} keepalive reads)",
        if keepalive { c.keepalive_reads() } else { 0 }
    );
    us.sort_unstable();
    Ok((us, bytes))
}

fn main() -> Result<()> {
    let a = parse();
    let _ctx = CudaCtx::new(0)?;
    let len = std::fs::metadata(&a.file)?.len();
    let rows = (len - 4096) / a.stride as u64;
    println!(
        "{} misses/step, {} ms idle between steps, {} steps, keepalive period {} ms",
        a.misses, a.gap_ms, a.steps, a.period_ms
    );
    // Distinct ids per timing arm (a drive-side cache must not favour the
    // second arm); the byte check re-faults arm 0's ids with the ticker on.
    let (seed_a, seed_b) = (0x2545_F491_4F6C_DD1D, 0x9E37_79B9_7F4A_7C15);
    let mut off_bytes = Vec::new();
    for (name, on, seed) in [("off", false, seed_a), ("on ", true, seed_b)] {
        let (us, bytes) = arm(&a, rows, on, seed)?;
        let p = |q: usize| us[(us.len() - 1) * q / 100];
        let mean = us.iter().sum::<u128>() / us.len() as u128;
        println!(
            "  keepalive {name}: resolve p10 {:>6} us  p50 {:>6} us  p90 {:>6} us  mean {:>6} us",
            p(10),
            p(50),
            p(90),
            mean
        );
        if !on {
            off_bytes = bytes;
        }
    }
    let (_, on_bytes) = arm(&a, rows, true, seed_a)?;
    let same = off_bytes == on_bytes;
    println!(
        "rows byte-identical across arms: {} ({} bytes)",
        if same { "yes" } else { "NO" },
        off_bytes.len()
    );
    anyhow::ensure!(same, "keepalive arm faulted different bytes");
    // FNV-1a over the faulted rows: compare across processes, e.g. with and
    // without ATLAS_PLE_FAULT_POOL=1.
    let h = off_bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    });
    println!("rows fnv1a64 {h:016x}");
    Ok(())
}
