// SPDX-License-Identifier: AGPL-3.0-only
//! Gate for the shared-scan decode attention (`ATLAS_ATTN_SHARED_SCAN`,
//! `attn_shared_scan` + `attn_shared_scan_reduce`) at the Qwen3.8-27B
//! attention shape (24 query heads over 4 KV heads, head dim 256, 16-token
//! KV blocks, BF16 NHD cache behind a permuted block table).
//!
//! ROW EXACTNESS (what `ATLAS_EXACT_VERIFY` needs). GOLDEN is serial decode:
//! each position of a chain run ALONE, in order, with every slot past it
//! still NaN (a stale slot must never reach the arithmetic). Each row of a
//! batched launch must then be BYTE-IDENTICAL to golden: the 16 rows
//! together (two groups), the first 8, a 3-row window, the chain interleaved
//! with another sequence's rows, and the chain in reverse order (no groups).
//! Chains start at positions chosen to straddle step, superblock and
//! size-class boundaries.
//!
//! CONTROLS. A changed value at a position a row attends must change that
//! row's bytes and leave the rows before it untouched.
//!
//! REFERENCE. The same rows through `paged_decode_attn` (scalar FP32): the
//! two kernels order their sums differently, so the gate is closeness, and
//! the largest difference is printed in BF16 steps of the element.
//!
//! `--time [TOKENS]` (default 210000) then times one layer at that depth,
//! both kernels, for 1 and 8 rows.
//!
//! GPU REQUIRED. On the GB10, with the qwen3.8-27b kernel set:
//!   ATLAS_TARGET_MODEL=qwen3.8-27b cargo run --release -p spark-model \
//!       --features cuda,gpu-examples --example attn_shared_scan_check
//! Exit codes: 0 = pass, 1 = FAIL, 2 = skipped (kernel not in this set).
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const NQ: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const BS: usize = 16;
const SLOT_BYTES: usize = NKV * HD * 2;
const Q_BYTES: usize = NQ * HD * 2;
const CHAIN: usize = 16;
/// Blocks of the main sequence and of the second one, in one pool.
const MAIN_BLOCKS: usize = 1280;
const SIDE_BLOCKS: usize = 768;
const MAX_BLOCKS: usize = MAIN_BLOCKS;

struct Lcg(u64);
impl Lcg {
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
    }
    fn bf16(&mut self, n: usize, lo: f64, hi: f64) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f64(self.r(lo, hi)).to_bits().to_le_bytes())
            .collect()
    }
}

fn main_block(i: usize) -> usize {
    (i * 37 + 5) % MAIN_BLOCKS
}
fn side_block(i: usize) -> usize {
    MAIN_BLOCKS + (i * 13 + 3) % SIDE_BLOCKS
}
/// Byte offset of the main sequence's position `p` in a pool.
fn main_slot(p: usize) -> usize {
    (main_block(p / BS) * BS + p % BS) * SLOT_BYTES
}

/// One row of a launch: which sequence, how long, and its query.
#[derive(Clone)]
struct Row {
    side: bool,
    len: usize,
    q: Vec<u8>,
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    old: KernelHandle,
    k: DevicePtr,
    v: DevicePtr,
    stream: u64,
}

impl Rig<'_> {
    /// Run `rows` in one launch; each row's [NQ, HD] BF16 output.
    fn run(&self, rows: &[Row], shared: bool) -> Result<Vec<Vec<u8>>> {
        let g = self.g;
        let n = rows.len();
        let mut tables = Vec::with_capacity(n * MAX_BLOCKS * 4);
        let mut lens = Vec::with_capacity(n * 4);
        let mut q = Vec::with_capacity(n * Q_BYTES);
        for row in rows {
            for i in 0..MAX_BLOCKS {
                let b = if row.side {
                    side_block(i % SIDE_BLOCKS)
                } else {
                    main_block(i)
                };
                tables.extend((b as i32).to_le_bytes());
            }
            lens.extend((row.len as i32).to_le_bytes());
            q.extend(&row.q);
        }
        let (dt, dl, dq) = (
            g.alloc(tables.len())?,
            g.alloc(lens.len())?,
            g.alloc(q.len())?,
        );
        let out = g.alloc(n * Q_BYTES)?;
        g.copy_h2d(&tables, dt)?;
        g.copy_h2d(&lens, dl)?;
        g.copy_h2d(&q, dq)?;
        g.memset(out, 0xEE, n * Q_BYTES)?;
        self.launch(dq, out, dt, dl, MAX_BLOCKS, n, shared)?;
        g.synchronize(self.stream)?;
        let mut bytes = vec![0u8; n * Q_BYTES];
        g.copy_d2h(out, &mut bytes)?;
        for p in [dt, dl, dq, out] {
            g.free(p)?;
        }
        Ok(bytes.chunks(Q_BYTES).map(<[u8]>::to_vec).collect())
    }

    fn launch(
        &self,
        q: DevicePtr,
        out: DevicePtr,
        tables: DevicePtr,
        lens: DevicePtr,
        max_blocks: usize,
        n: usize,
        shared: bool,
    ) -> Result<()> {
        let inv = 1.0 / (HD as f32).sqrt();
        let (nq, nkv, hd, bs) = (NQ as u32, NKV as u32, HD as u32, BS as u32);
        if shared {
            ops::attn_shared_scan_bf16(
                self.g,
                q,
                self.k,
                self.v,
                out,
                tables,
                lens,
                max_blocks as u32,
                n as u32,
                nq,
                nkv,
                hd,
                bs,
                inv,
                (NQ * HD) as u32,
                self.stream,
            )
        } else {
            ops::paged_decode_attn_bf16(
                self.g,
                self.old,
                q,
                self.k,
                self.v,
                out,
                tables,
                lens,
                max_blocks as u32,
                n as u32,
                nq,
                nkv,
                hd,
                bs,
                inv,
                (NQ * HD) as u32,
                0,
                self.stream,
            )
        }
    }
}

fn vals(b: &[u8]) -> Vec<f32> {
    b.chunks(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// Largest |a - b| in BF16 steps of the larger element, and how many differ.
fn closeness(a: &[u8], b: &[u8]) -> (f32, usize) {
    let (x, y) = (vals(a), vals(b));
    let mut worst = 0f32;
    let mut differ = 0;
    for (p, q) in x.iter().zip(&y) {
        if p.to_bits() != q.to_bits() {
            differ += 1;
        }
        let scale = p.abs().max(q.abs()).max(1e-3);
        // A BF16 step at magnitude `scale` is 2^(exponent - 7).
        let step = 2f32.powi(scale.log2().floor() as i32 - 7);
        worst = worst.max((p - q).abs() / step);
    }
    (worst, differ)
}

fn check_base(
    rig: &Rig,
    base: usize,
    hk: &[u8],
    hv: &[u8],
    side: &[Row],
    fails: &mut Vec<String>,
) -> Result<(f32, usize)> {
    let g = rig.g;
    let mut r = Lcg(0x5EED ^ base as u64);
    let chain: Vec<Row> = (0..CHAIN)
        .map(|i| Row {
            side: false,
            len: base + i + 1,
            q: r.bf16(NQ * HD, -4.0, 4.0),
        })
        .collect();
    // Serial decode: every slot from `base` on starts as NaN and position
    // base + i is written just before step i.
    let span = (CHAIN + 2 * BS).div_ceil(BS) * BS;
    let nan: Vec<u8> = std::iter::repeat_n(0x7FC0u16.to_le_bytes(), NKV * HD)
        .flatten()
        .collect();
    for p in base..base + span {
        g.copy_h2d(&nan, rig.k.offset(main_slot(p)))?;
        g.copy_h2d(&nan, rig.v.offset(main_slot(p)))?;
    }
    let mut golden = Vec::new();
    let mut reference = Vec::new();
    for (i, row) in chain.iter().enumerate() {
        let s = main_slot(base + i);
        g.copy_h2d(&hk[s..s + SLOT_BYTES], rig.k.offset(s))?;
        g.copy_h2d(&hv[s..s + SLOT_BYTES], rig.v.offset(s))?;
        golden.push(rig.run(std::slice::from_ref(row), true)?.remove(0));
        reference.push(rig.run(std::slice::from_ref(row), false)?.remove(0));
    }
    for (i, out) in golden.iter().enumerate() {
        if vals(out).iter().any(|x| !x.is_finite()) {
            fails.push(format!(
                "base {base}: serial row {i} is not finite (a stale slot reached the arithmetic)"
            ));
        }
    }
    // Batched launches against golden.
    let mut mixed: Vec<Row> = vec![side[0].clone()];
    mixed.extend(chain[..3].iter().cloned());
    mixed.push(side[1].clone());
    mixed.extend(chain[3..11].iter().cloned());
    let mixed_ix: Vec<Option<usize>> = [None]
        .into_iter()
        .chain((0..3).map(Some))
        .chain([None])
        .chain((3..11).map(Some))
        .collect();
    let reversed: Vec<Row> = chain[..8].iter().rev().cloned().collect();
    let cases: Vec<(&str, Vec<Row>, Vec<Option<usize>>)> = vec![
        ("16 rows", chain.clone(), (0..16).map(Some).collect()),
        ("first 8", chain[..8].to_vec(), (0..8).map(Some).collect()),
        (
            "rows 5..8",
            chain[5..8].to_vec(),
            (5..8).map(Some).collect(),
        ),
        ("interleaved", mixed, mixed_ix),
        ("reversed 8", reversed, (0..8).rev().map(Some).collect()),
    ];
    for (name, rows, ix) in &cases {
        let outs = rig.run(rows, true)?;
        for (slot, which) in ix.iter().enumerate() {
            let Some(i) = *which else { continue };
            if outs[slot] != golden[i] {
                let (w, d) = closeness(&outs[slot], &golden[i]);
                fails.push(format!("base {base} [{name}]: row {i} differs from serial ({d} elements, worst {w:.2} steps)"));
            }
        }
    }
    // Control: change V at position base + 2 (KV head 0); rows 0 and 1 do
    // not attend it, rows 2.. do.
    let s = main_slot(base + 2);
    let bumped: Vec<u8> = std::iter::repeat_n(bf16::from_f32(3.0).to_bits().to_le_bytes(), HD)
        .flatten()
        .collect();
    g.copy_h2d(&bumped, rig.v.offset(s))?;
    let outs = rig.run(&chain[..8], true)?;
    for (i, out) in outs.iter().enumerate() {
        let same = *out == golden[i];
        if (i < 2) != same {
            fails.push(format!(
                "base {base} control: row {i} {} after a change at position base+2",
                if same { "did not change" } else { "changed" }
            ));
        }
    }
    // Leave the pool as it was for the next chain.
    for p in base..base + span {
        let s = main_slot(p);
        g.copy_h2d(&hk[s..s + SLOT_BYTES], rig.k.offset(s))?;
        g.copy_h2d(&hv[s..s + SLOT_BYTES], rig.v.offset(s))?;
    }
    // Closeness to the scalar kernel.
    let mut worst = 0f32;
    let mut differ = 0;
    for (a, b) in golden.iter().zip(&reference) {
        let (w, d) = closeness(a, b);
        worst = worst.max(w);
        differ += d;
    }
    Ok((worst, differ))
}

fn time_layer(g: &dyn GpuBackend, old: KernelHandle, tokens: usize) -> Result<()> {
    let blocks = tokens.div_ceil(BS) + 2;
    let bytes = blocks * BS * SLOT_BYTES;
    let chunk = Lcg(7).bf16(1 << 20, -1.0, 1.0);
    let host: Vec<u8> = chunk.iter().copied().cycle().take(bytes).collect();
    let (k, v) = (g.alloc(bytes)?, g.alloc(bytes)?);
    g.copy_h2d(&host, k)?;
    g.copy_h2d(&host, v)?;
    let rig = Rig {
        g,
        old,
        k,
        v,
        stream: g.default_stream(),
    };
    eprintln!(
        "timing one layer at {tokens} tokens ({} MiB of K+V):",
        (2 * bytes) >> 20
    );
    for rows in [1usize, 8] {
        let mut tables = Vec::new();
        let mut lens = Vec::new();
        for i in 0..rows {
            for b in 0..blocks {
                tables.extend((b as i32).to_le_bytes());
            }
            lens.extend(((tokens - rows + i + 1) as i32).to_le_bytes());
        }
        let q = Lcg(11).bf16(rows * NQ * HD, -4.0, 4.0);
        let (dt, dl, dq, out) = (
            g.alloc(tables.len())?,
            g.alloc(lens.len())?,
            g.alloc(q.len())?,
            g.alloc(rows * Q_BYTES)?,
        );
        g.copy_h2d(&tables, dt)?;
        g.copy_h2d(&lens, dl)?;
        g.copy_h2d(&q, dq)?;
        for shared in [false, true] {
            let run = |n: usize| -> Result<f64> {
                let t = std::time::Instant::now();
                for _ in 0..n {
                    rig.launch(dq, out, dt, dl, blocks, rows, shared)?;
                }
                g.synchronize(rig.stream)?;
                Ok(t.elapsed().as_secs_f64() * 1e3 / n as f64)
            };
            run(2)?;
            let ms = run(10)?;
            let gbs = (2 * bytes) as f64 / 1e9 / (ms / 1e3);
            eprintln!(
                "  {rows} row(s) {:>11}: {ms:8.3} ms a layer, {:7.1} ms for 16 layers, {gbs:6.1} GB/s of K+V",
                if shared { "shared scan" } else { "scalar" },
                ms * 16.0
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b or qwen3.6-27b kernel set in this build");
    eprintln!("kernel set: {}", set.target.model);
    let gpu = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &gpu;
    if !ops::attn_shared_scan_available(g) {
        eprintln!("SKIPPED: attn_shared_scan is not in this kernel set");
        std::process::exit(2);
    }
    let old = g.kernel("paged_decode", "paged_decode_attn")?;

    let pool = (MAIN_BLOCKS + SIDE_BLOCKS) * BS * SLOT_BYTES;
    let mut r = Lcg(0xA77E);
    let hk = r.bf16(pool / 2, -1.0, 1.0);
    let hv = r.bf16(pool / 2, -2.0, 2.0);
    let (k, v) = (g.alloc(pool)?, g.alloc(pool)?);
    g.copy_h2d(&hk, k)?;
    g.copy_h2d(&hv, v)?;
    let rig = Rig {
        g,
        old,
        k,
        v,
        stream: g.default_stream(),
    };
    ensure!(
        MAIN_BLOCKS * BS >= 20_480,
        "main sequence too short for the bases"
    );

    // A second sequence's rows, and their output alone.
    let side: Vec<Row> = [3000usize, 9001]
        .iter()
        .map(|&len| Row {
            side: true,
            len,
            q: r.bf16(NQ * HD, -4.0, 4.0),
        })
        .collect();
    let mut fails = Vec::new();
    let side_alone: Vec<Vec<u8>> = side
        .iter()
        .map(|s| Ok(rig.run(std::slice::from_ref(s), true)?.remove(0)))
        .collect::<Result<_>>()?;
    let together = rig.run(&side, true)?;
    if together != side_alone {
        fails.push("second sequence: rows together differ from rows alone".into());
    }

    // Chain starts around step (16), superblock (64, 256, 512) and size-class
    // (2048, 4096, 8192, 16384) boundaries.
    let bases = [
        0usize, 1, 14, 16, 56, 250, 255, 256, 500, 2040, 4090, 8185, 8192, 12345, 16375, 16384,
        20000,
    ];
    let mut worst = 0f32;
    let mut differ = 0usize;
    for &base in &bases {
        let before = fails.len();
        let (w, d) = check_base(&rig, base, &hk, &hv, &side, &mut fails)?;
        worst = worst.max(w);
        differ += d;
        eprintln!(
            "base {base:>6}: {} (vs scalar: worst {w:.2} BF16 steps)",
            if fails.len() == before {
                "rows byte-identical to serial"
            } else {
                "FAIL"
            }
        );
    }
    let total = bases.len() * CHAIN * NQ * HD;
    eprintln!(
        "vs scalar kernel: {differ} of {total} elements differ ({:.2}%), worst {worst:.2} BF16 steps",
        100.0 * differ as f64 / total as f64
    );
    if worst > 2.0 {
        fails.push(format!(
            "shared scan is {worst:.2} BF16 steps from the scalar kernel (gate 2.0)"
        ));
    }

    if let Some(i) = args.iter().position(|a| a == "--time") {
        let tokens = args
            .get(i + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(210_000);
        time_layer(g, old, tokens)?;
    }

    if fails.is_empty() {
        eprintln!("PASS");
        Ok(())
    } else {
        for f in &fails {
            eprintln!("FAIL: {f}");
        }
        std::process::exit(1);
    }
}
