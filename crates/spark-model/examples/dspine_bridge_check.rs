// SPDX-License-Identifier: AGPL-3.0-only
//! Parity gate for the DSpine drafter kernels (`common/dspine_bridge.cu`) at
//! Qwen3.8-27B-DSpine-b16 shapes (hidden 5120, code 1024, rank 512, γ 16).
//!
//! The drafter only proposes — the exact verify decides every token — so
//! these kernels need no serial-decode twin. What this pins down:
//! - BYTE-EXACT against a CPU replica of each kernel's arithmetic (same
//!   rounding points, same reduction order, --fmad=false):
//!   `dspine_bridge_read`, `dspine_bridge_codes`, `dspine_bridge_write`,
//!   `dspine_scale`, `dspine_serial_walk`.
//! - `dspine_bridge_message` and `dspine_serial_table` call `expf`, whose
//!   CUDA rounding the CPU cannot reproduce: they must be BYTE-IDENTICAL run
//!   to run and between a two-block launch and each block launched alone
//!   (a sequence's drafts do not depend on its batch), and agree with the
//!   CPU to one BF16 ulp (message) / pick for pick (table).
//! NEGATIVE CONTROL: each exact leg re-run from an input one BF16 ulp away in
//! one element must differ from its golden.
//!
//! GPU REQUIRED. On the GB10, with the qwen3.8-27b kernel set:
//!   ATLAS_TARGET_MODEL=qwen3.8-27b cargo run --release -p spark-model \
//!       --features cuda,gpu-examples --example dspine_bridge_check
//! Exit codes: 0 = pass, 1 = FAIL, 2 = skipped (kernel not in this set).
use anyhow::{Result, ensure};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const H: usize = 5120;
const C: usize = 1024;
const R: usize = 512;
const G: usize = 16;
const BLOCKS: usize = 2;
const ROWS: usize = G * BLOCKS;
const V: usize = 2048;
const K: usize = 16;
const EPS: f32 = 1e-6;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn bf(&mut self, n: usize, scale: f32) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f32(self.f() * scale)).collect()
    }
}

fn up(g: &dyn GpuBackend, v: &[bf16]) -> Result<DevicePtr> {
    let p = g.alloc(v.len() * 2)?;
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_u32(g: &dyn GpuBackend, v: &[u32]) -> Result<DevicePtr> {
    let p = g.alloc(v.len() * 4)?;
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.synchronize(g.default_stream())?;
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
fn down32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u32>> {
    let mut b = vec![0u8; n * 4];
    g.synchronize(g.default_stream())?;
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn bits(v: &[bf16]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}
/// One BF16 ulp up in element `i`.
fn bump(v: &[bf16], i: usize) -> Vec<bf16> {
    let mut w = v.to_vec();
    w[i] = bf16::from_bits(w[i].to_bits() + 1);
    w
}

/// `dsp_block_sum` over `blockDim` threads: strided per-thread partials, an
/// xor butterfly per warp, then warp 0's butterfly over the warp sums.
fn block_sum(vals: &[f32], threads: usize) -> f32 {
    let mut part = vec![0.0f32; threads];
    for (t, p) in part.iter_mut().enumerate() {
        let mut i = t;
        while i < vals.len() {
            *p += vals[i];
            i += threads;
        }
    }
    let fly = |v: &mut [f32]| {
        for o in [16usize, 8, 4, 2, 1] {
            let old = v.to_vec();
            for l in 0..32 {
                v[l] = old[l] + old[l ^ o];
            }
        }
    };
    let mut warp = vec![0.0f32; 32];
    for w in 0..threads / 32 {
        let mut v = part[w * 32..w * 32 + 32].to_vec();
        fly(&mut v);
        warp[w] = v[0];
    }
    fly(&mut warp);
    warp[0]
}

struct Ks {
    read: KernelHandle,
    codes: KernelHandle,
    message: KernelHandle,
    write: KernelHandle,
    scale: KernelHandle,
    table: KernelHandle,
    walk: KernelHandle,
    topk: KernelHandle,
}

fn read(g: &dyn GpuBackend, k: &Ks, x: &[bf16], rows: usize) -> Result<(Vec<u16>, Vec<u32>)> {
    let (xp, n, r) = (up(g, x)?, g.alloc(rows * H * 2)?, g.alloc(rows * 4)?);
    KernelLaunch::new(g, k.read)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(xp)
        .arg_ptr(n)
        .arg_ptr(r)
        .arg_u32(H as u32)
        .arg_f32(EPS)
        .launch(g.default_stream())?;
    Ok((down(g, n, rows * H)?, down32(g, r, rows)?))
}

fn read_cpu(x: &[bf16], rows: usize) -> (Vec<u16>, Vec<u32>) {
    let (mut n, mut r) = (Vec::new(), Vec::new());
    for row in x.chunks_exact(H).take(rows) {
        let sq: Vec<f32> = row.iter().map(|v| v.to_f32() * v.to_f32()).collect();
        let rms = (block_sum(&sq, 256) / H as f32)
            .max(f32::MIN_POSITIVE)
            .sqrt();
        r.push(rms.to_bits());
        n.extend(
            row.iter()
                .map(|v| bf16::from_f32(v.to_f32() / (rms + EPS)).to_bits()),
        );
    }
    (n, r)
}

#[allow(clippy::too_many_arguments)]
fn codes(
    g: &dyn GpuBackend,
    k: &Ks,
    p: &[bf16],
    cb: DevicePtr,
    anchors: DevicePtr,
    stride: u32,
    rows: usize,
) -> Result<Vec<u16>> {
    let (pp, out) = (up(g, p)?, g.alloc(rows * C * 2)?);
    KernelLaunch::new(g, k.codes)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(pp)
        .arg_ptr(cb)
        .arg_ptr(anchors)
        .arg_u32(stride)
        .arg_ptr(out)
        .arg_u32(G as u32)
        .arg_u32(C as u32)
        .arg_u32(V as u32)
        .arg_f32(EPS)
        .launch(g.default_stream())?;
    down(g, out, rows * C)
}

fn codes_cpu(p: &[bf16], cb: &[bf16], anchors: &[u32]) -> Vec<u16> {
    let sc = (C as f32).sqrt();
    let mut out = Vec::new();
    for (row, pr) in p.chunks_exact(C).enumerate() {
        if row % G == 0 {
            let a = anchors[row / G] as usize;
            out.extend(
                cb[a * C..a * C + C]
                    .iter()
                    .map(|v| bf16::from_f32(sc * v.to_f32()).to_bits()),
            );
            continue;
        }
        let sq: Vec<f32> = pr.iter().map(|v| v.to_f32() * v.to_f32()).collect();
        let d = block_sum(&sq, 256).sqrt() + EPS;
        out.extend(
            pr.iter()
                .map(|v| bf16::from_f32(sc * v.to_f32() / d).to_bits()),
        );
    }
    out
}

fn message(
    g: &dyn GpuBackend,
    k: &Ks,
    packed: &[bf16],
    bias: DevicePtr,
    rows: usize,
) -> Result<Vec<u16>> {
    let (pp, out) = (up(g, packed)?, g.alloc(rows * R * 2)?);
    KernelLaunch::new(g, k.message)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(pp)
        .arg_ptr(bias)
        .arg_ptr(out)
        .arg_u32(G as u32)
        .arg_u32(R as u32)
        .launch(g.default_stream())?;
    down(g, out, rows * R)
}

fn message_cpu(packed: &[bf16], bias: &[bf16]) -> Vec<f32> {
    let mut out = Vec::new();
    for row in 0..packed.len() / (3 * R) {
        for i in 0..R {
            if row % G == 0 {
                out.push(0.0);
                continue;
            }
            let own = &packed[row * 3 * R..];
            let pred = &packed[(row - 1) * 3 * R..];
            let z = own[i].to_f32() + pred[R + i].to_f32() + bias[i].to_f32();
            out.push((1.0 / (1.0 + (-z).exp())) * pred[2 * R + i].to_f32());
        }
    }
    out
}

fn write(
    g: &dyn GpuBackend,
    k: &Ks,
    x: &[bf16],
    rms: &[f32],
    delta: &[bf16],
    rows: usize,
) -> Result<(Vec<u16>, Vec<u16>)> {
    let (xp, dp, sp) = (up(g, x)?, up(g, delta)?, g.alloc(rows * H * 2)?);
    let rp = g.alloc(rows * 4)?;
    let rb: Vec<u8> = rms.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&rb, rp)?;
    KernelLaunch::new(g, k.write)
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(xp)
        .arg_ptr(rp)
        .arg_ptr(dp)
        .arg_ptr(sp)
        .arg_u32(G as u32)
        .arg_u32(H as u32)
        .launch(g.default_stream())?;
    Ok((down(g, xp, rows * H)?, down(g, sp, rows * H)?))
}

fn write_cpu(x: &[bf16], rms: &[f32], delta: &[bf16]) -> Vec<u16> {
    x.iter()
        .zip(delta)
        .enumerate()
        .map(|(i, (v, d))| {
            let row = i / H;
            if row % G == 0 {
                v.to_bits()
            } else {
                bf16::from_f32(v.to_f32() + rms[row] * d.to_f32()).to_bits()
            }
        })
        .collect()
}

fn check(name: &str, ok: bool, fails: &mut u32) {
    eprintln!("{} {name}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        *fails += 1;
    }
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b or qwen3.6-27b kernel set in this build");
    let gpu = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let m = |f: &str| g.kernel("dspine_bridge", f);
    if m("dspine_bridge_read").is_err() {
        eprintln!("SKIPPED: dspine_bridge is not in this kernel set");
        std::process::exit(2);
    }
    let k = Ks {
        read: m("dspine_bridge_read")?,
        codes: m("dspine_bridge_codes")?,
        message: m("dspine_bridge_message")?,
        write: m("dspine_bridge_write")?,
        scale: m("dspine_scale")?,
        table: m("dspine_serial_table")?,
        walk: m("dspine_serial_walk")?,
        topk: g.kernel("dflash2_candidate_selector", "dflash2_selector_topk")?,
    };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut fails = 0u32;

    // read
    let x = rng.bf(ROWS * H, 4.0);
    let (n, r) = read(g, &k, &x, ROWS)?;
    let (n0, r0) = read_cpu(&x, ROWS);
    check(
        "read: normalized + rms byte-exact vs CPU",
        n == n0 && r == r0,
        &mut fails,
    );
    let (nb, rb) = read(g, &k, &bump(&x, 3 * H + 77), ROWS)?;
    check(
        "read: 1-ulp input differs (negative control)",
        nb != n0 || rb != r0,
        &mut fails,
    );

    // codes
    let cb = rng.bf(V * C, 0.05);
    let cbp = up(g, &cb)?;
    let anchors = [17u32, 1999];
    let ap = up_u32(g, &anchors)?;
    let p = rng.bf(ROWS * C, 2.0);
    let c = codes(g, &k, &p, cbp, ap, 1, ROWS)?;
    check(
        "codes: byte-exact vs CPU",
        c == codes_cpu(&p, &cb, &anchors),
        &mut fails,
    );
    let cbump = codes(g, &k, &bump(&p, C + 5), cbp, ap, 1, ROWS)?;
    check(
        "codes: 1-ulp input differs (negative control)",
        cbump != c,
        &mut fails,
    );

    // message: determinism, batch independence, CPU within 1 BF16 ulp
    let packed = rng.bf(ROWS * 3 * R, 3.0);
    let bias = rng.bf(R, 1.0);
    let bp = up(g, &bias)?;
    let m1 = message(g, &k, &packed, bp, ROWS)?;
    let m2 = message(g, &k, &packed, bp, ROWS)?;
    let alone = message(g, &k, &packed[G * 3 * R..], bp, G)?;
    check("message: run-to-run byte-identical", m1 == m2, &mut fails);
    check(
        "message: block alone == block in batch",
        alone == m1[G * R..],
        &mut fails,
    );
    let cpu = message_cpu(&packed, &bias);
    let off = m1
        .iter()
        .zip(&cpu)
        .filter(|(a, b)| (**a as i32 - bf16::from_f32(**b).to_bits() as i32).abs() > 1)
        .count();
    check(
        &format!("message: CPU within 1 BF16 ulp ({off} off)"),
        off == 0,
        &mut fails,
    );
    let mb = message(g, &k, &bump(&packed, 5 * 3 * R + R + 9), bp, ROWS)?;
    check(
        "message: 1-ulp input differs (negative control)",
        mb != m1,
        &mut fails,
    );

    // write (with stash)
    let rms: Vec<f32> = (0..ROWS).map(|_| rng.f().abs() + 0.5).collect();
    let delta = rng.bf(ROWS * H, 1.0);
    let (w, stash) = write(g, &k, &x, &rms, &delta, ROWS)?;
    check(
        "write: byte-exact vs CPU",
        w == write_cpu(&x, &rms, &delta),
        &mut fails,
    );
    check(
        "write: stash holds the pre-write rows",
        stash == bits(&x),
        &mut fails,
    );
    // An anchor row passes through unwritten, so a 1-ulp input there must
    // reach the output (a 1-ulp delta can round away in `x + rms * delta`).
    let (wb, _) = write(g, &k, &bump(&x, G * H + 1), &rms, &delta, ROWS)?;
    check(
        "write: 1-ulp input differs (negative control)",
        wb != w,
        &mut fails,
    );

    // scale (column and scalar)
    let col = rng.bf(H, 2.0);
    let (src, colp, dst) = (up(g, &x)?, up(g, &col)?, g.alloc(ROWS * H * 2)?);
    let launch = |colp: DevicePtr, s: f32| {
        KernelLaunch::new(g, k.scale)
            .grid([64, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u64((ROWS * H) as u64)
            .arg_u32(H as u32)
            .arg_f32(s)
            .arg_ptr(colp)
            .launch(g.default_stream())
    };
    launch(colp, 0.0)?;
    let sc = down(g, dst, ROWS * H)?;
    let sc0: Vec<u16> = x
        .iter()
        .enumerate()
        .map(|(i, v)| bf16::from_f32(v.to_f32() * col[i % H].to_f32()).to_bits())
        .collect();
    launch(DevicePtr::NULL, 32.0)?;
    let ss = down(g, dst, ROWS * H)?;
    let ss0: Vec<u16> = x
        .iter()
        .map(|v| bf16::from_f32(v.to_f32() * 32.0).to_bits())
        .collect();
    check(
        "scale: byte-exact vs CPU (column, scalar)",
        sc == sc0 && ss == ss0,
        &mut fails,
    );

    fails += table_legs(g, &k, &mut rng)?;
    if fails > 0 {
        eprintln!("DSPINE BRIDGE CHECK: {fails} FAIL");
        std::process::exit(1);
    }
    eprintln!("DSPINE BRIDGE CHECK: all legs PASS");
    Ok(())
}

#[path = "dspine_bridge_check_parts/table_legs.rs"]
mod table_legs;
use table_legs::table_legs;
