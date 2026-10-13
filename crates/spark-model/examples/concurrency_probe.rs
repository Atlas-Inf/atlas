// SPDX-License-Identifier: AGPL-3.0-only
//! How much of a GLM decode (verify) workload hides under a concurrent
//! prefill workload on one GB10, with real kernels at serving shapes:
//!
//! * prefill proxy (low-priority stream): K128W MoE gate + down over 16K
//!   sorted rows on 144 local experts (DRAM-bound) and the sparse MLA
//!   attention of a 4096-row chunk (compute-bound);
//! * decode proxy (high-priority stream): the four-owner verify's routed MoE
//!   (24 rows over ~40 local experts), KDA recurrence and split sparse
//!   attention.
//!
//! Prints each alone, both concurrently, and the fraction of the shorter
//! workload hidden. Random data; timing only.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example concurrency_probe

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

unsafe extern "C" {
    fn cuStreamCreateWithPriority(stream: *mut u64, flags: u32, priority: i32) -> i32;
    fn cuCtxGetStreamPriorityRange(least: *mut i32, greatest: *mut i32) -> i32;
}

const EXPERTS: usize = 144;
const GRID_EXPERTS: u32 = 288;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn rand_bytes(rng: &mut Lcg, n: usize, lo: u8, span: u64) -> Vec<u8> {
    (0..n)
        .map(|_| lo.wrapping_add((rng.next() % span) as u8))
        .collect()
}

/// Per-expert transposed NVFP4 tables `[K/2, N]` + `[K/16, N]` for the local
/// half of a 288-expert grid; returns (packed ptrs, scale ptrs, scale2).
fn tables(g: &dyn GpuBackend, rng: &mut Lcg, n: usize, k: usize) -> Result<[DevicePtr; 3]> {
    let (mut pp, mut sp) = (Vec::new(), Vec::new());
    for _ in 0..EXPERTS {
        let w = up(g, &rand_bytes(rng, n * k / 2, 0, 256))?;
        let s = up(g, &rand_bytes(rng, n * k / 16, 0x30, 16))?;
        pp.extend_from_slice(&w.0.to_le_bytes());
        sp.extend_from_slice(&s.0.to_le_bytes());
    }
    pp.resize(GRID_EXPERTS as usize * 8, 0);
    sp.resize(GRID_EXPERTS as usize * 8, 0);
    let s2: Vec<u8> = (0..GRID_EXPERTS)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect();
    Ok([up(g, &pp)?, up(g, &sp)?, up(g, &s2)?])
}

/// Expert offsets over 288 experts: `local` rows spread over the first
/// `active` local experts, and as many rows on the remote half.
fn offsets(g: &dyn GpuBackend, active: usize, local: usize) -> Result<(DevicePtr, usize)> {
    let mut rows = vec![0usize; GRID_EXPERTS as usize];
    for i in 0..local {
        rows[i % active] += 1;
        rows[EXPERTS + i % active] += 1;
    }
    let mut off = vec![0i32];
    for r in &rows {
        off.push(off.last().unwrap() + *r as i32);
    }
    let b: Vec<u8> = off.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok((up(g, &b)?, *off.last().unwrap() as usize))
}

struct Moe {
    kernel: KernelHandle,
    prefix_k: KernelHandle,
    a: DevicePtr,
    a_s: DevicePtr,
    w: [DevicePtr; 3],
    out: DevicePtr,
    off: DevicePtr,
    prefix: DevicePtr,
    bound: u32,
    n: u32,
    k: u32,
}

impl Moe {
    #[allow(clippy::too_many_arguments)]
    fn new(
        g: &dyn GpuBackend,
        rng: &mut Lcg,
        w: [DevicePtr; 3],
        n: usize,
        k: usize,
        active: usize,
        local: usize,
    ) -> Result<Self> {
        let (off, rows) = offsets(g, active, local)?;
        Ok(Self {
            kernel: g.kernel(
                "moe_w4a16",
                "moe_w4a4_grouped_gemm_prequant_t_k128w_compact",
            )?,
            prefix_k: g.kernel("moe_w4a16", "moe_mtile_prefix")?,
            a: up(g, &rand_bytes(rng, rows * k / 2, 0, 256))?,
            a_s: up(g, &rand_bytes(rng, rows * k / 16, 0x30, 16))?,
            w,
            out: g.alloc(rows * n * 2)?,
            off,
            prefix: g.alloc((GRID_EXPERTS as usize + 1) * 4)?,
            bound: (local.div_ceil(64) + EXPERTS) as u32,
            n: n as u32,
            k: k as u32,
        })
    }

    fn run(&self, g: &dyn GpuBackend, stream: u64) -> Result<()> {
        KernelLaunch::new(g, self.prefix_k)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(self.off)
            .arg_ptr(self.w[0])
            .arg_ptr(self.prefix)
            .arg_u32(GRID_EXPERTS)
            .launch(stream)?;
        KernelLaunch::new(g, self.kernel)
            .grid([self.n / 256, self.bound, 1])
            .block([256, 1, 1])
            .arg_ptr(self.a)
            .arg_ptr(self.a_s)
            .arg_ptr(self.w[0])
            .arg_ptr(self.w[1])
            .arg_ptr(self.w[2])
            .arg_ptr(self.out)
            .arg_ptr(self.off)
            .arg_ptr(DevicePtr(0))
            .arg_u32(GRID_EXPERTS)
            .arg_u32(self.n)
            .arg_u32(self.k)
            .arg_ptr(self.prefix)
            .launch(stream)
    }
}

/// Sparse MLA attention (BF16 cache) over `rows` query rows; split when
/// `splits > 1`.
struct Attn {
    unsplit: KernelHandle,
    split: KernelHandle,
    merge: KernelHandle,
    q: DevicePtr,
    cache: DevicePtr,
    table: DevicePtr,
    idx: DevicePtr,
    out: DevicePtr,
    part: DevicePtr,
    lse: DevicePtr,
    rows: u32,
    splits: u32,
}

impl Attn {
    fn new(g: &dyn GpuBackend, rng: &mut Lcg, rows: u32, splits: u32) -> Result<Self> {
        let (tokens, width) = (32768usize, 2051u32);
        let table: Vec<u8> = (0..tokens as u32 / 16)
            .flat_map(|b| b.to_le_bytes())
            .collect();
        let idx: Vec<u8> = (0..rows * width)
            .flat_map(|_| ((rng.next() % tokens as u64) as i32).to_le_bytes())
            .collect();
        let rh = rows as usize * 32;
        Ok(Self {
            unsplit: g.kernel(
                "glm_sparse_prefill_kv_reuse",
                "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad",
            )?,
            split: g.kernel(
                "glm_sparse_prefill_kv_reuse",
                "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split",
            )?,
            merge: g.kernel(
                "glm_sparse_decode_split_merge",
                "glm_sparse_decode_split_merge",
            )?,
            q: up(g, &rand_bytes(rng, rh * 512 * 2, 0x10, 32))?,
            cache: up(g, &rand_bytes(rng, tokens * 512 * 2, 0x10, 32))?,
            table: up(g, &table)?,
            idx: up(g, &idx)?,
            out: g.alloc(rh * 512 * 2)?,
            part: g.alloc(splits as usize * rh * 512 * 4)?,
            lse: g.alloc((splits as usize + 1) * rh * 4)?,
            rows,
            splits,
        })
    }

    fn run(&self, g: &dyn GpuBackend, stream: u64) -> Result<()> {
        let split = self.splits > 1;
        let mut l = KernelLaunch::new(g, if split { self.split } else { self.unsplit })
            .grid([1, self.rows, self.splits])
            .block([256, 1, 1])
            .shared_mem(69376)
            .arg_ptr(self.q)
            .arg_ptr(self.cache)
            .arg_ptr(self.cache)
            .arg_ptr(self.idx)
            .arg_ptr(self.out)
            .arg_ptr(self.table)
            .arg_u32(self.rows)
            .arg_u32(32)
            .arg_u32(512)
            .arg_u32(2051)
            .arg_u32(16)
            .arg_f32(1.0 / 16.0);
        if split {
            l = l.arg_ptr(self.part).arg_ptr(self.lse);
        }
        l.launch(stream)?;
        if split {
            let rh = self.rows as usize * 32;
            KernelLaunch::new(g, self.merge)
                .grid([self.rows * 32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.part)
                .arg_ptr(self.lse)
                .arg_ptr(self.out)
                .arg_ptr(self.lse.offset(self.splits as usize * rh * 4))
                .arg_u32(self.rows)
                .arg_u32(32)
                .arg_u32(512)
                .arg_u32(self.splits)
                .launch(stream)?;
        }
        Ok(())
    }
}

/// Four owners x 6 rows of the owner-batched KDA verify recurrence.
struct Kda {
    kernel: KernelHandle,
    bufs: [DevicePtr; 8],
    states: [DevicePtr; 4],
    inter: [DevicePtr; 4],
}

impl Kda {
    fn new(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<Self> {
        let (n, hd) = (24usize, 32 * 128usize);
        let state = 32 * 128 * 128 * 4;
        let mut states = [DevicePtr(0); 4];
        let mut inter = [DevicePtr(0); 4];
        for o in 0..4 {
            states[o] = g.alloc(state)?;
            g.memset(states[o], 0, state)?;
            inter[o] = g.alloc(5 * state)?;
        }
        Ok(Self {
            kernel: g.kernel("kda", "kda_recurrent_bf16_verify_snap_owners")?,
            bufs: [
                up(g, &rand_bytes(rng, n * 3 * hd * 2, 0x10, 32))?,
                up(g, &rand_bytes(rng, n * hd * 2, 0x10, 32))?,
                up(g, &rand_bytes(rng, n * 32 * 2, 0x10, 32))?,
                up(g, &vec![0u8; 32 * 4])?,
                up(g, &vec![0u8; hd * 4])?,
                g.alloc(n * hd * 2)?,
                DevicePtr(0),
                DevicePtr(0),
            ],
            states,
            inter,
        })
    }

    fn run(&self, g: &dyn GpuBackend, stream: u64) -> Result<()> {
        let mut l = KernelLaunch::new(g, self.kernel)
            .grid([32, 4, 1])
            .block([128, 1, 1]);
        for p in &self.bufs[..6] {
            l = l.arg_ptr(*p);
        }
        for p in self.states.iter().chain(&self.inter) {
            l = l.arg_ptr(*p);
        }
        l.arg_u64(32 * 128 * 128)
            .arg_u32(6)
            .arg_u32(32)
            .arg_u32(128)
            .arg_f32(-5.0)
            .launch(stream)
    }
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (mut least, mut greatest) = (0i32, 0i32);
    ensure!(unsafe { cuCtxGetStreamPriorityRange(&mut least, &mut greatest) } == 0);
    let (mut lo, mut hi) = (0u64, 0u64);
    ensure!(unsafe { cuStreamCreateWithPriority(&mut lo, 1, least) } == 0);
    ensure!(unsafe { cuStreamCreateWithPriority(&mut hi, 1, greatest) } == 0);
    let mut rng = Lcg(0xC0C0);
    let gate_w = tables(g, &mut rng, 2048, 4096)?;
    let down_w = tables(g, &mut rng, 4096, 2048)?;
    // Prefill: 16K local sorted rows over all 144 local experts.
    let p_gate = Moe::new(g, &mut rng, gate_w, 2048, 4096, EXPERTS, 16384)?;
    let p_down = Moe::new(g, &mut rng, down_w, 4096, 2048, EXPERTS, 16384)?;
    let p_attn = Attn::new(g, &mut rng, 4096, 1)?;
    // Decode: 24 rows x 8 routes -> ~96 local slots over ~40 experts.
    let d_gate = Moe::new(g, &mut rng, gate_w, 2048, 4096, 40, 96)?;
    let d_down = Moe::new(g, &mut rng, down_w, 4096, 2048, 40, 96)?;
    let d_attn = Attn::new(g, &mut rng, 24, 2)?;
    let d_kda = Kda::new(g, &mut rng)?;

    let prefill = |s: u64, reps: usize| -> Result<()> {
        for _ in 0..reps {
            p_gate.run(g, s)?;
            p_gate.run(g, s)?;
            p_attn.run(g, s)?;
            p_down.run(g, s)?;
        }
        Ok(())
    };
    let decode = |s: u64, reps: usize| -> Result<()> {
        for _ in 0..reps {
            d_gate.run(g, s)?;
            d_gate.run(g, s)?;
            d_down.run(g, s)?;
            d_kda.run(g, s)?;
            d_attn.run(g, s)?;
        }
        Ok(())
    };
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        f()?;
        g.synchronize(lo)?;
        g.synchronize(hi)?;
        let t0 = std::time::Instant::now();
        f()?;
        g.synchronize(lo)?;
        g.synchronize(hi)?;
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    };
    let one_p = time(&|| prefill(lo, 1))?;
    let one_d = time(&|| decode(hi, 1))?;
    // Size the decode run to about the prefill run.
    let (rp, rd) = (10usize, ((10.0 * one_p / one_d).round() as usize).max(1));
    let tp = time(&|| prefill(lo, rp))?;
    let td = time(&|| decode(hi, rd))?;
    let both = time(&|| {
        prefill(lo, rp)?;
        decode(hi, rd)
    })?;
    let hidden = (tp + td - both) / tp.min(td);
    println!(
        "prefill proxy {rp} x {one_p:.2} ms = {tp:.1} ms; decode proxy {rd} x {one_d:.2} ms = {td:.1} ms"
    );
    println!(
        "concurrent {both:.1} ms (serial {:.1}); hidden {:.0}% of the shorter",
        tp + td,
        hidden * 100.0
    );
    // The decode stream's own pace under contention.
    let t0 = std::time::Instant::now();
    prefill(lo, rp)?;
    decode(hi, rd)?;
    g.synchronize(hi)?;
    let d_done = t0.elapsed().as_secs_f64() * 1e3;
    g.synchronize(lo)?;
    println!(
        "decode proxy under prefill: {d_done:.1} ms ({:.2}x its solo time)",
        d_done / td
    );
    Ok(())
}
