// SPDX-License-Identifier: AGPL-3.0-only
//! GLM-5 KDA verify recurrence: per-owner `kda_recurrent_bf16_verify_snap`
//! launches against one owner-batched `kda_recurrent_bf16_verify_snap_owners`
//! launch. Final states and rollback snapshots must match bit for bit; BF16
//! outputs may round differently in rare elements (reported). Reports both
//! times.
//!
//! Exit: 0 pass, 1 mismatch.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example kda_verify_bench

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const HEADS: usize = 32;
const DIM: usize = 128;
const STATE: usize = HEADS * DIM * DIM;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn fetch(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let single = g.kernel("kda", "kda_recurrent_bf16_verify_snap")?;
    let batched = g.kernel("kda", "kda_recurrent_bf16_verify_snap_owners")?;
    let mut rng = Lcg(0xDA);
    let mut fail = false;
    for (owners, rows) in [(1usize, 8usize), (4, 6), (4, 8)] {
        let n = owners * rows;
        let qkv: Vec<u8> = (0..n * 3 * HEADS * DIM)
            .flat_map(|_| bf16(rng.f()))
            .collect();
        let gate: Vec<u8> = (0..n * HEADS * DIM)
            .flat_map(|_| bf16(rng.f() * 3.0))
            .collect();
        let beta: Vec<u8> = (0..n * HEADS).flat_map(|_| bf16(rng.f() * 2.0)).collect();
        let a_log: Vec<u8> = (0..HEADS)
            .flat_map(|_| (rng.f() * 0.5).to_le_bytes())
            .collect();
        let dt: Vec<u8> = (0..HEADS * DIM)
            .flat_map(|_| (rng.f()).to_le_bytes())
            .collect();
        let states: Vec<u8> = (0..owners * STATE)
            .flat_map(|_| (rng.f() * 0.1).to_le_bytes())
            .collect();
        let (d_qkv, d_gate, d_beta, d_alog, d_dt) = (
            up(g, &qkv)?,
            up(g, &gate)?,
            up(g, &beta)?,
            up(g, &a_log)?,
            up(g, &dt)?,
        );
        let d_pristine = up(g, &states)?;
        let inter_bytes = (rows - 1).max(1) * STATE * 4;
        // Per variant: states, rollback slabs, outputs.
        let mk = || -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
            Ok((
                g.alloc(owners * STATE * 4)?,
                g.alloc(owners * inter_bytes)?,
                g.alloc(n * HEADS * DIM * 2)?,
            ))
        };
        let (s_a, i_a, o_a) = mk()?;
        let (s_b, i_b, o_b) = mk()?;
        let lower = -5.0f32;
        let run_single = || -> Result<()> {
            for o in 0..owners {
                KernelLaunch::new(g, single)
                    .grid([HEADS as u32, 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(d_qkv.offset(o * rows * 3 * HEADS * DIM * 2))
                    .arg_ptr(d_gate.offset(o * rows * HEADS * DIM * 2))
                    .arg_ptr(d_beta.offset(o * rows * HEADS * 2))
                    .arg_ptr(d_alog)
                    .arg_ptr(d_dt)
                    .arg_ptr(s_a.offset(o * STATE * 4))
                    .arg_ptr(o_a.offset(o * rows * HEADS * DIM * 2))
                    .arg_ptr(i_a.offset(o * inter_bytes))
                    .arg_u64(STATE as u64)
                    .arg_u32(rows as u32)
                    .arg_u32(HEADS as u32)
                    .arg_u32(DIM as u32)
                    .arg_f32(lower)
                    .launch(0)?;
            }
            Ok(())
        };
        let run_batched = || -> Result<()> {
            let mut l = KernelLaunch::new(g, batched)
                .grid([HEADS as u32, owners as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(d_qkv)
                .arg_ptr(d_gate)
                .arg_ptr(d_beta)
                .arg_ptr(d_alog)
                .arg_ptr(d_dt)
                .arg_ptr(o_b);
            for o in 0..4 {
                l = l.arg_ptr(if o < owners {
                    s_b.offset(o * STATE * 4)
                } else {
                    DevicePtr(0)
                });
            }
            for o in 0..4 {
                l = l.arg_ptr(if o < owners {
                    i_b.offset(o * inter_bytes)
                } else {
                    DevicePtr(0)
                });
            }
            l.arg_u64(STATE as u64)
                .arg_u32(rows as u32)
                .arg_u32(HEADS as u32)
                .arg_u32(DIM as u32)
                .arg_f32(lower)
                .launch(0)
        };
        let timed = |f: &dyn Fn() -> Result<()>, s: DevicePtr| -> Result<f64> {
            g.copy_d2d(d_pristine, s, owners * STATE * 4)?;
            f()?;
            g.synchronize(0)?;
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                f()?;
            }
            g.synchronize(0)?;
            let t = t0.elapsed().as_secs_f64() / 20.0;
            // The check runs once from the pristine states.
            g.copy_d2d(d_pristine, s, owners * STATE * 4)?;
            f()?;
            g.synchronize(0)?;
            Ok(t)
        };
        let (t_a, t_b) = (timed(&run_single, s_a)?, timed(&run_batched, s_b)?);
        let mut diff = 0;
        let mut worst = 0f32;
        for (name, a, b, bytes) in [
            ("state", s_a, s_b, owners * STATE * 4),
            ("output", o_a, o_b, n * HEADS * DIM * 2),
            (
                "snapshots",
                i_a,
                i_b,
                if rows > 1 { owners * inter_bytes } else { 0 },
            ),
        ] {
            let (x, y) = (fetch(g, a, bytes)?, fetch(g, b, bytes)?);
            let d = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            if d > 0 && name != "output" {
                for (p, q) in x.chunks_exact(4).zip(y.chunks_exact(4)) {
                    let (p, q) = (
                        f32::from_le_bytes(p.try_into()?),
                        f32::from_le_bytes(q.try_into()?),
                    );
                    if p != q {
                        worst = worst.max((p - q).abs() / p.abs().max(1e-30));
                    }
                }
            }
            if d > 0 {
                println!("   {name}: {d} bytes differ");
            }
            if name != "output" {
                diff += d;
            }
        }
        println!("   worst relative FP32 state difference {worst:.2e}");
        fail |= diff != 0;
        println!(
            "owners {owners} x {rows} rows: per-owner verify_snap {:7.1}us, owner-batched {:7.1}us  {}",
            t_a * 1e6,
            t_b * 1e6,
            if diff == 0 {
                "states+snapshots bitwise".to_string()
            } else {
                format!("MISMATCH {diff} bytes")
            }
        );
    }
    std::process::exit(if fail { 1 } else { 0 });
}
