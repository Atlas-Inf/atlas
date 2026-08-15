// SPDX-License-Identifier: AGPL-3.0-only

//! cfg4-vs-m8 Marlin A/B rig. Synthetic raw NVFP4 weights are repacked into
//! the Marlin B layout by the REAL atlas_marlin_repack_w4, then the m8 and
//! cfg4 kernels compute the same GEMM. C rows must match exactly.
//!
//! Run: cargo run -p spark-model --release --example marlin_cfg4_diff \
//!        --features cuda,gpu-examples

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const K: u32 = 2688;
const N: u32 = 1856;
const GROUP: u32 = 16;
const SMS: u32 = 48;
const SMEM: u32 = 96 * 1024;

fn fill_random(g: &dyn GpuBackend, dst: DevicePtr, bytes: usize, seed: u8) -> Result<()> {
    let mut v = vec![0u8; bytes];
    let mut x: u32 = 0x1234_5678u32 ^ ((seed as u32) << 24);
    for b in v.iter_mut() {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *b = (x >> 24) as u8;
    }
    g.copy_h2d(&v, dst)?;
    Ok(())
}

fn fill_sane_scales(g: &dyn GpuBackend, dst: DevicePtr, n: usize) -> Result<()> {
    let pat = [0x30u8, 0x38, 0x40, 0x34, 0x3C]; // e4m3 0.5, 1.0, 2.0, 0.75, 1.5
    let mut v = vec![0u8; n];
    for (i, b) in v.iter_mut().enumerate() {
        *b = pat[i % pat.len()];
    }
    g.copy_h2d(&v, dst)?;
    Ok(())
}

fn fill_sane_bf16(g: &dyn GpuBackend, dst: DevicePtr, n: usize) -> Result<()> {
    // deterministic small-magnitude bf16 pattern: (i % 9) - 4, half steps
    let mut v = vec![0u16; n];
    for (i, w) in v.iter_mut().enumerate() {
        let val = ((i % 9) as f32) - 4.0 + 0.5 * (((i / 9) % 2) as f32);
        *w = (val.to_bits() >> 16) as u16;
    }
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 2)
    };
    g.copy_h2d(bytes, dst)?;
    Ok(())
}

fn dump(g: &dyn GpuBackend, path: &str, src: DevicePtr, bytes: usize) -> Result<()> {
    let mut v = vec![0u8; bytes];
    g.copy_d2h(src, &mut v)?;
    std::fs::write(path, &v)?;
    println!("wrote {path} ({bytes} B)");
    Ok(())
}

fn run_marlin(
    g: &dyn GpuBackend,
    k: KernelHandle,
    threads: u32,
    kd: u32,
    nd: u32,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    c_tmp: DevicePtr,
    s: DevicePtr,
    gs: DevicePtr,
    locks: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    g.memset_async(locks, 0, (SMS * 16) as usize, stream)?;
    KernelLaunch::new(g, k)
        .grid([SMS, 1, 1])
        .block([threads, 1, 1])
        .shared_mem(SMEM)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_ptr(c_tmp)
        .arg_ptr(DevicePtr::NULL) // b_bias
        .arg_ptr(DevicePtr::NULL) // a_scales
        .arg_ptr(s)
        .arg_ptr(gs)
        .arg_ptr(DevicePtr::NULL) // zp
        .arg_ptr(DevicePtr::NULL) // g_idx
        .arg_i32((kd / GROUP) as i32) // num_groups
        .arg_i32(m as i32)
        .arg_i32(nd as i32)
        .arg_i32(kd as i32)
        .arg_i32(kd as i32) // lda
        .arg_ptr(locks)
        .arg_i32(0) // has_bias
        .arg_i32(1) // use_atomic_add
        .arg_i32(1) // use_fp32_reduce
        .arg_i32(SMEM as i32)
        .launch(stream)?;
    Ok(())
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let stream = g.create_stream()?;

    let m8: KernelHandle = g
        .kernel("marlin_nvfp4_gemm", "atlas_marlin_nvfp4_m8")
        .map_err(|e| anyhow::anyhow!("m8 missing: {e}"))?;
    let cfg4: KernelHandle = g
        .kernel("marlin_nvfp4_gemm", "atlas_marlin_nvfp4_cfg4")
        .map_err(|e| anyhow::anyhow!("cfg4 missing: {e}"))?;
    let repack: KernelHandle = g
        .kernel("marlin_repack", "atlas_marlin_repack_w4")
        .map_err(|e| anyhow::anyhow!("repack missing: {e}"))?;

    // ── synthetic weights (raw modelopt-packed [N, K/2] u8) ──
    let raw = g.alloc((N * K / 2) as usize)?;
    fill_random(g, raw, (N * K / 2) as usize, 0x11)?;
    let scales = g.alloc((N * K / GROUP) as usize)?;
    fill_sane_scales(g, scales, (N * K / GROUP) as usize)?;
    let gs = g.alloc(4)?;
    {
        let v = [1.0f32];
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4) };
        g.copy_h2d(bytes, gs)?;
    }

    // ── repack: input must be the [K/8, N] int32 view (transpose on host) ──
    let raw_host = {
        let mut v = vec![0u8; (N * K / 2) as usize];
        g.copy_d2h(raw, &mut v)?;
        v
    };
    let mut b_in = vec![0u32; ((K / 8) * N) as usize];
    for k32 in 0..(K / 8) as usize {
        for n in 0..N as usize {
            b_in[k32 * N as usize + n] =
                u32::from_le_bytes(raw_host[(n * (K / 2) as usize + k32 * 4)..][..4].try_into().unwrap());
        }
    }
    let b_in_dev = g.alloc(b_in.len() * 4)?;
    {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(b_in.as_ptr() as *const u8, b_in.len() * 4)
        };
        g.copy_h2d(bytes, b_in_dev)?;
    }
    let b_marlin = g.alloc(b_in.len() * 4)?;
    g.memset_async(b_marlin, 0, b_in.len() * 4, stream)?;
    KernelLaunch::new(g, repack)
        .grid([SMS, 1, 1])
        .block([256, 1, 1])
        .shared_mem(SMEM)
        .arg_ptr(b_in_dev)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(b_marlin)
        .arg_i32(K as i32)
        .arg_i32(N as i32)
        .launch(stream)?;
    g.synchronize(stream)?;
    println!("repack ok");

    // ── A: 64 sane rows ──
    let a = g.alloc((64 * K * 2) as usize)?;
    fill_sane_bf16(g, a, (64 * K) as usize)?;
    let c_tmp = g.alloc((32 * N * 4) as usize)?;
    let locks = g.alloc((SMS * 16 * 4) as usize)?;

    // ── m8 reference: rows 0..8, 8..16, 16..24, 24..32 ──
    let c_ref = g.alloc((32 * N * 2) as usize)?;
    for t in 0..4 {
        run_marlin(
            g,
            m8,
            128,
            K,
            N,
            a.offset(t * 8 * K as usize * 2),
            b_marlin,
            c_ref.offset(t * 8 * N as usize * 2),
            c_tmp,
            scales,
            gs,
            locks,
            8,
            stream,
        )?;
        g.synchronize(stream)?;
        println!("m8 tile {t} ok");
    }
    // ── cfg4 (128-thread cfg3-style, M=32) ──
    let c_got = g.alloc((32 * N * 2) as usize)?;
    run_marlin(
        g, cfg4, 128, K, N, a, b_marlin, c_got, c_tmp, scales, gs, locks, 32, stream,
    )?;
    g.synchronize(stream)?;
    println!("cfg4 ok");

    // ── M=64 leg: the serve pads to (M_e+31)&!31 — multi-tile (parallel=2) ──
    let c64_got = g.alloc((64 * N * 2) as usize)?;
    let c_tmp64 = g.alloc((64 * N * 4) as usize)?;
    let c64_ref = g.alloc((64 * N * 2) as usize)?;
    for t in 0..8 {
        run_marlin(
            g,
            m8,
            128,
            K,
            N,
            a.offset(t * 8 * K as usize * 2),
            b_marlin,
            c64_ref.offset(t * 8 * N as usize * 2),
            c_tmp64,
            scales,
            gs,
            locks,
            8,
            stream,
        )?;
    }
    run_marlin(
        g, cfg4, 128, K, N, a, b_marlin, c64_got, c_tmp64, scales, gs, locks, 64, stream,
    )?;
    g.synchronize(stream)?;
    println!("cfg4 m=64 ok");
    dump(g, "/tmp/cfg4m64_c_ref.bin", c64_ref, (64 * N * 2) as usize)?;
    dump(g, "/tmp/cfg4m64_c_got.bin", c64_got, (64 * N * 2) as usize)?;

    // ── OFFSET leg: A/C at row 16 (non-zero base like the serve's off[e]) ──
    let coff_ref = g.alloc((32 * N * 2) as usize)?;
    let coff_got = g.alloc((32 * N * 2) as usize)?;
    for t in 0..4 {
        run_marlin(
            g,
            m8,
            128,
            K,
            N,
            a.offset((16 + t * 8) * K as usize * 2),
            b_marlin,
            coff_ref.offset(t * 8 * N as usize * 2),
            c_tmp,
            scales,
            gs,
            locks,
            8,
            stream,
        )?;
    }
    run_marlin(
        g,
        cfg4,
        128,
        K,
        N,
        a.offset(16 * K as usize * 2),
        b_marlin,
        coff_got,
        c_tmp,
        scales,
        gs,
        locks,
        32,
        stream,
    )?;
    g.synchronize(stream)?;
    println!("cfg4 offset ok");
    dump(g, "/tmp/cfg4o_c_ref.bin", coff_ref, (32 * N * 2) as usize)?;
    dump(g, "/tmp/cfg4o_c_got.bin", coff_got, (32 * N * 2) as usize)?;

    // ── DOWN leg: K=1856, N=2688, k64n128 kernels ──
    let m8d: KernelHandle = g
        .kernel("marlin_nvfp4_gemm", "atlas_marlin_nvfp4_m8_k64n128")
        .map_err(|e| anyhow::anyhow!("m8d missing: {e}"))?;
    let cfg4d: KernelHandle = g
        .kernel("marlin_nvfp4_gemm", "atlas_marlin_nvfp4_cfg4_k64n128")
        .map_err(|e| anyhow::anyhow!("cfg4d missing: {e}"))?;
    const KD: u32 = 1856;
    const ND: u32 = 2688;
    let rawd = g.alloc((ND * KD / 2) as usize)?;
    fill_random(g, rawd, (ND * KD / 2) as usize, 0x77)?;
    let sd = g.alloc((ND * KD / GROUP) as usize)?;
    fill_sane_scales(g, sd, (ND * KD / GROUP) as usize)?;
    let rawd_host = {
        let mut v = vec![0u8; (ND * KD / 2) as usize];
        g.copy_d2h(rawd, &mut v)?;
        v
    };
    let mut b_ind = vec![0u32; ((KD / 8) * ND) as usize];
    for k32 in 0..(KD / 8) as usize {
        for n in 0..ND as usize {
            b_ind[k32 * ND as usize + n] = u32::from_le_bytes(
                rawd_host[(n * (KD / 2) as usize + k32 * 4)..][..4]
                    .try_into()
                    .unwrap(),
            );
        }
    }
    let b_ind_dev = g.alloc(b_ind.len() * 4)?;
    {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b_ind.as_ptr() as *const u8, b_ind.len() * 4) };
        g.copy_h2d(bytes, b_ind_dev)?;
    }
    let bd_marlin = g.alloc(b_ind.len() * 4)?;
    g.memset_async(bd_marlin, 0, b_ind.len() * 4, stream)?;
    KernelLaunch::new(g, repack)
        .grid([SMS, 1, 1])
        .block([256, 1, 1])
        .shared_mem(SMEM)
        .arg_ptr(b_ind_dev)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(bd_marlin)
        .arg_i32(KD as i32)
        .arg_i32(ND as i32)
        .launch(stream)?;
    g.synchronize(stream)?;
    println!("down repack ok");
    let ad = g.alloc((64 * KD * 2) as usize)?;
    fill_sane_bf16(g, ad, (64 * KD) as usize)?;
    let c_tmpd = g.alloc((32 * ND * 4) as usize)?;
    let cd_ref = g.alloc((32 * ND * 2) as usize)?;
    for t in 0..4 {
        run_marlin(
            g,
            m8d,
            128,
            KD,
            ND,
            ad.offset(t * 8 * KD as usize * 2),
            bd_marlin,
            cd_ref.offset(t * 8 * ND as usize * 2),
            c_tmpd,
            sd,
            gs,
            locks,
            8,
            stream,
        )?;
        g.synchronize(stream)?;
        println!("m8d tile {t} ok");
    }
    let cd_got = g.alloc((32 * ND * 2) as usize)?;
    run_marlin(
        g, cfg4d, 128, KD, ND, ad, bd_marlin, cd_got, c_tmpd, sd, gs, locks, 32, stream,
    )?;
    g.synchronize(stream)?;
    println!("cfg4d ok");

    dump(g, "/tmp/cfg4d_c_ref.bin", cd_ref, (32 * ND * 2) as usize)?;
    dump(g, "/tmp/cfg4d_c_got.bin", cd_got, (32 * ND * 2) as usize)?;
    println!("dumps written");
    Ok(())
}
