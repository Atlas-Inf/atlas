// SPDX-License-Identifier: AGPL-3.0-only
//! Secondary bitwise gates of `moe_fp4_prefill_bench`: fused gate/up+SiLU
//! against its three-kernel reference, and the tile worklist builder against
//! a host walk.

use super::*;

/// The fused gate/up + SiLU·mul + NVFP4 kernel against its three-kernel
/// reference (K128W gate, K128W up, `silu_mul_quant_nvfp4`): the packed E2M1
/// and E4M3 scale bytes must match. Returns true on mismatch.
pub(super) fn gate_up_silu(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    d_off: DevicePtr,
    d_prefix: DevicePtr,
    tile_bound: u32,
    all_rows: usize,
) -> Result<bool> {
    let (n, k) = (super::inter(), 4096u32);
    let (nu, ku) = (n as usize, k as usize);
    let (Ok(fused), Ok(wide), Ok(silu), Ok(prefix_k)) = (
        g.kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w"),
        g.kernel(MODULE, "moe_w4a4_grouped_gemm_prequant_t_k128w_compact"),
        g.kernel("moe_silu_mul", "silu_mul_quant_nvfp4"),
        g.kernel(MODULE, "moe_mtile_prefix"),
    ) else {
        println!("gate/up + SiLU quant: absent");
        return Ok(false);
    };
    let a: Vec<u8> = (0..all_rows * ku / 2).map(|_| rng.next() as u8).collect();
    let a_s: Vec<u8> = (0..all_rows * ku / 16)
        .map(|_| 0x30 + (rng.next() % 16) as u8)
        .collect();
    let (d_a, d_as) = (up(g, &a)?, up(g, &a_s)?);
    let ([g_pp, g_sp, ..], mut owned) = expert_tables(g, rng, nu, ku)?;
    let ([u_pp, u_sp, ..], u_owned) = expert_tables(g, rng, nu, ku)?;
    owned.extend(u_owned);
    // Output scales that put the SiLU inputs around the clamp and below it.
    let s2 = |base: f32| -> Vec<u8> {
        (0..GRID_EXPERTS)
            .flat_map(|e| (base * (1.0 + (e % 5) as f32 * 0.25)).to_le_bytes())
            .collect()
    };
    let (g_s2, u_s2) = (up(g, &s2(1.0 / 256.0))?, up(g, &s2(1.0 / 128.0))?);
    let (c_g, c_u) = (g.alloc(all_rows * nu * 2)?, g.alloc(all_rows * nu * 2)?);
    let (q_bytes, s_bytes) = (all_rows * nu / 2, all_rows * nu / 16);
    let (ref_q, ref_s, new_q, new_s) = (
        g.alloc(q_bytes)?,
        g.alloc(s_bytes)?,
        g.alloc(q_bytes)?,
        g.alloc(s_bytes)?,
    );
    for (p, b) in [
        (c_g, all_rows * nu * 2),
        (c_u, all_rows * nu * 2),
        (ref_q, q_bytes),
        (ref_s, s_bytes),
        (new_q, q_bytes),
        (new_s, s_bytes),
    ] {
        g.memset(p, 0, b)?;
    }
    let prefix = |pp| {
        KernelLaunch::new(g, prefix_k)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(d_off)
            .arg_ptr(pp)
            .arg_ptr(d_prefix)
            .arg_u32(GRID_EXPERTS as u32)
            .launch(0)
    };
    let run_ref = || -> Result<()> {
        prefix(g_pp)?;
        for (pp, sp, s2p, c) in [(g_pp, g_sp, g_s2, c_g), (u_pp, u_sp, u_s2, c_u)] {
            launch(
                g,
                wide,
                256,
                256,
                &[d_a, d_as, pp, sp, s2p, c, d_off, DevicePtr(0)],
                n,
                k,
                tile_bound,
                Some(d_prefix),
            )?;
        }
        KernelLaunch::new(g, silu)
            .grid([all_rows as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(c_g)
            .arg_ptr(c_u)
            .arg_ptr(ref_q)
            .arg_ptr(ref_s)
            .arg_ptr(DevicePtr(0))
            .arg_u32(all_rows as u32)
            .arg_u32(n)
            .launch(0)
    };
    let run_fused = || -> Result<()> {
        prefix(g_pp)?;
        KernelLaunch::new(g, fused)
            .grid([n / 128, tile_bound, 1])
            .block([256, 1, 1])
            .arg_ptr(d_a)
            .arg_ptr(d_as)
            .arg_ptr(g_pp)
            .arg_ptr(g_sp)
            .arg_ptr(g_s2)
            .arg_ptr(DevicePtr(0))
            .arg_ptr(d_off)
            .arg_ptr(DevicePtr(0))
            .arg_u32(GRID_EXPERTS as u32)
            .arg_u32(n)
            .arg_u32(k)
            .arg_ptr(d_prefix)
            .arg_ptr(u_pp)
            .arg_ptr(u_sp)
            .arg_ptr(u_s2)
            .arg_ptr(new_q)
            .arg_ptr(new_s)
            .launch(0)
    };
    let (t_ref, t_new) = (time(g, &run_ref)?, time(g, &run_fused)?);
    let fetch = |p, b| -> Result<Vec<u8>> {
        let mut v = vec![0u8; b];
        g.copy_d2h(p, &mut v)?;
        Ok(v)
    };
    let (rq, rs, nq, ns) = (
        fetch(ref_q, q_bytes)?,
        fetch(ref_s, s_bytes)?,
        fetch(new_q, q_bytes)?,
        fetch(new_s, s_bytes)?,
    );
    let diff = rq.iter().zip(&nq).filter(|(a, b)| a != b).count()
        + rs.iter().zip(&ns).filter(|(a, b)| a != b).count();
    let zero_scales = rs.iter().filter(|&&b| b == 0).count();
    println!(
        "gate/up + SiLU quant: gate+up+silu {:7.1}us, fused {:7.1}us  {} ({zero_scales} of {s_bytes} reference scales zero)",
        t_ref * 1e6,
        t_new * 1e6,
        if diff == 0 {
            "bitwise".to_string()
        } else {
            format!("MISMATCH {diff} bytes")
        }
    );
    for p in owned
        .into_iter()
        .chain([d_a, d_as, g_s2, u_s2, c_g, c_u, ref_q, ref_s, new_q, new_s])
    {
        g.free(p)?;
    }
    Ok(diff != 0)
}

/// E4M3 value of `b` (OCP FP8 E4M3: bias 7, no infinities).
fn e4m3(b: u8) -> f32 {
    let (s, e, m) = (b >> 7, (b >> 3) & 15, b & 7);
    let v = if e == 0 {
        m as f32 / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}

/// The lossy E4M3 down grid (`_e4m3`) against the BF16 reference `want`:
/// its time and the dequantized values' error relative to each 64-column
/// group's amax. Returns true when the kernel's error exceeds E4M3's
/// rounding (2^-4 of the group amax).
pub(super) fn down_e4m3(
    g: &dyn GpuBackend,
    args: &[DevicePtr; 8],
    want: &[u8],
    [n, k]: [u32; 2],
    tile_bound: u32,
    flop: f64,
) -> Result<bool> {
    let Ok(prefix_k) = g.kernel(MODULE, "moe_mtile_prefix") else {
        return Ok(false);
    };
    let rows = want.len() / 2 / n as usize;
    let d_sc = g.alloc(rows * n as usize / 64 * 4)?;
    let d_prefix = g.alloc((GRID_EXPERTS + 1) * 4)?;
    let d_next = g.alloc(16)?;
    let mut fail = false;
    for (name, persist) in [
        ("moe_w4a4_grouped_gemm_prequant_t_k128w_compact_e4m3", false),
        (
            "moe_w4a4_grouped_gemm_prequant_t_k128w_compact_persist_e4m3",
            true,
        ),
    ] {
        let Ok(kernel) = g.kernel(MODULE, name) else {
            println!("  {name}: absent");
            continue;
        };
        g.memset(args[5], 0, rows * n as usize)?;
        let run = || -> Result<()> {
            KernelLaunch::new(g, prefix_k)
                .grid([1, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(args[6])
                .arg_ptr(args[2])
                .arg_ptr(d_prefix)
                .arg_u32(GRID_EXPERTS as u32)
                .launch(0)?;
            let grid = if persist {
                g.memset_async(d_next, 0, 4, 0)?;
                [2 * g.sm_count()?, 1, 1]
            } else {
                [n / 256, tile_bound, 1]
            };
            let mut l = KernelLaunch::new(g, kernel).grid(grid).block([256, 1, 1]);
            for p in args {
                l = l.arg_ptr(*p);
            }
            l = l
                .arg_u32(GRID_EXPERTS as u32)
                .arg_u32(n)
                .arg_u32(k)
                .arg_ptr(d_prefix);
            if persist {
                l = l.arg_ptr(d_next);
            }
            l.arg_ptr(d_sc).launch(0)
        };
        let t = time(g, &run)?;
        let mut q = vec![0u8; rows * n as usize];
        g.copy_d2h(args[5], &mut q)?;
        let mut sc = vec![0u8; rows * n as usize / 64 * 4];
        g.copy_d2h(d_sc, &mut sc)?;
        let (mut worst, mut sum, mut count) = (0f64, 0f64, 0usize);
        for (r, (wrow, qrow)) in want
            .chunks_exact(2 * n as usize)
            .zip(q.chunks_exact(n as usize))
            .enumerate()
        {
            for gi in 0..n as usize / 64 {
                let ref_vals: Vec<f32> = (0..64)
                    .map(|j| {
                        let c = gi * 64 + j;
                        half::bf16::from_le_bytes([wrow[2 * c], wrow[2 * c + 1]]).to_f32()
                    })
                    .collect();
                let amax = ref_vals.iter().fold(0f32, |m, v| m.max(v.abs()));
                if amax == 0.0 {
                    continue;
                }
                let o = (r * (n as usize / 64) + gi) * 4;
                let s = f32::from_le_bytes(sc[o..o + 4].try_into()?);
                for (j, v) in ref_vals.iter().enumerate() {
                    let e = ((e4m3(qrow[gi * 64 + j]) * s - v) / amax).abs() as f64;
                    worst = worst.max(e);
                    sum += e;
                    count += 1;
                }
            }
        }
        println!(
            "  {name}: {:7.1}us {:5.1} TFLOPS  lossy: error / group amax max {worst:.4} mean {:.5} ({count} values)",
            t * 1e6,
            flop / t / 1e12,
            sum / count.max(1) as f64
        );
        // BF16 reference rounding (2^-9) + E4M3 rounding of the FP32 values (2^-4).
        fail |= worst > 1.0 / 16.0 + 1.0 / 256.0;
    }
    for p in [d_sc, d_prefix, d_next] {
        g.free(p)?;
    }
    Ok(fail)
}

/// `moe_build_tile_worklist` (block scan) against a host serial walk of the
/// same offsets, local experts only, at the decode compact shape (M64 tiles,
/// 16 N tiles). Returns true on mismatch.
pub(super) fn tile_worklist(g: &dyn GpuBackend, offsets: &[i32]) -> Result<bool> {
    let Ok(builder) = g.kernel("moe", "moe_build_tile_worklist") else {
        println!("tile worklist: absent");
        return Ok(false);
    };
    let (n_tiles, m_tile) = (16u32, 64u32);
    let ptrs: Vec<u8> = (0..GRID_EXPERTS)
        .flat_map(|e| (if e < experts() { 0x1000u64 } else { 0 }).to_le_bytes())
        .collect();
    let mut want = Vec::new();
    for e in 0..experts() {
        let rows = (offsets[e + 1] - offsets[e]) as u32;
        for mt in 0..rows.div_ceil(m_tile) {
            for nt in 0..n_tiles {
                want.extend([e as u32, (mt << 6) | nt]);
            }
        }
    }
    let off_bytes: Vec<u8> = offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (d_off, d_ptrs) = (up(g, &off_bytes)?, up(g, &ptrs)?);
    let (d_list, d_total) = (g.alloc(want.len() * 4 + 64)?, g.alloc(4)?);
    let run = || {
        KernelLaunch::new(g, builder)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_off)
            .arg_ptr(d_ptrs)
            .arg_ptr(d_list)
            .arg_ptr(d_total)
            .arg_u32(GRID_EXPERTS as u32)
            .arg_u32(n_tiles)
            .arg_u32(m_tile)
            .launch(0)
    };
    let t = time(g, &run)?;
    let mut total = [0u8; 4];
    g.copy_d2h(d_total, &mut total)?;
    let mut got = vec![0u8; want.len() * 4];
    g.copy_d2h(d_list, &mut got)?;
    let got: Vec<u32> = got
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let ok = i32::from_le_bytes(total) as usize * 2 == want.len() && got == want;
    println!(
        "tile worklist: {} items {:5.1}us  {}",
        want.len() / 2,
        t * 1e6,
        if ok { "exact" } else { "MISMATCH" }
    );
    for p in [d_off, d_ptrs, d_list, d_total] {
        g.free(p)?;
    }
    Ok(!ok)
}
