// SPDX-License-Identifier: AGPL-3.0-only
//! Load-time host-side Marlin NVFP4 weight preparation (vLLM
//! `nvfp4_marlin_process_scales` / `_process_global_scale` port), shared by
//! the Nemotron MoE sidecar and the Marlin microbenchmarks.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::marlin_nvfp4::marlin_repack_w4;

/// Load-time Marlin packing of one modelopt NVFP4 projection: `w_host` is the
/// [N, K/2] packed E2M1 weight, `s_host` the [N, K/16] E4M3 block scales,
/// `scale2` the per-tensor f32 scale. B goes through `atlas_marlin_repack_w4`
/// (via `tmp_in`/`tmp_out`, each >= N*K/2 bytes); the scales are permuted and
/// rescaled, and the rescale is folded into the f32 global scale.
#[allow(clippy::too_many_arguments)]
pub fn marlin_pack_nvfp4(
    gpu: &dyn GpuBackend,
    repack: KernelHandle,
    w_host: &[u8],
    s_host: &[u8],
    scale2: f32,
    n: usize,
    k: usize,
    tmp_in: DevicePtr,
    tmp_out: DevicePtr,
    dest_w: DevicePtr,
    dest_s: DevicePtr,
    dest_gs: DevicePtr,
    sms: u32,
    smem: u32,
) -> Result<()> {
    let t = transpose_u32(w_host, n, k / 2);
    gpu.copy_h2d(&t, tmp_in)?;
    marlin_repack_w4(
        gpu, repack, tmp_in, tmp_out, k as i32, n as i32, sms, smem, 0,
    )?;
    gpu.synchronize(0)?;
    gpu.copy_d2d(tmp_out, dest_w, (k / 16) * (n * 16 / 8) * 4)?;
    let (proc, sf) = process_scales(s_host, n, k);
    gpu.copy_h2d(&proc, dest_s)?;
    gpu.copy_h2d(&process_global(scale2, sf).to_le_bytes(), dest_gs)?;
    Ok(())
}

const GROUP: usize = 16;

fn e4m3_to_f32(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0xf;
    let m = b & 7;
    let v = if e == 0 {
        m as f32 * 0.001953125
    } else if e == 15 && m == 7 {
        0.0
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7fffff;
    if exp == 255 {
        return sign | 0x7c00 | ((man >> 13) as u16);
    }
    let exp16 = exp - 127 + 15;
    if exp16 <= 0 {
        return sign;
    }
    if exp16 >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp16 as u16) << 10) | ((man >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = h & 0x3ff;
    let sign = h >> 15;
    let mut v = if exp == 0 {
        (man as f32 / 1024.0) * 2f32.powi(-14)
    } else {
        (1.0 + man as f32 / 1024.0) * 2f32.powi(exp - 15)
    };
    if sign == 1 {
        v = -v;
    }
    v
}

fn scale_perm() -> [usize; 64] {
    let mut p = [0usize; 64];
    let mut k = 0;
    for i in 0..8 {
        for j in 0..8 {
            p[k] = i + 8 * j;
            k += 1;
        }
    }
    p
}

fn process_scales(src_e4m3: &[u8], n: usize, k: usize) -> (Vec<u8>, f32) {
    let ng = k / GROUP;
    let mut t = vec![0f32; ng * n];
    for r in 0..n {
        for g in 0..ng {
            t[g * n + r] = e4m3_to_f32(src_e4m3[r * ng + g]);
        }
    }
    let perm = scale_perm();
    let mut perm_f = vec![0f32; ng * n];
    let cols = 64;
    let rows = (ng * n) / cols;
    for i in 0..rows {
        for j in 0..cols {
            perm_f[i * cols + j] = t[i * cols + perm[j]];
        }
    }
    let mut maxv = 0f32;
    for &v in &perm_f {
        if v > 0.0 {
            maxv = maxv.max(v * 128.0);
        }
    }
    let mut sf = 1.0f32;
    if maxv > 0.0 && maxv < 448.0 * 128.0 {
        sf = 2f32.powf((448.0 * 128.0 / maxv).log2().floor());
    }
    let mut half = vec![0u16; perm_f.len()];
    for (i, v) in perm_f.iter().enumerate() {
        half[i] = f32_to_f16_bits(v * sf);
    }
    let mut sw = half.clone();
    for i in (0..sw.len()).step_by(4) {
        sw[i] = half[i];
        sw[i + 1] = half[i + 2];
        sw[i + 2] = half[i + 1];
        sw[i + 3] = half[i + 3];
    }
    let mut e4 = vec![0u8; sw.len() * 2];
    for (i, h) in sw.iter().enumerate() {
        let f = f16_bits_to_f32(*h) * 128.0;
        let c = if f < 2.0 { 0.0 } else { f };
        let sh = f32_to_f16_bits(c) << 1;
        let b = sh.to_le_bytes();
        e4[i * 2] = b[0];
        e4[i * 2 + 1] = b[1];
    }
    let mut out = vec![0u8; ng * n];
    for row in 0..ng {
        for col in 0..n {
            out[row * n + col] = e4[row * (2 * n) + col * 2 + 1];
        }
    }
    (out, sf)
}

fn process_global(gs: f32, sf: f32) -> f32 {
    // vLLM nvfp4_marlin_process_global_scale for BF16:
    // exponent_bias = 2^(8-1) - 2^(2-1) = 126; then 2^(126-7) = 2^119.
    gs * 2f32.powi(126 - 7) / sf
}

fn transpose_u32(src: &[u8], rows: usize, cols_u8: usize) -> Vec<u8> {
    let cols = cols_u8 / 4;
    let mut dst = vec![0u8; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            let s = (r * cols + c) * 4;
            let d = (c * rows + r) * 4;
            dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
        }
    }
    dst
}
