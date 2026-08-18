// SPDX-License-Identifier: AGPL-3.0-only

use super::{GROUP, MarlinSidecar, SMEM, SMS, SORTED_CAP};
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

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

fn pack_proj(
    gpu: &dyn GpuBackend,
    w: &QuantizedWeight,
    n: usize,
    k: usize,
    tmp_in: DevicePtr,
    tmp_out: DevicePtr,
    dest_w: DevicePtr,
    dest_s: DevicePtr,
    dest_gs: DevicePtr,
    repack: spark_runtime::gpu::KernelHandle,
) -> Result<()> {
    let packed_u8 = n * (k / 2);
    let mut host = vec![0u8; packed_u8];
    gpu.copy_d2h(w.weight, &mut host)?;
    let t = transpose_u32(&host, n, k / 2);
    gpu.copy_h2d(&t, tmp_in)?;
    ops::marlin_repack_w4(
        gpu, repack, tmp_in, tmp_out, k as i32, n as i32, SMS, SMEM, 0,
    )?;
    gpu.synchronize(0)?;
    let w_bytes = (k / 16) * (n * 16 / 8) * 4;
    gpu.copy_d2d(tmp_out, dest_w, w_bytes)?;

    let ng = k / GROUP;
    let mut sc = vec![0u8; n * ng];
    gpu.copy_d2h(w.weight_scale, &mut sc)?;
    let (proc, sf) = process_scales(&sc, n, k);
    gpu.copy_h2d(&proc, dest_s)?;
    let gs = process_global(w.weight_scale_2, sf);
    gpu.copy_h2d(&gs.to_le_bytes(), dest_gs)?;
    Ok(())
}

impl MarlinSidecar {
    pub fn try_build(
        gpu: &dyn GpuBackend,
        experts: &[crate::weight_map::NemotronExpertWeight],
        up_n: usize,
        up_k: usize,
        down_n: usize,
        down_k: usize,
    ) -> Result<Option<Self>> {
        if std::env::var_os("ATLAS_MOE_MARLIN").is_none() {
            return Ok(None);
        }
        let moe_up =
            crate::layers::try_kernel(gpu, "marlin_moe_nvfp4", "atlas_marlin_moe_nvfp4_m8");
        let moe_down =
            crate::layers::try_kernel(gpu, "marlin_moe_nvfp4", "atlas_marlin_moe_nvfp4_m8_k64n128");
        let lin_up = crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_m8");
        let lin_down =
            crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_m8_k64n128");
        let cfg4_up =
            crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_cfg4");
        let cfg4_down =
            crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_cfg4_k64n128");
        let slot_up =
            crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_m8_allslots");
        let slot_dn = crate::layers::try_kernel(
            gpu,
            "marlin_nvfp4_gemm",
            "atlas_marlin_nvfp4_m8_k64n128_allslots",
        );
        let pack = crate::layers::try_kernel(gpu, "marlin_pack_slots", "atlas_marlin_pack_slots");
        let scatter =
            crate::layers::try_kernel(gpu, "marlin_scatter_slots", "atlas_marlin_scatter_slots");
        let gather =
            crate::layers::try_kernel(gpu, "marlin_scatter_slots", "atlas_marlin_gather_slots");
        let repack = crate::layers::try_kernel(gpu, "marlin_repack", "atlas_marlin_repack_w4");
        let align = crate::layers::try_kernel(gpu, "marlin_align", "atlas_marlin_align_block8");
        let repeat = crate::layers::try_kernel(gpu, "marlin_row_repeat", "atlas_row_repeat_bf16");
        let pack_rows =
            crate::layers::try_kernel(gpu, "marlin_pack_rows", "atlas_marlin_pack_rows");
        if lin_up.0 == 0
            || lin_down.0 == 0
            || moe_up.0 == 0
            || moe_down.0 == 0
            || repack.0 == 0
            || align.0 == 0
            || repeat.0 == 0
            || pack_rows.0 == 0
        {
            tracing::warn!("ATLAS_MOE_MARLIN set but kernels missing; leaving GEMV");
            return Ok(None);
        }
        let e = experts.len();
        let up_w_b = (up_k / 16) * (up_n * 16 / 8) * 4;
        let down_w_b = (down_k / 16) * (down_n * 16 / 8) * 4;
        let up_s_b = (up_k / GROUP) * up_n;
        let down_s_b = (down_k / GROUP) * down_n;
        let up_w = gpu.alloc(e * up_w_b)?;
        let down_w = gpu.alloc(e * down_w_b)?;
        let up_s = gpu.alloc(e * up_s_b)?;
        let down_s = gpu.alloc(e * down_s_b)?;
        let up_gs = gpu.alloc(e * 4)?;
        let down_gs = gpu.alloc(e * 4)?;
        let tmp_in = gpu.alloc(up_w_b.max(down_w_b))?;
        let tmp_out = gpu.alloc(up_w_b.max(down_w_b))?;
        for (i, ex) in experts.iter().enumerate() {
            pack_proj(
                gpu,
                &ex.up_proj,
                up_n,
                up_k,
                tmp_in,
                tmp_out,
                DevicePtr(up_w.0 + (i * up_w_b) as u64),
                DevicePtr(up_s.0 + (i * up_s_b) as u64),
                DevicePtr(up_gs.0 + (i * 4) as u64),
                repack,
            )?;
            pack_proj(
                gpu,
                &ex.down_proj,
                down_n,
                down_k,
                tmp_in,
                tmp_out,
                DevicePtr(down_w.0 + (i * down_w_b) as u64),
                DevicePtr(down_s.0 + (i * down_s_b) as u64),
                DevicePtr(down_gs.0 + (i * 4) as u64),
                repack,
            )?;
        }
        let _ = gpu.free(tmp_in);
        let _ = gpu.free(tmp_out);
        tracing::info!("Marlin sidecar packed {e} experts UP {up_n}x{up_k} DOWN {down_n}x{down_k}");
        Ok(Some(Self {
            up_w,
            up_s,
            up_gs,
            down_w,
            down_s,
            down_gs,
            locks: gpu.alloc(ops::MARLIN_SLOTS as usize * 256 * 4)?,
            c_tmp: gpu.alloc(ops::MARLIN_SLOTS as usize * 16 * down_n.max(up_n) * 4)?,
            sorted_ids: gpu.alloc(SORTED_CAP * 4)?,
            expert_ids: gpu.alloc(256 * 4)?,
            n_post: gpu.alloc(4)?,
            a_exp: gpu.alloc(16 * 6 * up_k * 2)?,
            moe_up_k: moe_up,
            moe_down_k: moe_down,
            lin_up_k: lin_up,
            lin_down_k: lin_down,
            cfg4_up_k: cfg4_up,
            cfg4_down_k: cfg4_down,
            pack_rows_k: pack_rows,
            lin_up_out: gpu.alloc(8 * up_n * 2)?,
            lin_dn_out: gpu.alloc(8 * down_n * 2)?,
            pack_k: pack,
            scatter_k: scatter,
            gather_k: gather,
            slot_up_k: slot_up,
            slot_dn_k: slot_dn,
            slot_eids: gpu.alloc(ops::MARLIN_SLOTS as usize * 4)?,
            slot_map: gpu.alloc(ops::MARLIN_SLOTS as usize * ops::MARLIN_M_TILE as usize * 4)?,
            slot_a: gpu
                .alloc(ops::MARLIN_SLOTS as usize * ops::MARLIN_M_TILE as usize * up_k * 2)?,
            slot_up: gpu
                .alloc(ops::MARLIN_SLOTS as usize * ops::MARLIN_M_TILE as usize * up_n * 2)?,
            slot_dn: gpu
                .alloc(ops::MARLIN_SLOTS as usize * ops::MARLIN_M_TILE as usize * down_n * 2)?,
            slot_bars: gpu.alloc(ops::MARLIN_SLOTS as usize * 4)?,
            align_k: align,
            repeat_k: repeat,
            up_n: up_n as i32,
            up_k: up_k as i32,
            down_n: down_n as i32,
            down_k: down_k as i32,
            e: e as i32,
        }))
    }
}
