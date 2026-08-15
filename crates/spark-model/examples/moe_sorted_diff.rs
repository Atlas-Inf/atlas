// SPDX-License-Identifier: AGPL-3.0-only

//! G9 execution-level diff rig for the nemotron sorted-MoE prefill pipeline.
//!
//! Launches the REAL kernels — moe_sort_by_expert, moe_w4a16_grouped_gemm_ptrtable,
//! moe_unpermute_reduce_indexed — on synthetic single-expert data and dumps the
//! routed output for a host-side numpy diff against the reference dequant+GEMV.
//!
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=nemotron-3-nano-30b-a3b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!       --example moe_sorted_diff --features cuda,gpu-examples

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const N_TOK: u32 = 64; // tokens, random routing across 128 experts
const K: u32 = 2688; // hidden (expert UP input dim)
const INTER: u32 = 1856; // moe inter (UP output / DOWN input)
const H: u32 = 2688; // hidden (DOWN output dim)
const TOPK: u32 = 6;
const N_EXP: u32 = 128;
const TE: u32 = N_TOK * TOPK; // total expanded = 384
const GROUP: u32 = 16;

#[allow(clippy::too_many_arguments)]
fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let stream = g.create_stream()?;

    // ── kernel handles (module "moe" / "moe_w4a16") ──
    let sort_k: KernelHandle = g
        .kernel("moe", "moe_sort_by_expert")
        .map_err(|e| anyhow::anyhow!("moe_sort_by_expert missing: {e}"))?;
    let up_k: KernelHandle = g
        .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")
        .map_err(|e| anyhow::anyhow!("grouped_gemm_ptrtable missing: {e}"))?;
    let unperm_k: KernelHandle = g
        .kernel("moe", "moe_unpermute_reduce_indexed")
        .map_err(|e| anyhow::anyhow!("unpermute missing: {e}"))?;
    let relu2_k: KernelHandle = g
        .kernel("relu2", "relu_squared_inplace")
        .map_err(|e| anyhow::anyhow!("relu_squared_inplace missing: {e}"))?;

    // ── synthetic weights: ONE expert's tables sized [n_exp] with expert 0..3
    //    holding distinct random weights; 4..128 alias expert 0 (empty weight
    //    coverage is enough — routing correctness is what this leg tests).
    let (inter, k, h, n_exp, te) = (INTER as usize, K as usize, H as usize, N_EXP as usize, TE as usize);
    let b_up = g.alloc(inter * k / 2)?;
    let s_up = g.alloc(inter * k / GROUP as usize)?;
    let b_dn = g.alloc(h * inter / 2)?;
    let s_dn = g.alloc(h * inter / GROUP as usize)?;
    // random-ish fill on host
    fill_random(g, b_up, inter * k / 2, 0x11)?;
    fill_sane_scales(g, s_up, inter * k / GROUP as usize)?;
    fill_random(g, b_dn, h * inter / 2, 0x33)?;
    fill_sane_scales(g, s_dn, h * inter / GROUP as usize)?;

    // scale2 arrays (per expert f32)
    let scale2 = g.alloc(n_exp * 4)?;
    {
        let mut v = vec![1.0f32; n_exp];
        g.copy_h2d(as_bytes(&mut v), scale2)?;
    }
    // ptr tables: [packed_ptr(u64), scale_ptr(u64)] — experts 4..128 alias expert 0
    let b_ptrs = g.alloc(n_exp * 8)?;
    let s_ptrs = g.alloc(n_exp * 8)?;
    {
        let mut bp = vec![0u64; n_exp];
        let mut sp = vec![0u64; n_exp];
        bp[0] = b_up.0 as u64;
        sp[0] = s_up.0 as u64;
        for e in 4..n_exp {
            bp[e] = b_up.0 as u64;
            sp[e] = s_up.0 as u64;
        }
        g.copy_h2d(as_bytes(&mut bp), b_ptrs)?;
        g.copy_h2d(as_bytes(&mut sp), s_ptrs)?;
    }
    let b_ptrs_dn = g.alloc(n_exp * 8)?;
    let s_ptrs_dn = g.alloc(n_exp * 8)?;
    {
        let mut bp = vec![0u64; n_exp];
        let mut sp = vec![0u64; n_exp];
        bp[0] = b_dn.0 as u64;
        sp[0] = s_dn.0 as u64;
        for e in 4..n_exp {
            bp[e] = b_dn.0 as u64;
            sp[e] = s_dn.0 as u64;
        }
        g.copy_h2d(as_bytes(&mut bp), b_ptrs_dn)?;
        g.copy_h2d(as_bytes(&mut sp), s_ptrs_dn)?;
    }

    // ── synthetic input A [4, K] + topk ids/weights ──
    let a = g.alloc(n_toks() * k * 2)?;
    fill_sane_bf16(g, a, n_toks() * k)?;
    let topk_ids = g.alloc(te * 4)?; // u32, deterministic pseudo-random routing
    {
        let mut ids = vec![0u32; te];
        let mut x: u32 = 0xDEAD_BEEF;
        for v in ids.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (x >> 8) % N_EXP;
        }
        g.copy_h2d(as_bytes(&mut ids), topk_ids)?;
    }
    let topk_w = g.alloc(te * 4)?; // f32, 1/6
    {
        let mut w = vec![1.0f32 / TOPK as f32; te];
        g.copy_h2d(as_bytes(&mut w), topk_w)?;
    }

    // ── sort arrays ──
    let sorted_tokens = g.alloc(te * 4)?;
    let sorted_experts = g.alloc(te * 4)?;
    let expert_offsets = g.alloc((n_exp + 1) * 4)?;
    let token_to_perm = g.alloc(te * 4)?;

    KernelLaunch::new(g, sort_k)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(topk_ids)
        .arg_ptr(sorted_tokens)
        .arg_ptr(sorted_experts)
        .arg_ptr(expert_offsets)
        .arg_ptr(token_to_perm)
        .arg_u32(TE)
        .arg_u32(N_EXP)
        .arg_u32(TOPK)
        .launch(stream)?;

    // ── grouped UP GEMM: C_up [te, inter] — max_m_tiles = worst case (all tokens one expert)
    let c_up = g.alloc(te * inter * 2)?;
    let max_m_tiles = if std::env::var("ATLAS_M_TILES_1").is_ok() {
        1
    } else {
        div_ceil(TE, 64).max(1)
    };
    KernelLaunch::new(g, up_k)
        .grid([div_ceil(INTER, 64), max_m_tiles, N_EXP])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b_ptrs)
        .arg_ptr(s_ptrs)
        .arg_ptr(scale2)
        .arg_ptr(c_up)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_tokens)
        .arg_u32(N_EXP)
        .arg_u32(INTER)
        .arg_u32(K)
        .launch(stream)?;

    // relu^2 — env ATLAS_NO_RELU2 skips (diagnostic: separate GEMM vs relu2 bugs)
    if std::env::var("ATLAS_NO_RELU2").is_err() {
        KernelLaunch::new(g, relu2_k)
            .grid([div_ceil(te as u32 * INTER, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c_up)
            .arg_u32(te as u32 * INTER)
            .launch(stream)?;
    }

    // ── grouped DOWN GEMM: C_dn [te, h], A = c_up (sorted rows), no gather ──
    let c_dn = g.alloc(te * h * 2)?;
    KernelLaunch::new(g, up_k)
        .grid([div_ceil(H, 64), max_m_tiles, N_EXP])
        .block([128, 1, 1])
        .arg_ptr(c_up)
        .arg_ptr(b_ptrs_dn)
        .arg_ptr(s_ptrs_dn)
        .arg_ptr(scale2)
        .arg_ptr(c_dn)
        .arg_ptr(expert_offsets)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(N_EXP)
        .arg_u32(H)
        .arg_u32(INTER)
        .launch(stream)?;

    // ── unpermute+reduce → out [4, h] ──
    let out = g.alloc(n_toks() * h * 2)?;
    KernelLaunch::new(g, unperm_k)
        .grid([N_TOK, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(c_dn)
        .arg_ptr(out)
        .arg_ptr(token_to_perm)
        .arg_ptr(topk_w)
        .arg_u32(H)
        .arg_u32(N_TOK)
        .arg_u32(TOPK)
        .launch(stream)?;

    g.synchronize(stream)?;

    // ── dump: c_up, c_dn, out, sort arrays, weights, A ──
    dump(g, "/tmp/g9_sorted_c_up.bin", c_up, te * inter * 2)?;
    dump(g, "/tmp/g9_sorted_c_dn.bin", c_dn, te * h * 2)?;
    dump(g, "/tmp/g9_sorted_out.bin", out, n_toks() * h * 2)?;
    dump(g, "/tmp/g9_sorted_tokens.bin", sorted_tokens, te * 4)?;
    dump(g, "/tmp/g9_sorted_offsets.bin", expert_offsets, (n_exp + 1) * 4)?;
    dump(g, "/tmp/g9_sorted_perm.bin", token_to_perm, te * 4)?;
    dump(g, "/tmp/g9_sorted_a.bin", a, n_toks() * k * 2)?;
    dump(g, "/tmp/g9_w_up.bin", b_up, inter * k / 2)?;
    dump(g, "/tmp/g9_s_up.bin", s_up, inter * k / GROUP as usize)?;
    dump(g, "/tmp/g9_w_dn.bin", b_dn, h * inter / 2)?;
    dump(g, "/tmp/g9_s_dn.bin", s_dn, h * inter / GROUP as usize)?;
    dump(g, "/tmp/g9_topk_w.bin", topk_w, te * 4)?;
    dump(g, "/tmp/g9_topk_ids.bin", topk_ids, te * 4)?;

    println!("dumps written to /tmp/g9_*.bin");
    Ok(())
}

fn n_toks() -> usize {
    N_TOK as usize
}

fn as_bytes<T: Copy>(v: &mut Vec<T>) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            v.as_mut_ptr() as *mut u8,
            v.len() * std::mem::size_of::<T>(),
        )
    }
}

/// E4M3-valid scale bytes cycling 0.5 / 1.0 / 2.0 (0x30 / 0x38 / 0x40).
fn fill_sane_scales(g: &dyn GpuBackend, dst: DevicePtr, bytes: usize) -> Result<()> {
    let pat = [0x30u8, 0x38, 0x40];
    let mut v = vec![0u8; bytes];
    for (i, b) in v.iter_mut().enumerate() {
        *b = pat[i % 3];
    }
    g.copy_h2d(&mut v[..], dst)?;
    Ok(())
}

/// Deterministic BF16 pattern: value = ((i*7)%13 - 6) as BF16.
fn fill_sane_bf16(g: &dyn GpuBackend, dst: DevicePtr, elems: usize) -> Result<()> {
    let mut v = vec![0u16; elems];
    for (i, w) in v.iter_mut().enumerate() {
        let x = ((i as i32 * 7) % 13) - 6;
        *w = f32_to_bf16(x as f32);
    }
    g.copy_h2d(as_bytes(&mut v), dst)?;
    Ok(())
}

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn fill_random(g: &dyn GpuBackend, dst: DevicePtr, bytes: usize, seed: u8) -> Result<()> {
    let mut v = vec![0u8; bytes];
    let mut x: u32 = 0x1234_5678u32 ^ ((seed as u32) << 24);
    for b in v.iter_mut() {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (x >> 16) as u8;
    }
    g.copy_h2d(&mut v[..], dst)?;
    Ok(())
}

fn dump(g: &dyn GpuBackend, path: &str, src: DevicePtr, bytes: usize) -> Result<()> {
    let mut v = vec![0u8; bytes];
    g.copy_d2h(src, &mut v[..])?;
    std::fs::write(path, &v)?;
    println!("wrote {path} ({bytes} B)");
    Ok(())
}
