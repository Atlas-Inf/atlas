// SPDX-License-Identifier: AGPL-3.0-only

//! Precision check for the `bf16a` transposed-K64 MoE prefill kernels (#74).
//!
//! The default `_t_k64` kernels quantize bf16 activations to UNSCALED e4m3
//! and requantize the dequantized NVFP4 weights to e4m3 before
//! `mma.m16n8k32.e4m3`. The `_bf16a` clones keep bf16 activations and
//! bf16-dequanted weights with `mma.m16n8k16.bf16` — the same operand
//! formats as the untransposed `moe_w4a16_grouped_gemm_ptrtable` kernel.
//!
//! This test runs all three on identical inputs over ragged expert row
//! counts and reports, per kernel, the max error vs the untransposed
//! bf16-MMA reference. Expectation: bf16a lands at bf16 rounding (~1e-2
//! relative or bitwise), e4m3 lands materially above it.
//!
//! GPU test: `#[ignore]` per repo convention. Run on the GB10 host with
//! ```text
//! cargo test -p spark-model --release --features cuda \
//!   moe_t_bf16a -- --ignored --nocapture
//! ```

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

/// Synthetic grouped-GEMM shape: K % 64 == 0 (K64 pipeline), N % 128 == 0
/// (N_TILE_LG boundary). Small enough to keep the test fast, large enough
/// to span two m-tiles on the 70-row expert.
const K: u32 = 256;
const N: u32 = 256;
const BF16: usize = 2;

/// Ragged per-expert row counts: one full tile, one spilling to a second
/// m-tile (exercises the persistent stride when grid_m=1), one partial.
const EXPERT_ROWS: [u32; 3] = [64, 70, 5];

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    let lsb = (b >> 16) & 1;
    let r = b.wrapping_add(0x7FFF + lsb);
    (r >> 16) as u16
}

fn bf16_at(buf: &[u8], elem: usize) -> f32 {
    let lo = buf[2 * elem] as u32;
    let hi = buf[2 * elem + 1] as u32;
    f32::from_bits(((hi << 8) | lo) << 16)
}

fn upload(gpu: &dyn GpuBackend, b: &[u8]) -> DevicePtr {
    let p = gpu.alloc(b.len().max(256)).unwrap();
    gpu.copy_h2d_async(b, p, gpu.default_stream()).unwrap();
    p
}

fn download(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut raw = vec![0u8; n];
    gpu.copy_d2h(p, &mut raw).unwrap();
    raw
}

/// One expert's NVFP4 table in CHECKPOINT layout: packed e2m1 bytes
/// `[N][K/2]` (lo nibble = even k) + e4m3 scale bytes `[N][K/16]`.
/// Scale bytes are masked to 0x01..=0x7E so no byte decodes to NaN.
struct Nvfp4Expert {
    packed: Vec<u8>,
    scales: Vec<u8>,
    scale2: f32,
}

fn synth_expert(rng: &mut Rng) -> Nvfp4Expert {
    let packed: Vec<u8> = (0..(N * K / 2) as usize)
        .map(|_| (rng.next_u64() & 0xFF) as u8)
        .collect();
    let scales: Vec<u8> = (0..(N * K / 16) as usize)
        .map(|_| 0x01 | ((rng.next_u64() as u8) & 0x7E))
        .collect();
    Nvfp4Expert {
        packed,
        scales,
        scale2: 1.0 + (rng.next_u64() % 4) as f32 * 0.25,
    }
}

/// Host-side equivalent of the `_t` transpose: `[N][K/2]` packed bytes and
/// `[N][K/16]` scale bytes become `[K/2][N]` and `[K/16][N]` — the exact
/// layouts `*_t_k64` kernels index (`B[(k/2)*N + n]`, `S[(k/16)*N + n]`).
fn transpose_table(src: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let mut t = vec![0u8; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = src[r * cols + c];
        }
    }
    t
}

struct Tables {
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    allocs: Vec<DevicePtr>,
}

/// Upload per-expert tables and build the u64 device pointer arrays.
fn make_tables(gpu: &dyn GpuBackend, per_expert: &[(Vec<u8>, Vec<u8>)], scale2: &[f32]) -> Tables {
    let mut packed_ptrs = Vec::new();
    let mut scale_ptrs = Vec::new();
    let mut allocs = Vec::new();
    for (packed, scales) in per_expert {
        let p = upload(gpu, packed);
        let s = upload(gpu, scales);
        packed_ptrs.extend_from_slice(&p.0.to_le_bytes());
        scale_ptrs.extend_from_slice(&s.0.to_le_bytes());
        allocs.extend_from_slice(&[p, s]);
    }
    let scale2_bytes: Vec<u8> = scale2.iter().flat_map(|f| f.to_le_bytes()).collect();
    Tables {
        packed_ptrs: upload(gpu, &packed_ptrs),
        scale_ptrs: upload(gpu, &scale_ptrs),
        scale2_vals: upload(gpu, &scale2_bytes),
        allocs,
    }
}

/// max|Δ| and max relative error between two bf16 outputs.
fn max_err(got: &[u8], want: &[u8]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut max_abs, mut max_rel) = (0.0f64, 0.0f64);
    for i in 0..got.len() / 2 {
        let (g, w) = (bf16_at(got, i) as f64, bf16_at(want, i) as f64);
        let d = (g - w).abs();
        max_abs = f64::max(max_abs, d);
        max_rel = f64::max(max_rel, d / w.abs().max(1e-6));
    }
    (max_abs, max_rel)
}

fn report(label: &str, got: &[u8], want: &[u8]) {
    let (max_abs, max_rel) = max_err(got, want);
    println!(
        "  {label}: {} (max|Δ| {max_abs:.4e}, max rel {max_rel:.4e})",
        if got == want { "IDENTICAL" } else { "DIFFER" },
    );
}

#[test]
#[ignore]
fn moe_t_bf16a_matches_untransposed_bf16() {
    // qwen3.8-flash-next/nvfp4 ships the moe_w4a16 module with the bf16a
    // clones (its .cu is the shared qwen3.6-35b-a3b source).
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 target");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let stream = gpu.default_stream();
    let m = "moe_w4a16";
    let k_ref = gpu.kernel(m, "moe_w4a16_grouped_gemm_ptrtable").unwrap();
    let k_t = gpu
        .kernel(m, "moe_w4a16_grouped_gemm_ptrtable_t_k64")
        .unwrap();
    let k_t_bf16a = gpu
        .kernel(m, "moe_w4a16_grouped_gemm_ptrtable_t_k64_bf16a")
        .unwrap();
    let k_fused = gpu.kernel(m, "moe_w4a16_fused_gate_up_t_k64").unwrap();
    let k_fused_bf16a = gpu
        .kernel(m, "moe_w4a16_fused_gate_up_t_k64_bf16a")
        .unwrap();

    let ne = EXPERT_ROWS.len() as u32;
    let total: u32 = EXPERT_ROWS.iter().sum();
    let max_m_tiles = EXPERT_ROWS.iter().max().unwrap().div_ceil(64);

    // Ragged expert offsets + identity sorted_token_ids (A is already in
    // sorted order — the gather is a no-op, keeping the test self-contained).
    let mut eoff = vec![0i32];
    let mut acc = 0u32;
    for &r in &EXPERT_ROWS {
        acc += r;
        eoff.push(acc as i32);
    }
    let eoff_dev = upload(
        &gpu,
        &eoff
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let tok_dev = upload(
        &gpu,
        &(0..total as i32)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );

    // Activations [total, K] bf16 in [-4, 4].
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let a_host: Vec<u8> = (0..total as usize * K as usize)
        .flat_map(|_| {
            let u = rng.next_u64();
            let v = ((u >> 40) as f64 / (1u64 << 24) as f64) * 8.0 - 4.0;
            bf16(v as f32).to_le_bytes()
        })
        .collect();
    let a = upload(&gpu, &a_host);

    // Down-projection-style GEMM: B [N, K/2] NVFP4 per expert.
    let experts: Vec<Nvfp4Expert> = (0..ne).map(|_| synth_expert(&mut rng)).collect();
    let scale2: Vec<f32> = experts.iter().map(|e| e.scale2).collect();
    let orig = make_tables(
        &gpu,
        &experts
            .iter()
            .map(|e| (e.packed.clone(), e.scales.clone()))
            .collect::<Vec<_>>(),
        &scale2,
    );
    let trans = make_tables(
        &gpu,
        &experts
            .iter()
            .map(|e| {
                (
                    transpose_table(&e.packed, N as usize, (K / 2) as usize),
                    transpose_table(&e.scales, N as usize, (K / 16) as usize),
                )
            })
            .collect::<Vec<_>>(),
        &scale2,
    );

    let out_bytes = (total * N) as usize * BF16;
    let c_ref = gpu.alloc(out_bytes).unwrap();
    let c_e4m3 = gpu.alloc(out_bytes).unwrap();
    let c_bf16a = gpu.alloc(out_bytes).unwrap();

    // ── Down GEMM: untransposed bf16 reference vs e4m3 vs bf16a ─────────
    ops::moe_w4a16_grouped_gemm_ptrtable(
        &gpu,
        k_ref,
        a,
        orig.packed_ptrs,
        orig.scale_ptrs,
        orig.scale2_vals,
        c_ref,
        eoff_dev,
        tok_dev,
        ne,
        N,
        K,
        max_m_tiles,
        stream,
    )
    .unwrap();
    for (k, c, name) in [
        (k_t, c_e4m3, "t_k64 (e4m3)"),
        (k_t_bf16a, c_bf16a, "t_k64_bf16a"),
    ] {
        ops::moe_w4a16_grouped_gemm_ptrtable_n128(
            &gpu,
            k,
            a,
            trans.packed_ptrs,
            trans.scale_ptrs,
            trans.scale2_vals,
            c,
            eoff_dev,
            tok_dev,
            ne,
            N,
            K,
            max_m_tiles,
            stream,
        )
        .unwrap();
        gpu.synchronize(stream).unwrap();
        report(
            &format!("down {name} vs untransposed bf16"),
            &download(&gpu, c, out_bytes),
            &download(&gpu, c_ref, out_bytes),
        );
    }

    // ── Strided-grid equivalence: bf16a must honor the #68 m-tile striding
    // (grid.y = 1 still covers the 70-row expert via the persistent loop).
    ops::moe_w4a16_grouped_gemm_ptrtable_n128(
        &gpu,
        k_t_bf16a,
        a,
        trans.packed_ptrs,
        trans.scale_ptrs,
        trans.scale2_vals,
        c_bf16a,
        eoff_dev,
        tok_dev,
        ne,
        N,
        K,
        1,
        stream,
    )
    .unwrap();
    gpu.synchronize(stream).unwrap();
    report(
        "down t_k64_bf16a grid_m=1 (strided) vs untransposed bf16",
        &download(&gpu, c_bf16a, out_bytes),
        &download(&gpu, c_ref, out_bytes),
    );

    // ── Fused gate+up: two fresh experts, bf16a fused vs per-projection
    // untransposed references ─────────────────────────────────────────────
    let gates: Vec<Nvfp4Expert> = (0..ne).map(|_| synth_expert(&mut rng)).collect();
    let ups: Vec<Nvfp4Expert> = (0..ne).map(|_| synth_expert(&mut rng)).collect();
    let tab = |xs: &[Nvfp4Expert], tr: bool| {
        make_tables(
            &gpu,
            &xs.iter()
                .map(|e| {
                    if tr {
                        (
                            transpose_table(&e.packed, N as usize, (K / 2) as usize),
                            transpose_table(&e.scales, N as usize, (K / 16) as usize),
                        )
                    } else {
                        (e.packed.clone(), e.scales.clone())
                    }
                })
                .collect::<Vec<_>>(),
            &xs.iter().map(|e| e.scale2).collect::<Vec<_>>(),
        )
    };
    let g_orig = tab(&gates, false);
    let u_orig = tab(&ups, false);
    let g_t = tab(&gates, true);
    let u_t = tab(&ups, true);

    let cg_ref = gpu.alloc(out_bytes).unwrap();
    let cu_ref = gpu.alloc(out_bytes).unwrap();
    let cg = gpu.alloc(out_bytes).unwrap();
    let cu = gpu.alloc(out_bytes).unwrap();
    ops::moe_w4a16_grouped_gemm_ptrtable(
        &gpu,
        k_ref,
        a,
        g_orig.packed_ptrs,
        g_orig.scale_ptrs,
        g_orig.scale2_vals,
        cg_ref,
        eoff_dev,
        tok_dev,
        ne,
        N,
        K,
        max_m_tiles,
        stream,
    )
    .unwrap();
    ops::moe_w4a16_grouped_gemm_ptrtable(
        &gpu,
        k_ref,
        a,
        u_orig.packed_ptrs,
        u_orig.scale_ptrs,
        u_orig.scale2_vals,
        cu_ref,
        eoff_dev,
        tok_dev,
        ne,
        N,
        K,
        max_m_tiles,
        stream,
    )
    .unwrap();
    for (k, name) in [
        (k_fused, "fused t_k64 (e4m3)"),
        (k_fused_bf16a, "fused t_k64_bf16a"),
    ] {
        ops::moe_w4a16_fused_gate_up_k64_n128(
            &gpu,
            k,
            a,
            g_t.packed_ptrs,
            g_t.scale_ptrs,
            g_t.scale2_vals,
            u_t.packed_ptrs,
            u_t.scale_ptrs,
            u_t.scale2_vals,
            cg,
            cu,
            eoff_dev,
            tok_dev,
            ne,
            N,
            K,
            max_m_tiles,
            stream,
        )
        .unwrap();
        gpu.synchronize(stream).unwrap();
        report(
            &format!("gate {name} vs untransposed bf16"),
            &download(&gpu, cg, out_bytes),
            &download(&gpu, cg_ref, out_bytes),
        );
        report(
            &format!("up {name} vs untransposed bf16"),
            &download(&gpu, cu, out_bytes),
            &download(&gpu, cu_ref, out_bytes),
        );
    }

    // ── Verdict: bf16a must sit at bf16 rounding; e4m3 materially above ──
    let (bf16a_abs, bf16a_rel) = max_err(
        &download(&gpu, c_bf16a, out_bytes),
        &download(&gpu, c_ref, out_bytes),
    );
    let (e4m3_abs, e4m3_rel) = max_err(
        &download(&gpu, c_e4m3, out_bytes),
        &download(&gpu, c_ref, out_bytes),
    );
    println!(
        "VERDICT down: bf16a rel {bf16a_rel:.3e} vs e4m3 rel {e4m3_rel:.3e} \
         (abs {bf16a_abs:.3e} vs {e4m3_abs:.3e})"
    );
    assert!(
        bf16a_rel <= 2e-2,
        "bf16a rel error {bf16a_rel:.3e} exceeds bf16-rounding bound"
    );
    assert!(
        bf16a_rel < e4m3_rel,
        "expected bf16a ({bf16a_rel:.3e}) < e4m3 ({e4m3_rel:.3e})"
    );

    for p in [
        a,
        eoff_dev,
        tok_dev,
        orig.packed_ptrs,
        orig.scale_ptrs,
        orig.scale2_vals,
        trans.packed_ptrs,
        trans.scale_ptrs,
        trans.scale2_vals,
        g_orig.packed_ptrs,
        u_orig.packed_ptrs,
        g_t.packed_ptrs,
        u_t.packed_ptrs,
        c_ref,
        c_e4m3,
        c_bf16a,
        cg_ref,
        cu_ref,
        cg,
        cu,
    ] {
        gpu.free(p).ok();
    }
    for p in orig
        .allocs
        .iter()
        .chain(trans.allocs.iter())
        .chain(g_orig.allocs.iter())
        .chain(u_orig.allocs.iter())
        .chain(g_t.allocs.iter())
        .chain(u_t.allocs.iter())
    {
        gpu.free(*p).ok();
    }
}
