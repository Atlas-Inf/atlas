// SPDX-License-Identifier: AGPL-3.0-only

//! M-invariance of the NVFP4 W4A4 MMQ dense-FFN prefill pipeline.
//!
//! The question this answers (job 354): for IDENTICAL bf16 activation rows,
//! do the `atlas_nvfp4_mmq{16,32,64,128}_{nc,wc}` tiles plus the
//! `atlas_nvfp4_quantize_bf16` activation quantizer produce bitwise-identical
//! per-row outputs regardless of M (rows in the launch) and of the M tile
//! picked? The 27B's chunked prefill runs the dense FFN at M=53 (mmq64 tile)
//! vs M=5 (mmq16 tile) depending on `--max-prefill-tokens`; if the tile is
//! not M-invariant, that alone injects the observed `moe_out` drift.
//!
//! What is proven here, in order:
//!   1. `atlas_nvfp4_quantize_bf16` is token-local: a 53-row launch and a
//!      5-row launch on a sub-range produce byte-identical y rows.
//!   2. Tile invariance on a shared quantized buffer: each tile's slice
//!      launches (rows [32,48) and [48,53)) reproduce the mmq64 full-launch
//!      rows bitwise, and the all-rows mmq128 launch matches mmq64 too.
//!   3. End-to-end: quantize at the chunk's own M (the production pattern)
//!      then the tile that M selects — slices reproduce the contiguous
//!      reference rows bitwise.
//!
//! GPU test: `#[ignore]` per repo convention. Run on the GB10 host with
//! ```text
//! cargo test -p spark-model --release --features cuda \
//!   nvfp4_mmq_m_invariance -- --ignored --nocapture
//! ```

use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;

/// The 27B dense-FFN gate/up shape: [inter=17408, hidden=5120].
const K: u32 = 5120;
const N: u32 = 17408;
const M: u32 = 53;
const BF16: usize = 2;

/// Deterministic PRNG (xorshift64*): fixed bit patterns beat realism here —
/// the test is about launch-shape invariance, not distribution.
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

/// f32 in [-4, 4] (the SiLU input range) → bf16 RNE bits.
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

struct Mmq {
    repack: KernelHandle,
    quant: KernelHandle,
    nc: [KernelHandle; 4], // tiles 16, 32, 64, 128
    wc: [KernelHandle; 4],
}

impl Mmq {
    fn load(gpu: &dyn GpuBackend) -> Self {
        let k = |func: &str| gpu.kernel("nvfp4_mmq", func).expect(func);
        Self {
            repack: k("atlas_nvfp4_repack"),
            quant: k("atlas_nvfp4_quantize_bf16"),
            nc: [
                k("atlas_nvfp4_mmq16_nc"),
                k("atlas_nvfp4_mmq32_nc"),
                k("atlas_nvfp4_mmq64_nc"),
                k("atlas_nvfp4_mmq128_nc"),
            ],
            wc: [
                k("atlas_nvfp4_mmq16_wc"),
                k("atlas_nvfp4_mmq32_wc"),
                k("atlas_nvfp4_mmq64_wc"),
                k("atlas_nvfp4_mmq128_wc"),
            ],
        }
    }

    /// The tile index `dense_ffn.rs`'s dispatch picks for this M (small
    /// tiles ON): m≤16 → 16, ≤32 → 32, ≤64 → 64, else 128.
    fn tile_for(m: u32) -> usize {
        if m <= 16 {
            0
        } else if m <= 32 {
            1
        } else if m <= 64 {
            2
        } else {
            3
        }
    }

    fn mmq_x(tile: usize) -> u32 {
        [16, 32, 64, 128][tile]
    }
}

/// `block_fp4_mmq` size: 4 ue4m3 scale words (16 B) + 128 packed e2m1 bytes.
const FP4_BLOCK_BYTES: usize = 144;

/// K-blocks per row: `block_fp4_mmq` covers 256 activation values.
fn k_blocks(k: u32) -> usize {
    k.div_ceil(256) as usize
}

/// Activation y bytes for one logical row, gathered per k-block. The
/// quantizer writes `[k_block][ne1]` block-column-major — `ib =
/// k_block * ne1 + i1` (`atlas_nvfp4_quantize_bf16`, nvfp4_mmq.cu) — so
/// logical row `r`'s k-block `j` lives at `(j * ne1 + r) * 144`, NOT at
/// `r * k_blocks * 144`. The 53-row and 5-row launches therefore lay the
/// same rows at different byte offsets even when every quantized value
/// is identical.
fn y_row_gather(y: &[u8], ne1: usize, k: u32, r: usize) -> Vec<u8> {
    let nkb = k_blocks(k);
    let mut out = Vec::with_capacity(nkb * FP4_BLOCK_BYTES);
    for j in 0..nkb {
        let off = (j * ne1 + r) * FP4_BLOCK_BYTES;
        out.extend_from_slice(&y[off..off + FP4_BLOCK_BYTES]);
    }
    out
}

/// Gather `rows` logical rows starting at `r0` out of a y buffer produced
/// by an `ne1`-row quantize launch, into the `[k_block][rows]` layout a
/// `ncols_y = rows` mmq launch expects (mul_mat_q_process_tile walks y as
/// `y + ncols_y * (kb0 * qk / ne_block) * sz`, q4k_vendor/mmq.cuh:3619).
fn y_slice_gather(y: &[u8], ne1: usize, k: u32, r0: usize, rows: usize) -> Vec<u8> {
    let nkb = k_blocks(k);
    let mut out = vec![0u8; nkb * rows * FP4_BLOCK_BYTES];
    for j in 0..nkb {
        for i in 0..rows {
            let src = (j * ne1 + r0 + i) * FP4_BLOCK_BYTES;
            let dst = (j * rows + i) * FP4_BLOCK_BYTES;
            out[dst..dst + FP4_BLOCK_BYTES].copy_from_slice(&y[src..src + FP4_BLOCK_BYTES]);
        }
    }
    out
}

/// Quantize `rows` rows of `x` ([M, K] bf16, contiguous) starting at row
/// `r0` into `y` at offset 0 — exactly the `nvfp4_mmq_quantize_act` call
/// `forward_prefill_inner` makes (contiguous rows: s01 == ne00 == K).
fn quantize(
    gpu: &dyn GpuBackend,
    k: &Mmq,
    x: DevicePtr,
    r0: u32,
    rows: u32,
    y: DevicePtr,
    stream: u64,
) {
    let x_off = x.offset((r0 as usize) * K as usize * BF16);
    ops::nvfp4_mmq_quantize_act(gpu, k.quant, x_off, y, rows, K, stream).unwrap();
}

/// One `nvfp4_mmq_gemm_tiled` launch on a y buffer already laid out for
/// `ncols_y = rows` (`[k_block][rows]` block-column-major — see
/// `y_slice_gather`), writing `out` [rows, N] bf16. `grid.y` stays 1 —
/// matching production, which only calls this when `m <= mmq_x`.
fn gemm(
    gpu: &dyn GpuBackend,
    k: &Mmq,
    tile: usize,
    y: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    rows: u32,
    stream: u64,
) {
    ops::nvfp4_mmq_gemm_tiled(
        gpu,
        k.nc[tile],
        k.wc[tile],
        Mmq::mmq_x(tile),
        y,
        w,
        out,
        rows,
        N,
        K,
        stream,
    )
    .unwrap();
}

/// Bitwise + max|Δ| comparison of `rows` rows of N bf16 values. Records a
/// failure line instead of asserting so every stage's comparison prints
/// before the final verdict.
fn check_rows_identical(
    label: &str,
    got: &[u8],
    want: &[u8],
    rows: usize,
    failures: &mut Vec<String>,
) {
    let n = rows * N as usize;
    if got.len() != want.len() || got.len() != n * BF16 {
        failures.push(format!(
            "{label}: length mismatch got={} want={} expected={}",
            got.len(),
            want.len(),
            n * BF16
        ));
        return;
    }
    let mut mismatched = 0usize;
    let (mut worst, mut worst_d) = (0usize, 0.0f64);
    for i in 0..n {
        if got[2 * i] != want[2 * i] || got[2 * i + 1] != want[2 * i + 1] {
            mismatched += 1;
            let d = (bf16_at(got, i) - bf16_at(want, i)).abs() as f64;
            if d > worst_d {
                worst_d = d;
                worst = i;
            }
        }
    }
    println!(
        "  {label}: {} (mismatched elems {mismatched}/{n}, max|Δ| {worst_d:.4e} at row {} col {})",
        if got == want { "IDENTICAL" } else { "DIFFER" },
        worst / N as usize,
        worst % N as usize,
    );
    if got != want {
        failures.push(format!(
            "{label}: {mismatched} bf16 elements differ, max|Δ| {worst_d:.4e}"
        ));
    }
}

#[test]
#[ignore]
fn nvfp4_mmq_m_invariance() {
    // qwen3.6-27b/nvfp4 is the kernel dir qwen3.8-27b resolves to (the two
    // targets are config-identical; only the 27B tree carries nvfp4_mmq.cu).
    let set = atlas_kernels::ptx_for_exact_target("qwen3.6-27b", "nvfp4")
        .expect("qwen3.6-27b/nvfp4 target with nvfp4_mmq");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let kk = Mmq::load(&gpu);
    let stream = gpu.default_stream();

    // ── Deterministic activations: [53, 5120] bf16 ──────────────────────
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let x_host: Vec<u16> = (0..M as usize * K as usize)
        .map(|_| {
            let u = rng.next_u64();
            let v = ((u >> 40) as f64 / (1u64 << 24) as f64) * 8.0 - 4.0;
            bf16(v as f32)
        })
        .collect();
    let x_host: Vec<u8> = x_host.iter().flat_map(|b| b.to_le_bytes()).collect();
    let x = upload(&gpu, &x_host);

    // ── Synthetic NVFP4 weight in CHECKPOINT layout, repacked through the
    // production `atlas_nvfp4_repack` (same path as ensure_nvfp4_mmq_weight):
    // packed [N, K/2] e2m1 nibbles + scales [N, K/16] e4m3 bytes.
    let packed_host: Vec<u8> = (0..N as usize * (K as usize / 2))
        .map(|_| (rng.next_u64() >> 32) as u8)
        .collect();
    // Keep scale bytes in 0x01..=0x7E — e4m3 normals/denorms, never NaN.
    let scales_host: Vec<u8> = (0..N as usize * (K as usize / 16))
        .map(|_| 1 + ((rng.next_u64() >> 40) % 0x7E) as u8)
        .collect();
    let packed = upload(&gpu, &packed_host);
    let scales = upload(&gpu, &scales_host);
    let w = gpu.alloc(ops::nvfp4_mmq_weight_bytes(N, K)).unwrap();
    ops::nvfp4_mmq_repack(&gpu, kk.repack, packed, scales, w, N, K, stream).unwrap();
    gpu.synchronize(stream).unwrap();

    let y_bytes = ops::fp4_act_scratch_bytes(M, K);
    let out_bytes = M as usize * N as usize * BF16;

    let mut failures: Vec<String> = Vec::new();

    // ── 1. Quantizer is token-local ──────────────────────────────────────
    // y_full rows [0,53) vs y_part rows [0,5) quantized from x[48..53).
    // The y buffer is `[k_block][ne1]` block-column-major — `ib =
    // k_block * ne1 + i1` — so the same logical row lands at different
    // byte offsets in the two launches. Gather per logical row before
    // comparing (quantizer amax/scale are computed per 16-element
    // sub-block of the row itself — no cross-row dependence by
    // construction — so byte-identical blocks are the correct verdict).
    let y_full = gpu.alloc(y_bytes).unwrap();
    let y_part = gpu.alloc(y_bytes).unwrap();
    quantize(&gpu, &kk, x, 0, M, y_full, stream);
    quantize(&gpu, &kk, x, 48, 5, y_part, stream);
    gpu.synchronize(stream).unwrap();
    let yf = download(&gpu, y_full, y_bytes);
    let yp = download(&gpu, y_part, y_bytes);
    for r in 0..5usize {
        let got = y_row_gather(&yp, 5, K, r);
        let want = y_row_gather(&yf, M as usize, K, 48 + r);
        if got != want {
            let first = got
                .iter()
                .zip(want.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            failures.push(format!(
                "quantize row {} differs between the 53-row and 5-row launches \
                 (first diff at byte {first}, k-block {}) — a row's quantized \
                 output depends on launch M, not just the row",
                48 + r,
                first / FP4_BLOCK_BYTES
            ));
        }
    }
    println!(
        "quantize: rows 48..52 of the 53-row launch vs the 5-row launch, gathered per logical row: {}",
        if failures.is_empty() {
            "BYTE-IDENTICAL"
        } else {
            "DIFFER"
        }
    );

    // ── 2. Tile invariance on the shared quantized buffer ────────────────
    // Reference: mmq64 over all 53 rows (the tile M=53 picks).
    let out_ref = gpu.alloc(out_bytes).unwrap();
    gemm(&gpu, &kk, 2, y_full, w, out_ref, M, stream);
    gpu.synchronize(stream).unwrap();
    let reference = download(&gpu, out_ref, out_bytes);
    let want_rows = |r0: usize, rows: usize| -> Vec<u8> {
        (r0..r0 + rows)
            .flat_map(|r| reference[r * N as usize * BF16..(r + 1) * N as usize * BF16].to_vec())
            .collect()
    };

    // (d) all 53 rows through the 128 tile — the lever-off shape.
    gemm(&gpu, &kk, 3, y_full, w, out_ref, M, stream);
    gpu.synchronize(stream).unwrap();
    let full128 = download(&gpu, out_ref, out_bytes);
    check_rows_identical(
        "mmq128 all-53-rows vs mmq64 all-53-rows",
        &full128,
        &reference,
        53,
        &mut failures,
    );

    // (b)/(c): slice launches through each tile vs the reference rows. The
    // slice's y must be gathered into `[k_block][rows]` for its own
    // `ncols_y = rows` — a raw byte offset into the 53-row buffer would
    // misindex rows across k-block boundaries (the M-dependent layout).
    let out_slice = gpu.alloc(out_bytes).unwrap();
    for &tile in &[0usize, 1, 2, 3] {
        for &(r0, rows) in &[(32u32, 16u32), (48, 5)] {
            let y_slice = upload(
                &gpu,
                &y_slice_gather(&yf, M as usize, K, r0 as usize, rows as usize),
            );
            gemm(&gpu, &kk, tile, y_slice, w, out_slice, rows, stream);
            gpu.synchronize(stream).unwrap();
            gpu.free(y_slice).ok();
            let got = download(&gpu, out_slice, rows as usize * N as usize * BF16);
            check_rows_identical(
                &format!(
                    "mmq{} rows[{r0}..{}): slice launch vs mmq64 reference rows",
                    Mmq::mmq_x(tile),
                    r0 + rows
                ),
                &got,
                &want_rows(r0 as usize, rows as usize),
                rows as usize,
                &mut failures,
            );
        }
    }

    // ── 3. End-to-end: each chunk's own quantize+tile dispatch ───────────
    // Production runs quantize at the chunk's M and picks the tile for that
    // M — mirror that instead of sharing y_full.
    for &(r0, rows) in &[(32u32, 16u32), (48, 5)] {
        let y_c = gpu.alloc(y_bytes).unwrap();
        let out_c = gpu.alloc(out_bytes).unwrap();
        quantize(&gpu, &kk, x, r0, rows, y_c, stream);
        gemm(&gpu, &kk, Mmq::tile_for(rows), y_c, w, out_c, rows, stream);
        gpu.synchronize(stream).unwrap();
        let got = download(&gpu, out_c, rows as usize * N as usize * BF16);
        check_rows_identical(
            &format!(
                "end-to-end rows[{r0}..{}): chunk dispatch (m={rows}→mmq{}) vs contiguous mmq64",
                r0 + rows,
                Mmq::mmq_x(Mmq::tile_for(rows))
            ),
            &got,
            &want_rows(r0 as usize, rows as usize),
            rows as usize,
            &mut failures,
        );
        gpu.free(y_c).ok();
        gpu.free(out_c).ok();
    }

    for p in [x, packed, scales, w, y_full, y_part, out_ref, out_slice] {
        gpu.free(p).ok();
    }

    assert!(
        failures.is_empty(),
        "nvfp4_mmq M-invariance failures:\n  - {}",
        failures.join("\n  - ")
    );
}
