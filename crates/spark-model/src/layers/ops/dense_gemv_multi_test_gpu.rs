// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`; a GB10, a glm-5.3-flash kernel build and CUTLASS,
//! `CUTLASS_HOME` at build time): `ATLAS_GLM_KDA_SIDE_PROJ_LT`'s packed KDA
//! beta | f_a | g_a GEMM against the pipelined triple it replaces, at the
//! TP2 shapes (N = 32 / 128 / 128, K = 4096). Every plane must be the
//! triple's bytes at every row count, and rows computed in one call must equal
//! the same rows computed in pieces (prefix-cache warm vs cold, chunk sizes).
//!
//!   cargo test --release -p spark-model --lib packed_side_proj \
//!     -- --ignored --nocapture

use super::*;
use anyhow::bail;
use spark_runtime::cuda_backend::AtlasCudaBackend;

const H: u32 = 4096;
const HEADS: u32 = 32;
const DIM: u32 = 128;
const N: u32 = HEADS + 2 * DIM;

fn random_bf16(count: usize, seed: &mut u64) -> Vec<u8> {
    (0..count)
        .flat_map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((*seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 2.0;
            (((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(g.default_stream())?;
    let mut v = vec![0u8; bytes];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// The three dense planes `[rows, n]` of one output arena.
fn planes(base: DevicePtr, rows: u32) -> [DevicePtr; 3] {
    let fa = base.offset((rows * HEADS) as usize * 2);
    [base, fa, fa.offset((rows * DIM) as usize * 2)]
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and CUTLASS"]
fn packed_side_proj_is_the_pipelined_triple_bytes_at_every_row_count() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let triple = g.kernel("gemm", "dense_gemm_bf16_pipelined_triple_n")?;
    let stream = g.default_stream();
    let mut seed = 0x51de_9a0cu64;
    let max_m = 16384u32;
    let act = upload(g, &random_bf16((max_m * H) as usize, &mut seed))?;
    let packed = upload(g, &random_bf16((N * H) as usize, &mut seed))?;
    let row = H as usize * 2;
    let weights = [0, HEADS, HEADS + DIM].map(|r| DenseWeight {
        weight: packed.offset(r as usize * row),
    });
    let out_bytes = (max_m * N) as usize * 2;
    let (want, got, scratch) = (
        g.alloc(out_bytes)?,
        g.alloc(out_bytes)?,
        g.alloc(out_bytes)?,
    );
    for m in [1u32, 31, 2048, 5387, 8192, 16384] {
        dense_gemm_pipelined_triple_n(
            g,
            triple,
            act,
            [&weights[0], &weights[1], &weights[2]],
            planes(want, m),
            m,
            [HEADS, DIM],
            H,
            stream,
        )?;
        let ran = dense_gemm_packed_triple_n(
            g,
            act,
            packed,
            scratch,
            planes(got, m),
            m,
            [HEADS, DIM],
            H,
            stream,
        )?;
        if !ran {
            bail!("CUTLASS rejected the packed {m}x{N}x{H} GEMM (built without CUTLASS?)");
        }
        let bytes = (m * N) as usize * 2;
        let (w, o) = (read(g, want, bytes)?, read(g, got, bytes)?);
        let differing = w.chunks(2).zip(o.chunks(2)).filter(|(a, b)| a != b).count();
        println!("m={m}: {differing} differing BF16 outputs");
        assert_eq!(differing, 0, "m={m}: packed GEMM differs from the triple");
    }
    // Chunk invariance: 16384 rows at once vs 5387 + 2048 + 8949 rows.
    let whole = read(g, got, out_bytes)?;
    let mut row0 = 0u32;
    for rows in [5387u32, 2048, 8949] {
        let part = g.alloc((rows * N) as usize * 2)?;
        let input = act.offset((row0 * H) as usize * 2);
        let ok = dense_gemm_packed_triple_n(
            g,
            input,
            packed,
            scratch,
            planes(part, rows),
            rows,
            [HEADS, DIM],
            H,
            stream,
        )?;
        assert!(ok);
        let piece = read(g, part, (rows * N) as usize * 2)?;
        for (plane, width) in [HEADS, DIM, DIM].into_iter().enumerate() {
            let (w, start) = (width as usize * 2, [0, HEADS, HEADS + DIM][plane]);
            let whole_plane = (start * max_m) as usize * 2;
            let piece_plane = (start * rows) as usize * 2;
            let a = &whole[whole_plane + row0 as usize * w..][..rows as usize * w];
            let b = &piece[piece_plane..][..rows as usize * w];
            assert_eq!(
                a, b,
                "plane {plane}: rows {row0}..+{rows} differ split vs whole"
            );
        }
        g.free(part)?;
        row0 += rows;
    }
    for p in [act, packed, want, got, scratch] {
        g.free(p)?;
    }
    Ok(())
}
