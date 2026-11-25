// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): the projections that do several launches' work
//! in one, against those launches, on the release's weights and activations
//! spread over many binades.
//!
//!   ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::parts -- --ignored --nocapture --test-threads=1

use anyhow::Result;

use super::*;
use crate::layers::deepseek_v41::ops::{Kernels, fp8_quant_dequant};
use crate::layers::deepseek_v41::test_support::{
    Rng, activations, backend, le_bytes, load_fp8, upload,
};

const TRIALS: usize = 4;

fn bytes(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(gpu.default_stream())?;
    let mut out = vec![0u8; n];
    gpu.copy_d2h(p, &mut out)?;
    Ok(out)
}

/// Rows `first..` of a loaded FP8 projection of `k` columns.
fn rows_from(w: &crate::weight_map::Mxfp8Weight, first: usize, k: usize) -> Slice {
    Slice {
        data: w.data.offset(first * k),
        scales: w.scales.offset(first * k / 32),
    }
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target and ATLAS_DSV41_CKPT"]
fn fused_projections_leave_the_bytes_of_their_launches() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let (k, cache) = (PartKernels::load(gpu)?, Kernels::load(gpu)?);
    let mut rng = Rng(0x5eed_d5e4);
    let mut values = |n: usize| -> Result<DevicePtr> {
        upload(gpu, &le_bytes(&activations(&mut rng, n), u16::to_le_bytes))
    };

    // Rounding on the way in, with the sum rounded and not: a quarter of
    // the heads' projection.
    let (wq_b, n, width) = load_fp8(gpu, "layers.0.attn.wq_b")?;
    let rows = n / PARTS;
    let w = rows_from(&wq_b, rows, width);
    let (rounded, plain, fused) = (
        gpu.alloc(width * 2)?,
        gpu.alloc(rows * 4)?,
        gpu.alloc(rows * 4)?,
    );
    for trial in 0..TRIALS {
        let x = values(width)?;
        fp8_quant_dequant(gpu, cache.fp8_quant_dequant, x, rounded, width, stream)?;
        for (one, two, size) in [(k.fp8, k.fp8_q, 2), (k.fp8_f32, k.fp8_f32_q, 4)] {
            project(
                gpu, one, rounded, w.data, w.scales, plain, rows, width, stream,
            )?;
            project(gpu, two, x, w.data, w.scales, fused, rows, width, stream)?;
            assert!(
                bytes(gpu, plain, rows * size)? == bytes(gpu, fused, rows * size)?,
                "trial {trial}: rounding on the way in differs ({size} bytes a value)"
            );
        }
        gpu.free(x)?;
    }

    // Two projections of one vector.
    let (wq_a, n_a, k_a) = load_fp8(gpu, "layers.0.attn.wq_a")?;
    let (wkv, n_b, k_b) = load_fp8(gpu, "layers.0.attn.wkv")?;
    anyhow::ensure!(k_a == k_b, "wq_a reads {k_a} values and wkv {k_b}");
    let (a, b) = (rows_from(&wq_a, 0, k_a), rows_from(&wkv, 0, k_a));
    let both = (n_a + n_b) * 2;
    let (plain, fused, rounded) = (gpu.alloc(both)?, gpu.alloc(both)?, gpu.alloc(k_a * 2)?);
    for trial in 0..TRIALS {
        let x = values(k_a)?;
        fp8_quant_dequant(gpu, cache.fp8_quant_dequant, x, rounded, k_a, stream)?;
        project(
            gpu, k.fp8, rounded, a.data, a.scales, plain, n_a, k_a, stream,
        )?;
        let second = plain.offset(n_a * 2);
        project(
            gpu, k.fp8, rounded, b.data, b.scales, second, n_b, k_a, stream,
        )?;
        project_two(
            gpu,
            k.fp8_two,
            rounded,
            (a, n_a, fused),
            (b, n_b, fused.offset(n_a * 2)),
            k_a,
            1,
            stream,
        )?;
        assert!(
            bytes(gpu, plain, both)? == bytes(gpu, fused, both)?,
            "trial {trial}: two projections in a launch differ"
        );
        gpu.free(x)?;
    }

    // A block-diagonal projection: two of wo_a's eight groups.
    const GROUPS: usize = 8;
    let (wo_a, n, group_in) = load_fp8(gpu, "layers.0.attn.wo_a")?;
    let (rank, first, groups) = (n / GROUPS, 2, 2);
    let (plain, fused) = (gpu.alloc(groups * rank * 2)?, gpu.alloc(groups * rank * 2)?);
    for trial in 0..TRIALS {
        let x = values(groups * group_in)?;
        for group in 0..groups {
            let w = rows_from(&wo_a, (first + group) * rank, group_in);
            project(
                gpu,
                k.fp8_q,
                x.offset(group * group_in * 2),
                w.data,
                w.scales,
                plain.offset(group * rank * 2),
                rank,
                group_in,
                stream,
            )?;
        }
        // The fused launch reads its input rounded already.
        let values = groups * group_in;
        fp8_quant_dequant(gpu, cache.fp8_quant_dequant, x, x, values, stream)?;
        project_groups(
            gpu,
            k.fp8_groups,
            x,
            rows_from(&wo_a, first * rank, group_in),
            fused,
            1,
            groups,
            rank,
            group_in,
            stream,
        )?;
        assert!(
            bytes(gpu, plain, groups * rank * 2)? == bytes(gpu, fused, groups * rank * 2)?,
            "trial {trial}: the block-diagonal projection differs"
        );
        gpu.free(x)?;
    }
    println!(
        "rounding on the way in, two projections in a launch and a block-diagonal one leave the bytes of their launches, {TRIALS} trials each"
    );
    Ok(())
}

/// wo_b cut by its columns (`block_out.rs`) against wo_b whole: each part's
/// FP32 partial of two rows from its quarter of the input, the four added
/// and rounded into rows 1 and 2 of a batch of four. The sums are the whole
/// projection's in another order: a value is the whole's, or the BF16 next
/// to it, or where its terms cancel as far from it as FP32 sums of the row's
/// size differ.
#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target and ATLAS_DSV41_CKPT"]
fn wo_b_by_its_columns_is_wo_b_whole() -> Result<()> {
    use crate::layers::deepseek_v41::cache_recipe::bf16_to_f32;
    use crate::layers::deepseek_v41::weights::Fp8Proj;
    use crate::layers::deepseek_v41::weights_columns::{columns, free_columns};

    const ROWS: usize = 2;
    const FIRST: usize = 1;
    const BATCH: usize = 4;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let (k, cache) = (PartKernels::load(gpu)?, Kernels::load(gpu)?);
    let (w, n, width) = load_fp8(gpu, "layers.0.attn.wo_b")?;
    let whole = Fp8Proj { w, n, k: width };
    let cut = columns(gpu, &k, &whole, Held::ALL, stream)?;
    let (of_part, quarter) = (width / PARTS, n / PARTS);

    let mut rng = Rng(0xc01_5eed);
    let x = upload(
        gpu,
        &le_bytes(&activations(&mut rng, ROWS * width), u16::to_le_bytes),
    )?;
    fp8_quant_dequant(gpu, cache.fp8_quant_dequant, x, x, ROWS * width, stream)?;
    let rounded = bytes(gpu, x, ROWS * width * 2)?;

    // Whole: a row at a time.
    let plain = gpu.alloc(ROWS * n * 2)?;
    for row in 0..ROWS {
        let (from, to) = (x.offset(row * width * 2), plain.offset(row * n * 2));
        project(gpu, k.fp8, from, w.data, w.scales, to, n, width, stream)?;
    }
    let plain = bytes(gpu, plain, ROWS * n * 2)?;

    // By columns: a part's input is its quarter of every row.
    let partial = gpu.alloc(PARTS * ROWS * n * 4)?;
    for (part, columns) in cut.iter().enumerate() {
        let columns = columns.expect("every part is held");
        let mine: Vec<u8> = rounded
            .chunks_exact(width * 2)
            .flat_map(|row| row[part * of_part * 2..(part + 1) * of_part * 2].to_vec())
            .collect();
        let mine = upload(gpu, &mine)?;
        project_rows(
            gpu,
            k.fp8_f32,
            mine,
            columns.data,
            columns.scales,
            partial.offset(part * ROWS * n * 4),
            4,
            n,
            ROWS,
            n,
            of_part,
            stream,
        )?;
        gpu.synchronize(stream)?;
        gpu.free(mine)?;
    }
    // Rows the launch does not write keep what they held.
    const KEPT: u16 = 0x7fc1;
    let out = upload(gpu, &le_bytes(&vec![KEPT; BATCH * n], u16::to_le_bytes))?;
    round(
        gpu,
        k.round,
        partial,
        out,
        ROWS * n,
        ROWS,
        n,
        FIRST,
        BATCH,
        stream,
    )?;
    let out = bytes(gpu, out, BATCH * n * 2)?;
    let bf16 = |bytes: &[u8], at: usize| u16::from_le_bytes([bytes[2 * at], bytes[2 * at + 1]]);

    let largest: Vec<f32> = plain
        .chunks_exact(n * 2)
        .map(|row| {
            (0..n)
                .map(|d| bf16_to_f32(bf16(row, d)).abs())
                .fold(0.0, f32::max)
        })
        .collect();
    let (mut same, mut worst) = (0usize, 0.0f32);
    for row in 0..BATCH {
        for d in 0..n {
            let got = bf16(&out, ((d / quarter) * BATCH + row) * quarter + d % quarter);
            let Some(of) = row.checked_sub(FIRST).filter(|&of| of < ROWS) else {
                assert_eq!(got, KEPT, "row {row} value {d} was written");
                continue;
            };
            let want = bf16(&plain, of * n + d);
            same += usize::from(got == want);
            let (got, want) = (bf16_to_f32(got), bf16_to_f32(want));
            let (apart, scale) = ((got - want).abs(), largest[of]);
            worst = worst.max(apart / scale);
            assert!(
                apart <= (want.abs() / 128.0).max(scale * 1e-4),
                "row {row} value {d}: {got} by columns, {want} whole, in a row up to {scale}"
            );
        }
    }
    let share = same as f64 / (ROWS * n) as f64;
    println!(
        "wo_b by its columns: {:.3}% of {} values the whole projection's bytes, the rest within {worst:.2e} of their row's largest",
        100.0 * share,
        ROWS * n
    );
    assert!(share > 0.99, "{share}: too few values are the whole's");
    free_columns(gpu, &cut)
}
