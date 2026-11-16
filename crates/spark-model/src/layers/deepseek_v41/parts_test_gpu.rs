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
        project_groups(
            gpu,
            k.fp8_groups_q,
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
