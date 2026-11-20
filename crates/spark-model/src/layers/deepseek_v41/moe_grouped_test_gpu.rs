// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): a few rows' pairs sorted by expert on the device
//! are the host's plan of them.
//!
//!   cargo test --release -p spark-model --lib deepseek_v41::moe_grouped -- --ignored --nocapture --test-threads=1

use anyhow::Result;

use super::*;
use crate::layers::deepseek_v41::test_support::{Rng, backend, le_bytes, upload};

fn words(gpu: &dyn GpuBackend, p: DevicePtr, count: usize) -> Result<Vec<u32>> {
    let mut bytes = vec![0u8; count * 4];
    gpu.copy_d2h(p, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn the_devices_order_of_a_few_rows_is_the_hosts() -> Result<()> {
    let gpu = backend()?;
    let (stream, kernels) = (gpu.default_stream(), MoeKernels::load(gpu)?);
    let mut rng = Rng(0x91a7_70e5);
    // The model's routing and a draft stage's; few experts, so that rows
    // share them, and many.
    for (rows, top_k, experts) in [
        (6usize, 6usize, 384usize),
        (16, 6, 12),
        (5, 3, 128),
        (32, 6, 40),
        (2, 6, 6),
    ] {
        let slots = top_k + 1;
        // A row's experts ascending and distinct, as the router leaves them.
        let chosen: Vec<u32> = (0..rows)
            .flat_map(|_| {
                let mut row: Vec<u32> = Vec::new();
                while row.len() < top_k {
                    let expert = (rng.next() % experts as u64) as u32;
                    if !row.contains(&expert) {
                        row.push(expert);
                    }
                }
                row.sort_unstable();
                row
            })
            .collect();
        let weights: Vec<f32> = (0..rows * top_k).map(|i| 0.25 + i as f32).collect();
        let host = Plan::new(&chosen, &weights, top_k, slots, experts);
        let scratch = GroupedScratch::alloc(gpu, rows, top_k, experts, 64)?;
        let on_device = (
            upload(gpu, &le_bytes(&chosen, u32::to_le_bytes))?,
            upload(gpu, &le_bytes(&weights, f32::to_le_bytes))?,
        );
        let counts = plan_on_device(
            gpu,
            &kernels,
            &scratch,
            on_device.0,
            on_device.1,
            rows,
            top_k,
            slots,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let pairs = rows * top_k;
        assert_eq!((counts.tiles, counts.pairs), (pairs, host.pairs()));
        // The host's tiles, then none.
        let mut tiles = host.tiles.clone();
        tiles.resize(pairs * 3, 0);
        assert_eq!(
            words(gpu, scratch.tiles, pairs * 3)?,
            tiles,
            "{rows} rows: tiles"
        );
        let signed = |v: &[i32]| -> Vec<u32> { v.iter().map(|&i| i as u32).collect() };
        assert_eq!(words(gpu, scratch.row_of, pairs)?, signed(&host.row_of));
        assert_eq!(words(gpu, scratch.out_row, pairs)?, signed(&host.out_row));
        let bits: Vec<u32> = host.weights.iter().map(|w| w.to_bits()).collect();
        assert_eq!(words(gpu, scratch.weights, pairs)?, bits);
        for p in [
            on_device.0,
            on_device.1,
            scratch.tiles,
            scratch.row_of,
            scratch.out_row,
            scratch.weights,
            scratch.gate_up,
        ] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
