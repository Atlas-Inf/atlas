// SPDX-License-Identifier: AGPL-3.0-only
//! Byte-level repack oracle on GB10; no checkpoint and under 1 MiB of tensors.
use super::*;

#[test]
#[ignore = "requires GB10 and compiled GLM NVFP4 kernels; no checkpoint"]
fn mmq_cuda_repack_preserves_nibbles_scales_and_checkpoint_ownership() -> Result<()> {
    let target = atlas_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .ok_or_else(|| anyhow::anyhow!("compile the glm-5.3-flash/nvfp4 kernel target"))?;
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &target.modules)?;
    let (mut layer, mut store, config) = layer_fixture(&gpu);
    let mut originals = Vec::new();
    for (projection, name) in ["gate", "up", "down"].iter().enumerate() {
        // Nonuniform nibbles exercise both packed halves and every row/K tile.
        let packed: Vec<u8> = (0usize..256 * 128)
            .map(|i| (i.wrapping_mul(73) ^ (i >> 7) ^ (projection * 31)) as u8)
            .collect();
        let scales: Vec<u8> = (0..256 * 16)
            .map(|i| ((i * 13 + (i >> 4) + projection * 7) % 126 + 1) as u8)
            .collect();
        gpu.copy_h2d(&packed, store.get(&format!("{name}.weight"))?.ptr)?;
        gpu.copy_h2d(&scales, store.get(&format!("{name}.weight_scale"))?.ptr)?;
        originals.push((packed, scales));
    }
    layer.repack_nvfp4_mmq_unified(&gpu, &config, &store)?;
    assert_eq!(store.reclaimed_count(), 6);
    // This teardown used to free all six source allocations again. It must
    // succeed before we read the new slabs (which can reuse their addresses).
    store.release(&gpu)?;
    for (projection, table) in [&layer.gate_ptrs, &layer.up_ptrs, &layer.down_ptrs]
        .into_iter()
        .enumerate()
    {
        let slab = layer._nvfp4_mmq_owned[projection];
        let mut pointers = [0u8; 16];
        gpu.copy_d2h(table.packed_ptrs, &mut pointers)?;
        assert_eq!(u64::from_le_bytes(pointers[..8].try_into()?), 0);
        assert_eq!(u64::from_le_bytes(pointers[8..].try_into()?), slab.0);
        let mut actual = vec![0; 256 * 144];
        gpu.copy_d2h(slab, &mut actual)?;
        let (packed, scales) = &originals[projection];
        for row in 0..256 {
            for col in 0..256 {
                let expected = (packed[row * 128 + col / 2] >> (col % 2 * 4)) & 15;
                let output_byte = actual[row * 144 + col / 16 * 8 + col % 8];
                let got = (output_byte >> (col % 16 / 8 * 4)) & 15;
                assert_eq!(got, expected, "projection={projection} row={row} col={col}");
            }
            assert_eq!(
                &actual[row * 144 + 128..(row + 1) * 144],
                &scales[row * 16..(row + 1) * 16],
                "projection={projection} row={row} scales"
            );
        }
        free_expert_table(&gpu, table)?;
        gpu.free(slab)?;
    }
    gpu.free(layer.zero_accum_dev)?;
    assert_eq!(gpu.sweep_unreleased(), 0, "tiny fixture leaked allocations");
    Ok(())
}
