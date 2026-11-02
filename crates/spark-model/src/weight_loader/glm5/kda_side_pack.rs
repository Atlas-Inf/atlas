// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_GLM_KDA_SIDE_PROJ_LT`: the KDA b / f_a / g_a weights moved into one
//! allocation, rows back to back, so prefill projects them as one
//! `[n_b + 2 * n_a, hidden]` GEMM (`ops::dense_gemm_packed_triple_n`). The
//! three `DenseWeight`s point into it, so every other path reads the same
//! bytes, and the sources are released: the resident size is unchanged.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::retirement::{LoadedDense, RetirementLog};
use crate::tp_shard::TpGdnDims;
use crate::weight_map::DenseWeight;

/// The layer's `[b, f_a, g_a]` packed: `b` is this rank's `[local_nk, h]`
/// shard, f_a / g_a (`{prefix}.{f,g}_a_proj.weight`) the replicated `[kd, h]`.
pub(super) fn pack_side<'a>(
    store: &'a WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    retirement: Option<&'a RetirementLog<'a>>,
    b: LoadedDense<'a>,
    dims: &TpGdnDims,
) -> Result<[DenseWeight; 3]> {
    let side = |name: &str| {
        LoadedDense::load(
            store,
            &format!("{prefix}.{name}.weight"),
            gpu,
            retirement,
            false,
        )
    };
    let a_bytes = dims.kd * dims.h * 2;
    pack(
        gpu,
        [b, side("f_a_proj")?, side("g_a_proj")?],
        [dims.local_nk * dims.h * 2, a_bytes, a_bytes],
    )
}

/// `parts[i]` (`bytes[i]` long) copied to consecutive offsets of one new
/// allocation, then released.
fn pack(
    gpu: &dyn GpuBackend,
    parts: [LoadedDense<'_>; 3],
    bytes: [usize; 3],
) -> Result<[DenseWeight; 3]> {
    let base = gpu.alloc(bytes.iter().sum())?;
    let mut offset = 0;
    let mut packed = [DenseWeight { weight: base }; 3];
    for ((part, len), slot) in parts.into_iter().zip(bytes).zip(&mut packed) {
        slot.weight = base.offset(offset);
        gpu.copy_d2d(part.dense.weight, slot.weight, len)?;
        part.release(gpu)?;
        offset += len;
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::{WeightDtype, WeightTensor};
    use std::collections::HashMap;

    #[test]
    fn packed_side_weights_hold_the_sources_back_to_back() {
        let gpu = MockGpuBackend::new();
        let sizes = [64usize, 256, 256];
        let names = ["b.weight", "fa.weight", "ga.weight"];
        let mut map = HashMap::new();
        let mut want = Vec::new();
        for (i, (name, bytes)) in names.iter().zip(sizes).enumerate() {
            let data: Vec<u8> = (0..bytes).map(|j| (j * 3 + i * 101) as u8).collect();
            let ptr = gpu.alloc(bytes).unwrap();
            gpu.copy_h2d(&data, ptr).unwrap();
            want.extend_from_slice(&data);
            map.insert(
                (*name).into(),
                WeightTensor {
                    ptr,
                    shape: vec![bytes / 8, 4],
                    dtype: WeightDtype::BF16,
                },
            );
        }
        let store = WeightStore::from_map(map);
        let parts = names.map(|n| LoadedDense::load(&store, n, &gpu, None, false).unwrap());
        let [b, fa, ga] = pack(&gpu, parts, sizes).unwrap();
        assert_eq!(fa.weight, b.weight.offset(sizes[0]));
        assert_eq!(ga.weight, fa.weight.offset(sizes[1]));
        let mut got = vec![0u8; want.len()];
        gpu.copy_d2h(b.weight, &mut got).unwrap();
        assert_eq!(got, want);
    }

    /// The loader's form: b from the layer's shard, f_a / g_a by name, every
    /// source released through the retirement log (or the store) exactly
    /// once, the packed copy the only allocation left.
    #[test]
    fn pack_side_retires_each_source_once_and_keeps_only_the_packed_copy() {
        for tracked in [false, true] {
            let gpu = MockGpuBackend::new();
            let (h, kd, nk) = (16usize, 8usize, 2usize);
            let mut map = HashMap::new();
            let mut want = Vec::new();
            for (i, (name, rows)) in [("b_proj", nk), ("f_a_proj", kd), ("g_a_proj", kd)]
                .into_iter()
                .enumerate()
            {
                let data: Vec<u8> = (0..rows * h * 2).map(|j| (j + 37 * i) as u8).collect();
                let ptr = gpu.alloc(data.len()).unwrap();
                gpu.copy_h2d(&data, ptr).unwrap();
                want.extend_from_slice(&data);
                map.insert(
                    format!("l.self_attn.{name}.weight"),
                    WeightTensor {
                        ptr,
                        shape: vec![rows, h],
                        dtype: WeightDtype::BF16,
                    },
                );
            }
            let mut store = WeightStore::from_map(map);
            let owned = RetirementLog::new(&store, &gpu).unwrap();
            let log = tracked.then_some(&owned);
            let dims = TpGdnDims {
                tp_rank: 0,
                tp_size: 1,
                h,
                kd,
                vd: kd,
                local_nk: nk,
                full_nk: nk,
                local_nv: nk,
                full_nv: nk,
            };
            let b = LoadedDense::load(&store, "l.self_attn.b_proj.weight", &gpu, log, false);
            let [b, fa, ga] =
                pack_side(&store, "l.self_attn", &gpu, log, b.unwrap(), &dims).unwrap();
            assert_eq!(fa.weight, b.weight.offset(nk * h * 2));
            assert_eq!(ga.weight, fa.weight.offset(kd * h * 2));
            let mut got = vec![0u8; want.len()];
            gpu.copy_d2h(b.weight, &mut got).unwrap();
            assert_eq!(got, want);
            if tracked {
                // The rebuilt store no longer names the retired sources.
                owned.finish().rebuild(&mut store, &gpu).unwrap();
                assert_eq!(store.len(), 0);
            }
            assert_eq!(gpu.alloc_count(), 1, "tracked={tracked}");
        }
    }
}
