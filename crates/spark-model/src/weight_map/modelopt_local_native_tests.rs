// SPDX-License-Identifier: AGPL-3.0-only

//! Native MXFP8 modules through the fast loader: left to the shard path, and
//! checked where they already sit.

use super::*;

/// Native modules load through the shard path like any other tensor: only
/// what the build must expand stays deferred.
#[test]
fn deferred_modules_leave_native_modules_to_the_shard_path() {
    const INDEXER: &str = "model.language_model.layers.3.self_attn.indexer.index_qk_proj";
    let mut config = fp4_local_config();
    let qc = config.quantization_config.as_mut().unwrap();
    qc.quantized_layers
        .insert(INDEXER.into(), spec("MXFP8", 32));
    let got = deferred_modules(&config);
    let want: HashSet<String> = [INDEXER.to_string()].into_iter().collect();
    assert_eq!(got, want);
    assert!(host_decode_modules(&config).contains(GDN_OUT));
    assert!(deferred_modules(&ModelConfig::qwen3_next_80b_nvfp4()).is_empty());
}

/// The fast loader already put a native module on the device: it is checked
/// and left there, never read back and uploaded a second time.
#[test]
fn resident_native_mxfp8_stays_where_it_is() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let weight: Vec<u8> = (0..64u8).collect();
    for (suffix, data, shape, dtype) in [
        (
            ".weight",
            weight.as_slice(),
            vec![2, 32],
            WeightDtype::FP8E4M3,
        ),
        (
            ".weight_scale",
            [120u8, 125].as_slice(),
            vec![2, 1],
            WeightDtype::UInt8,
        ),
    ] {
        let ptr = gpu.alloc(data.len()).unwrap();
        gpu.copy_h2d(data, ptr).unwrap();
        let tensor = WeightTensor { ptr, shape, dtype };
        store.insert(format!("{GDN_OUT}{suffix}"), tensor).unwrap();
    }
    let before = store.get(&format!("{GDN_OUT}.weight")).unwrap().ptr;
    let allocs = gpu.alloc_count();
    let done = materialize(&mut store, &fp4_local_config(), &gpu).unwrap();
    assert_eq!((done.mxfp8_native, done.uploaded_bytes), (1, 0));
    assert_eq!(gpu.alloc_count(), allocs);
    assert_eq!(store.get(&format!("{GDN_OUT}.weight")).unwrap().ptr, before);
}

/// A resident native module is checked like a deferred one.
#[test]
fn resident_mxfp8_with_wrong_scale_shape_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    for (suffix, len, shape, dtype) in [
        (".weight", 64, vec![2, 32], WeightDtype::FP8E4M3),
        (".weight_scale", 3, vec![3, 1], WeightDtype::UInt8),
    ] {
        let ptr = gpu.alloc(len).unwrap();
        let tensor = WeightTensor { ptr, shape, dtype };
        store.insert(format!("{GDN_OUT}{suffix}"), tensor).unwrap();
    }
    let err = materialize(&mut store, &fp4_local_config(), &gpu).unwrap_err();
    assert!(err.to_string().contains("expected MXFP8"), "{err:#}");
}
