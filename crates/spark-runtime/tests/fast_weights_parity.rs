// SPDX-License-Identifier: AGPL-3.0-only

//! Parity test: FastSafetensorsLoader must produce byte-identical weights
//! to the mmap-based SafetensorsLoader for the same file.
//!
//! Builds a tiny synthetic safetensors file in a tempdir, loads it with both
//! loaders against a MockGpuBackend, and asserts every tensor's bytes match.

#![cfg(unix)]

use spark_runtime::fast_weights::FastSafetensorsLoader;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader, WeightStore};
use std::io::Write;

/// Build a minimal `model.safetensors` with two BF16 tensors and one U8 tensor.
/// Layout written by hand so the test doesn't depend on the safetensors crate
/// for encoding (decoding is still needed, used by the baseline loader).
fn write_test_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
    // Tensor A: BF16, shape [4, 8] = 64 bytes.
    // Tensor B: BF16, shape [2, 2] = 8 bytes.
    // Tensor C: U8,   shape [16]   = 16 bytes.
    let a_bytes: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let b_bytes: Vec<u8> = (0..8).map(|i| (128 + i) as u8).collect();
    let c_bytes: Vec<u8> = (0..16).map(|i| (200 + i) as u8).collect();

    let header = serde_json::json!({
        "a": { "dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64] },
        "b": { "dtype": "BF16", "shape": [2, 2], "data_offsets": [64, 72] },
        "c": { "dtype": "U8",   "shape": [16],   "data_offsets": [72, 88] },
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();

    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(&header_bytes).unwrap();
    f.write_all(&a_bytes).unwrap();
    f.write_all(&b_bytes).unwrap();
    f.write_all(&c_bytes).unwrap();
    f.sync_all().unwrap();
    path
}

#[test]
fn fast_and_mmap_loaders_agree() {
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new()
        .load(&tmp, &gpu_base, 0)
        .expect("baseline load");
    assert_eq!(base.len(), 3);

    let gpu_fast = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    // Force the buffered-read path: tmpfs rejects O_DIRECT on most kernels,
    // but we disable it explicitly so the test is deterministic.
    fast.try_direct_io = false;
    let new = fast.load(&tmp, &gpu_fast, 0).expect("fast load");
    assert_eq!(new.len(), 3);

    for name in ["a", "b", "c"] {
        let wb = base.get(name).unwrap();
        let wn = new.get(name).unwrap();
        assert_eq!(wb.shape, wn.shape, "shape mismatch for {name}");
        assert_eq!(wb.dtype, wn.dtype, "dtype mismatch for {name}");
        let bb = gpu_base.read_alloc(wb.ptr).unwrap();
        let bn = gpu_fast.read_alloc(wn.ptr).unwrap();
        assert_eq!(bb, bn, "byte mismatch for {name}");
    }

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn fast_loader_with_direct_io_if_supported() {
    // Best-effort O_DIRECT test — silently succeeds (by falling back to
    // buffered) if the filesystem rejects O_DIRECT.
    let tmp = tempdir_like();
    write_test_safetensors(&tmp);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new().load(&tmp, &gpu_base, 0).unwrap();

    let gpu_fast = MockGpuBackend::new();
    let fast = FastSafetensorsLoader::new(); // try_direct_io = true by default
    let new = fast
        .load(&tmp, &gpu_fast, 0)
        .expect("fast load with O_DIRECT attempted");
    assert_eq!(new.len(), 3);

    for name in ["a", "b", "c"] {
        let bb = gpu_base.read_alloc(base.get(name).unwrap().ptr).unwrap();
        let bn = gpu_fast.read_alloc(new.get(name).unwrap().ptr).unwrap();
        assert_eq!(bb, bn, "byte mismatch for {name} (O_DIRECT path)");
    }

    std::fs::remove_dir_all(&tmp).ok();
}

/// One tensor to write: (name, safetensors dtype string, shape, raw bytes).
type TensorSpec<'a> = (&'a str, &'a str, Vec<usize>, Vec<u8>);

/// Write a minimal safetensors file by hand: 8-byte LE header length, the JSON
/// header with `data_offsets`, then the concatenated tensor bytes.
fn write_st(path: &std::path::Path, tensors: &[TensorSpec<'_>]) {
    let mut obj = serde_json::Map::new();
    let mut offset = 0usize;
    let mut blobs: Vec<&[u8]> = Vec::with_capacity(tensors.len());
    for (name, dtype, shape, bytes) in tensors {
        obj.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes.len()],
            }),
        );
        offset += bytes.len();
        blobs.push(bytes);
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(obj)).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(&header_bytes).unwrap();
    for blob in blobs {
        f.write_all(blob).unwrap();
    }
    f.sync_all().unwrap();
}

/// Distinct non-zero bytes per tensor (k picks the tensor so tensors differ).
fn pattern_bytes(n: usize, k: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 7 + k) as u8).collect()
}

#[test]
fn exl3_checkpoint_loads_identically_on_both_loaders() {
    let tmp = tempdir_like();
    std::fs::write(
        tmp.join("config.json"),
        r#"{"quantization_config": {"quant_method": "exl3"}}"#,
    )
    .unwrap();

    // The indexed shard: trellis (I16), raw-F16 scale vectors, the I32 mul1
    // tag, and one ordinary F16 tensor that must still become BF16.
    let shard_tensors: Vec<TensorSpec<'_>> = vec![
        ("p.trellis", "I16", vec![1, 1, 64], pattern_bytes(128, 0)),
        ("p.suh", "F16", vec![16], pattern_bytes(32, 1)),
        ("p.svh", "F16", vec![16], pattern_bytes(32, 2)),
        (
            "p.mul1",
            "I32",
            vec![],
            0x83DC_D12Du32.to_le_bytes().to_vec(),
        ),
        ("e.weight", "F16", vec![4], pattern_bytes(8, 3)),
    ];
    write_st(
        &tmp.join("model-00001-of-00001.safetensors"),
        &shard_tensors,
    );

    let weight_map: serde_json::Map<String, serde_json::Value> = shard_tensors
        .iter()
        .map(|(n, _, _, _)| {
            (
                (*n).to_string(),
                serde_json::Value::String("model-00001-of-00001.safetensors".into()),
            )
        })
        .collect();
    std::fs::write(
        tmp.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();

    // Side file NOT in the index: EXL3 ships it, both loaders must merge it.
    let side_tensors: Vec<TensorSpec<'_>> = vec![
        (
            "m.ple.ple_embedding.ngram_embedding.trellis",
            "I16",
            vec![4, 3],
            pattern_bytes(24, 4),
        ),
        (
            "m.ple.ple_embedding.ngram_embedding.head_bias",
            "F16",
            vec![2],
            pattern_bytes(4, 5),
        ),
    ];
    write_st(&tmp.join("ngram_embedding.safetensors"), &side_tensors);

    let gpu_base = MockGpuBackend::new();
    let base = SafetensorsLoader::new()
        .load(&tmp, &gpu_base, 0)
        .expect("baseline load");
    let gpu_fast = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let new = fast.load(&tmp, &gpu_fast, 0).expect("fast load");

    // Per-loader expectations, run identically on both stores.
    check_exl3_store("SafetensorsLoader", &base, &gpu_base);
    check_exl3_store("FastSafetensorsLoader", &new, &gpu_fast);

    // The two stores must agree tensor by tensor.
    let mut base_names: Vec<&str> = base.names().collect();
    let mut fast_names: Vec<&str> = new.names().collect();
    base_names.sort_unstable();
    fast_names.sort_unstable();
    assert_eq!(
        base_names, fast_names,
        "resident tensor sets differ (base {base_names:?} vs fast {fast_names:?})"
    );
    for name in base_names {
        let wb = base.get(name).unwrap();
        let wn = new.get(name).unwrap();
        assert_eq!(wb.shape, wn.shape, "shape mismatch for {name}");
        assert_eq!(wb.dtype, wn.dtype, "dtype mismatch for {name}");
        assert_eq!(
            gpu_base.read_alloc(wb.ptr).unwrap(),
            gpu_fast.read_alloc(wn.ptr).unwrap(),
            "byte mismatch for {name}"
        );
    }

    std::fs::remove_dir_all(&tmp).ok();
}

/// Every EXL3-shaped expectation, asserted identically against one loader's
/// store (so a divergence between the loaders names the loader and the tensor).
fn check_exl3_store(tag: &str, store: &WeightStore, gpu: &MockGpuBackend) {
    use spark_runtime::weights::WeightDtype::{BF16, FP16, Int16, Int32};

    let trellis = store
        .get("p.trellis")
        .unwrap_or_else(|_| panic!("{tag}: p.trellis resident"));
    assert_eq!(trellis.dtype, Int16, "{tag}: p.trellis dtype");
    assert_eq!(trellis.shape, vec![1, 1, 64], "{tag}: p.trellis shape");
    assert_eq!(
        gpu.read_alloc(trellis.ptr).unwrap(),
        pattern_bytes(128, 0),
        "{tag}: p.trellis bytes"
    );

    let mul1_t = store
        .get("p.mul1")
        .unwrap_or_else(|_| panic!("{tag}: p.mul1 resident"));
    assert_eq!(mul1_t.dtype, Int32, "{tag}: p.mul1 dtype");
    assert_eq!(mul1_t.shape, Vec::<usize>::new(), "{tag}: p.mul1 shape");
    assert_eq!(
        gpu.read_alloc(mul1_t.ptr).unwrap(),
        0x83DC_D12Du32.to_le_bytes().to_vec(),
        "{tag}: p.mul1 bytes"
    );

    for name in ["p.suh", "p.svh"] {
        let t = store
            .get(name)
            .unwrap_or_else(|_| panic!("{tag}: {name} resident"));
        assert_eq!(t.dtype, FP16, "{tag}: {name} must keep raw FP16 bytes");
        assert_eq!(
            gpu.read_alloc(t.ptr).unwrap(),
            pattern_bytes(32, if name == "p.suh" { 1 } else { 2 }),
            "{tag}: {name} raw bytes"
        );
    }

    let e = store
        .get("e.weight")
        .unwrap_or_else(|_| panic!("{tag}: e.weight resident"));
    assert_eq!(e.dtype, BF16, "{tag}: ordinary F16 still converts to BF16");
    assert_ne!(
        gpu.read_alloc(e.ptr).unwrap(),
        pattern_bytes(8, 3),
        "{tag}: e.weight bytes unchanged"
    );

    let head_bias = store
        .get("m.ple.ple_embedding.ngram_embedding.head_bias")
        .unwrap_or_else(|_| panic!("{tag}: head_bias must be resident"));
    assert_eq!(head_bias.dtype, FP16, "{tag}: head_bias dtype");
    assert_eq!(
        gpu.read_alloc(head_bias.ptr).unwrap(),
        pattern_bytes(4, 5),
        "{tag}: head_bias raw bytes"
    );

    assert!(
        store
            .get("m.ple.ple_embedding.ngram_embedding.trellis")
            .is_err(),
        "{tag}: n-gram trellis must NOT be resident"
    );
    let d = store
        .deferred("m.ple.ple_embedding.ngram_embedding.trellis")
        .unwrap_or_else(|| panic!("{tag}: n-gram trellis must be deferred"));
    assert_eq!(d.dtype, Int16, "{tag}: deferred trellis dtype");
    assert_eq!(d.shape, vec![4, 3], "{tag}: deferred trellis shape");
}

/// Creates a unique temp directory without pulling in the tempfile crate.
fn tempdir_like() -> std::path::PathBuf {
    let pid = std::process::id();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("atlas-fwp-{pid}-{ns}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}
