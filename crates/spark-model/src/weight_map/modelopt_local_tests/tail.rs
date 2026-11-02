// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// A CSF scale stream whose declared encoding is not `"csf"` must fail at
/// load naming the encoding — never guessed.
#[test]
fn unknown_csf_encoding_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let mut shard = Shard::new("enc");
    let module = "model.language_model.layers.0.mlp.experts";
    shard.add(
        &format!("{module}.weight_scale.nvfp4_csf_fixed"),
        &[0; 8],
        &[2, 4],
        WeightDtype::UInt8,
    );
    shard.add(
        &format!("{module}.weight_scale.nvfp4_csf_exceptions"),
        &[0; 4],
        &[4],
        WeightDtype::UInt8,
    );
    shard.defer_into(&mut store);

    let mut cfg = fp4_local_config();
    cfg.quantization_config
        .as_mut()
        .unwrap()
        .quantized_layers
        .get_mut(module)
        .unwrap()
        .weight_scale_encoding = Some("csf_32x2_v1".into());
    let err = materialize(&mut store, &cfg, &gpu).unwrap_err().to_string();
    assert!(
        err.contains("csf_32x2_v1"),
        "error must name the encoding: {err}"
    );
}

/// A module whose algo has no registered consumer — here a W4A16 NVFP4 on
/// a non-expert, non-vision path — falls through to the exact BF16
/// expansion.
#[test]
fn unregistered_nvfp4_module_expands() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let mut shard = Shard::new("unreg");
    let module = "model.language_model.layers.0.mlp.down_proj";
    // NVFP4 [1, 16]: nibble 2 (=1.0) everywhere, block scale 2.0, global 0.5
    // → all ones BF16.
    shard.add(
        &format!("{module}.weight"),
        &[0x22; 8],
        &[1, 8],
        WeightDtype::UInt8,
    );
    shard.add(
        &format!("{module}.weight_scale"),
        &[0x40],
        &[1, 1],
        WeightDtype::FP8E4M3,
    );
    shard.defer_into(&mut store);
    resident(
        &mut store,
        &gpu,
        &format!("{module}.weight_scale_2"),
        &0.5f32.to_le_bytes(),
        WeightDtype::FP32,
    );

    let mut cfg = fp4_local_config();
    cfg.quantization_config
        .as_mut()
        .unwrap()
        .quantized_layers
        .insert(module.into(), spec("W4A16_NVFP4", 16));
    let done = materialize(&mut store, &cfg, &gpu).unwrap();
    assert_eq!(done.nvfp4_to_bf16, 1);
    let w = store.get(&format!("{module}.weight")).unwrap();
    assert_eq!((w.dtype, w.shape.clone()), (WeightDtype::BF16, vec![1, 16]));
    let v = gpu.read_alloc(w.ptr).unwrap();
    assert!((0..16).all(|i| {
        f32::from_bits(u32::from(u16::from_le_bytes([v[2 * i], v[2 * i + 1]])) << 16) == 1.0
    }));
}
