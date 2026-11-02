// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::config::{QuantLayerSpec, QuantizationConfig};
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::DeferredTensor;

const ATTN: &str = "model.language_model.layers.3.self_attn.q_proj";
const SHARED: &str = "model.language_model.layers.3.mlp.shared_expert.up_proj";
const VIS: &str = "model.visual.blocks.0.mlp.linear_fc1";
const HC_NORM: &str = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight";
const PLE_NORM: &str = "model.language_model.layers.1.ple.norm_conv.weight";
const EXPERT: &str = "model.language_model.layers.0.mlp.experts.3.down_proj.weight_scale";
const GDN_OUT: &str = "model.language_model.layers.0.linear_attn.out_proj";

fn spec(algo: &str, group: usize) -> QuantLayerSpec {
    QuantLayerSpec {
        quant_algo: algo.into(),
        group_size: group,
        weight_scale_encoding: None,
        weight_block_size: vec![],
    }
}

/// A config carrying the FP4-Local `quantized_layers` shape: CSF NVFP4
/// experts, an MXFP8 projection, a W4A16 NVFP4 vision MLP, and the PLE table.
fn fp4_local_config() -> ModelConfig {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.model_type = "qwen4_exp".into();
    let mut map = std::collections::BTreeMap::new();
    map.insert("model.language_model.layers.0.mlp.experts".into(), {
        let mut s = spec("NVFP4", 16);
        s.weight_scale_encoding = Some("csf".into());
        s
    });
    map.insert(SHARED.into(), spec("MXFP8", 32));
    map.insert(GDN_OUT.into(), spec("MXFP8", 32));
    map.insert(VIS.into(), spec("W4A16_NVFP4", 16));
    map.insert(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding".into(),
        spec("NVFP4", 16),
    );
    map.insert("mtp.layers.0.mlp.experts".into(), spec("W4A16_NVFP4", 16));
    cfg.quantization_config = Some(QuantizationConfig {
        quant_method: "modelopt".into(),
        quant_algo: "MIXED_PRECISION".into(),
        format: String::new(),
        ignore_modules: vec![],
        weight_block_size: vec![],
        group_size: 0,
        quantized_layers: map,
        exl3: None,
    });
    cfg
}

/// Only MXFP8 modules are host-expanded: the trunk experts, MTP experts, PLE
/// table and the vision tower (native NVFP4 dequant on-GPU) keep their own
/// paths.
/// Under a GLM native policy `weight_loader/glm5` reads the checkpoint's own
/// MXFP8 and NVFP4 bytes: expanding them here would leave it BF16 copies and
/// every native path on its fallback. Without one, and for any other model,
/// the rule above stands.
#[test]
fn a_glm_native_policy_leaves_every_module_to_its_loaders() {
    let on =
        NativePolicy::from_lookup(|name| (name == NativePolicy::UMBRELLA).then(|| "1".to_string()));
    let off = NativePolicy::default();
    let mut glm = fp4_local_config();
    glm.model_type = "glm5_next".into();
    assert!(host_decode_modules_for(&glm, on).is_empty());
    assert!(!host_decode_modules_for(&glm, off).is_empty());
    let qwen = fp4_local_config();
    assert_eq!(
        host_decode_modules_for(&qwen, on),
        host_decode_modules_for(&qwen, off)
    );
}

/// GLM-5.3 FP4-Local declares `FP8_PB_WO` (128x128-block FP32 scales) on its
/// shared experts: those are NOT MXFP8 per-32 E8M0 and must stay resident for
/// the layer's own loader (`shared_expert_requantizes_from_fp8_blocks`), not
/// host-expanded. Before this fix the "no native consumer" arm scooped them
/// into `mxfp8_to_bf16` and the 2048-byte scale failed its shape check at
/// model build.
#[test]
fn host_decode_modules_leaves_fp8_block_scales_to_the_layer_loader() {
    let mut cfg = fp4_local_config();
    cfg.model_type = "glm5_next".into();
    let shared = "model.language_model.layers.10.mlp.shared_experts.down_proj";
    cfg.quantization_config
        .as_mut()
        .unwrap()
        .quantized_layers
        .insert(shared.into(), spec("FP8_PB_WO", 128));
    // glm5_next registers no native consumers: MXFP8 and NVFP4-family
    // modules keep the pre-existing defer-and-expand path (BF16, as in the
    // 494-module build log), while FP8_PB_WO shared experts stay resident
    // for the GLM loader's own fp8 handling.
    let got = host_decode_modules(&cfg);
    assert!(got.contains(SHARED) && got.contains(GDN_OUT));
    assert!(got.contains(VIS), "NVFP4 vision modules still host-expand");
    assert!(!got.contains(shared), "FP8_PB_WO must not be host-expanded");
}

/// Per-layer the same suffix may carry DIFFERENT algos (GLM-5.3's
/// `shared_experts.down_proj`: FP8_PB_WO on most layers, MXFP8 on the MTP
/// layer). Classification must key the exact per-layer path — an FP8_PB_WO
/// layer is never host-expanded, an MXFP8 layer is.
#[test]
fn the_same_suffix_classifies_per_layer_not_per_suffix() {
    let mut cfg = fp4_local_config();
    cfg.model_type = "glm5_next".into();
    let m = "model.language_model.layers";
    let block = spec("FP8_PB_WO", 128);
    let e8m0 = spec("MXFP8", 32);
    let ql = &mut cfg.quantization_config.as_mut().unwrap().quantized_layers;
    ql.insert(
        format!("{m}.10.mlp.shared_experts.down_proj"),
        block.clone(),
    );
    ql.insert(format!("{m}.45.mlp.shared_experts.down_proj"), e8m0);
    let got = host_decode_modules(&cfg);
    assert!(
        !got.contains(&format!("{m}.10.mlp.shared_experts.down_proj")),
        "the FP8_PB_WO layer stays for the GLM loader"
    );
    assert!(
        got.contains(&format!("{m}.45.mlp.shared_experts.down_proj")),
        "the MXFP8 layer is host-expanded"
    );
}

#[test]
fn host_decode_modules_pick_mxfp8_and_vision_nvfp4_only() {
    let got = host_decode_modules(&fp4_local_config());
    let want: HashSet<String> = [SHARED, GDN_OUT].iter().map(|s| s.to_string()).collect();
    assert_eq!(got, want);
    assert!(host_decode_modules(&ModelConfig::qwen3_next_80b_nvfp4()).is_empty());
}

/// Writes tensors into one scratch "shard" and defers each at its offset,
/// exactly as the fast loader records them.
struct Shard {
    path: std::path::PathBuf,
    bytes: Vec<u8>,
    entries: Vec<(String, u64, Vec<usize>, WeightDtype)>,
}
impl Shard {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("atlas-fp4l-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self {
            path: dir.join("s.bin"),
            bytes: vec![0; 8],
            entries: Vec::new(),
        }
    }
    fn add(&mut self, name: &str, data: &[u8], shape: &[usize], dtype: WeightDtype) {
        self.entries
            .push((name.into(), self.bytes.len() as u64, shape.to_vec(), dtype));
        self.bytes.extend_from_slice(data);
    }
    fn defer_into(self, store: &mut WeightStore) {
        std::fs::write(&self.path, &self.bytes).unwrap();
        for (name, offset, shape, dtype) in self.entries {
            store.defer(
                name,
                DeferredTensor {
                    path: self.path.clone(),
                    offset,
                    shape,
                    dtype,
                },
            );
        }
    }
}

fn resident(
    store: &mut WeightStore,
    gpu: &MockGpuBackend,
    name: &str,
    data: &[u8],
    dtype: WeightDtype,
) {
    let ptr = gpu.alloc(data.len()).unwrap();
    gpu.copy_h2d(data, ptr).unwrap();
    store
        .insert(
            name.into(),
            WeightTensor {
                ptr,
                shape: vec![],
                dtype,
            },
        )
        .unwrap();
}

#[test]
fn materialize_expands_every_fp4_local_format() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let mut shard = Shard::new("all");

    // CSF: a [16, 4] scale matrix whose bytes are all 0x38 except one outlier.
    let mut fixed = vec![0u8; 16 * 3];
    fixed[..16].fill(0x38); // bases; every code nibble 0
    let outlier = ((0x7Eu32 << 24) | 5).to_le_bytes();
    shard.add(
        &format!("{EXPERT}.nvfp4_csf_fixed"),
        &fixed,
        &[1, 48],
        WeightDtype::UInt8,
    );
    shard.add(
        &format!("{EXPERT}.nvfp4_csf_exceptions"),
        &outlier,
        &[1],
        WeightDtype::Int32,
    );

    // MXFP8 [1, 32]: E4M3 1.0 (0x38) everywhere, E8M0 scale 2^-3.
    shard.add(
        &format!("{SHARED}.weight"),
        &[0x38; 32],
        &[1, 32],
        WeightDtype::FP8E4M3,
    );
    shard.add(
        &format!("{SHARED}.weight_scale"),
        &[124],
        &[1, 1],
        WeightDtype::UInt8,
    );

    // Vision NVFP4 [1, 16]: nibble 2 (=1.0) everywhere, block 2.0, global 0.5.
    shard.add(
        &format!("{VIS}.weight"),
        &[0x22; 8],
        &[1, 8],
        WeightDtype::UInt8,
    );
    shard.add(
        &format!("{VIS}.weight_scale"),
        &[0x40],
        &[1, 1],
        WeightDtype::FP8E4M3,
    );
    shard.defer_into(&mut store);
    resident(
        &mut store,
        &gpu,
        &format!("{VIS}.weight_scale_2"),
        &0.5f32.to_le_bytes(),
        WeightDtype::FP32,
    );
    resident(
        &mut store,
        &gpu,
        &format!("{VIS}.input_scale"),
        &1f32.to_le_bytes(),
        WeightDtype::FP32,
    );

    // An F32 hc_norm [2] = [1.0, 2.0], as the pack ships it.
    let norm: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = gpu.alloc(norm.len()).unwrap();
    gpu.copy_h2d(&norm, ptr).unwrap();
    let t = WeightTensor {
        ptr,
        shape: vec![2],
        dtype: WeightDtype::FP32,
    };
    store.insert(HC_NORM.into(), t).unwrap();
    let ptr = gpu.alloc(norm.len()).unwrap();
    gpu.copy_h2d(&norm, ptr).unwrap();
    let t = WeightTensor {
        ptr,
        shape: vec![2],
        dtype: WeightDtype::FP32,
    };
    store.insert(PLE_NORM.into(), t).unwrap();
    // A 1-D F32 that is not a norm (a bias, say) is left alone.
    let ptr = gpu.alloc(norm.len()).unwrap();
    let t = WeightTensor {
        ptr,
        shape: vec![2],
        dtype: WeightDtype::FP32,
    };
    store
        .insert("model.language_model.layers.1.ple.conv1d.bias".into(), t)
        .unwrap();
    let done = materialize(&mut store, &fp4_local_config(), &gpu).unwrap();
    assert_eq!(
        (
            done.csf_scales,
            done.mxfp8_native,
            done.mxfp8_to_bf16,
            done.nvfp4_to_bf16,
            done.f32_norms_native
        ),
        (1, 1, 0, 0, 2)
    );
    // The shared-expert projection is a native consumer now: bytes as shipped.
    let se = store.get(&format!("{SHARED}.weight")).unwrap();
    assert_eq!(
        (se.dtype, se.shape.clone()),
        (WeightDtype::FP8E4M3, vec![1, 32])
    );
    assert_eq!(gpu.read_alloc(se.ptr).unwrap(), [0x38u8; 32]);
    let se_s = store.get(&format!("{SHARED}.weight_scale")).unwrap();
    assert_eq!(
        (se_s.dtype, se_s.shape.clone()),
        (WeightDtype::UInt8, vec![1, 1])
    );
    // The F32 norms are served as shipped: same dtype, byte-identical data.
    let norm = store.get(HC_NORM).unwrap();
    assert_eq!(norm.dtype, WeightDtype::FP32);
    assert_eq!(
        gpu.read_alloc(norm.ptr).unwrap(),
        [1.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    );
    let norm = store.get(PLE_NORM).unwrap();
    assert_eq!(norm.dtype, WeightDtype::FP32);
    assert_eq!(
        gpu.read_alloc(norm.ptr).unwrap(),
        [1.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    );
    let bias = store
        .get("model.language_model.layers.1.ple.conv1d.bias")
        .unwrap();
    assert_eq!(bias.dtype, WeightDtype::FP32);
    // The vision NVFP4 tensors are NOT consumed: they are a native GPU
    // dequant path now, read as deferred shards by the vision loader.
    assert_eq!(
        store.deferred_sorted().len(),
        2,
        "vision tensors stay deferred"
    );

    let scale = store.get(EXPERT).unwrap();
    assert_eq!(
        (scale.dtype, scale.shape.clone()),
        (WeightDtype::FP8E4M3, vec![16, 4])
    );
    let mut want = vec![0x38u8; 64];
    want[5] = 0x7E;
    assert_eq!(gpu.read_alloc(scale.ptr).unwrap(), want);

    let vis = store.deferred(&format!("{VIS}.weight")).unwrap();
    assert_eq!(
        (vis.dtype, vis.shape.clone()),
        (WeightDtype::UInt8, vec![1, 8])
    );
    // Nothing consumes the vision NVFP4s at materialize: weight+scale stay
    // deferred for the vision loader; the resident scalars are untouched.
    assert!(store.deferred(&format!("{VIS}.weight_scale")).is_some());
    assert!(store.get(&format!("{VIS}.weight_scale_2")).is_ok());
    assert!(store.get(&format!("{VIS}.input_scale")).is_ok());
}

/// The GDN projections have a native MXFP8 kernel: their bytes are uploaded
/// exactly as shipped, weight and scale, and nothing is widened.
#[test]
fn gdn_mxfp8_stays_native() {
    use crate::weight_map::modelopt_plan::{QuantAlgo, native_consumers};
    let n = native_consumers("qwen4_exp");
    let mx = |m: &str| n.covers(m, &QuantAlgo::Mxfp8);
    assert!(mx(GDN_OUT));
    assert!(mx("model.language_model.layers.7.linear_attn.in_proj_qkv"));
    assert!(mx(ATTN));
    assert!(!mx(
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj"
    ));
    assert!(mx("model.visual.blocks.0.attn.qkv"));
    assert!(mx("model.visual.blocks.0.attn.proj"));
    assert!(!mx("model.visual.blocks.0.mlp.linear_fc1"));
    assert!(!mx("model.language_model.layers.7.linear_attn.in_proj_a"));

    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let mut shard = Shard::new("gdn");
    let weight: Vec<u8> = (0..64u8).collect();
    shard.add(
        &format!("{GDN_OUT}.weight"),
        &weight,
        &[2, 32],
        WeightDtype::FP8E4M3,
    );
    shard.add(
        &format!("{GDN_OUT}.weight_scale"),
        &[120, 125],
        &[2, 1],
        WeightDtype::UInt8,
    );
    shard.defer_into(&mut store);
    let done = materialize(&mut store, &fp4_local_config(), &gpu).unwrap();
    assert_eq!((done.mxfp8_native, done.mxfp8_to_bf16), (1, 0));
    let w = store.get(&format!("{GDN_OUT}.weight")).unwrap();
    assert_eq!(
        (w.dtype, w.shape.clone()),
        (WeightDtype::FP8E4M3, vec![2, 32])
    );
    assert_eq!(gpu.read_alloc(w.ptr).unwrap(), weight);
    let sc = store.get(&format!("{GDN_OUT}.weight_scale")).unwrap();
    assert_eq!(
        (sc.dtype, sc.shape.clone()),
        (WeightDtype::UInt8, vec![2, 1])
    );
    assert_eq!(gpu.read_alloc(sc.ptr).unwrap(), [120, 125]);
}

/// A malformed MXFP8 scale is refused rather than uploaded.
#[test]
fn gdn_mxfp8_with_wrong_scale_shape_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let mut shard = Shard::new("gdnbad");
    shard.add(
        &format!("{GDN_OUT}.weight"),
        &[0; 64],
        &[2, 32],
        WeightDtype::FP8E4M3,
    );
    shard.add(
        &format!("{GDN_OUT}.weight_scale"),
        &[0; 4],
        &[2, 2],
        WeightDtype::UInt8,
    );
    shard.defer_into(&mut store);
    let err = materialize(&mut store, &fp4_local_config(), &gpu).unwrap_err();
    assert!(format!("{err:#}").contains("expected MXFP8"), "{err:#}");
}

/// A CSF fixed stream with no exceptions partner is a broken checkpoint, not
/// something to serve with a half-decoded scale.
#[test]
fn unpaired_csf_stream_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let mut shard = Shard::new("unpaired");
    shard.add(
        &format!("{EXPERT}.nvfp4_csf_fixed"),
        &[0x38; 48],
        &[1, 48],
        WeightDtype::UInt8,
    );
    shard.defer_into(&mut store);
    let err = materialize(&mut store, &fp4_local_config(), &gpu).unwrap_err();
    assert!(
        format!("{err:#}").contains("without its exceptions partner"),
        "{err:#}"
    );
}

/// Every other checkpoint passes through untouched.
#[test]
fn materialize_is_a_no_op_without_fp4_local_tensors() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    resident(&mut store, &gpu, "a.weight", &[0; 4], WeightDtype::BF16);
    resident(
        &mut store,
        &gpu,
        "b.norm.weight",
        &[0; 4],
        WeightDtype::FP32,
    );
    let done = materialize(&mut store, &ModelConfig::qwen3_next_80b_nvfp4(), &gpu).unwrap();
    assert_eq!(done, Materialized::default());
    assert!(store.contains("a.weight"));
    // Another release's F32 tensors are never touched.
    assert_eq!(store.get("b.norm.weight").unwrap().dtype, WeightDtype::FP32);
}

#[path = "modelopt_local_tests/tail.rs"]
mod tail;

#[path = "modelopt_local_native_tests.rs"]
mod native;
