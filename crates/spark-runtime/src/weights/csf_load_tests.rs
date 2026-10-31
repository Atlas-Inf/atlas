// SPDX-License-Identifier: AGPL-3.0-only

//! Both safetensors loaders turn CSF pairs into decoded `weight_scale` tensors.
//!
//! The checkpoint built here is shaped like FP4-Local where it matters: the
//! index routes to shards under a `tensors/` SUBDIRECTORY, and each CSF pair is
//! split ACROSS the two shards (so a loader that decodes per shard fails). The
//! bytes are the real golden pairs, so "decoded" is checked against vLLM's
//! reference by sha256, not against this crate's own decoder.
#![cfg(unix)]

use super::*;
use crate::fast_weights::FastSafetensorsLoader;
use crate::gpu::mock::MockGpuBackend;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

const GATE: &str = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale";
const DOWN: &str = "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale";
const PLAIN: &str = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
const DOWN_W: &str = "model.language_model.layers.0.mlp.experts.1.down_proj.weight";
/// The packed NVFP4 weights the decoded scales belong to: `[n, k/2]` for a
/// `[n, k/16]` scale.
const GATE_W_SHAPE: [usize; 2] = [640, 1280];
const DOWN_W_SHAPE: [usize; 2] = [2560, 320];

pub(crate) struct Dir(pub(crate) PathBuf);
impl Dir {
    pub(crate) fn new(tag: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("atlas-csf-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(p.join("tensors")).unwrap();
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/nvfp4_csf")
            .join(name),
    )
    .unwrap()
}

fn golden_sha(tag: &str) -> String {
    let doc: serde_json::Value = serde_json::from_slice(&fixture("golden.json")).unwrap();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["tag"] == tag)
        .unwrap()["decoded_sha256"]
        .as_str()
        .unwrap()
        .to_string()
}

pub(crate) type T<'a> = (&'a str, &'a str, Vec<usize>, Vec<u8>);

pub(crate) fn write_shard(path: &Path, tensors: &[T<'_>]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}),
        );
        off += bytes.len();
    }
    let json = serde_json::to_vec(&header).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&json).unwrap();
    for (_, _, _, bytes) in tensors {
        f.write_all(bytes).unwrap();
    }
}

/// Two shards under `tensors/`, each CSF pair split across them.
fn checkpoint(dir: &Path, with_gate_exceptions: bool) {
    checkpoint_with(dir, with_gate_exceptions, GATE_W_SHAPE, &[]);
}

/// [`checkpoint`] with the gate weight at `gate_w` and `extra_b` added to
/// shard b.
fn checkpoint_with(dir: &Path, with_gate_exceptions: bool, gate_w: [usize; 2], extra_b: &[T<'_>]) {
    let gate_f = fixture("l0_e0_gate.fixed.bin");
    let gate_e = fixture("l0_e0_gate.exceptions.bin");
    let down_f = fixture("l0_e0_down.fixed.bin");
    let down_e = fixture("l0_e0_down.exceptions.bin");
    let fx = |n: &str| format!("{n}.nvfp4_csf_fixed");
    let ex = |n: &str| format!("{n}.nvfp4_csf_exceptions");
    let (gf, ge, df, de) = (fx(GATE), ex(GATE), fx(DOWN), ex(DOWN));
    write_shard(
        &dir.join("tensors/a.safetensors"),
        &[
            (&gf, "U8", vec![40, 1296], gate_f),
            (&de, "U32", vec![down_e.len() / 4], down_e),
            (PLAIN, "U8", gate_w.to_vec(), vec![7; gate_w[0] * gate_w[1]]),
        ],
    );
    let mut b: Vec<T<'_>> = vec![
        (&df, "U8", vec![160, 336], down_f),
        (DOWN_W, "U8", DOWN_W_SHAPE.to_vec(), vec![9; 2560 * 320]),
    ];
    if with_gate_exceptions {
        b.push((&ge, "U32", vec![gate_e.len() / 4], gate_e));
    }
    b.extend(extra_b.iter().cloned());
    write_shard(&dir.join("tensors/b.safetensors"), &b);
    let mut map = serde_json::Map::new();
    for (n, s) in [
        (&gf, "a"),
        (&de, "a"),
        (&PLAIN.to_string(), "a"),
        (&df, "b"),
        (&DOWN_W.to_string(), "b"),
    ] {
        map.insert(n.clone(), format!("tensors/{s}.safetensors").into());
    }
    for (n, ..) in extra_b {
        map.insert(n.to_string(), "tensors/b.safetensors".into());
    }
    if with_gate_exceptions {
        map.insert(ge.clone(), "tensors/b.safetensors".into());
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap(),
    )
    .unwrap();
}

fn check(store: &WeightStore, gpu: &MockGpuBackend, which: &str) {
    for (name, tag, shape) in [
        (GATE, "l0_e0_gate", [640, 160]),
        (DOWN, "l0_e0_down", [2560, 40]),
    ] {
        let t = store.get(name).unwrap_or_else(|e| panic!("{which}: {e}"));
        assert_eq!(t.dtype, WeightDtype::FP8E4M3, "{which}: {name}");
        assert_eq!(t.shape, shape.to_vec(), "{which}: {name}");
        let bytes = gpu.read_alloc(t.ptr).unwrap();
        let sha: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(sha, golden_sha(tag), "{which}: {name} decoded wrong");
    }
    assert!(store.contains(PLAIN), "{which}: plain tensor lost");
    assert!(
        store.names().all(|n| nvfp4_csf::csf_part(n).is_none()),
        "{which}: a CSF stream reached the store"
    );
    assert_eq!(store.len(), 4, "{which}");
}

#[test]
fn both_loaders_decode_cross_shard_pairs_from_a_subdirectory_index() {
    let dir = Dir::new("ok");
    checkpoint(&dir.0, true);

    let gpu = MockGpuBackend::new();
    let store = SafetensorsLoader::new().load(&dir.0, &gpu, 0).unwrap();
    check(&store, &gpu, "mmap");

    let gpu = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let store = fast.load(&dir.0, &gpu, 0).unwrap();
    check(&store, &gpu, "fast");

    // The pre-flight counts each pair at its decoded size.
    let shards = [
        dir.0.join("tensors/a.safetensors"),
        dir.0.join("tensors/b.safetensors"),
    ];
    assert_eq!(
        estimate_load_bytes(&shards, &|_| false).unwrap(),
        640 * 160 + 2560 * 40 + 640 * 1280 + 2560 * 320
    );
}

#[test]
fn a_lone_csf_half_refuses_the_load() {
    let dir = Dir::new("lone");
    checkpoint(&dir.0, false);
    let gpu = MockGpuBackend::new();
    let e = format!(
        "{:#}",
        SafetensorsLoader::new()
            .load(&dir.0, &gpu, 0)
            .err()
            .unwrap()
    );
    assert!(e.contains("without a partner"), "{e}");
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let e = format!("{:#}", fast.load(&dir.0, &gpu, 0).err().unwrap());
    assert!(e.contains("without a partner"), "{e}");
}

/// Under EP each rank keeps its own experts: both halves of a remote expert's
/// pair are skipped together, so nothing is left unpaired.
#[test]
fn ep_skips_both_halves_of_a_remote_pair() {
    let dir = Dir::new("ep");
    checkpoint(&dir.0, true);
    let gpu = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::with_ep(1, 2, 2);
    fast.try_direct_io = false;
    let store = fast.load(&dir.0, &gpu, 0).unwrap();
    assert!(store.contains(DOWN) && !store.contains(GATE) && !store.contains(PLAIN));
    let store = SafetensorsLoader::with_ep(1, 2, 2)
        .load(&dir.0, &gpu, 0)
        .unwrap();
    assert!(store.contains(DOWN) && !store.contains(GATE));
}

/// Both loaders, a load that must fail with `needle`.
fn both_refuse(dir: &Path, needle: &str) {
    let gpu = MockGpuBackend::new();
    let e = format!(
        "{:#}",
        SafetensorsLoader::new().load(dir, &gpu, 0).err().unwrap()
    );
    assert!(e.contains(needle), "mmap: {e}");
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let e = format!("{:#}", fast.load(dir, &gpu, 0).err().unwrap());
    assert!(e.contains(needle), "fast: {e}");
}

/// A decoded scale that is not the `[n, k/16]` its weight needs fails the
/// load, whichever shard the weight is in.
#[test]
fn a_scale_that_does_not_fit_its_weight_refuses_the_load() {
    let dir = Dir::new("misfit");
    checkpoint_with(&dir.0, true, [640, 1288], &[]);
    both_refuse(&dir.0, "needs");
}

/// A plain `weight_scale` beside a CSF pair for the same name fails the
/// load. The plain copy sits in shard b after the pair's fixed half, so the
/// pair completes first: the order the loaders used to let through (they
/// checked only a pair arriving after its plain twin). Either order is
/// covered at the assembler level by
/// `nvfp4_csf::tests::a_scale_both_plain_and_csf_is_refused_in_either_order`.
#[test]
fn a_scale_both_plain_and_csf_refuses_the_load() {
    let dir = Dir::new("dup");
    let plain = vec![0x38u8; 2560 * 40];
    checkpoint_with(
        &dir.0,
        true,
        GATE_W_SHAPE,
        &[(DOWN, "F8_E4M3", vec![2560, 40], plain)],
    );
    both_refuse(&dir.0, "both plain and CSF");
}

// ---- csf_32x2_v1 --------------------------------------------------------------

const C32_DOWN: &str = "model.language_model.layers.3.mlp.experts.0.down_proj.weight_scale";
const C32_SYN: &str = "model.language_model.layers.3.mlp.experts.1.up_proj.weight_scale";

fn csf32_file(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/nvfp4_csf32")
            .join(name),
    )
    .unwrap()
}

/// A tensor with an owned name: `(name, dtype, shape, bytes)`.
type Owned = (String, &'static str, Vec<usize>, Vec<u8>);

/// Two shards under `tensors/`: a real GLM down_proj scale ([4096, 128]) and
/// the synthetic odd-`C` one ([160, 133]), each scale's five components
/// alternating between the shards, the weights in the shard opposite their
/// first component. `drop` leaves one component of the down scale out.
fn csf32_checkpoint(dir: &Path, drop: Option<&str>) -> Vec<PathBuf> {
    let comps = [
        ("bases", "U8", 1),
        ("codes", "U8", 1),
        ("page_bases", "U32", 4),
        ("offsets", "U16", 2),
        ("exceptions", "U16", 2),
    ];
    let mut owned: [Vec<Owned>; 2] = Default::default();
    for (scale, tag, w_shape) in [
        (C32_DOWN, "l3_e0_down", [4096, 1024]),
        (C32_SYN, "syn_160x133", [160, 1064]),
    ] {
        for (i, (part, dtype, elem)) in comps.into_iter().enumerate() {
            if scale == C32_DOWN && drop == Some(part) {
                continue;
            }
            let bytes = csf32_file(&format!("{tag}.{part}.bin"));
            let shape = vec![bytes.len() / elem];
            owned[i % 2].push((format!("{scale}.csf32_{part}"), dtype, shape, bytes));
        }
        let w = scale.strip_suffix("_scale").unwrap().to_string();
        owned[1].push((w, "U8", w_shape.to_vec(), vec![3; w_shape[0] * w_shape[1]]));
    }
    let mut map = serde_json::Map::new();
    let mut paths = Vec::new();
    for (s, tensors) in ["a", "b"].iter().zip(&owned) {
        let shard = format!("tensors/{s}.safetensors");
        let t: Vec<T<'_>> = tensors
            .iter()
            .map(|(n, d, sh, b)| (n.as_str(), *d, sh.clone(), b.clone()))
            .collect();
        write_shard(&dir.join(&shard), &t);
        for (n, ..) in tensors {
            map.insert(n.clone(), shard.clone().into());
        }
        paths.push(dir.join(&shard));
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap(),
    )
    .unwrap();
    paths
}

fn csf32_golden_sha(tag: &str) -> String {
    let doc: serde_json::Value = serde_json::from_slice(&csf32_file("golden.json")).unwrap();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["tag"] == tag)
        .unwrap()["decoded_sha256"]
        .as_str()
        .unwrap()
        .to_string()
}

fn check_csf32(store: &WeightStore, gpu: &MockGpuBackend, which: &str) {
    for (name, tag, shape) in [
        (C32_DOWN, "l3_e0_down", [4096, 128]),
        (C32_SYN, "syn_160x133", [160, 133]),
    ] {
        let t = store.get(name).unwrap_or_else(|e| panic!("{which}: {e}"));
        assert_eq!(t.dtype, WeightDtype::FP8E4M3, "{which}: {name}");
        assert_eq!(t.shape, shape.to_vec(), "{which}: {name}");
        let sha: String = Sha256::digest(gpu.read_alloc(t.ptr).unwrap())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(sha, csf32_golden_sha(tag), "{which}: {name} decoded wrong");
    }
    assert!(
        store.names().all(|n| nvfp4_csf::csf_part(n).is_none()),
        "{which}: a CSF component reached the store"
    );
    assert_eq!(store.len(), 4, "{which}: two scales + two weights");
}

/// The GLM layout: five components per scale, split across shards, U16 and
/// U32 dtypes the store never carries. Both loaders present each scale as
/// the plain F8_E4M3 `weight_scale` the reference decodes it to, and the
/// pre-flight counts it at its decoded size.
#[test]
fn both_loaders_decode_csf32_scales_split_across_shards() {
    let dir = Dir::new("c32");
    let shards = csf32_checkpoint(&dir.0, None);

    let gpu = MockGpuBackend::new();
    let store = SafetensorsLoader::new().load(&dir.0, &gpu, 0).unwrap();
    check_csf32(&store, &gpu, "mmap");

    let gpu = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    let store = fast.load(&dir.0, &gpu, 0).unwrap();
    check_csf32(&store, &gpu, "fast");

    assert_eq!(
        estimate_load_bytes(&shards, &|_| false).unwrap(),
        4096 * 128 + 160 * 133 + 4096 * 1024 + 160 * 1064
    );
}

/// A scale missing any one of its five components fails the load.
#[test]
fn a_csf32_scale_missing_a_component_refuses_the_load() {
    for part in ["bases", "page_bases", "exceptions"] {
        let dir = Dir::new("c32lone");
        csf32_checkpoint(&dir.0, Some(part));
        both_refuse(&dir.0, "without a partner");
    }
}

#[test]
fn slab_packing_preserves_decoded_csf32_scales() {
    let dir = Dir::new("c32slab");
    csf32_checkpoint(&dir.0, None);
    let gpu = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::new();
    fast.try_direct_io = false;
    fast.pack_slab = true;
    let store = fast.load(&dir.0, &gpu, 0).unwrap();
    check_csf32(&store, &gpu, "packed fast");
}

/// Under EP the remote expert's five components are all skipped.
#[test]
fn ep_skips_every_component_of_a_remote_csf32_scale() {
    let dir = Dir::new("c32ep");
    csf32_checkpoint(&dir.0, None);
    let gpu = MockGpuBackend::new();
    let mut fast = FastSafetensorsLoader::with_ep(1, 2, 2);
    fast.try_direct_io = false;
    let store = fast.load(&dir.0, &gpu, 0).unwrap();
    assert!(store.contains(C32_SYN) && !store.contains(C32_DOWN));
    let store = SafetensorsLoader::with_ep(0, 2, 2)
        .load(&dir.0, &gpu, 0)
        .unwrap();
    assert!(store.contains(C32_DOWN) && !store.contains(C32_SYN));
}
