// SPDX-License-Identifier: AGPL-3.0-only

//! The fast loader turns CSF pairs into resident `weight_scale` tensors.
//!
//! The checkpoint here is shaped like FP4-Local where it matters: the index
//! routes to shards under a `tensors/` subdirectory, and a pair can sit across
//! two shards, in either order.

use super::FastSafetensorsLoader;
use crate::gpu::mock::MockGpuBackend;
use crate::weights::{WeightDtype, WeightLoader};
use atlas_core::modelopt_codec::decode_csf_scales;
use std::io::Write;
use std::path::{Path, PathBuf};

const GATE: &str = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale";
const DOWN: &str = "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale";
const PLAIN: &str = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
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

type T = (String, &'static str, Vec<usize>, Vec<u8>);

fn write_shard(path: &Path, tensors: &[T]) {
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}),
        );
        off += bytes.len();
    }
    let json = serde_json::to_vec(&header).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&json).unwrap();
    for (.., bytes) in tensors {
        f.write_all(bytes).unwrap();
    }
}

/// Write `shards` under `tensors/` with an index routing each tensor there.
fn checkpoint(dir: &Path, shards: &[Vec<T>]) {
    let mut map = serde_json::Map::new();
    for (i, tensors) in shards.iter().enumerate() {
        let file = format!("tensors/s{i}.safetensors");
        write_shard(&dir.join(&file), tensors);
        for (name, ..) in tensors {
            map.insert(name.clone(), file.clone().into());
        }
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap(),
    )
    .unwrap();
}

/// One 16-row slab of a two-column scale: base `seed + row`, codes 1 and 2.
fn fixed(scale: &str, seed: u8) -> T {
    let mut bytes: Vec<u8> = (0..16).map(|row| seed + row).collect();
    bytes.extend([0x21u8; 16]);
    (format!("{scale}.nvfp4_csf_fixed"), "U8", vec![1, 32], bytes)
}

/// One exception: position 3 becomes `0xAB`.
fn exceptions(scale: &str) -> T {
    let word = (0xABu32 << 24) | 3;
    (
        format!("{scale}.nvfp4_csf_exceptions"),
        "U32",
        vec![1],
        word.to_le_bytes().to_vec(),
    )
}

fn plain(name: &str, bytes: Vec<u8>) -> T {
    (name.to_string(), "U8", vec![bytes.len()], bytes)
}

fn load(dir: &Path, gpu: &MockGpuBackend) -> anyhow::Result<crate::weights::WeightStore> {
    FastSafetensorsLoader::new().load(dir, gpu, 0)
}

/// The scale as the codec decodes it, independently of the loader.
fn want(seed: u8) -> Vec<u8> {
    let (name, _, _, bytes) = fixed("x", seed);
    assert!(name.ends_with("_fixed"));
    let word = exceptions("x").3;
    decode_csf_scales(&bytes, 1, 32, &word).unwrap()
}

#[test]
fn a_pair_across_shards_becomes_one_resident_scale_in_either_order() {
    let dir = Dir::new("order");
    checkpoint(
        &dir.0,
        &[
            vec![fixed(GATE, 10), exceptions(DOWN), plain(PLAIN, vec![7; 64])],
            vec![exceptions(GATE), fixed(DOWN, 30)],
        ],
    );
    let gpu = MockGpuBackend::new();
    let store = load(&dir.0, &gpu).unwrap();
    for (name, seed) in [(GATE, 10), (DOWN, 30)] {
        let t = store.get(name).unwrap();
        assert_eq!(t.dtype, WeightDtype::FP8E4M3, "{name}");
        assert_eq!(t.shape, vec![16, 2], "{name}");
        assert_eq!(gpu.read_alloc(t.ptr).unwrap(), want(seed), "{name}");
    }
    assert!(store.contains(PLAIN));
    assert_eq!(store.len(), 3, "a stream reached the store");
    assert!(
        store.deferred_sorted().is_empty(),
        "a stream stayed on disk"
    );
}

#[test]
fn a_lone_stream_fails_the_load() {
    let dir = Dir::new("lone");
    checkpoint(&dir.0, &[vec![fixed(GATE, 10), plain(PLAIN, vec![7; 64])]]);
    let err = load(&dir.0, &MockGpuBackend::new()).err().unwrap();
    assert!(err.to_string().contains("without a partner"), "{err:#}");
}

#[test]
fn a_scale_both_plain_and_compressed_is_refused_whichever_comes_first() {
    for plain_first in [true, false] {
        let dir = Dir::new("both");
        let scale = plain(GATE, vec![1; 32]);
        let pair = vec![fixed(GATE, 10), exceptions(GATE)];
        let shards = if plain_first {
            [vec![scale], pair]
        } else {
            [pair, vec![scale]]
        };
        checkpoint(&dir.0, &shards);
        let err = load(&dir.0, &MockGpuBackend::new()).err().unwrap();
        assert!(
            format!("{err:#}").contains("both plain and CSF-compressed"),
            "{err:#}"
        );
    }
}

/// More tensors than the readers may hold ahead, in lengths that make them
/// finish out of order: every one lands under its own name, intact.
#[test]
fn every_tensor_lands_intact_past_the_read_ahead() {
    let dir = Dir::new("order");
    let tensors: Vec<T> = (0..200u32)
        .map(|i| {
            let len = 1 + (i as usize * 7919) % 200_000;
            plain(&format!("t{i}"), vec![i as u8; len])
        })
        .collect();
    checkpoint(&dir.0, std::slice::from_ref(&tensors));
    let gpu = MockGpuBackend::new();
    let store = load(&dir.0, &gpu).unwrap();
    assert_eq!(store.len(), tensors.len());
    for (name, _, _, bytes) in &tensors {
        let t = store.get(name).unwrap();
        assert_eq!(&gpu.read_alloc(t.ptr).unwrap(), bytes, "{name}");
    }
}

/// A slab reserves nothing for streams: only the decoded scale is uploaded.
#[test]
fn a_slab_holds_no_stream_bytes() {
    let dir = Dir::new("slab");
    checkpoint(
        &dir.0,
        &[vec![
            fixed(GATE, 10),
            exceptions(GATE),
            plain(PLAIN, vec![7; 64]),
        ]],
    );
    let gpu = MockGpuBackend::new();
    let mut loader = FastSafetensorsLoader::new();
    loader.pack_slab = true;
    let store = loader.load(&dir.0, &gpu, 0).unwrap();
    let scale = store.get(GATE).unwrap();
    assert_eq!(gpu.read_alloc(scale.ptr).unwrap(), want(10));
    // One slab for the plain tensor, one allocation for the scale.
    assert_eq!(gpu.alloc_count(), 2);
}
