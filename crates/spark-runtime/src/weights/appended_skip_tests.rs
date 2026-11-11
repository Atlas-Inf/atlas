// SPDX-License-Identifier: AGPL-3.0-only
//! Actual file loading and byte estimates agree when an appended layer is unused.
#![cfg(unix)]
use super::*;
use crate::fast_weights::FastSafetensorsLoader;
use crate::gpu::mock::MockGpuBackend;
use std::io::Write;

const PREFIX: &str = "model.language_model.layers.45.";

struct TestDir(std::path::PathBuf);
impl TestDir {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let counter = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "atlas-appended-skip-{}-{counter}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn checkpoint(path: &std::path::Path) {
    write_checkpoint(
        path,
        &[
            "model.language_model.layers.44.mlp.experts.0.weight",
            "model.language_model.layers.44.mlp.experts.1.weight",
            "model.language_model.layers.45.mlp.experts.0.weight",
            "model.language_model.layers.45.mlp.experts.1.weight",
            "model.language_model.layers.45.enorm.weight",
            "model.language_model.layers.450.norm.weight",
            "other.model.language_model.layers.45.norm.weight",
        ],
    );
}

fn write_checkpoint(path: &std::path::Path, names: &[&str]) {
    let mut header = serde_json::Map::new();
    for (i, name) in names.iter().copied().enumerate() {
        header.insert(
            name.into(),
            serde_json::json!({"dtype":"BF16", "shape":[2], "data_offsets":[i*4,(i+1)*4]}),
        );
    }
    let bytes = serde_json::to_vec(&header).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&bytes).unwrap();
    file.write_all(&(0..(names.len() * 4) as u8).collect::<Vec<_>>())
        .unwrap();
}

#[test]
fn unused_appended_layer_skips_load_and_estimate_on_both_ep_ranks() {
    let tmp = TestDir::new();
    let path = tmp.path().join("model.safetensors");
    checkpoint(&path);
    for rank in 0..2 {
        let mut baseline = SafetensorsLoader::with_ep(rank, 2, 2);
        let mut fast = FastSafetensorsLoader::with_ep(rank, 2, 2);
        fast.try_direct_io = false;
        baseline.skip_layer_prefix = Some(PREFIX.into());
        fast.skip_layer_prefix = Some(PREFIX.into());
        let estimated = estimate_load_bytes(std::slice::from_ref(&path), &|name| {
            baseline.should_skip_tensor(name)
        })
        .unwrap();
        assert_eq!(
            estimated, 12,
            "only one target expert plus two boundary controls"
        );
        let a = MockGpuBackend::new();
        let b = MockGpuBackend::new();
        let lhs = baseline.load(tmp.path(), &a, 0).unwrap();
        let rhs = fast.load(tmp.path(), &b, 0).unwrap();
        assert_eq!(lhs.len(), 3);
        assert_eq!(rhs.len(), 3);
        assert_eq!(a.alloc_count(), 3);
        assert_eq!(b.alloc_count(), 3);
        for name in lhs.names() {
            assert!(!name.starts_with(PREFIX));
            assert_eq!(
                a.read_alloc(lhs.get(name).unwrap().ptr),
                b.read_alloc(rhs.get(name).unwrap().ptr)
            );
        }
    }
}

#[test]
fn appended_mtp_on_retains_existing_rank0_and_replicated_experts() {
    let tmp = TestDir::new();
    checkpoint(&tmp.path().join("model.safetensors"));
    for rank in 0..2 {
        for distributed in [false, true] {
            let mut baseline = SafetensorsLoader::with_ep(rank, 2, 2);
            let mut fast = FastSafetensorsLoader::with_ep(rank, 2, 2);
            fast.try_direct_io = false;
            if distributed {
                baseline.replicated_expert_prefix = Some(".layers.45.".into());
                fast.replicated_expert_prefix = Some(".layers.45.".into());
            } else {
                baseline.rank0_only_expert_prefix = Some(".layers.45.".into());
                fast.rank0_only_expert_prefix = Some(".layers.45.".into());
            }
            let a = MockGpuBackend::new();
            let b = MockGpuBackend::new();
            let lhs = baseline.load(tmp.path(), &a, 0).unwrap();
            let rhs = fast.load(tmp.path(), &b, 0).unwrap();
            let expected = if distributed || rank == 0 { 6 } else { 4 };
            assert_eq!(lhs.len(), expected);
            assert_eq!(rhs.len(), expected);
            assert!(lhs.contains("model.language_model.layers.45.enorm.weight"));
            for name in lhs.names() {
                assert_eq!(
                    a.read_alloc(lhs.get(name).unwrap().ptr),
                    b.read_alloc(rhs.get(name).unwrap().ptr)
                );
            }
        }
    }
}

#[test]
fn unused_appended_layer_is_not_reintroduced_by_extra_weights() {
    let tmp = TestDir::new();
    checkpoint(&tmp.path().join("model.safetensors"));
    write_checkpoint(
        &tmp.path().join("extra_weights.safetensors"),
        &["model.language_model.layers.45.enorm.weight", "extra.keep"],
    );
    for rank in 0..2 {
        let mut baseline = SafetensorsLoader::with_ep(rank, 2, 2);
        let mut fast = FastSafetensorsLoader::with_ep(rank, 2, 2);
        baseline.skip_layer_prefix = Some(PREFIX.into());
        fast.skip_layer_prefix = Some(PREFIX.into());
        fast.try_direct_io = false;
        let a = MockGpuBackend::new();
        let b = MockGpuBackend::new();
        let lhs = baseline.load(tmp.path(), &a, 0).unwrap();
        let rhs = fast.load(tmp.path(), &b, 0).unwrap();
        assert_eq!(lhs.len(), 4);
        assert_eq!(rhs.len(), 4);
        assert!(!lhs.contains("model.language_model.layers.45.enorm.weight"));
        assert!(!rhs.contains("model.language_model.layers.45.enorm.weight"));
        assert_eq!(
            a.read_alloc(lhs.get("extra.keep").unwrap().ptr),
            Some(vec![4, 5, 6, 7])
        );
        assert_eq!(
            b.read_alloc(rhs.get("extra.keep").unwrap().ptr),
            Some(vec![4, 5, 6, 7])
        );
    }
}
