// SPDX-License-Identifier: AGPL-3.0-only

//! Tiny real-device/file check: one GPU exercises every TP4 rank geometry.
//! No model, collectives, or inference kernels are needed.
//!
//! ATLAS_NVME_TEST_DIR=/nvme cargo test -p spark-runtime --lib \
//!   tp4_nvme_device_file_round_trip -- --ignored --nocapture

use super::*;
use crate::cuda_backend::AtlasCudaBackend;

struct Scratch(std::path::PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires a CUDA device and ATLAS_NVME_TEST_DIR on a real O_DIRECT filesystem"]
fn tp4_nvme_device_file_round_trip() {
    let root = std::path::PathBuf::from(
        std::env::var_os("ATLAS_NVME_TEST_DIR")
            .expect("set ATLAS_NVME_TEST_DIR to an NVMe directory"),
    );
    assert!(root.is_dir(), "test directory must already exist");
    assert_eq!(atlas_tier::unsuitable_swap_fs(&root), None);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let scratch = Scratch(root.join(format!("atlas-tp4-nvme-{}-{nonce}", std::process::id())));
    std::fs::create_dir(&scratch.0).unwrap();
    // This check only uses allocation and DMA, so no PTX modules are loaded.
    let gpu = AtlasCudaBackend::new(0, &[]).unwrap();
    let stream = gpu.default_stream();
    let mut checked = 0usize;
    for fast in [false, true] {
        for rank in 0..4 {
            let mut c = sharded_world(&gpu, rank, 4, 32, None);
            for class in 0..4 {
                let path = scratch.0.join(format!("r{rank}.c{class}.fast{fast}.swap"));
                let bytes = c.nvme_class_record_bytes(class);
                if fast {
                    let store = atlas_tier::SharedRecordFile::create(&path, bytes).unwrap();
                    assert!(store.reserve(4 * bytes as u64).unwrap());
                    c.attach_nvme_fast(Arc::new(store), &gpu).unwrap();
                } else {
                    let store = atlas_tier::DirectSwapFile::create_new(&path, bytes).unwrap();
                    c.attach_nvme_spill(Box::new(store), &gpu).unwrap();
                }
                std::fs::remove_file(path).unwrap();
            }
            assert!(c.nvme_attached());
            let source: Vec<u32> = (0..8).map(|l| c.alloc_block_at(l).unwrap()).collect();
            let want: Vec<Vec<u8>> = source
                .iter()
                .enumerate()
                .map(|(l, &b)| {
                    fill(&c, &gpu, b, 31 + l as u8);
                    dump(&c, &gpu, b)
                })
                .collect();
            let orders: Vec<SpillOrder> = source
                .iter()
                .enumerate()
                .rev()
                .map(|(l, &block)| SpillOrder {
                    block,
                    slot: l as u32,
                    tag: tag(l as u32),
                })
                .collect();
            assert!(c.nvme_write(&orders, &gpu, stream).is_empty());
            for &b in &source {
                fill(&c, &gpu, b, 0xE7);
            }
            let mut disk: Vec<DiskRef> = (0..8u32)
                .map(|slot| DiskRef {
                    slot,
                    tag: tag(slot),
                })
                .collect();
            let mut targets: Vec<u32> = (0..8).map(|l| c.alloc_block_at(l).unwrap()).collect();
            // Exercise fast lane-local sorting without changing any residue.
            targets.rotate_left(4);
            assert_eq!(c.nvme_read(&disk, &mut targets, &gpu, stream), (8, false));
            for (l, &b) in targets.iter().enumerate() {
                assert_eq!(b as usize % 4, l % 4);
                assert_eq!(
                    dump(&c, &gpu, b),
                    want[l],
                    "rank={rank} fast={fast} block={l}"
                );
                checked += want[l].len();
            }
            // A stale class-3 tag bounds the adopted prefix even though other
            // lanes can complete records beyond that logical position.
            disk[3].tag ^= 1;
            assert_eq!(c.nvme_read(&disk, &mut targets, &gpu, stream), (3, true));
            for (l, &b) in targets[..3].iter().enumerate() {
                assert_eq!(dump(&c, &gpu, b), want[l]);
            }
            assert!(c.nvme_take_failed().is_empty());
            c.release(&gpu).unwrap();
            println!("TP4 NVMe CUDA/file: rank={rank} fast={fast} bytes=exact bad_tag_prefix=3");
        }
    }
    assert_eq!(std::fs::read_dir(&scratch.0).unwrap().count(), 0);
    println!(
        "TP4 NVMe CUDA/file PASS: {checked} bytes, eight rank/path cases, no named files retained"
    );
}
