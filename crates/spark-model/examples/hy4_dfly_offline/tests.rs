// SPDX-License-Identifier: AGPL-3.0-only

use super::io::CheckpointIo;
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(dtype: &str, truncated: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "atlas-dfly-rows-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let raw: Vec<u8> = (0..12)
            .flat_map(|v| {
                if dtype == "BF16" {
                    half::bf16::from_f32(v as f32).to_le_bytes().to_vec()
                } else {
                    (v as f32).to_le_bytes().to_vec()
                }
            })
            .collect();
        let header = serde_json::to_vec(&serde_json::json!({
            "embedding": {"dtype":dtype, "shape":[4,3], "data_offsets":[0,raw.len()]}
        }))
        .unwrap();
        let mut f = fs::File::create(root.join("weights.safetensors")).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&raw[..raw.len() - usize::from(truncated)])
            .unwrap();
        fs::write(
            root.join("model.safetensors.index.json"),
            br#"{"weight_map":{"embedding":"weights.safetensors"}}"#,
        )
        .unwrap();
        Self(root)
    }
    fn open(&self) -> CheckpointIo {
        CheckpointIo::open(&self.0).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn reads_only_requested_rows_including_repeated_tokens_for_both_dtypes() {
    for dtype in ["BF16", "F32"] {
        let f = Fixture::new(dtype, false);
        let mut actual = Vec::new();
        for id in [2, 0, 2] {
            f.open()
                .copy_rows("embedding", [4, 3], id, 1, |offset, bytes| {
                    assert_eq!(offset, 0);
                    assert_eq!(bytes.len(), 6);
                    actual.extend_from_slice(bytes);
                    Ok(())
                })
                .unwrap();
        }
        let expected: Vec<_> = [6., 7., 8., 0., 1., 2., 6., 7., 8.]
            .into_iter()
            .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn rejects_empty_invalid_truncated_and_failed_copy_requests() {
    let f = Fixture::new("BF16", false);
    for (shape, start, rows) in [
        ([4, 3], 0, 0),
        ([4, 3], 4, 1),
        ([4, 3], 3, 2),
        ([4, 4], 0, 1),
    ] {
        assert!(
            f.open()
                .copy_rows("embedding", shape, start, rows, |_, _| panic!(
                    "invalid copy"
                ))
                .is_err()
        );
    }
    assert!(
        f.open()
            .copy_rows("missing", [4, 3], 0, 1, |_, _| Ok(()))
            .is_err()
    );
    assert!(
        f.open()
            .copy_rows("embedding", [4, 3], 0, 1, |_, _| anyhow::bail!(
                "copy failed"
            ))
            .is_err()
    );
    let truncated = Fixture::new("BF16", true);
    assert!(
        truncated
            .open()
            .copy_rows("embedding", [4, 3], 0, 1, |_, _| Ok(()))
            .is_err()
    );
}
