// SPDX-License-Identifier: AGPL-3.0-only

//! Merging of the unindexed `*.safetensors` side files of an EXL3 checkpoint
//! into the index `weight_map`.

use super::merge_exl3_side_files;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Temp dir unique per test name and process; removed at the end of each test.
fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas-side-files-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write a safetensors file whose tensors are all zero bytes. `tensors` is
/// `(name, dtype, shape)`; every dtype is 2 bytes per element.
fn write_st(path: &Path, tensors: &[(&str, &str, &[usize])]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for &(name, dtype, shape) in tensors {
        let len = shape.iter().product::<usize>() * 2;
        header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len] }),
        );
        offset += len;
    }
    let json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&json).unwrap();
    f.write_all(&vec![0u8; offset]).unwrap();
}

fn write_config(dir: &Path, body: &str) {
    std::fs::write(dir.join("config.json"), body).unwrap();
}

/// An indexed model shard plus an unindexed n-gram side file.
fn exl3_dir(tag: &str, side_tensors: &[(&str, &str, &[usize])]) -> PathBuf {
    let dir = temp_dir(tag);
    write_config(&dir, r#"{"quantization_config": {"quant_method": "exl3"}}"#);
    write_st(
        &dir.join("model-00001.safetensors"),
        &[("a.weight", "BF16", &[2])],
    );
    write_st(&dir.join("ngram_embedding.safetensors"), side_tensors);
    dir
}

fn mapped() -> HashMap<String, String> {
    HashMap::from([(
        "a.weight".to_string(),
        "model-00001.safetensors".to_string(),
    )])
}

#[test]
fn exl3_side_file_tensors_are_merged() {
    let dir = exl3_dir(
        "merge",
        &[
            ("x.ngram_embedding.trellis", "I16", &[4, 51]),
            ("x.ngram_embedding.head_bias", "F16", &[2]),
        ],
    );
    let mut map = mapped();
    let merged = merge_exl3_side_files(&dir, &mut map).unwrap();

    assert_eq!(merged, vec!["ngram_embedding.safetensors".to_string()]);
    assert_eq!(
        map.get("x.ngram_embedding.trellis").map(String::as_str),
        Some("ngram_embedding.safetensors")
    );
    assert_eq!(
        map.get("x.ngram_embedding.head_bias").map(String::as_str),
        Some("ngram_embedding.safetensors")
    );
    assert_eq!(
        map.get("a.weight").map(String::as_str),
        Some("model-00001.safetensors")
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn indexed_names_win() {
    let dir = exl3_dir(
        "shadow",
        &[
            ("a.weight", "BF16", &[2]),
            ("x.ngram_embedding.trellis", "I16", &[4, 51]),
        ],
    );
    let mut map = mapped();
    let merged = merge_exl3_side_files(&dir, &mut map).unwrap();

    assert_eq!(merged, vec!["ngram_embedding.safetensors".to_string()]);
    assert_eq!(
        map.get("a.weight").map(String::as_str),
        Some("model-00001.safetensors")
    );
    assert_eq!(
        map.get("x.ngram_embedding.trellis").map(String::as_str),
        Some("ngram_embedding.safetensors")
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn extra_weights_is_left_alone() {
    let dir = exl3_dir("extra", &[("x.ngram_embedding.trellis", "I16", &[4, 51])]);
    write_st(
        &dir.join("extra_weights.safetensors"),
        &[("mtp.weight", "BF16", &[2])],
    );
    let mut map = mapped();
    let merged = merge_exl3_side_files(&dir, &mut map).unwrap();

    assert_eq!(merged, vec!["ngram_embedding.safetensors".to_string()]);
    assert!(
        !map.contains_key("mtp.weight"),
        "extra_weights has its own path"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn non_exl3_checkpoints_are_untouched() {
    let dir = temp_dir("fp8");
    write_config(&dir, r#"{"quantization_config": {"quant_method": "fp8"}}"#);
    write_st(
        &dir.join("model-00001.safetensors"),
        &[("a.weight", "BF16", &[2])],
    );
    write_st(
        &dir.join("consolidated.safetensors"),
        &[("b.weight", "BF16", &[2])],
    );
    let mut map = mapped();
    let merged = merge_exl3_side_files(&dir, &mut map).unwrap();

    assert!(merged.is_empty());
    assert_eq!(map, mapped());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_config_is_a_no_op() {
    let dir = temp_dir("noconfig");
    write_st(
        &dir.join("model-00001.safetensors"),
        &[("a.weight", "BF16", &[2])],
    );
    write_st(
        &dir.join("consolidated.safetensors"),
        &[("b.weight", "BF16", &[2])],
    );
    let mut map = mapped();
    let merged = merge_exl3_side_files(&dir, &mut map).unwrap();

    assert!(merged.is_empty());
    assert_eq!(map, mapped());
    std::fs::remove_dir_all(&dir).unwrap();
}
