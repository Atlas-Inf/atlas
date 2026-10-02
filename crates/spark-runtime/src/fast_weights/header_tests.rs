// SPDX-License-Identifier: AGPL-3.0-only

//! `parse_header` dtype staging for EXL3 checkpoints: trellis (I16) and mul1
//! (I32) are store-legal raw dtypes, the suh/svh scale vectors keep their raw
//! F16 bytes, and every other F16 tensor still stages as BF16 for conversion.

use super::*;
use std::io::Write;

/// Writes a safetensors file whose tensors are all zero bytes and returns it
/// opened for reading. `tensors` is `(name, dtype, shape, element bytes)`.
fn write_safetensors(tag: &str, tensors: &[(&str, &str, &[usize], usize)]) -> File {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for &(name, dtype, shape, elem) in tensors {
        let len = shape.iter().product::<usize>() * elem;
        header.insert(
            name.to_string(),
            serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len] }),
        );
        offset += len;
    }
    let json = serde_json::to_vec(&Value::Object(header)).unwrap();
    let path = std::env::temp_dir().join(format!(
        "atlas-exl3-header-{tag}-{}.safetensors",
        std::process::id()
    ));
    let mut f = File::create(&path).unwrap();
    f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&json).unwrap();
    f.write_all(&vec![0u8; offset]).unwrap();
    drop(f);
    let file = File::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    file
}

#[test]
fn exl3_tensors_stage_with_raw_dtypes() {
    let mut file = write_safetensors(
        "dtypes",
        &[
            ("a.trellis", "I16", &[2, 3], 2),
            ("a.mul1", "I32", &[], 4),
            ("a.suh", "F16", &[4], 2),
            ("a.svh", "F16", &[4], 2),
            ("b.weight", "F16", &[4], 2),
        ],
    );
    let metas = parse_header(&mut file).unwrap();
    let get = |name: &str| metas.iter().find(|t| t.name == name).unwrap();

    assert_eq!(get("a.trellis").dtype, WeightDtype::Int16);
    assert!(!get("a.trellis").from_f16);
    assert_eq!(get("a.trellis").len, 12);
    // A 0-d scalar is one element, not zero.
    assert_eq!(get("a.mul1").dtype, WeightDtype::Int32);
    assert_eq!(get("a.mul1").len, 4);
    for scale in ["a.suh", "a.svh"] {
        assert_eq!(get(scale).dtype, WeightDtype::FP16, "{scale}");
        assert!(!get(scale).from_f16, "{scale} must not be converted");
    }
    // Any other F16 tensor keeps the BF16 staging and the byte conversion.
    assert_eq!(get("b.weight").dtype, WeightDtype::BF16);
    assert!(get("b.weight").from_f16);
}

#[test]
fn unknown_dtype_is_still_refused() {
    let mut file = write_safetensors("unknown", &[("x.weight", "C64", &[1], 8)]);
    let Err(err) = parse_header(&mut file) else {
        panic!("a C64 tensor must be refused");
    };
    assert!(format!("{err:#}").contains("C64"), "{err:#}");
}
