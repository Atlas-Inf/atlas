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

/// FP4-Local's CSF exception streams are U32; they stage as a raw 4-byte
/// container exactly as the buffered loader maps them.
#[test]
fn u32_stages_as_a_raw_four_byte_container() {
    let mut file = write_safetensors(
        "u32",
        &[("e.weight_scale.nvfp4_csf_exceptions", "U32", &[81], 4)],
    );
    let metas = parse_header(&mut file).unwrap();
    assert_eq!(metas[0].dtype, WeightDtype::Int32);
    assert_eq!(metas[0].len, 324);
    assert!(!metas[0].from_f16);
}

/// FP4-Local's index names its shards under `tensors/`; each shard must still
/// receive exactly the tensors routed to it.
#[test]
fn shards_in_a_subdirectory_receive_their_tensors() {
    let dir = Path::new("/models/fp4-local");
    let map: HashMap<String, String> = [
        ("a.weight", "tensors/model-00001-of-00002.safetensors"),
        ("b.weight", "tensors/model-00002-of-00002.safetensors"),
        ("c.weight", "tensors/model-00001-of-00002.safetensors"),
    ]
    .into_iter()
    .map(|(t, s)| (t.to_string(), s.to_string()))
    .collect();
    let mut got = tensors_routed_to(
        &map,
        dir,
        &dir.join("tensors/model-00001-of-00002.safetensors"),
    );
    got.sort();
    assert_eq!(got, ["a.weight", "c.weight"]);
    // A flat index still works, and a same-named file elsewhere matches nothing.
    let flat: HashMap<String, String> = [("x.weight".to_string(), "model.safetensors".to_string())]
        .into_iter()
        .collect();
    assert_eq!(
        tensors_routed_to(&flat, dir, &dir.join("model.safetensors")),
        ["x.weight"]
    );
    assert!(tensors_routed_to(&map, dir, &dir.join("model-00001-of-00002.safetensors")).is_empty());
}
