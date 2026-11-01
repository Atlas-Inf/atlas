// SPDX-License-Identifier: AGPL-3.0-only
use super::WeightDtype;

#[test]
fn from_safetensors_str_matches_disk_mapping() {
    // The RDMA weight peer publishes these raw header strings; the client
    // must resolve them to the exact WeightDtype the disk loaders use, else
    // byte_size/shape diverge and logits break. Locks the closed mapping.
    use WeightDtype::*;
    for (s, want) in [
        ("F32", FP32),
        ("BF16", BF16),
        ("U8", UInt8),
        ("I8", UInt8), // packed NVFP4 raw container
        ("F8_E4M3", FP8E4M3),
        ("F8_E8M0", FP8E8M0),
        ("I64", Int64),
    ] {
        assert_eq!(
            WeightDtype::from_safetensors_str(s).unwrap(),
            want,
            "dtype {s}"
        );
    }
    // F16 is converted to BF16 at disk-load; a store (and therefore a
    // peer manifest) can never contain it, so the wire mapping rejects it.
    assert!(WeightDtype::from_safetensors_str("F16").is_err());
    assert!(WeightDtype::from_safetensors_str("bogus").is_err());
}

#[test]
fn f16_bytes_convert_to_bf16_via_f32() {
    use half::{bf16, f16};
    // Cover sign, exact powers of two, a value needing mantissa rounding
    // (f16 has 10 mantissa bits, bf16 only 7), f16 max, and a subnormal.
    let vals = [0.0f32, 1.0, -1.5, 0.1, 65504.0, -6.1035156e-5];
    let src: Vec<u8> = vals
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_le_bytes())
        .collect();
    let out = super::f16_to_bf16_bytes(&src);
    assert_eq!(out.len(), src.len());
    for (i, v) in vals.iter().enumerate() {
        let got = bf16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
        let want = bf16::from_f32(f16::from_f32(*v).to_f32());
        assert_eq!(got, want, "value {v}");
    }
}
