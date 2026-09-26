// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 runtime-dtype tests (M2a): the new I16/I32/F16 mappings, the raw-F16
//! name rule, and the EXL3 n-gram deferral names. Split out of `weights.rs`
//! (≤500 LoC cap), following the `packed_q2_tests` pattern.

use super::*;

#[test]
fn exl3_dtype_byte_sizes() {
    assert_eq!(WeightDtype::FP16.byte_size(), 2);
    assert_eq!(WeightDtype::Int16.byte_size(), 2);
    assert_eq!(WeightDtype::Int32.byte_size(), 4);
}

#[test]
fn exl3_dtypes_from_safetensors_enum() {
    assert_eq!(
        WeightDtype::from_safetensors(safetensors::Dtype::I16).unwrap(),
        WeightDtype::Int16
    );
    assert_eq!(
        WeightDtype::from_safetensors(safetensors::Dtype::I32).unwrap(),
        WeightDtype::Int32
    );
}

#[test]
fn exl3_dtypes_from_safetensors_str_and_f16_stays_rejected() {
    // The wire (RDMA peer manifest) must resolve I16/I32 exactly like disk.
    assert_eq!(
        WeightDtype::from_safetensors_str("I16").unwrap(),
        WeightDtype::Int16
    );
    assert_eq!(
        WeightDtype::from_safetensors_str("I32").unwrap(),
        WeightDtype::Int32
    );
    // F16 deliberately stays an error on the wire: peers stage BF16 only, and
    // EXL3's raw-F16 scale vectors never reach it (see `from_safetensors_str`).
    assert!(WeightDtype::from_safetensors_str("F16").is_err());
}

#[test]
fn keeps_raw_f16_matches_only_exl3_scale_suffixes() {
    assert!(keeps_raw_f16("model.layers.0.self_attn.q_proj.suh"));
    assert!(keeps_raw_f16("model.layers.3.self_attn.o_proj.svh"));
    assert!(keeps_raw_f16(
        "model.language_model.layers.3.ple.ple_embedding.ngram_embedding.head_bias"
    ));
    // Ordinary F16 weights and lookalikes must keep the BF16 conversion.
    assert!(!keeps_raw_f16("model.norm.weight"));
    assert!(!keeps_raw_f16("model.layers.0.linear_attn.A_log"));
    assert!(!keeps_raw_f16("x.suh.weight"));
    assert!(!keeps_raw_f16("x.suhx"));
    // A bare component without the prefix dot is not an EXL3 scale name.
    assert!(!keeps_raw_f16("head_bias"));
}

#[test]
fn is_ngram_table_matches_both_exl3_layouts() {
    let base = "model.language_model.layers.3.ple.ple_embedding.ngram_embedding";
    // Layout 1: ONE I16 tensor.
    assert!(is_ngram_table(&format!("{base}.trellis")));
    // Layout 2: 128 I16 shards.
    assert!(is_ngram_table(&format!("{base}.shard_0.trellis")));
    assert!(is_ngram_table(&format!("{base}.shard_127.trellis")));

    // Small aux tensors stay resident (read on host).
    assert!(!is_ngram_table(&format!("{base}.head_bias")));
    assert!(!is_ngram_table(&format!("{base}.head_offsets")));
    assert!(!is_ngram_table(&format!("{base}.layer_multipliers")));
    // A per-projection trellis is an ordinary quantized weight, not a table.
    assert!(!is_ngram_table("model.layers.0.self_attn.q_proj.trellis"));
    // Malformed shard tails must not match (fail loudly, not silently).
    assert!(!is_ngram_table(&format!("{base}.shard_.trellis")));
    assert!(!is_ngram_table(&format!("{base}.shard_1a.trellis")));
}
