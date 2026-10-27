// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn native() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen4_exp".into();
    c.max_position_embeddings = NATIVE_CONTEXT;
    c.head_dim = 256;
    c.partial_rotary_factor = 0.25;
    c.rope_theta = 10_000_000.0;
    c
}

#[test]
fn extended_context_requires_explicit_scaling() {
    assert!(configure_context(&mut native(), None, 262144).is_ok());
    assert!(configure_context(&mut native(), None, 262145).is_err());
    assert!(configure_context(&mut native(), Some(2), 524288).is_ok());
    assert!(configure_context(&mut native(), Some(2), 524289).is_err());
    assert!(configure_context(&mut native(), Some(4), 1000000).is_ok());
    for factor in [0, 1, 3, 255] {
        assert!(configure_context(&mut native(), Some(factor), 32768).is_err());
    }
    assert!(configure_context(&mut native(), Some(2), 0).is_err());
    let mut other = native();
    other.model_type = "glm5_next".into();
    assert!(configure_context(&mut other, None, 1048576).is_ok());
    assert!(configure_context(&mut other, Some(4), 1048576).is_err());
}

#[test]
fn native_config_is_unchanged_and_conflicting_scaling_is_rejected() {
    let mut c = native();
    let before = format!("{c:?}");
    configure_context(&mut c, None, 32768).unwrap();
    assert_eq!(format!("{c:?}"), before);
    configure_context(&mut c, Some(2), 524288).unwrap();
    let scaled = format!("{c:?}");
    configure_context(&mut c, Some(2), 524288).unwrap();
    assert_eq!(format!("{c:?}"), scaled);
    assert!(configure_context(&mut c, Some(4), 1000000).is_err());
}

#[test]
fn frequency_ramp_preserves_local_resolution_and_stretches_low_frequencies() {
    for factor in [2, 4] {
        let mut c = native();
        configure_context(&mut c, Some(factor), 524288).unwrap();
        let frequencies = inverse_frequencies(&c).unwrap();
        assert_eq!(frequencies.len(), 32);
        assert_eq!(frequencies[0], 1.0);
        // For dim=64, theta=1e7, L=262144: correction floor/ceil are 14/22.
        // Test all bands against this independently calculated interval.
        for (j, &actual) in frequencies.iter().enumerate() {
            let base = 10_000_000f64.powf(-(j as f64) / 32.0);
            let fraction = ((j as f64 - 14.0) / 8.0).clamp(0.0, 1.0);
            let expected = base * (1.0 - fraction + fraction / f64::from(factor));
            assert!(
                (f64::from(actual) - expected).abs() <= expected * 1e-6,
                "pair {j}"
            );
        }
        assert!(frequencies.windows(2).all(|w| w[0] > w[1]));
    }
}

#[test]
fn rank_agreement_uses_effective_scaling() {
    let mut c = native();
    assert!(rank_settings(&c).iter().all(|(_, v)| *v == 0));
    configure_context(&mut c, Some(2), 524288).unwrap();
    let baseline = rank_settings(&c);
    c.yarn_attention_factor += 0.01;
    assert_ne!(baseline, rank_settings(&c));
    c.yarn_factor = 4.0;
    assert_ne!(baseline[0], rank_settings(&c)[0]);
}

#[test]
fn frequency_table_is_within_two_ulps_of_float32_reference() {
    // Float32 translation of Hugging Face modeling_rope_utils.compute_yarn_parameters:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
    // theta**(arange(0,64,2)/64), inverse native/interpolated frequencies,
    // then the [14,22] ramp blend, with every intermediate rounded to F32.
    // Rust uses F64 intermediates; this checks numerical proximity, not bit equality.
    let tables = [
        "3f800000 3f1ab32b 3ebaf81b 3e61f835 3e088d77 3da50956 3d47763f 3cf11177
         3c91ad39 3c301052 3bd4ca15 3b80967d 3b1b690d 3abbd3ed 3a6301e2 3a009b22
         399111f4 3922ce9d 38b5a1aa 384939ae 37dd1727 37707cc9 37012dab 369c1fc4
         363cb0c1 35e40cc6 3589cf4a 35268e4c 34c94c57 3473499d 3413048e 33b1af44",
        "3f800000 3f1ab32b 3ebaf81b 3e61f835 3e088d77 3da50956 3d47763f 3cf11177
         3c91ad39 3c301052 3bd4ca15 3b80967d 3b1b690d 3abbd3ed 3a6301e2 39f8a364
         3986b53e 3910058b 38975c0d 381b7e06 379ac368 3712f6ed 36812dab 361c1fc4
         35bcb0c1 35640cc6 3509cf4a 34a68e4c 34494c57 33f3499d 3393048e 3331af44",
    ];
    for (factor, table) in [2, 4].into_iter().zip(tables) {
        let mut c = native();
        configure_context(&mut c, Some(factor), 524288).unwrap();
        let expected: Vec<_> = table
            .split_whitespace()
            .map(|bits| u32::from_str_radix(bits, 16).unwrap())
            .collect();
        let actual = inverse_frequencies(&c).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (j, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.to_bits().abs_diff(expected) <= 2,
                "factor {factor}, pair {j}"
            );
        }
    }
}
