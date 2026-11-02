// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn lookup<'a>(flags: &'a [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |key| {
        flags
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn modes_reject_unknown_values() {
    assert_eq!(ServingMode::parse(None).unwrap(), None);
    assert_eq!(
        ServingMode::parse(Some("strict")).unwrap(),
        Some(ServingMode::Strict)
    );
    assert_eq!(
        ServingMode::parse(Some("spark")).unwrap(),
        Some(ServingMode::Spark)
    );
    for value in ["", "fast", "STRICT"] {
        assert!(ServingMode::parse(Some(value)).is_err());
    }
}

fn flags(tables: &[&[(&'static str, &'static str)]]) -> Vec<(&'static str, &'static str)> {
    tables.concat()
}

const STRICT: &[&[(&str, &str)]] = &[NATIVE_FLAGS, STRICT_NATIVE_FLAGS, STRICT_FLAGS];
const SPARK: &[&[(&str, &str)]] = &[NATIVE_FLAGS, SPARK_FLAGS];

/// Every listed flag is required at exactly its value.
fn requires_each_exactly(
    mode: ServingMode,
    native_mtp: bool,
    all: &[(&'static str, &'static str)],
) {
    mode.validate_flags(native_mtp, lookup(all)).unwrap();
    for &(changed, _) in all {
        let missing: Vec<_> = all.iter().copied().filter(|(k, _)| *k != changed).collect();
        assert!(
            mode.validate_flags(native_mtp, lookup(&missing)).is_err(),
            "{changed}"
        );
        let wrong: Vec<_> = all
            .iter()
            .map(|&(k, v)| (k, if k == changed { "invalid" } else { v }))
            .collect();
        assert!(
            mode.validate_flags(native_mtp, lookup(&wrong)).is_err(),
            "{changed}"
        );
    }
}

#[test]
fn strict_requires_every_drafter_and_target_flag_exactly() {
    requires_each_exactly(ServingMode::Strict, false, &flags(STRICT));
}

#[test]
fn spark_requires_the_full_shared_output_and_the_nvfp4_drafter() {
    let spark = ServingMode::Spark;
    requires_each_exactly(spark, false, &flags(SPARK));
    let mut bf16_drafter = flags(SPARK);
    bf16_drafter.push(("ATLAS_NO_DFLASH_DRAFTER_NVFP4", "1"));
    assert!(spark.validate_flags(false, lookup(&bf16_drafter)).is_err());
}

#[test]
fn both_modes_serve_the_fp4_local_native_formats() {
    // SparkGLM 0.2.0 serve.py: one native target, two decode policies.
    for (key, value) in [
        ("ATLAS_GLM_FP4L_NATIVE", "1"),
        ("ATLAS_GLM_FP4L_MLA_FP8", "1"),
        ("ATLAS_GLM_FP4L_DENSE_MXFP8", "1"),
        ("ATLAS_GLM_FP4L_KDA_W8A8_PREFILL", "1"),
        ("ATLAS_GLM_FP4L_INPUT_SCALES", "1"),
        ("ATLAS_GLM_FP4L_FP8_PREFILL", "w8a8"),
        ("ATLAS_GLM_LM_HEAD_MXFP8", "1"),
        ("ATLAS_GLM_FP4L_KDA_TILED_TWIN", "0"),
    ] {
        for all in [flags(STRICT), flags(SPARK)] {
            assert!(all.contains(&(key, value)), "{key}");
        }
    }
    let strict = flags(STRICT);
    let spark = flags(SPARK);
    for key in [
        "ATLAS_GLM_FP4L_KDA_MXFP8_DECODE",
        "ATLAS_GLM_FP4L_DECODE_A16",
    ] {
        assert!(strict.contains(&(key, "1")), "{key}");
        assert!(spark.contains(&(key, "0")), "{key}");
    }
    assert!(strict.contains(&("ATLAS_GLM_FP4L_SHARED_FP8", "1")));
}

#[test]
fn spark_takes_the_shared_expert_format_from_the_profile() {
    for value in ["0", "1"] {
        let mut all = flags(SPARK);
        all.push(("ATLAS_GLM_FP4L_SHARED_FP8", value));
        ServingMode::Spark
            .validate_flags(false, lookup(&all))
            .unwrap();
    }
}

#[test]
fn native_mtp_is_strict_only_and_replaces_the_drafter_policy() {
    requires_each_exactly(
        ServingMode::Strict,
        true,
        &flags(&[NATIVE_FLAGS, STRICT_NATIVE_FLAGS]),
    );
    assert!(
        ServingMode::Spark
            .validate_flags(true, lookup(&flags(SPARK)))
            .is_err()
    );
}

#[test]
fn strict_is_serial_target_decoding() {
    assert!(ServingMode::Strict.validate_speculation(true).is_err());
    ServingMode::Strict.validate_speculation(false).unwrap();
    ServingMode::Spark.validate_speculation(true).unwrap();
    assert!(strict_serial_for(Some(ServingMode::Strict), false));
    assert!(!strict_serial_for(Some(ServingMode::Strict), true));
    assert!(!strict_serial_for(Some(ServingMode::Spark), false));
    assert!(!strict_serial_for(None, false));
}

#[test]
fn modes_admit_pair_and_four_rank_targets_only() {
    assert!(supported_parallelism(2, 2, 2));
    assert!(supported_parallelism(4, 4, 4));
    for (world, tp, ep) in [
        (0, 0, 0),
        (1, 1, 1),
        (3, 3, 3),
        (8, 8, 8),
        (2, 1, 2),
        (4, 2, 4),
        (4, 4, 2),
        (2, 4, 4),
    ] {
        assert!(!supported_parallelism(world, tp, ep), "{world}/{tp}/{ep}");
    }
}

#[test]
fn drafter_copies_take_the_rne_weight_converter_only_under_a_mode() {
    assert_eq!(weight_quantizer_for(None), "quantize_bf16_to_nvfp4");
    for mode in [ServingMode::Strict, ServingMode::Spark] {
        assert_eq!(
            weight_quantizer_for(Some(mode)),
            "quantize_bf16_to_nvfp4_weight_rne"
        );
    }
}

#[test]
fn a_mode_refuses_a_fallback_conversion() {
    for mode in [ServingMode::Strict, ServingMode::Spark] {
        assert!(require_format_for(Some(mode), true, "weights").is_ok());
        let err = require_format_for(Some(mode), false, "x MXFP8 weights").unwrap_err();
        assert!(err.to_string().contains("x MXFP8 weights"), "{err}");
    }
    assert!(require_format_for(None, false, "weights").is_ok());
}

#[test]
fn missing_local_scales_cannot_masquerade_as_remote_experts() {
    assert!(validate_calibration(true, [true; 3], [true; 3]).is_ok());
    assert!(validate_calibration(true, [false; 3], [false; 3]).is_ok());
    assert!(validate_calibration(true, [true; 3], [false; 3]).is_err());
    assert!(validate_calibration(true, [true; 3], [true, true, false]).is_err());
    assert!(validate_calibration(true, [true, false, true], [true; 3]).is_err());
    assert!(validate_calibration(false, [true; 3], [false; 3]).is_ok());
}
