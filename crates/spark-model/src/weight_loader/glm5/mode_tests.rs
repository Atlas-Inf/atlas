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

#[test]
fn strict_requires_every_drafter_and_target_flag_exactly() {
    let strict = ServingMode::Strict;
    strict.validate_flags(false, lookup(STRICT_FLAGS)).unwrap();
    for &(changed, _) in STRICT_FLAGS {
        let missing: Vec<_> = STRICT_FLAGS
            .iter()
            .copied()
            .filter(|(k, _)| *k != changed)
            .collect();
        assert!(strict.validate_flags(false, lookup(&missing)).is_err());
        let wrong: Vec<_> = STRICT_FLAGS
            .iter()
            .map(|&(k, v)| (k, if k == changed { "invalid" } else { v }))
            .collect();
        assert!(strict.validate_flags(false, lookup(&wrong)).is_err());
    }
}

#[test]
fn spark_requires_the_full_shared_output_and_the_nvfp4_drafter() {
    let spark = ServingMode::Spark;
    spark.validate_flags(false, lookup(SPARK_FLAGS)).unwrap();
    assert!(spark.validate_flags(false, lookup(&[])).is_err());
    assert!(
        spark
            .validate_flags(false, lookup(&[("ATLAS_GLM_SHARED_TP_SPLIT", "1")]))
            .is_err()
    );
    let bf16_drafter = [
        ("ATLAS_GLM_SHARED_TP_SPLIT", "0"),
        ("ATLAS_NO_DFLASH_DRAFTER_NVFP4", "1"),
    ];
    assert!(spark.validate_flags(false, lookup(&bf16_drafter)).is_err());
}

#[test]
fn native_mtp_is_strict_only_and_replaces_the_drafter_policy() {
    ServingMode::Strict
        .validate_flags(true, lookup(&[]))
        .unwrap();
    assert!(
        ServingMode::Spark
            .validate_flags(true, lookup(SPARK_FLAGS))
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
