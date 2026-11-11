// SPDX-License-Identifier: AGPL-3.0-only
//! Mode/topology errors must fail before the backend allocates GPU memory.
use super::*;
use spark_model::weight_loader::qwen4exp_mode::{REQUIRED_FLAGS, ServingMode};

fn qwen_config() -> ModelConfig {
    atlas_core::config::parse_config(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen4_exp_flash_next_fp4local_config.json"
    )))
    .unwrap()
}

fn check_mode(name: &str, mode: ServingMode) {
    let mut flags = REQUIRED_FLAGS.to_vec();
    flags.extend(mode.drafter_flags());
    flags.push((
        "ATLAS_QWEN4EXP_SERVING_MODE",
        match mode {
            ServingMode::Strict => "strict",
            ServingMode::Spark => "spark",
        },
    ));
    if isolated(&format!("qwen_modes::{name}"), &flags) {
        return;
    }
    for rank in 0..2 {
        let mut a = args(4, rank);
        a.speculative = true;
        let mut cfg = qwen_config();
        let heads = cfg.num_attention_heads;
        let topology = super::super::prepare_topology(&a, &mut cfg)
            .unwrap()
            .unwrap();
        assert_eq!(topology.tp_size, 2);
        assert_eq!(topology.ep_size, 2);
        assert_eq!(cfg.tp_rank, rank);
        assert_eq!(cfg.num_attention_heads, heads / 2);
        for bad in 0..3 {
            let mut invalid = a.clone();
            match bad {
                0 => invalid.no_fast_load = true,
                1 => invalid.speculative = false,
                _ => invalid.dflash = true,
            }
            let mut cfg = qwen_config();
            let err =
                super::super::prepare_reserve(&invalid, &mut cfg, || -> Result<((), usize)> {
                    panic!("invalid Qwen mode reached backend initialization")
                })
                .err()
                .expect("invalid mode must fail");
            assert!(
                err.to_string().to_ascii_lowercase().contains("qwen"),
                "{err:#}"
            );
        }
    }
}

#[test]
fn strict_preflight_retains_mtp_and_resolves_tp_before_allocation() {
    check_mode(
        "strict_preflight_retains_mtp_and_resolves_tp_before_allocation",
        ServingMode::Strict,
    );
}

#[test]
fn spark_preflight_retains_mtp_and_resolves_tp_before_allocation() {
    check_mode(
        "spark_preflight_retains_mtp_and_resolves_tp_before_allocation",
        ServingMode::Spark,
    );
}
