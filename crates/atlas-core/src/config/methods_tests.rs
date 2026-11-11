// SPDX-License-Identifier: AGPL-3.0-only

use super::ModelConfig;

#[test]
fn any_compressed_deepseek_v4_layer_is_not_kv_cache_complete() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".to_string();

    for ratios in [vec![4, 0, 0], vec![0, 4, 0], vec![0, 0, 128]] {
        config.compress_ratios = ratios;
        assert!(!config.kv_only_prefix_cache_is_safe());
    }
}

#[test]
fn kv_complete_models_can_use_the_prefix_cache() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(config.kv_only_prefix_cache_is_safe());

    config.model_type = "deepseek_v4".to_string();
    config.compress_ratios = vec![0; 3];
    assert!(config.kv_only_prefix_cache_is_safe());
}

#[test]
fn final_norm_identity_only_for_qwen4_exp() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(!config.final_norm_is_identity());

    config.model_type = "qwen4_exp".to_string();
    assert!(config.final_norm_is_identity());

    config.model_type = "deepseek_v4".to_string();
    assert!(!config.final_norm_is_identity());
}

/// A TP=EP=2 rank's view restores the full head counts and drops the
/// split; a single-rank config is its own view.
#[test]
fn tp1_view_undoes_the_tp_head_split() {
    let full = ModelConfig::qwen3_next_80b_nvfp4();
    let mut rank1 = full.clone();
    rank1.num_attention_heads /= 2;
    rank1.num_key_value_heads /= 2;
    rank1.linear_num_key_heads /= 2;
    rank1.linear_num_value_heads /= 2;
    (rank1.tp_rank, rank1.tp_world_size) = (1, 2);
    (rank1.ep_rank, rank1.ep_world_size) = (1, 2);
    let view = rank1.tp1_view();
    let heads = |c: &ModelConfig| {
        (
            c.num_attention_heads,
            c.num_key_value_heads,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
        )
    };
    assert_eq!(heads(&view), heads(&full));
    assert_eq!((view.tp_rank, view.tp_world_size), (0, 1));
    assert_eq!((view.ep_rank, view.ep_world_size), (0, 1));
    assert_eq!(view.local_expert_range(), (0, full.num_experts));

    for world in [0, 1] {
        let mut single = full.clone();
        (single.tp_world_size, single.ep_world_size) = (world, world);
        let view = single.tp1_view();
        assert_eq!(heads(&view), heads(&single));
        assert_eq!((view.tp_world_size, view.ep_world_size), (world, world));
    }
}

#[test]
fn nemotron_mamba2_ssm_bytes_are_not_gdn_zero() {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.mamba_num_heads = 64;
    c.mamba_head_dim = 64;
    c.ssm_state_size = 128;
    c.n_groups = 8;
    c.linear_conv_kernel_dim = 4;
    c.linear_num_value_heads = 0;
    c.linear_value_head_dim = 0;
    c.linear_num_key_heads = 0;
    c.linear_key_head_dim = 0;
    assert_eq!(c.ssm_h_state_bytes(), 64 * 64 * 128 * 4);
    assert_eq!(c.ssm_conv_state_bytes(), (64 * 64 + 2 * 8 * 128) * 4 * 4);
    assert_eq!(
        c.linear_num_value_heads * c.linear_value_head_dim * c.linear_key_head_dim * 4,
        0
    );
}
