// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferSizes;

fn config() -> ModelConfig {
    ModelConfig {
        model_type: "glm5_next".into(),
        hidden_size: 4096,
        moe_intermediate_size: 2048,
        shared_expert_intermediate_size: 2048,
        num_experts: 288,
        num_experts_per_tok: 8,
        tp_world_size: 2,
        ep_world_size: 2,
        scoring_func: "sigmoid".into(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}

#[test]
fn four_row_arenas_cover_grouped_and_scalar_scratch_lifetimes() {
    let config = config();
    let sizes = BufferSizes::from_config(&config, 4, 2048, 16, 4);
    validate_c4_moe_arenas(&config, &sizes).unwrap();
    let requirements = c4_moe_arenas(&config, &sizes).unwrap();
    let required = |name| requirements.iter().find(|x| x.0 == name).unwrap().2;
    assert_eq!(required("routed gate"), 131072);
    assert_eq!(required("routed up"), 131072);
    assert_eq!(required("routed down / FP4 staging"), 262144);
    assert_eq!(required("compact worklist"), 4112);
    assert_eq!(required("router / sort metadata"), 2304);
    assert_eq!(required("shared gate"), 16384);
    assert_eq!(required("shared down"), 32768);
    assert_eq!(required("output"), 32768);
}

#[test]
fn every_borrowed_arena_is_checked_independently() {
    let config = config();
    let changes: [fn(&mut BufferSizes); 12] = [
        |s| s.norm_output = 32767,
        |s| s.moe_output = 32767,
        |s| s.gate_logits = 2303,
        |s| s.moe_router_in_f32 = 4111,
        |s| s.expert_gate_out = 131071,
        |s| s.expert_up_out = 131071,
        |s| s.expert_down_out = 262143,
        |s| s.ssm_deinterleaved = 16383,
        |s| s.ssm_qkvz = 16383,
        |s| s.attn_output = 32767,
        |s| s.logits = 4095,
        |s| s.scratch = 255,
    ];
    for change in changes {
        let mut sizes = BufferSizes::from_config(&config, 4, 2048, 16, 4);
        change(&mut sizes);
        assert!(validate_c4_moe_arenas(&config, &sizes).is_err());
    }
}

#[test]
fn arena_arithmetic_rejects_overflow() {
    let mut config = config();
    let sizes = BufferSizes::from_config(&config, 4, 2048, 16, 4);
    config.hidden_size = usize::MAX;
    assert!(c4_moe_arenas(&config, &sizes).is_err());
}

#[test]
fn reverse_scalar_copy_keeps_every_finished_row_live() {
    let input = [11u32, 22, 33, 44];
    let original = input;
    let mut output = [0; 4];
    for row in c4_scalar_rows() {
        // The scalar forward and its collective write only row zero.
        output[0] = input[row] + 100;
        if row > 0 {
            output[row] = output[0];
        }
    }
    assert_eq!(input, original);
    assert_eq!(output, [111, 122, 133, 144]);
}

#[test]
fn compact_grid_covers_concentrated_distinct_and_remote_routes() {
    let bound = super::super::prequant_fp4::compact_gate_up_worklist_bytes(4, 8, 2048);
    for (loads, local) in [
        (vec![32usize], vec![true]),
        (vec![1; 32], vec![true; 32]),
        (vec![1; 32], vec![false; 32]),
        (vec![16, 16], vec![true, false]),
    ] {
        let tiles = loads
            .iter()
            .zip(local)
            .filter(|(_, yes)| *yes)
            .map(|(&count, _)| count.div_ceil(64) * 16)
            .sum::<usize>();
        assert!(16 + tiles * 8 <= bound);
    }
}
