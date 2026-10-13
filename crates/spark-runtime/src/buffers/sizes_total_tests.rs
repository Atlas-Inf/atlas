// SPDX-License-Identifier: AGPL-3.0-only

//! Reserve accounting must equal actual arena allocations, including placeholders.
use super::*;
use crate::buffers::BufferArena;
use crate::gpu::mock::MockGpuBackend;
use atlas_core::scope::ModelResource;

fn assert_actual_arena_total(config: &ModelConfig, rows: usize) {
    let gpu = MockGpuBackend::new();
    let mut arena = BufferArena::new(config, rows, 256, 16, 4, &gpu).unwrap();
    let pointers = [
        arena.hidden_states,
        arena.residual,
        arena.norm_output,
        arena.qkv_output,
        arena.attn_output,
        arena.gate_logits,
        arena.gate_logits_f32,
        arena.moe_router_in_f32,
        arena.moe_output,
        arena.logits,
        arena.ssm_qkvz,
        arena.ssm_ba,
        arena.ssm_deinterleaved,
        arena.ssm_gates,
        arena.ssm_conv_out_f32,
        arena.scratch,
        arena.expert_gate_out,
        arena.expert_up_out,
        arena.hc_lowrank_scratch,
        arena.qsa_select_scratch,
        arena.expert_down_out,
        arena.splitk_workspace,
        arena.o_latent,
        arena.norm_unit_w,
        arena.hc_streams,
        arena.hc_post,
        arena.hc_comb,
        arena.gdn_fla_scratch,
        arena.ssd_scratch,
        arena.token_ids,
        arena.ffn_act_q8,
        arena.ffn_act_a,
        arena.ffn_act_scale,
        arena.fp8_act,
        arena.fp8_act_scale,
        arena.lora_xa,
        arena.lora_delta,
        arena.lora_hact,
        arena.lora_seq_slot,
        arena.q2_dequant_scratch,
        arena.q2_act_q8,
    ];
    let live: std::collections::HashSet<_> = pointers
        .into_iter()
        .filter(|p| !p.is_null())
        .map(|p| p.0)
        .collect();
    assert_eq!(
        live.len(),
        gpu.alloc_count(),
        "fixture must cover every actual allocation"
    );
    let actual: usize = live
        .into_iter()
        .map(|p| {
            gpu.read_alloc(crate::gpu::DevicePtr(p))
                .expect("arena owns this allocation")
                .len()
        })
        .sum();
    assert_eq!(
        arena.sizes().total_bytes(),
        actual,
        "reserve total must include o_latent={} and norm_unit_w={}",
        arena.sizes().o_latent,
        arena.sizes().norm_unit_w
    );
    arena.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn buffer_total_matches_actual_mock_allocations_with_placeholder_o_latent() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(
        BufferSizes::from_config(&config, 1, 256, 16, 4).o_latent,
        256
    );
    assert_actual_arena_total(&config, 1);
}

#[test]
fn buffer_total_matches_actual_mock_allocations_with_grouped_o_and_adapters() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.o_groups = 4;
    config.o_lora_rank = 128;
    config.adapter_max_rank = 8;
    config.mlp_only_layers = vec![0];
    assert_eq!(
        BufferSizes::from_config(&config, 5, 256, 16, 4).o_latent,
        5120
    );
    assert_actual_arena_total(&config, 5);
}
