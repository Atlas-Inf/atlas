// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM-5.3-Flash pair (TP2/EP2) arena: the three members sized for what
//! its passes read instead of an FP32 width, and the same members of every
//! other shape, unchanged.
use super::*;
use crate::buffers::BufferArena;
use crate::gpu::GpuBackend;
use crate::gpu::mock::MockGpuBackend;

/// One rank's config of the serving pair, as the launcher's topology step
/// leaves it (TP-local head counts, BF16 highway).
fn glm_pair() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.intermediate_size = 12288;
    c.vocab_size = 154_880;
    c.attn_gated = false;
    c.num_attention_heads = 32;
    c.num_key_value_heads = 32;
    c.head_dim = 256;
    c.kv_lora_rank = 512;
    c.q_lora_rank = 1536;
    c.qk_rope_head_dim = 0;
    c.linear_num_key_heads = 32;
    c.linear_num_value_heads = 32;
    c.linear_key_head_dim = 128;
    c.linear_value_head_dim = 128;
    c.num_experts = 288;
    c.num_experts_per_tok = 8;
    c.moe_intermediate_size = 2048;
    c.shared_expert_intermediate_size = 2048;
    c.use_routing_bias = true;
    c.router_logits_f32 = true;
    c.mlp_only_layers = vec![0, 1, 2];
    c.hc_mult = 4;
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.expert_tp = true;
    c.hc_highway_bf16 = true;
    c
}

/// The serving arena: 1M-token context, 16-token blocks, four sequences.
fn sized(c: &ModelConfig, rows: usize) -> BufferSizes {
    BufferSizes::from_config(c, rows, 1_048_576, 16, 4)
}

/// Sizing reads these; a run that sets one sizes another arena.
fn env_is_default() -> bool {
    [
        "ATLAS_FP32_ROUTING",
        "ATLAS_FFN_MMQ",
        "ATLAS_INT8_PREFILL",
        "ATLAS_FP4_PREFILL",
        "ATLAS_GGUF_NATIVE_Q2",
        "ATLAS_GGUF_NATIVE_Q2_MMQ",
    ]
    .iter()
    .all(|name| std::env::var_os(name).is_none())
}

#[test]
fn glm_pair_members_hold_what_its_passes_read() {
    if !env_is_default() {
        return;
    }
    let c = glm_pair();
    for rows in [8196usize, 16384] {
        let s = sized(&c, rows);
        // BF16 highway: [rows, 4 streams, 4096].
        assert_eq!(s.hc_streams, rows * 4 * 4096 * 2);
        // Worklist scratch: a count and one item per row, route and tile.
        assert_eq!(s.moe_router_in_f32, 16 + rows * 8 * 16 * 8);
        assert_eq!(
            s.moe_router_in_f32,
            router_worklist_bytes(rows, 8, 2048),
            "every worklist of at most `rows` rows fits"
        );
        // KDA convolution: [rows, 3 * 32 heads * 128] BF16, and no less than
        // the FP32 width over one attention piece.
        let kda = rows * 3 * 4096 * 2;
        let piece = GLM_ATTENTION_PIECE_ROWS * 4 * 4096 * 4;
        assert_eq!(s.ssm_conv_out_f32, kda.max(piece));
    }
    assert_eq!(sized(&c, 8196).ssm_conv_out_f32, 4100 * 65536);
    assert_eq!(sized(&c, 16384).ssm_conv_out_f32, 16384 * 24576);
}

/// What a row of the pair's arena costs past the piece floor, and the two
/// arenas a 16K-row chunk is weighed against. Before these members were
/// trimmed a row cost 479,828 bytes (89,088 more) and the 8196-row arena
/// 4,007,445,456 bytes, the 3821.8 MiB the server logged.
#[test]
fn glm_pair_arena_bytes_a_row_and_in_all() {
    if !env_is_default() {
        return;
    }
    let c = glm_pair();
    let rows_only = |rows: usize| {
        let s = sized(&c, rows);
        s.total_bytes() - s.scratch
    };
    assert_eq!((rows_only(16384) - rows_only(12288)) / 4096, PER_ROW);
    assert_eq!(sized(&c, 8196).total_bytes(), ARENA_8196);
    assert_eq!(sized(&c, 16384).total_bytes(), ARENA_16384);
}

/// Bytes a row, without the metadata scratch (76 more).
const PER_ROW: usize = 390_740;
/// 3189.6 MiB.
const ARENA_8196: usize = 3_344_557_008;
/// 6176.6 MiB.
const ARENA_16384: usize = 6_476_628_096;

/// Each trim is tied to the property that makes the bytes dead; without it
/// the member keeps the size it always had.
#[test]
fn each_trim_needs_its_own_condition() {
    if !env_is_default() {
        return;
    }
    let rows = 16384;
    let full = |c: &ModelConfig| sized(c, rows);

    // An FP32 highway (every other model, and GLM without the switch).
    let mut c = glm_pair();
    c.hc_highway_bf16 = false;
    assert_eq!(full(&c).hc_streams, rows * 4 * 4096 * 4);

    // A router without a correction bias may read an FP32 input here.
    let mut c = glm_pair();
    c.use_routing_bias = false;
    assert_eq!(full(&c).moe_router_in_f32, rows * 4096 * 4);
    // Per-layer expert shapes: the worklist bound is stated for one shape.
    let mut c = glm_pair();
    c.moe_intermediate_sizes = vec![2048];
    assert_eq!(full(&c).moe_router_in_f32, rows * 4096 * 4);
    let mut c = glm_pair();
    c.num_experts_per_toks = vec![8];
    assert_eq!(full(&c).moe_router_in_f32, rows * 4096 * 4);

    // GDN (not KDA) convolves FP32 q|k|v|z; an MLA with a RoPE partition or
    // a low-rank O projection also scales this buffer with the rows.
    let fp32_conv = rows * 4 * 4096 * 4;
    let mut c = glm_pair();
    c.model_type = "qwen3_next".into();
    assert_eq!(full(&c).ssm_conv_out_f32, fp32_conv);
    let mut c = glm_pair();
    c.qk_rope_head_dim = 64;
    assert_eq!(full(&c).ssm_conv_out_f32, fp32_conv);
    let mut c = glm_pair();
    c.o_lora_rank = 1024;
    assert_eq!(full(&c).ssm_conv_out_f32, fp32_conv);
}

#[test]
fn other_models_keep_their_sizes() {
    if !env_is_default() {
        return;
    }
    let c = ModelConfig::qwen3_next_80b_nvfp4();
    for rows in [1usize, 128, 8192, 16384] {
        let s = BufferSizes::from_config(&c, rows, 4096, 16, 32);
        assert_eq!(s.moe_router_in_f32, rows * c.hidden_size * 4);
        assert_eq!(s.ssm_conv_out_f32, (rows * c.ssm_qkvz_size() * 4).max(256));
        assert_eq!(s.hc_streams, 256);
    }
    // A one-row correction-bias arena keeps its 16 KiB of router scratch.
    assert_eq!(sized(&glm_pair(), 1).moe_router_in_f32, 4096 * 4);
}

/// `zero_dirty` shares the conv scratch out by the rows it holds at the FP32
/// row a pass may reach, so a pass of `n` rows has `n` such rows zeroed, as
/// before the trim; an untrimmed buffer holds the arena's rows.
#[test]
fn the_conv_scratch_is_zeroed_by_its_wide_rows() {
    if !env_is_default() {
        return;
    }
    let c = glm_pair();
    for (rows, wide) in [
        (64usize, 64usize),
        (4100, 4100),
        (8196, 4100),
        (16384, 6144),
    ] {
        assert_eq!(
            sized(&c, rows).conv_wide_rows(&c, rows),
            wide,
            "{rows} rows"
        );
    }
    let c = ModelConfig::qwen3_next_80b_nvfp4();
    for rows in [1usize, 128, 16384] {
        let s = BufferSizes::from_config(&c, rows, 4096, 16, 32);
        assert_eq!(s.conv_wide_rows(&c, rows), rows);
    }
}

/// The mHC TF32 prewarm reads an FP32 activation of every arena row and
/// keeps its scratch in `gate_logits_f32`; over a BF16 highway it borrows the
/// activation from the span past the never-zeroed members (`lend_tail`).
#[test]
fn a_bf16_highway_leaves_the_prewarm_an_fp32_activation_to_borrow() {
    if !env_is_default() {
        return;
    }
    let c = glm_pair();
    for rows in [4100usize, 8196, 16384] {
        let s = sized(&c, rows);
        let activation = rows * 4 * 4096 * 4;
        assert!(s.hc_streams < activation, "the highway no longer holds it");
        let lent = s.lent_sizes();
        let span = crate::buffers::lend::layout(&lent).1;
        let prefix = crate::buffers::lend::layout(&lent[..crate::buffers::lend::UNZEROED]).1;
        assert!(span - prefix >= activation, "{rows} rows");
        // Its scratch: the [24, 16384] FP32 mix weight and [rows, 24] output.
        assert!(
            s.gate_logits_f32 >= 24 * 16384 * 4 + rows * 24 * 4,
            "{rows} rows"
        );
    }
}

/// The arena allocates what the sizes say, and lends its span past the
/// members it never zeroes: the borrow starts there, cannot exceed what is
/// left, and comes back zeroed.
#[test]
fn glm_pair_arena_allocates_its_sizes_and_lends_its_tail() {
    if !env_is_default() {
        return;
    }
    let gpu = MockGpuBackend::new();
    let arena = BufferArena::new(&glm_pair(), 64, 4096, 16, 4, &gpu).unwrap();
    let s = arena.sizes();
    let len = |ptr| gpu.read_alloc(ptr).expect("own allocation").len();
    assert_eq!(len(arena.hc_streams()), s.hc_streams);
    assert_eq!(s.hc_streams, 64 * 4 * 4096 * 2);

    let prefix = arena.lend_unzeroed;
    let tail = arena.lendable_bytes() - prefix;
    assert_eq!(arena.qkv_output().0, arena.lend_base().0 + prefix as u64);
    assert!(arena.gate_logits_f32().0 + s.gate_logits_f32 as u64 <= arena.qkv_output().0);
    assert!(arena.lend_tail(tail + 1, &gpu).is_err());
    let lent = arena.lend_tail(tail, &gpu).unwrap();
    assert_eq!(lent, arena.qkv_output());
    arena.zero_all(&gpu, 0).unwrap();
    gpu.memset(lent, 0x5A, tail).unwrap();
    arena.return_lent_tail(tail, &gpu, 0).unwrap();
    let mut host = vec![0xFFu8; tail];
    gpu.copy_d2h(lent, &mut host).unwrap();
    assert!(host.iter().all(|&b| b == 0));
}
