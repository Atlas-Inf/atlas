// SPDX-License-Identifier: AGPL-3.0-only

use super::LIGHTNING_SERVED_GAMMA;
use super::batch_execution::{paged_slot_mapping, resolve_lane_id};
use super::batch_inputs::DsparkBatchInputError;
use super::batch_inputs_tests::valid;

#[test]
fn packed_queries_are_sequence_major_anchor_then_masks() {
    for batch in [1, 2, 4, 8] {
        let input = valid(batch);
        let packed = input.packed_query_tokens(990);
        assert_eq!(packed.len(), batch * LIGHTNING_SERVED_GAMMA);
        for sequence in 0..batch {
            let range = input.sequence_row_range(sequence).unwrap();
            assert_eq!(packed[range.start], 100 + sequence as u32);
            assert!(packed[range.start + 1..range.end].iter().all(|&t| t == 990));
        }
    }
}

#[test]
fn packed_positions_preserve_each_sequence_absolute_base() {
    let input = valid(2);
    assert_eq!(
        input.packed_positions().unwrap(),
        vec![1000, 1001, 1002, 1003, 1017, 1018, 1019, 1020]
    );
}

#[test]
fn markov_depth_rows_are_batch_wide_and_sequence_major() {
    let input = valid(4);
    assert_eq!(input.rows_at_query(0).unwrap(), vec![0, 4, 8, 12]);
    assert_eq!(input.rows_at_query(1).unwrap(), vec![1, 5, 9, 13]);
    assert_eq!(input.rows_at_query(3).unwrap(), vec![3, 7, 11, 15]);
}

#[test]
fn sampled_rows_reorder_masks_before_unbiased_anchor() {
    let input = valid(2);
    let sampled = vec![10, 11, 12, 13, 20, 21, 22, 23];
    assert_eq!(
        input.reorder_sampled_rows(&sampled).unwrap(),
        vec![vec![11, 12, 13, 10], vec![21, 22, 23, 20]]
    );
    assert!(matches!(
        input.reorder_sampled_rows(&sampled[..7]),
        Err(DsparkBatchInputError::LengthMismatch {
            field: "sampled_rows",
            expected: 8,
            found: 7
        })
    ));
}

#[test]
fn paged_slots_are_sequence_major_and_cross_blocks_exactly() {
    let tables = vec![vec![5, 6], vec![8, 9, 10]];
    let slots = paged_slot_mapping(&tables, &[14, 31], 4, 16)
        .unwrap()
        .unwrap();
    assert_eq!(slots, vec![94, 95, 96, 97, 159, 160, 161, 162]);
    assert!(
        paged_slot_mapping(&[vec![5]], &[15], 4, 16)
            .unwrap()
            .is_none()
    );
}

#[test]
fn proposal_lane_resolution_rejects_invalid_indices_before_vector_access() {
    assert_eq!(resolve_lane_id(usize::MAX, 3).unwrap(), 0);
    assert_eq!(resolve_lane_id(2, 3).unwrap(), 2);
    assert!(matches!(
        resolve_lane_id(3, 3),
        Err(DsparkBatchInputError::LaneOutOfBounds {
            lane: 3,
            lane_count: 3
        })
    ));
    assert!(matches!(
        resolve_lane_id(usize::MAX, 0),
        Err(DsparkBatchInputError::LaneOutOfBounds {
            lane: 0,
            lane_count: 0
        })
    ));
}

#[test]
fn sequence_accessor_is_not_public_api() {
    let source = include_str!("batch_inputs.rs");
    assert!(source.contains("pub(super) fn sequence"));
    assert!(!source.contains("pub fn sequence(&self"));
}

#[test]
fn batched_backbone_reaches_every_remaining_layer_in_serial_operation_order() {
    // The staged dispatch (per-layer loop + tail/markov calls) lives in
    // batch_propose.rs since the dflash_head.rs split.
    let entry = include_str!("batch_propose.rs");
    assert!(entry.contains("for layer_idx in 0..self.layers.len()"));
    assert!(entry.contains("self.run_batched_layer_stage("));
    assert!(entry.contains("self.run_batched_tail_base("));
    assert!(entry.contains("self.run_batched_markov("));

    let source = include_str!("batch_forward.rs");
    let input_norm = source.find("&layer.input_layernorm").unwrap();
    let q_proj = source.find("&layer.q_proj").unwrap();
    let q_norm = source.find("&layer.q_norm").unwrap();
    let rope = source.find("ops::rope_yarn").unwrap();
    let attention = source.find("self.run_staged_attention").unwrap();
    let o_proj = source.find("&layer.o_proj").unwrap();
    let post_norm = source.find("&layer.post_attention_layernorm").unwrap();
    let gate = source.find("&layer.gate_proj").unwrap();
    let silu = source.find("ops::silu_mul").unwrap();
    let down = source.find("&layer.down_proj").unwrap();
    assert!(
        input_norm < q_proj
            && q_proj < q_norm
            && q_norm < rope
            && rope < attention
            && attention < o_proj
            && o_proj < post_norm
            && post_norm < gate
            && gate < silu
            && silu < down
    );
    let attention_source = include_str!("batch_attention.rs");
    assert!(attention_source.contains("prefill_attention_paged_dflash_bf16_indirect"));
    assert!(attention_source.contains("prefill_attention_paged_batched_sink"));
    let projection_source = include_str!("batch_projection.rs");
    assert!(projection_source.contains("(total_rows - row).min(16)"));
    assert!(!projection_source.contains("w4a16_gemv_batch32"));
    // The tail/markov methods moved to batch_forward_tail.rs with the
    // batch_forward.rs split; the relative-order assertions still hold
    // within the moved bodies.
    let tail_source = include_str!("batch_forward_tail.rs");
    let tail = &tail_source[tail_source.find("fn run_batched_tail_base").unwrap()..];
    let final_norm = tail.find("&self.norm").unwrap();
    let lm_head = tail.find("ops::w4a16_gemm").unwrap();
    let argmax = tail.find("ops::argmax_bf16_batch").unwrap();
    assert!(tail.contains("self.batch_logits"));
    assert!(tail.contains("self.batch_tokens"));
    assert!(tail.contains("ops::w4a16_gemv_batchm"));
    assert!(tail.contains("self.kernels.w4a16_gemv_batch16"));
    assert!(tail.contains("(batch_rows - row).min(16)"));
    assert!(tail.contains("self.startup.native_batch_authoritative"));
    assert!(tail.contains("exact NVFP4 LM-head batch kernel is unresolved"));
    assert!(final_norm < lm_head && lm_head < argmax);
    let markov = &tail_source[tail_source.find("fn run_batched_markov").unwrap()..];
    let depth_loop = markov.find("for depth in 1..gamma").unwrap();
    let embed = markov.find("ops::batched_embed").unwrap();
    let project = markov.find("ops::dense_gemv_batchm").unwrap();
    let bias = markov.find("ops::dflash_batch_add_depth_bias").unwrap();
    let sample = markov.find("ops::argmax_bf16_batch").unwrap();
    let store = markov.find("ops::dflash_batch_store_depth_tokens").unwrap();
    assert!(
        depth_loop < embed && embed < project && project < bias && bias < sample && sample < store
    );
}

#[test]
fn production_seam_prepares_then_returns_native_rows_before_generic_serial_dispatch() {
    // The prepare → native-rows → generic-serial seam lives in
    // batch_propose.rs since the dflash_head.rs split; the orchestrator
    // (propose_batch admission/fallback gates) stays in ../dflash_head.rs.
    let staged = include_str!("batch_propose.rs");
    let plan = staged.find("packed_query_tokens").unwrap();
    let upload = staged.find("copy_h2d(&query_bytes").unwrap();
    let embed = staged.find("ops::batched_embed").unwrap();

    let stage = staged.find("self.run_batched_layer_stage").unwrap();
    let tail = staged.find("self.run_batched_tail_base").unwrap();
    // The readback arm's binding is `let native = if native_staged …`
    // (rustfmt joined it onto one line when the body moved files).
    let native_readback = staged.find("let native = if native_staged").unwrap();
    let generic_serial = staged.find("Generic single-lane serial path").unwrap();
    assert!(
        plan < upload
            && upload < embed
            && embed < stage
            && stage < tail
            && tail < native_readback
            && native_readback < generic_serial
    );
    assert!(staged.contains("self.batch_capacity"));
    assert!(staged.contains("self.batch_query_ids_dev"));
    assert!(staged.contains("self.batch_position_ids"));
    assert!(staged.contains("self.batch_query_embed"));
    assert!(!staged.contains("ops::dflash_batch_anchor_add"));

    assert!(staged.contains("self.batch_block_table_ptrs"));
    assert!(staged.contains("self.batch_cu_seqlens"));
    assert!(staged.contains("self.batch_kv_lens"));
    assert!(staged.contains("self.batch_slot_mapping"));
    assert!(staged.contains("ctx_count_drafter"));
    assert!(!staged.contains(".ctx_len\n                    .checked_add(self.gamma)"));

    assert!(staged.contains("let native_staged = batch_slots_ready && self.lane_count() == 1"));

    assert!(staged.contains("batch_inputs.reorder_sampled_rows"));
    assert!(staged.contains("DFlash Bxgamma parity mismatch"));
    assert!(staged.contains("dstate.last_num_drafted = tokens.len()"));
    assert!(staged.contains("if self.startup.diagnostics.batch_parity"));
    assert!(staged.contains("DFlash Bxgamma parity cache gate"));

    // Orchestrator-side gates that stayed in dflash_head.rs's propose_batch.
    let orchestrator = include_str!("../dflash_head.rs");
    assert!(orchestrator.contains("self.startup.native_batch_authoritative"));
    assert!(orchestrator.contains("self.prepare_drafts_state"));
    assert!(orchestrator.contains("n == 1 && !(native_authoritative"));
    assert!(orchestrator.contains("DFlash Bxgamma parity dispatch"));

    let prepare_source = include_str!("propose.rs");
    assert!(prepare_source.contains("fn prepare_drafts_state"));
    assert!(prepare_source.contains("if prepare_only"));
    assert!(prepare_source.contains("native batch preparation requires Option B"));

    let projection_source = include_str!("batch_projection.rs");
    for tier in ["batch4", "batch8", "batch16"] {
        assert!(projection_source.contains(tier));
    }
    assert!(projection_source.contains("(total_rows - row).min(16)"));
    assert!(!projection_source.contains("w4a16_gemv_batch32"));
}
