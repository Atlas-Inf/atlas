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
    let entry = include_str!("../dflash_head.rs");
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
    let tail = &source[source.find("fn run_batched_tail_base").unwrap()..];
    let final_norm = tail.find("&self.norm").unwrap();
    let lm_head = tail.find("ops::w4a16_gemm").unwrap();
    let argmax = tail.find("ops::argmax_bf16_batch").unwrap();
    assert!(tail.contains("self.batch_logits"));
    assert!(tail.contains("self.batch_tokens"));
    assert!(final_norm < lm_head && lm_head < argmax);
    let markov = &source[source.find("fn run_batched_markov").unwrap()..];
    let depth_loop = markov.find("for depth in 1..self.gamma").unwrap();
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
fn production_seam_uploads_and_embeds_packed_queries_before_oracle_dispatch() {
    let source = include_str!("../dflash_head.rs");
    let plan = source.find("packed_query_tokens").unwrap();
    let upload = source.find("copy_h2d(&query_bytes").unwrap();
    let embed = source.find("ops::batched_embed").unwrap();

    let stage = source.find("self.run_batched_layer_stage").unwrap();
    let tail = source.find("self.run_batched_tail_base").unwrap();
    let oracle = source.find("let lanes_n = self.lane_count()").unwrap();
    assert!(plan < upload && upload < embed && embed < stage && stage < tail && tail < oracle);
    assert!(source.contains("self.batch_capacity"));
    assert!(source.contains("self.batch_query_ids_dev"));
    assert!(source.contains("self.batch_position_ids"));
    assert!(source.contains("self.batch_query_embed"));
    assert!(!source.contains("ops::dflash_batch_anchor_add"));

    assert!(source.contains("self.batch_block_table_ptrs"));
    assert!(source.contains("self.batch_cu_seqlens"));
    assert!(source.contains("self.batch_kv_lens"));
    assert!(source.contains("self.batch_slot_mapping"));
    assert!(source.contains("ctx_count_drafter"));
    assert!(!source.contains(".ctx_len\n                    .checked_add(self.gamma)"));

    assert!(source.contains("if batch_slots_ready && self.lane_count() == 1"));

    assert!(source.contains("diagnostics.batch_parity && batch_slots_ready"));
    assert!(source.contains("batch_inputs.reorder_sampled_rows"));
    assert!(source.contains("DFlash Bxgamma parity mismatch"));
    assert!(source.contains("if self.startup.diagnostics.batch_parity"));
    assert!(source.contains("self.batch_capacity"));
    assert!(source.contains("n == 1 && !self.startup.diagnostics.batch_parity"));
    assert!(source.contains("DFlash Bxgamma parity dispatch"));
    assert!(source.contains("DFlash Bxgamma parity cache gate"));
}
