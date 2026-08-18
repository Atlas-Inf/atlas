// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::DevicePtr;

use super::batch_execution::{paged_slot_mapping, resolve_lane_id};
use super::batch_inputs::{DsparkBatchInput, DsparkBatchInputError, validate_batch_input_lengths};
use super::{
    CaptureDescriptor, CaptureStatus, LIGHTNING_SERVED_GAMMA, LIGHTNING_TAPS, SequenceGeneration,
};

fn owner(slot: usize, generation: u64) -> SequenceGeneration {
    SequenceGeneration::new(slot, generation).unwrap()
}

fn lifecycle(owner: SequenceGeneration) -> CaptureDescriptor {
    CaptureDescriptor::bind(owner, 0, 0, LIGHTNING_SERVED_GAMMA, 64).unwrap()
}

fn valid_parts(
    batch: usize,
) -> (
    Vec<SequenceGeneration>,
    Vec<u32>,
    Vec<usize>,
    Vec<DevicePtr>,
    Vec<SequenceGeneration>,
    Vec<Option<CaptureDescriptor>>,
) {
    let owners: Vec<_> = (0..batch).map(|i| owner(i, (i + 1) as u64)).collect();
    let last_tokens: Vec<_> = (0..batch).map(|i| 100 + i as u32).collect();
    let positions: Vec<_> = (0..batch).map(|i| 1_000 + i * 17).collect();
    let target_hiddens: Vec<_> = (0..batch)
        .map(|i| DevicePtr(0x1000 + (i as u64) * 0x100))
        .collect();
    let lifecycles = owners.iter().copied().map(lifecycle).map(Some).collect();
    (
        owners.clone(),
        last_tokens,
        positions,
        target_hiddens,
        owners,
        lifecycles,
    )
}

fn valid(batch: usize) -> DsparkBatchInput {
    let (owners, last_tokens, positions, target_hiddens, expected, lifecycles) = valid_parts(batch);
    DsparkBatchInput::validate(
        LIGHTNING_SERVED_GAMMA,
        batch,
        &owners,
        &last_tokens,
        &positions,
        &target_hiddens,
        &expected,
        &lifecycles,
    )
    .unwrap()
}

#[test]
fn lightning_contract_consumes_the_existing_gamma_and_tap_ssot() {
    assert_eq!(LIGHTNING_SERVED_GAMMA, 4);
    assert_eq!(LIGHTNING_TAPS, [1, 5, 19, 29, 41, 51]);
    assert!(matches!(
        DsparkBatchInput::validate(0, 1, &[], &[], &[], &[], &[], &[]),
        Err(DsparkBatchInputError::GammaZero)
    ));
    assert!(matches!(
        DsparkBatchInput::validate(8, 1, &[], &[], &[], &[], &[], &[]),
        Err(DsparkBatchInputError::GammaMismatch {
            expected: LIGHTNING_SERVED_GAMMA,
            found: 8
        })
    ));
}

#[test]
fn b1_b2_b4_b8_use_sequence_then_gamma_rows() {
    for batch in [1, 2, 4, 8] {
        let input = valid(batch);
        assert_eq!(input.batch_len(), batch);
        assert_eq!(input.gamma(), LIGHTNING_SERVED_GAMMA);
        assert_eq!(input.total_rows(), batch * LIGHTNING_SERVED_GAMMA);
        for sequence in 0..batch {
            assert_eq!(
                input.sequence_row_range(sequence).unwrap(),
                sequence * LIGHTNING_SERVED_GAMMA..(sequence + 1) * LIGHTNING_SERVED_GAMMA
            );
            for query in 0..LIGHTNING_SERVED_GAMMA {
                assert_eq!(
                    input.row_index(sequence, query).unwrap(),
                    sequence * LIGHTNING_SERVED_GAMMA + query
                );
            }
        }
    }
}

#[test]
fn mixed_absolute_positions_and_owner_identity_survive_batch_reordering() {
    let (
        mut owners,
        mut last_tokens,
        mut positions,
        mut target_hiddens,
        mut expected,
        mut lifecycles,
    ) = valid_parts(2);
    positions[0] = 7;
    positions[1] = 99_999;
    last_tokens[0] = 3;
    last_tokens[1] = 4;
    target_hiddens[0] = DevicePtr(0xabc0);
    target_hiddens[1] = DevicePtr(0xdef0);
    let first = DsparkBatchInput::validate(
        LIGHTNING_SERVED_GAMMA,
        2,
        &owners,
        &last_tokens,
        &positions,
        &target_hiddens,
        &expected,
        &lifecycles,
    )
    .unwrap();

    owners.swap(0, 1);
    last_tokens.swap(0, 1);
    positions.swap(0, 1);
    target_hiddens.swap(0, 1);
    expected.swap(0, 1);
    lifecycles.swap(0, 1);
    let reordered = DsparkBatchInput::validate(
        LIGHTNING_SERVED_GAMMA,
        2,
        &owners,
        &last_tokens,
        &positions,
        &target_hiddens,
        &expected,
        &lifecycles,
    )
    .unwrap();

    assert_eq!(first.sequence(0).absolute_position, 7);
    assert_eq!(first.sequence(1).absolute_position, 99_999);
    assert_eq!(reordered.sequence(0).owner, first.sequence(1).owner);
    assert_eq!(reordered.sequence(0).target_hidden, DevicePtr(0xdef0));
}

#[test]
fn checked_byte_layout_uses_hidden_width_and_element_bytes() {
    let input = valid(2);
    assert_eq!(input.total_bytes(3, 2).unwrap(), 2 * 4 * 3 * 2);
    assert_eq!(input.row_byte_offset(1, 2, 3, 2).unwrap(), 36);
    assert_eq!(input.row_byte_range(1, 2, 3, 2).unwrap(), 36..42);
    assert_eq!(input.sequence_byte_range(1, 3, 2).unwrap(), 24..48);
    assert!(matches!(
        input.total_bytes(0, 2),
        Err(DsparkBatchInputError::ZeroDimension {
            field: "hidden_width"
        })
    ));
    assert!(matches!(
        input.total_bytes(3, 0),
        Err(DsparkBatchInputError::ZeroDimension {
            field: "element_bytes"
        })
    ));
}

#[test]
fn all_structural_lengths_must_match_the_batch_width() {
    let (owners, last_tokens, positions, target_hiddens, expected, lifecycles) = valid_parts(2);
    assert!(validate_batch_input_lengths(2, 2, 1, 2, 2, 2, 2).is_err());
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            2,
            &owners[..1],
            &last_tokens,
            &positions,
            &target_hiddens,
            &expected,
            &lifecycles,
        ),
        Err(DsparkBatchInputError::LengthMismatch {
            field: "owners",
            expected: 2,
            found: 1
        })
    ));
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            2,
            &owners,
            &last_tokens[..1],
            &positions,
            &target_hiddens,
            &expected,
            &lifecycles,
        ),
        Err(DsparkBatchInputError::LengthMismatch {
            field: "last_tokens",
            expected: 2,
            found: 1
        })
    ));
}

#[test]
fn empty_and_capacity_overflow_are_typed_failures() {
    assert!(matches!(
        DsparkBatchInput::validate(LIGHTNING_SERVED_GAMMA, 1, &[], &[], &[], &[], &[], &[],),
        Err(DsparkBatchInputError::EmptyBatch)
    ));
    let parts = valid_parts(2);
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            1,
            &parts.0,
            &parts.1,
            &parts.2,
            &parts.3,
            &parts.4,
            &parts.5,
        ),
        Err(DsparkBatchInputError::CapacityExceeded {
            capacity: 1,
            batch: 2
        })
    ));
}

#[test]
fn duplicate_expected_owners_are_rejected_even_when_rows_are_distinct() {
    let (mut owners, last_tokens, positions, target_hiddens, mut expected, mut lifecycles) =
        valid_parts(2);
    owners[1] = owners[0];
    expected[1] = expected[0];
    lifecycles[1] = Some(lifecycle(owners[0]));
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            2,
            &owners,
            &last_tokens,
            &positions,
            &target_hiddens,
            &expected,
            &lifecycles,
        ),
        Err(DsparkBatchInputError::DuplicateOwner {
            first: 0,
            second: 1,
            ..
        })
    ));
}

#[test]
fn expected_stale_retired_and_missing_lifecycle_owners_fail_closed() {
    let (owners, last_tokens, positions, target_hiddens, mut expected, mut lifecycles) =
        valid_parts(2);
    expected[1] = owner(1, 999);
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            2,
            &owners,
            &last_tokens,
            &positions,
            &target_hiddens,
            &expected,
            &lifecycles,
        ),
        Err(DsparkBatchInputError::ExpectedOwnerMismatch { sequence: 1, .. })
    ));

    expected[1] = owners[1];
    lifecycles[1] = Some(lifecycle(owner(1, 999)));
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            2,
            &owners,
            &last_tokens,
            &positions,
            &target_hiddens,
            &expected,
            &lifecycles,
        ),
        Err(DsparkBatchInputError::LifecycleOwnerMismatch { sequence: 1, .. })
    ));

    let mut retired = lifecycle(owners[1]);
    retired.retire(owners[1]).unwrap();
    assert_eq!(retired.status(), CaptureStatus::Retired);
    lifecycles[1] = Some(retired);
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            2,
            &owners,
            &last_tokens,
            &positions,
            &target_hiddens,
            &expected,
            &lifecycles,
        ),
        Err(DsparkBatchInputError::LifecycleNotLive { sequence: 1, .. })
    ));

    lifecycles[1] = None;
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            2,
            &owners,
            &last_tokens,
            &positions,
            &target_hiddens,
            &expected,
            &lifecycles,
        ),
        Err(DsparkBatchInputError::MissingLifecycle { sequence: 1, .. })
    ));
}

#[test]
fn zero_target_hidden_pointer_is_rejected_with_sequence_index() {
    let (owners, last_tokens, positions, mut target_hiddens, expected, lifecycles) = valid_parts(2);
    target_hiddens[1] = DevicePtr(0);
    assert!(matches!(
        DsparkBatchInput::validate(
            LIGHTNING_SERVED_GAMMA,
            2,
            &owners,
            &last_tokens,
            &positions,
            &target_hiddens,
            &expected,
            &lifecycles,
        ),
        Err(DsparkBatchInputError::ZeroTargetHidden { sequence: 1 })
    ));
}

#[test]
fn checked_arithmetic_rejects_byte_size_and_offset_overflow() {
    let input = valid(2);
    assert!(matches!(
        input.total_bytes(usize::MAX, 2),
        Err(DsparkBatchInputError::ArithmeticOverflow { .. })
    ));
    assert!(matches!(
        input.row_byte_offset(1, 0, usize::MAX, 2),
        Err(DsparkBatchInputError::ArithmeticOverflow { .. })
    ));
    assert!(matches!(
        input.sequence_byte_range(1, usize::MAX, 2),
        Err(DsparkBatchInputError::ArithmeticOverflow { .. })
    ));
}

#[test]
fn row_and_sequence_ranges_reject_out_of_bounds_indices() {
    let input = valid(1);
    assert!(matches!(
        input.row_index(1, 0),
        Err(DsparkBatchInputError::SequenceOutOfBounds {
            sequence: 1,
            batch: 1
        })
    ));
    assert!(matches!(
        input.row_index(0, LIGHTNING_SERVED_GAMMA),
        Err(DsparkBatchInputError::QueryOutOfBounds {
            query: LIGHTNING_SERVED_GAMMA,
            gamma: LIGHTNING_SERVED_GAMMA
        })
    ));
}

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
    assert!(entry.contains("for layer_idx in 1..self.layers.len()"));
    assert!(entry.contains("self.run_batched_layer_stage("));
    assert!(entry.contains("self.run_batched_tail_base("));

    let source = include_str!("batch_forward.rs");
    let input_norm = source.find("&layer.input_layernorm").unwrap();
    let q_proj = source.find("&layer.q_proj").unwrap();
    let q_norm = source.find("&layer.q_norm").unwrap();
    let rope = source.find("ops::rope_yarn").unwrap();
    let attention = source
        .find("ops::prefill_attention_paged_batched_sink")
        .unwrap();
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
    let tail = &source[source.find("fn run_batched_tail_base").unwrap()..];
    let final_norm = tail.find("&self.norm").unwrap();
    let lm_head = tail.find("ops::w4a16_gemm").unwrap();
    let argmax = tail.find("ops::argmax_bf16_batch").unwrap();
    assert!(tail.contains("self.batch_logits"));
    assert!(tail.contains("self.batch_tokens"));
    assert!(final_norm < lm_head && lm_head < argmax);
}

#[test]
fn production_seam_uploads_and_embeds_packed_queries_before_oracle_dispatch() {
    let source = include_str!("../dflash_head.rs");
    let plan = source.find("packed_query_tokens").unwrap();
    let upload = source.find("copy_h2d(&query_bytes").unwrap();
    let embed = source.find("ops::batched_embed").unwrap();
    let gather = source.find("batch_target_hidden.offset").unwrap();
    let fc = source.find("&self.fc").unwrap();
    let hidden_norm = source.find("&self.hidden_norm").unwrap();
    let anchor_add = source.find("ops::dflash_batch_anchor_add").unwrap();
    let layer_norm = source.find("&layer0.input_layernorm").unwrap();
    let q_proj = source.find("&layer0.q_proj").unwrap();
    let q_norm = source.find("&layer0.q_norm").unwrap();
    let rope = source.find("ops::rope_yarn").unwrap();
    let attention = source
        .find("ops::prefill_attention_paged_batched_sink")
        .unwrap();
    let o_proj = source.find("&layer0.o_proj").unwrap();
    let post_norm = source.find("&layer0.post_attention_layernorm").unwrap();
    let down_proj = source.find("&layer0.down_proj").unwrap();
    let oracle = source.find("let lanes_n = self.lane_count()").unwrap();
    assert!(
        plan < upload
            && upload < embed
            && embed < gather
            && gather < fc
            && fc < hidden_norm
            && hidden_norm < anchor_add
            && anchor_add < layer_norm
            && layer_norm < q_proj
            && q_proj < q_norm
            && q_norm < rope
            && rope < attention
            && attention < o_proj
            && o_proj < post_norm
            && post_norm < down_proj
            && down_proj < oracle
    );
    assert!(source.contains("self.batch_capacity"));
    assert!(source.contains("self.batch_query_ids_dev"));
    assert!(source.contains("self.batch_position_ids"));
    assert!(source.contains("self.batch_query_embed"));
    assert!(source.contains("self.batch_target_hidden"));
    assert!(source.contains("self.batch_fc_proj"));
    assert!(source.contains("self.batch_fc_norm"));
    assert!(source.contains("self.batch_norm"));
    assert!(source.contains("self.batch_q"));
    assert!(source.contains("self.batch_k"));
    assert!(source.contains("self.batch_v"));
    assert!(source.contains("self.batch_block_table_ptrs"));
    assert!(source.contains("self.batch_cu_seqlens"));
    assert!(source.contains("self.batch_kv_lens"));
    assert!(source.contains("self.batch_slot_mapping"));
    assert!(source.contains("ctx_count_drafter"));
    assert!(!source.contains(".ctx_len\n                    .checked_add(self.gamma)"));
    assert!(source.contains("batch_attn_out"));
    assert!(source.contains("self.batch_attn_proj"));
    assert!(source.contains("self.batch_mlp_gate"));
    assert!(source.contains("self.batch_mlp_up"));
    assert!(source.contains("self.batch_mlp_down"));
    assert!(source.contains("batch_logits"));
    assert!(source.contains("batch_tokens"));
    assert!(source.contains("batch_slots_ready\n            && self.lane_count() == 1"));
    assert!(source.contains("&& let Some(sinks)"));
    assert!(source.contains("sole source of returned drafts"));
}
