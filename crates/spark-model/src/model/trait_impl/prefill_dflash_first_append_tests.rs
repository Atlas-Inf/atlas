// SPDX-License-Identifier: AGPL-3.0-only
//! Exercise real non-GLM prefill capture, its cursor, and the first append.
use super::fixture::Fixture;
use crate::layers::dflash_head::{DflashProposerState, FirstAppend, prefill_test_support};
use crate::speculative::DraftProposer;
use std::sync::Arc;

#[test]
fn integration_non_glm_prefill_honors_first_append_variants() {
    for variant in [
        FirstAppend::Own,
        FirstAppend::Zero,
        FirstAppend::None,
        FirstAppend::Legacy,
    ] {
        let mut f = Fixture::new(1, 1, 0);
        f.model.config.model_type = "qwen3_next".into();
        f.model.config.hc_mult = 1;
        f.model.dflash_capture_layers = vec![0];
        let h = f.model.config.hidden_size;
        let slot = h * 2;
        let head = Arc::new(prefill_test_support::head(variant, h));
        f.seq.proposer_state = Some(head.alloc_state(f.model.gpu.as_ref()).unwrap());
        f.model.proposer = Some(head.clone());
        f.seq.tokens = vec![1, 2, 3, 4];
        f.seq.seq_len = 4;
        // Two chunks: the own row must track the LAST chunk's last row.
        for (start, byte) in [(0, 0x11), (2, 0x22)] {
            f.model
                .gpu
                .memset(f.model.buffers.hidden_states(), byte, 2 * slot)
                .unwrap();
            f.model
                .try_dflash_prefill_capture_layer(&mut f.seq, 0, start, 2, 7)
                .unwrap();
        }
        f.model
            .update_dflash_ctx_len_after_prefill(&mut f.seq, 2, 2)
            .unwrap();
        let state = f
            .seq
            .proposer_state
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .unwrap();
        assert_eq!(state.ctx_positions, vec![0, 1, 2, 3]);
        assert_eq!(state.ctx_prefill_origin, Some(0));
        assert_eq!(state.ctx_prefill_base, 0);
        // Model-global capture belongs to another request. Own must ignore it.
        let stack = f.model.gpu.alloc(slot).unwrap();
        f.model.gpu.memset(stack, 0xAB, slot).unwrap();
        prefill_test_support::append(&head, state, stack, 4, f.model.gpu.as_ref());
        let expected = match variant {
            FirstAppend::Own => Some(0x22),
            FirstAppend::Zero => Some(0),
            FirstAppend::Legacy => Some(0xAB),
            FirstAppend::None => None,
        };
        assert_eq!(
            state.ctx_len,
            4 + usize::from(expected.is_some()),
            "{variant:?}"
        );
        if let Some(byte) = expected {
            let mut actual = vec![0; slot];
            f.model
                .gpu
                .copy_d2h(state.ctx_hidden_acc.offset(4 * slot), &mut actual)
                .unwrap();
            assert_eq!(actual, vec![byte; slot], "{variant:?}");
            assert_eq!(state.ctx_positions, vec![0, 1, 2, 3, 3]);
        }
        // The first-append marker is one-shot even at the same position.
        if variant != FirstAppend::Legacy {
            let len = state.ctx_len;
            prefill_test_support::append(&head, state, stack, 4, f.model.gpu.as_ref());
            assert_eq!(state.ctx_len, len);
        }
    }
}
