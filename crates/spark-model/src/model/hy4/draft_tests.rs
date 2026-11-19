// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::traits::Model;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn prefix_commit_selects_captured_hidden_and_indices_without_running_a_forward() {
    for accepted in 1..=6 {
        let gpu = Box::new(MockGpuBackend::new());
        let weights = super::super::tests::store(&gpu);
        let model =
            Hy4Model::new(super::super::tests::config(), weights, gpu, None, 16, 4, 1).unwrap();
        let mut seq = model.alloc_sequence().unwrap();
        let stream = model.default_stream();
        model.prefill(&[1], &mut seq, stream).unwrap();
        model.checkpoint_ssm_states(&mut seq).unwrap();
        model
            .decode_verify(&[2, 3, 4, 5, 6, 7], &mut seq, stream)
            .unwrap();
        let runtime = model.borrow_dependent();
        let owner = model.borrow_owner();
        runtime
            .requests
            .lock()
            .get_mut(&seq.slot_idx)
            .unwrap()
            .with_dependent_mut(|cell_owner, data| {
                let checkpoint = data.checkpoint.as_mut().unwrap();
                for row in 0..6 {
                    let mut state = State::new(cell_owner.shared.gpu.as_ref(), 32).unwrap();
                    owner
                        .gpu
                        .copy_h2d(&vec![70 + row as u8; 64 + 5 * 8192], state.hidden)
                        .unwrap();
                    state.len = row + 2;
                    state.ready = true;
                    checkpoint.draft_prefix.push(state);
                }
                data.draft = Some(checkpoint.draft_prefix[5].fork(stream).unwrap());
                checkpoint.draft = data.draft.as_ref().map(|s| s.fork(stream).unwrap());
            });
        seq.tokens.truncate(1 + accepted);
        seq.seq_len = 1 + accepted;
        model.commit_accepted_prefix(&mut seq, accepted, 6).unwrap();
        assert_eq!(
            runtime.head_rows.load(std::sync::atomic::Ordering::Relaxed),
            6
        );
        let requests = runtime.requests.lock();
        let data = requests[&seq.slot_idx].borrow_dependent();
        assert_eq!(data.history, seq.tokens);
        assert_eq!(data.layers[0].attn.committed_len(), 1 + accepted);
        let state = data.draft.as_ref().unwrap();
        assert!(state.ready);
        assert_eq!(state.len, 1 + accepted);
        let mut bytes = vec![0; 64 + 5 * 8192];
        owner.gpu.copy_d2h(state.hidden, &mut bytes).unwrap();
        assert_eq!(bytes, vec![69 + accepted as u8; bytes.len()]);
    }
}

#[test]
fn duplicate_checkpoint_preserves_pre_draft_state_but_new_history_refreshes_it() {
    let gpu = Box::new(MockGpuBackend::new());
    let weights = super::super::tests::store(&gpu);
    let m = Hy4Model::new(super::super::tests::config(), weights, gpu, None, 8, 4, 1).unwrap();
    let mut seq = m.alloc_sequence().unwrap();
    let stream = m.default_stream();
    m.prefill(&[1], &mut seq, stream).unwrap();
    let s = m.borrow_owner();
    let r = m.borrow_dependent();
    r.requests
        .lock()
        .get_mut(&seq.slot_idx)
        .unwrap()
        .with_dependent_mut(|owner, data| {
            let mut state = State::new(owner.shared.gpu.as_ref(), 32).unwrap();
            s.gpu.copy_h2d(&[19; 64], state.hidden).unwrap();
            state.len = data.history.len();
            state.ready = true;
            data.draft = Some(state);
        });
    m.checkpoint_ssm_states(&mut seq).unwrap();
    r.requests
        .lock()
        .get_mut(&seq.slot_idx)
        .unwrap()
        .with_dependent_mut(|_, data| {
            s.gpu
                .copy_h2d(&[99; 64], data.draft.as_ref().unwrap().hidden)
                .unwrap();
        });
    m.checkpoint_ssm_states(&mut seq).unwrap();
    let saved_hidden = r.requests.lock()[&seq.slot_idx]
        .borrow_dependent()
        .checkpoint
        .as_ref()
        .unwrap()
        .draft
        .as_ref()
        .unwrap()
        .hidden;
    let mut bytes = [0; 64];
    s.gpu.copy_d2h(saved_hidden, &mut bytes).unwrap();
    assert_eq!(bytes, [19; 64]);
    let captured = r
        .requests
        .lock()
        .get_mut(&seq.slot_idx)
        .unwrap()
        .with_dependent_mut(|_, data| {
            let state = data.draft.take().unwrap();
            let mut captured = [0; 64];
            s.gpu.copy_d2h(state.hidden, &mut captured).unwrap();
            captured
        });
    m.decode(2, &mut seq, stream).unwrap();
    r.requests
        .lock()
        .get_mut(&seq.slot_idx)
        .unwrap()
        .with_dependent_mut(|owner, data| {
            let mut state = State::new(owner.shared.gpu.as_ref(), 32).unwrap();
            s.gpu.copy_h2d(&captured, state.hidden).unwrap();
            state.len = data.history.len();
            state.ready = true;
            data.draft = Some(state);
        });
    m.checkpoint_ssm_states(&mut seq).unwrap();
    s.gpu.copy_d2h(saved_hidden, &mut bytes).unwrap();
    assert_eq!(bytes, [99; 64]);
}

#[test]
fn live_architecture_validation_uses_tp_local_target_head_count() {
    let mut c = atlas_core::config::parse_config(include_str!(
        "../../../../atlas-core/src/config/fixtures/hy4/config.json"
    ))
    .unwrap();
    c.tp_world_size = 2;
    c.num_attention_heads /= 2;
    assert!(validate(&c).is_ok());
    c.num_attention_heads *= 2;
    assert!(validate(&c).is_err());
    c.num_attention_heads /= 2;
    c.tp_world_size = 1;
    assert!(validate(&c).is_err());
}

#[test]
fn captured_hidden_and_all_indices_survive_scratch_reuse_and_snapshot_restore() {
    let gpu = MockGpuBackend::new();
    let mut source = Owned::new(&gpu);
    let hidden = source.upload(&[19; 64]).unwrap();
    let caches: [TargetCache; 5] = (0..5)
        .map(|i| TargetCache {
            latent: DevicePtr(1),
            indices: source.upload(&vec![i as u8 + 1; 8192]).unwrap(),
            len: 57,
            capacity: 64,
        })
        .collect::<Vec<_>>()
        .try_into()
        .ok()
        .unwrap();
    let mut state = State::new(&gpu, 32).unwrap();
    assert!(!state.ready);
    state.capture(hidden, &caches, 0).unwrap();
    let saved = state.fork(0).unwrap();
    gpu.copy_h2d(&[99; 64], hidden).unwrap();
    let mut later = caches;
    for c in &mut later {
        gpu.copy_h2d(&vec![77; 8192], c.indices).unwrap();
        c.len = 63;
    }
    state.capture(hidden, &later, 0).unwrap();
    state.copy_from(&saved, 0).unwrap();
    assert!(state.ready);
    assert_eq!(state.len, 57);
    let mut bytes = [0; 64];
    gpu.copy_d2h(state.hidden, &mut bytes).unwrap();
    assert_eq!(bytes, [19; 64]);
    for i in 0..5 {
        let mut bytes = vec![0; 8192];
        gpu.copy_d2h(state.indices(i), &mut bytes).unwrap();
        assert_eq!(bytes, vec![i as u8 + 1; 8192]);
    }
}

#[test]
fn capture_rejects_empty_missing_inconsistent_state_and_wrong_snapshot_width() {
    let gpu = MockGpuBackend::new();
    let mut state = State::new(&gpu, 32).unwrap();
    let cache = TargetCache {
        latent: DevicePtr(1),
        indices: DevicePtr(2),
        len: 0,
        capacity: 64,
    };
    assert!(state.capture(DevicePtr(3), &[cache; 5], 0).is_err());
    assert!(state.capture(DevicePtr::NULL, &[cache; 5], 0).is_err());
    let mut caches = [TargetCache { len: 57, ..cache }; 5];
    caches[2].len = 58;
    assert!(state.capture(DevicePtr(3), &caches, 0).is_err());
    caches[2].len = 57;
    caches[2].indices = DevicePtr::NULL;
    assert!(state.capture(DevicePtr(3), &caches, 0).is_err());
    assert!(state.copy_from(&State::new(&gpu, 64).unwrap(), 0).is_err());
    assert!(!state.ready);
}
