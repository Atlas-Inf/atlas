// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn setup(n: usize, fp32: bool) -> (Stub, Vec<ActiveSeq>, Vec<RespRx>) {
    let mut model = Stub::new(false);
    model.fp32 = fp32;
    model.bootstrap_rows = (0..n)
        .map(|i| vec![0.0, 1.0 + i as f32 * 0.5, -2.0])
        .collect();
    let (mut active, rx) = seqs(vec![(10, vec![]); n]);
    for a in &mut active {
        a.top_logprobs = Some(3);
        a.logit_bias = vec![(2, 4.0)];
    }
    (model, active, rx)
}

fn assert_scored(a: &ActiveSeq, row: usize) {
    let expected = crate::scheduler::logprobs::extract_logprobs_from_f32(
        &[0.0, 1.0 + row as f32 * 0.5, 2.0],
        2,
        3,
    );
    assert_eq!(a.output_tokens, vec![10, 2]);
    assert_eq!(a.logprobs_data.len(), 1);
    let actual = &a.logprobs_data[0];
    assert_eq!(actual.token_id, 2);
    assert!(
        (actual.logprob - expected.logprob).abs() < 1e-6,
        "processed score: {} != {}",
        actual.logprob,
        expected.logprob
    );
    assert_eq!(
        actual.top, expected.top,
        "pipeline applies bias exactly once"
    );
}

#[test]
fn bootstrap_score_single_bf16_uses_processed_row() {
    let (model, mut active, _rx) = setup(1, false);
    step(&model, &mut active);
    assert_scored(&active[0], 0);
    assert_eq!(model.seen.lock().unwrap().other, vec!["decode"]);
}

#[test]
fn bootstrap_score_single_fp32_uses_actual_dtype() {
    let (model, mut active, _rx) = setup(1, true);
    step(&model, &mut active);
    assert_scored(&active[0], 0);
}

#[test]
fn bootstrap_score_batch_uses_each_processed_row() {
    let (model, mut active, _rx) = setup(2, false);
    step(&model, &mut active);
    for (i, a) in active.iter().enumerate() {
        assert_scored(a, i);
    }
    assert_eq!(model.seen.lock().unwrap().other, vec!["decode_batch"]);
}

#[test]
fn bootstrap_score_batch_read_failure_only_aborts_failed_member() {
    let (mut model, mut active, _rx) = setup(2, false);
    model.fail_row = Some(0);
    step(&model, &mut active);
    assert!(active[0].finished && active[0].engine_error.is_some());
    assert_eq!(active[0].output_tokens, vec![10]);
    assert!(active[0].logprobs_data.is_empty());
    assert!(active[0].pending_drafts.is_empty());
    assert_scored(&active[1], 1);
}

#[test]
fn bootstrap_score_unscored_fast_path_does_not_copy() {
    let mut model = Stub::new(false);
    model.fail_logits = true;
    let (mut active, _rx) = seqs(vec![(10, vec![])]);
    step(&model, &mut active);
    assert_eq!(active[0].output_tokens, vec![10, DECODED]);
    assert!(active[0].engine_error.is_none());
}

impl Stub {
    pub(super) fn copy_bootstrap_logits(&self, p: DevicePtr, d: &mut [u8]) -> Result<()> {
        if !self.bootstrap_rows.is_empty() {
            let elem = if self.fp32 { 4 } else { 2 };
            let row = p.0 as usize / (self.vocab_size() * elem);
            if self.fail_row == Some(row) {
                bail!("injected row read failure");
            }
            let bytes: Vec<u8> = self.bootstrap_rows[row]
                .iter()
                .flat_map(|v| {
                    if self.fp32 {
                        v.to_le_bytes().to_vec()
                    } else {
                        ((v.to_bits() >> 16) as u16).to_le_bytes().to_vec()
                    }
                })
                .collect();
            d.copy_from_slice(&bytes[..d.len()]);
        }
        Ok(())
    }
}
