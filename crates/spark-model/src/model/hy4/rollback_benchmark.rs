// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::traits::Model;

impl Hy4Model {
    pub(super) fn benchmark_rollback(
        &self,
        tokens: &[u32],
        slot: usize,
        stream: u64,
    ) -> Result<()> {
        let vocab = self.vocab_size() as u32;
        let max_seq = self.borrow_owner().max_seq;
        ensure!(
            tokens.len() + 10 <= max_seq,
            "Hy4 rollback benchmark capacity"
        );
        let long = Some(515).filter(|&n| n + 10 <= max_seq);
        for length in std::iter::once(tokens.len()).chain(long) {
            let prefix: Vec<_> = tokens.iter().copied().cycle().take(length).collect();
            let mut reference = self.benchmark_sequence(slot)?;
            let logits = self.prefill(&prefix, &mut reference, stream)?;
            let mut next = self.argmax_on_device(logits, stream)?;
            let mut inputs = Vec::new();
            let mut expected = Vec::new();
            for _ in 0..10 {
                inputs.push(next);
                let logits = self.decode(next, &mut reference, stream)?;
                next = self.argmax_on_device(logits, stream)?;
                expected.push(next);
            }
            for accepted in 1..=6 {
                let mut trial = self.benchmark_sequence(slot)?;
                self.prefill(&prefix, &mut trial, stream)?;
                self.checkpoint_ssm_states(&mut trial)?;
                let mut speculative = inputs[..6].to_vec();
                for token in &mut speculative[accepted..] {
                    *token = (*token + 13) % vocab;
                }
                self.decode_verify(&speculative, &mut trial, stream)?;
                trial.seq_len = length + accepted;
                trial.tokens.truncate(trial.seq_len);
                self.commit_accepted_prefix(&mut trial, accepted, 6)?;
                let mut actual = Vec::new();
                for &token in &expected[accepted - 1..accepted + 3] {
                    let logits = self.decode(token, &mut trial, stream)?;
                    actual.push(self.argmax_on_device(logits, stream)?);
                }
                let target = &expected[accepted..accepted + 4];
                let first_divergence = actual.iter().zip(target).position(|(a, b)| a != b);
                tracing::info!(
                    "HY4_ROLLBACK_PARITY prefix={length} accepted={accepted} rows=6 expected={target:?} actual={actual:?} first_divergence={first_divergence:?}"
                );
                ensure!(
                    first_divergence.is_none(),
                    "Hy4 rejected-suffix rollback divergence"
                );
            }
        }
        Ok(())
    }
}
