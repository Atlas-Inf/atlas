// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::traits::{Model, SequenceState};
use std::sync::atomic::Ordering;

pub(super) fn enabled() -> Result<bool> {
    match std::env::var("ATLAS_HY4_VERIFY_BENCH") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) if v == "0" => Ok(false),
        Ok(v) if v == "1" => Ok(true),
        _ => anyhow::bail!("ATLAS_HY4_VERIFY_BENCH must be 0 or 1"),
    }
}

impl Hy4Model {
    pub(super) fn maybe_verify_benchmark(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        let r = self.borrow_dependent();
        if !r.benchmark || r.benchmark_done.swap(true, Ordering::Relaxed) {
            return Ok(());
        }
        ensure!(seq.seq_len == 0, "Hy4 benchmark requires an empty sequence");
        let prefix = &tokens[..tokens.len().min(57).min(r.multi.capacity)];
        ensure!(
            !prefix.is_empty() && prefix.len() + 6 <= self.borrow_owner().max_seq,
            "Hy4 benchmark context capacity"
        );
        let mut reference = self.benchmark_sequence(seq.slot_idx)?;
        let mut logits = DevicePtr::NULL;
        for &token in prefix {
            logits = self.forward_token(token, &mut reference, stream)?;
        }
        let mut next = self.argmax_on_device(logits, stream)?;
        let mut block = Vec::new();
        let mut expected = Vec::new();
        for _ in 0..6 {
            block.push(next);
            logits = self.forward_token(next, &mut reference, stream)?;
            next = self.argmax_on_device(logits, stream)?;
            expected.push(next);
        }
        for rows in 1..=6 {
            let trials = if matches!(rows, 3 | 5) { 1 } else { 4 };
            for trial in 0..trials {
                let mut temporary = self.benchmark_sequence(seq.slot_idx)?;
                self.forward_rows(prefix, &mut temporary, super::batch::BATCH, stream)?;
                let begin = std::time::Instant::now();
                let ptr = self.forward_rows(
                    &block[..rows],
                    &mut temporary,
                    super::batch::VERIFY,
                    stream,
                )?;
                let ms = begin.elapsed().as_secs_f64() * 1000.;
                tracing::info!(
                    "HY4_VERIFY_BENCH rows={rows} prefix={} trial={trial} warmup={} forward_ms={ms:.6}",
                    prefix.len(),
                    trial == 0
                );
                if trial + 1 == trials {
                    let actual = self.argmax_batch(ptr, rows, stream)?;
                    let first_divergence = actual.iter().zip(&expected).position(|(a, b)| a != b);
                    tracing::info!(
                        "HY4_VERIFY_PARITY rows={rows} inputs={:?} expected={:?} actual={actual:?} first_divergence={first_divergence:?}",
                        &block[..rows],
                        &expected[..rows]
                    );
                }
            }
        }
        self.benchmark_sequence(seq.slot_idx)?;
        Ok(())
    }

    fn benchmark_sequence(&self, slot: usize) -> Result<SequenceState> {
        let mut temporary = SequenceState::host_only(slot);
        self.free_sequence(&mut temporary)?;
        let fresh = self.alloc_sequence()?;
        ensure!(fresh.slot_idx == slot, "Hy4 benchmark slot changed");
        Ok(fresh)
    }
}
