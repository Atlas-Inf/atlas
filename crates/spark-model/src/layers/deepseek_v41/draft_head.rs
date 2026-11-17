// SPDX-License-Identifier: AGPL-3.0-only

//! The end of a draft pass, on the host of the rank that serves the
//! request: the confidence head's reading of each draft, and how many of
//! them are worth verifying (`DSparkBlock.forward_head` in the release's
//! `inference/model.py`). The drafts themselves are chosen on the device
//! (`dsv41_markov_round`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::cache_recipe::bf16_to_f32;

/// The confidence head, and the Markov embeddings it reads, on the host.
pub struct HostHead {
    /// `[vocab, rank]` BF16: a token's Markov embedding.
    embed: Vec<u16>,
    rank: usize,
    /// `[hidden + rank]`: the confidence head's one row.
    confidence: Vec<f32>,
    hidden: usize,
}

fn host_bf16(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<Vec<u16>> {
    let t = store.get(name)?;
    ensure!(
        t.dtype == WeightDtype::BF16,
        "{name}: {:?}, expected BF16",
        t.dtype
    );
    let mut bytes = vec![0u8; t.num_elements() * 2];
    gpu.copy_d2h(t.ptr, &mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

impl HostHead {
    /// The heads of the draft stage named `stage`.
    pub fn load(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        stage: &str,
        hidden: usize,
        vocab: usize,
        rank: usize,
    ) -> Result<Self> {
        let embed = host_bf16(store, gpu, &format!("{stage}.markov_head.embed.weight"))?;
        let name = format!("{stage}.confidence_head.proj.weight");
        let confidence = host_bf16(store, gpu, &name)?;
        ensure!(
            embed.len() == vocab * rank && confidence.len() == hidden + rank,
            "{stage}: Markov embeddings of {} values and a confidence head of {}, for {vocab} tokens of rank {rank} and hidden {hidden}",
            embed.len(),
            confidence.len()
        );
        // Nothing reads the confidence head on the device.
        store.reclaim(gpu, &name)?;
        Ok(Self::new(
            embed,
            rank,
            confidence.into_iter().map(bf16_to_f32).collect(),
            hidden,
        ))
    }

    pub fn new(embed: Vec<u16>, rank: usize, confidence: Vec<f32>, hidden: usize) -> Self {
        Self {
            embed,
            rank,
            confidence,
            hidden,
        }
    }

    /// For each of `drafts`, the confidence head's logit that it is accepted
    /// given that the ones before it were: from the row the draft was read
    /// from (`x`, `[rows, hidden]` BF16, before the norm) and the Markov
    /// embedding of the token before it, which for the first is `token`.
    /// Shorter than `drafts` where a token is not the head's or `x` ends.
    pub fn confidence(&self, token: u32, drafts: &[u32], x: &[u16]) -> Vec<f32> {
        let (of_x, of_before) = self.confidence.split_at(self.hidden);
        let before = std::iter::once(token).chain(drafts.iter().copied());
        before
            .zip(x.chunks_exact(self.hidden))
            .take(drafts.len())
            .map_while(|(before, x)| {
                let at = before as usize * self.rank;
                let embed = self.embed.get(at..at + self.rank)?;
                let seen: f32 = x.iter().zip(of_x).map(|(&a, w)| bf16_to_f32(a) * w).sum();
                let said: f32 = embed
                    .iter()
                    .zip(of_before)
                    .map(|(&a, w)| bf16_to_f32(a) * w)
                    .sum();
                Some(seen + said)
            })
            .collect()
    }
}

/// How many of a pass's drafts to verify: while the chance that a draft
/// and all before it are accepted stays at `floor` or above, and at most
/// `most`. A verified row costs a row of the model, and a refused one buys
/// nothing.
pub fn worth(confidence: &[f32], floor: f32, most: usize) -> usize {
    let mut survives = 1.0f32;
    confidence
        .iter()
        .take(most)
        .take_while(|&&logit| {
            survives *= 1.0 / (1.0 + (-logit).exp());
            survives >= floor
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::{HostHead, worth};

    /// BF16 bits of a value BF16 holds exactly.
    fn bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// Three tokens of rank 2 and hidden 2.
    fn head() -> HostHead {
        let embed = [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let flat = embed.iter().flatten().map(|&v| bf16(v)).collect();
        HostHead::new(flat, 2, vec![1.0, 0.0, 0.5, -3.0], 2)
    }

    #[test]
    fn a_drafts_confidence_reads_its_row_and_the_token_before() {
        let x: Vec<u16> = [2.0, 9.0, 1.0, 9.0, 4.0, 9.0].map(bf16).to_vec();
        // x . [1, 0], and the embedding of the token before . [0.5, -3]:
        // token 0 before the first draft, then drafts 1 and 2.
        assert_eq!(
            head().confidence(0, &[1, 2, 0], &x),
            vec![2.0 + 0.5, 1.0 - 3.0, 4.0 + 1.0 - 6.0]
        );
    }

    #[test]
    fn confidence_ends_where_a_token_or_a_row_is_missing() {
        let x: Vec<u16> = [0.0; 4].map(bf16).to_vec();
        // Two rows of x for three drafts.
        assert_eq!(head().confidence(0, &[1, 2, 0], &x).len(), 2);
        // The second draft follows a token the head does not hold.
        assert_eq!(head().confidence(0, &[9, 1], &x).len(), 1);
        assert!(head().confidence(9, &[1], &x).is_empty());
    }

    #[test]
    fn drafts_are_verified_while_they_are_likely_to_stand() {
        // Chances 0.88, 0.73, 0.5: together 0.88, 0.64, 0.32.
        let confidence = [2.0, 1.0, 0.0, 5.0];
        assert_eq!(worth(&confidence, 0.5, 5), 2);
        assert_eq!(worth(&confidence, 0.3, 5), 4);
        assert_eq!(worth(&confidence, 0.3, 3), 3);
        assert_eq!(worth(&confidence, 0.9, 5), 0);
        assert_eq!(worth(&[], 0.1, 5), 0);
    }
}
