// SPDX-License-Identifier: AGPL-3.0-only

//! The end of a draft pass, on the host of the rank that serves the
//! request: the Markov head's chain over the block's rows, and how many of
//! the drafts are worth verifying (`DSparkBlock.forward_head` in the
//! release's `inference/model.py`).
//!
//! The reference adds the Markov head's bias to every logit of a row and
//! takes the largest. Here each rank keeps the few largest logits of its
//! share of the vocabulary and the chain is run over those: the bias moves
//! a token past the leaders only from near them. A draft that differs from
//! the reference's is one more the verify may refuse, never a wrong token.

use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::cache_recipe::bf16_to_f32;

/// The largest logits a rank keeps of each row.
pub const KEEP: usize = 8;

/// One kept logit: its value and its token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kept {
    pub logit: f32,
    pub token: u32,
}

/// The drafts of a pass and, for each, the confidence head's logit that it
/// is accepted given that the ones before it were.
#[derive(Debug, PartialEq)]
pub struct Chain {
    pub tokens: Vec<u32>,
    pub confidence: Vec<f32>,
}

/// The Markov and confidence heads, on the host.
pub struct HostHead {
    /// `[vocab, rank]` BF16 each: a token's embedding, and what each token's
    /// logit gains from the one before it.
    embed: Vec<u16>,
    head: Vec<u16>,
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
    /// The heads of the draft stage named `stage`, which the host then
    /// holds: the device's copies are let go.
    pub fn load(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        stage: &str,
        hidden: usize,
        vocab: usize,
        rank: usize,
    ) -> Result<Self> {
        let names = [
            "markov_head.embed",
            "markov_head.head",
            "confidence_head.proj",
        ]
        .map(|name| format!("{stage}.{name}.weight"));
        let embed = host_bf16(store, gpu, &names[0])?;
        let head = host_bf16(store, gpu, &names[1])?;
        let confidence = host_bf16(store, gpu, &names[2])?;
        ensure!(
            embed.len() == vocab * rank
                && head.len() == vocab * rank
                && confidence.len() == hidden + rank,
            "{stage}: Markov heads of {} and {} values and a confidence head of {}, for {vocab} tokens of rank {rank} and hidden {hidden}",
            embed.len(),
            head.len(),
            confidence.len()
        );
        for name in &names {
            store.reclaim(gpu, name)?;
        }
        Ok(Self::new(
            embed,
            head,
            rank,
            confidence.into_iter().map(bf16_to_f32).collect(),
            hidden,
        ))
    }

    pub fn new(
        embed: Vec<u16>,
        head: Vec<u16>,
        rank: usize,
        confidence: Vec<f32>,
        hidden: usize,
    ) -> Self {
        Self {
            embed,
            head,
            rank,
            confidence,
            hidden,
        }
    }

    fn row<'a>(&self, table: &'a [u16], token: u32) -> Option<&'a [u16]> {
        let at = token as usize * self.rank;
        table.get(at..at + self.rank)
    }

    /// The chain from `token`: row `i` of `kept` holds the largest logits of
    /// the block's row `i`, and `x` (`[rows, hidden]` BF16) the rows the
    /// logits were projected from, before their norm. A row whose logits
    /// name no token ends the chain.
    pub fn chain(&self, token: u32, kept: &[Vec<Kept>], x: &[u16]) -> Chain {
        let mut chain = Chain {
            tokens: Vec::with_capacity(kept.len()),
            confidence: Vec::with_capacity(kept.len()),
        };
        let mut last = token;
        for (row, kept) in kept.iter().enumerate() {
            let Some(from) = self.row(&self.embed, last) else {
                break;
            };
            let from: Vec<f32> = from.iter().copied().map(bf16_to_f32).collect();
            let bias = |token: u32| -> Option<f32> {
                let to = self.row(&self.head, token)?;
                Some(from.iter().zip(to).map(|(a, &b)| a * bf16_to_f32(b)).sum())
            };
            let best = kept
                .iter()
                .filter(|k| k.logit.is_finite())
                .filter_map(|k| Some((k.logit + bias(k.token)?, k.token)))
                .fold(None, |best: Option<(f32, u32)>, next| match best {
                    Some(best) if best.0 >= next.0 => Some(best),
                    _ => Some(next),
                });
            let (Some((_, next)), Some(x)) =
                (best, x.get(row * self.hidden..(row + 1) * self.hidden))
            else {
                break;
            };
            let (of_x, of_from) = self.confidence.split_at(self.hidden);
            let seen: f32 = x.iter().zip(of_x).map(|(&a, w)| bf16_to_f32(a) * w).sum();
            let said: f32 = from.iter().zip(of_from).map(|(a, w)| a * w).sum();
            chain.tokens.push(next);
            chain.confidence.push(seen + said);
            last = next;
        }
        chain
    }
}

/// How many of a chain's drafts to verify: while the chance that a draft
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
    use super::{Chain, HostHead, Kept, worth};

    /// BF16 bits of a value BF16 holds exactly.
    fn bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    /// Three tokens of rank 2 and hidden 2. Token 1 after token 0 gains 4.
    fn head() -> HostHead {
        let embed = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let head = [[0.0, 0.0], [4.0, 0.0], [0.0, 2.0]];
        let flat = |t: [[f32; 2]; 3]| t.iter().flatten().map(|&v| bf16(v)).collect();
        HostHead::new(flat(embed), flat(head), 2, vec![1.0, 0.0, 0.5, -3.0], 2)
    }

    fn kept(rows: &[&[(f32, u32)]]) -> Vec<Vec<Kept>> {
        rows.iter()
            .map(|row| {
                row.iter()
                    .map(|&(logit, token)| Kept { logit, token })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn the_token_before_moves_a_row_s_choice() {
        let x: Vec<u16> = [2.0, 9.0, 1.0, 9.0].map(bf16).to_vec();
        // Row 0 after token 0: token 1's logit 1 gains 4 and passes token
        // 2's 3. Row 1 after token 1: token 2 gains 2 and passes token 0.
        let rows = kept(&[&[(3.0, 2), (1.0, 1)], &[(5.0, 0), (4.0, 2)]]);
        assert_eq!(
            head().chain(0, &rows, &x),
            Chain {
                tokens: vec![1, 2],
                // x . [1, 0] and the token before's embedding . [0.5, -3].
                confidence: vec![2.0 + 0.5, 1.0 - 3.0],
            }
        );
    }

    #[test]
    fn a_row_with_nothing_kept_ends_the_chain() {
        let x: Vec<u16> = [0.0; 6].map(bf16).to_vec();
        let rows = kept(&[&[(1.0, 2)], &[(f32::NAN, 1)], &[(1.0, 0)]]);
        assert_eq!(head().chain(0, &rows, &x).tokens, vec![2]);
        // A token the heads do not hold is no candidate.
        let rows = kept(&[&[(9.0, 7), (1.0, 1)]]);
        assert_eq!(head().chain(0, &rows, &x).tokens, vec![1]);
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
