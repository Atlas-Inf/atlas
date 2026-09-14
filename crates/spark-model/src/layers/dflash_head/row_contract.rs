// SPDX-License-Identifier: AGPL-3.0-only

use std::fmt;

use super::{LIGHTNING_NUM_DRAFTS, LIGHTNING_SERVED_GAMMA};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightningRowContract {
    gamma: usize,
    num_drafts: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsparkProposal {
    drafts: Vec<u32>,
    bonus: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitProjection {
    pub committed_tokens: Vec<u32>,
    pub new_seq_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DsparkRowError {
    ExactUsize {
        field: &'static str,
        expected: usize,
        found: usize,
    },
    Empty {
        field: &'static str,
    },
    Ragged {
        field: &'static str,
        row: usize,
        expected: usize,
        found: usize,
    },
    TokenOutOfRange {
        field: &'static str,
        token: u32,
        vocab: usize,
    },
    NonFinite {
        field: &'static str,
        row: usize,
        column: usize,
    },
    AcceptedPrefix {
        accepted: usize,
        max: usize,
    },
    LengthOverflow,
}

impl fmt::Display for DsparkRowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactUsize {
                field,
                expected,
                found,
            } => write!(
                f,
                "DSpark row contract rejected {field}={found}; expected {expected}"
            ),
            Self::Empty { field } => write!(f, "DSpark row contract rejected empty {field}"),
            Self::Ragged {
                field,
                row,
                expected,
                found,
            } => write!(
                f,
                "DSpark row contract rejected {field} row {row} width {found}; expected {expected}"
            ),
            Self::TokenOutOfRange {
                field,
                token,
                vocab,
            } => write!(
                f,
                "DSpark row contract rejected {field} token {token}; vocab is {vocab}"
            ),
            Self::NonFinite { field, row, column } => write!(
                f,
                "DSpark row contract rejected non-finite {field}[{row}][{column}]"
            ),
            Self::AcceptedPrefix { accepted, max } => write!(
                f,
                "DSpark row contract rejected accepted prefix {accepted}; maximum is {max}"
            ),
            Self::LengthOverflow => f.write_str("DSpark row contract sequence length overflow"),
        }
    }
}

impl std::error::Error for DsparkRowError {}

impl DsparkProposal {
    pub fn drafts(&self) -> &[u32] {
        &self.drafts
    }

    pub fn bonus(&self) -> u32 {
        self.bonus
    }
}

impl LightningRowContract {
    pub fn new(gamma: usize, num_drafts: usize) -> Result<Self, DsparkRowError> {
        exact("gamma", gamma, LIGHTNING_SERVED_GAMMA)?;
        exact("num_drafts", num_drafts, LIGHTNING_NUM_DRAFTS)?;
        Ok(Self { gamma, num_drafts })
    }

    pub fn query_rows(&self, last_target: u32, mask: u32) -> Vec<u32> {
        let mut rows = vec![mask; self.gamma];
        rows[0] = last_target;
        rows
    }

    pub fn markov_sample(
        &self,
        logits: &[Vec<f32>],
        w1: &[Vec<f32>],
        w2: &[Vec<f32>],
        last_target: u32,
    ) -> Result<Vec<u32>, DsparkRowError> {
        let vocab = rectangular("logits", logits, self.gamma)?;
        let rank = rectangular("markov_w1", w1, vocab)?;
        let w2_rank = rectangular("markov_w2", w2, vocab)?;
        exact("markov_w2 rank", w2_rank, rank)?;
        finite("logits", logits)?;
        finite("markov_w1", w1)?;
        finite("markov_w2", w2)?;
        token("last_target", last_target, vocab)?;

        let mut sampled = Vec::with_capacity(self.gamma);
        sampled.push(argmax(&logits[0]) as u32);
        let mut previous = last_target as usize;
        for row in 1..self.gamma {
            let mut scores = Vec::with_capacity(vocab);
            for candidate in 0..vocab {
                let mut bias = 0.0f32;
                for r in 0..rank {
                    let product = w1[previous][r] * w2[candidate][r];
                    if !product.is_finite() {
                        return Err(DsparkRowError::NonFinite {
                            field: "markov product",
                            row,
                            column: candidate,
                        });
                    }
                    bias += product;
                    if !bias.is_finite() {
                        return Err(DsparkRowError::NonFinite {
                            field: "markov bias",
                            row,
                            column: candidate,
                        });
                    }
                }
                let score = logits[row][candidate] + bias;
                if !score.is_finite() {
                    return Err(DsparkRowError::NonFinite {
                        field: "markov score",
                        row,
                        column: candidate,
                    });
                }
                scores.push(score);
            }
            let next = argmax(&scores);
            sampled.push(next as u32);
            previous = next;
        }
        Ok(sampled)
    }

    pub fn reorder_and_split(
        &self,
        sampled_rows: &[u32],
    ) -> Result<DsparkProposal, DsparkRowError> {
        exact("sampled_rows", sampled_rows.len(), self.gamma)?;
        let drafts = sampled_rows[1..].to_vec();
        exact("reordered drafts", drafts.len(), self.num_drafts)?;
        Ok(DsparkProposal {
            drafts,
            bonus: sampled_rows[0],
        })
    }

    pub fn verify_input(
        &self,
        last_target: u32,
        proposal: &DsparkProposal,
    ) -> Result<Vec<u32>, DsparkRowError> {
        exact("proposal drafts", proposal.drafts.len(), self.num_drafts)?;
        let mut input = Vec::with_capacity(self.gamma);
        input.push(last_target);
        input.extend_from_slice(&proposal.drafts);
        exact("verify input", input.len(), self.gamma)?;
        Ok(input)
    }

    pub fn project_commit(
        &self,
        pre_verify_len: usize,
        accepted: usize,
        proposal: &DsparkProposal,
        target_rows: &[u32],
    ) -> Result<CommitProjection, DsparkRowError> {
        exact("proposal drafts", proposal.drafts.len(), self.num_drafts)?;
        exact("target rows", target_rows.len(), self.gamma)?;
        if accepted > self.num_drafts {
            return Err(DsparkRowError::AcceptedPrefix {
                accepted,
                max: self.num_drafts,
            });
        }
        let mut committed_tokens = proposal.drafts[..accepted].to_vec();
        committed_tokens.push(target_rows[accepted]);
        let new_seq_len = pre_verify_len
            .checked_add(accepted)
            .and_then(|n| n.checked_add(1))
            .ok_or(DsparkRowError::LengthOverflow)?;
        Ok(CommitProjection {
            committed_tokens,
            new_seq_len,
        })
    }
}

fn exact(field: &'static str, found: usize, expected: usize) -> Result<(), DsparkRowError> {
    if found == expected {
        Ok(())
    } else {
        Err(DsparkRowError::ExactUsize {
            field,
            expected,
            found,
        })
    }
}

fn rectangular(
    field: &'static str,
    rows: &[Vec<f32>],
    expected_rows: usize,
) -> Result<usize, DsparkRowError> {
    exact(field, rows.len(), expected_rows)?;
    let width = rows.first().ok_or(DsparkRowError::Empty { field })?.len();
    if width == 0 {
        return Err(DsparkRowError::Empty { field });
    }
    for (row, values) in rows.iter().enumerate() {
        if values.len() != width {
            return Err(DsparkRowError::Ragged {
                field,
                row,
                expected: width,
                found: values.len(),
            });
        }
    }
    Ok(width)
}

fn finite(field: &'static str, rows: &[Vec<f32>]) -> Result<(), DsparkRowError> {
    for (row, values) in rows.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DsparkRowError::NonFinite { field, row, column });
            }
        }
    }
    Ok(())
}

fn token(field: &'static str, value: u32, vocab: usize) -> Result<(), DsparkRowError> {
    if (value as usize) < vocab {
        Ok(())
    } else {
        Err(DsparkRowError::TokenOutOfRange {
            field,
            token: value,
            vocab,
        })
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}
