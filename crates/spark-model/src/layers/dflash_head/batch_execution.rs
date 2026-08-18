// SPDX-License-Identifier: AGPL-3.0-only

//! Pure `[B, gamma]` execution-plan projections derived from validated inputs.

use super::batch_inputs::{DsparkBatchInput, DsparkBatchInputError};

/// Build `[sequence][gamma]` physical cache slots from owner-local block tables.
/// `None` means lazy block allocation is not ready yet; callers must not launch.
pub(crate) fn paged_slot_mapping(
    block_tables: &[Vec<u32>],
    ctx_counts: &[usize],
    gamma: usize,
    block_size: usize,
) -> Result<Option<Vec<i64>>, DsparkBatchInputError> {
    if block_tables.len() != ctx_counts.len() {
        return Err(DsparkBatchInputError::LengthMismatch {
            field: "ctx_counts",
            expected: block_tables.len(),
            found: ctx_counts.len(),
        });
    }
    let mut slots = Vec::with_capacity(block_tables.len().saturating_mul(gamma));
    for (table, &ctx_count) in block_tables.iter().zip(ctx_counts.iter()) {
        for query in 0..gamma {
            let logical =
                ctx_count
                    .checked_add(query)
                    .ok_or(DsparkBatchInputError::ArithmeticOverflow {
                        operation: "paged slot position",
                        lhs: ctx_count,
                        rhs: query,
                    })?;
            let logical_block = logical / block_size;
            let offset = logical % block_size;
            let Some(&physical_block) = table.get(logical_block) else {
                return Ok(None);
            };
            let slot = usize::try_from(physical_block)
                .ok()
                .and_then(|block| block.checked_mul(block_size))
                .and_then(|base| base.checked_add(offset))
                .ok_or(DsparkBatchInputError::ArithmeticOverflow {
                    operation: "physical paged slot",
                    lhs: physical_block as usize,
                    rhs: block_size,
                })?;
            slots.push(i64::try_from(slot).map_err(|_| {
                DsparkBatchInputError::ArithmeticOverflow {
                    operation: "physical paged slot i64",
                    lhs: slot,
                    rhs: i64::MAX as usize,
                }
            })?);
        }
    }
    Ok(Some(slots))
}

impl DsparkBatchInput {
    /// Pack query IDs as `[sequence][gamma]`: anchor followed by mask rows.
    pub fn packed_query_tokens(&self, mask_token: u32) -> Vec<u32> {
        let mut packed = Vec::with_capacity(self.total_rows());
        for sequence in 0..self.batch_len() {
            packed.push(self.sequence(sequence).last_token);
            packed.extend(std::iter::repeat_n(mask_token, self.gamma() - 1));
        }
        packed
    }

    /// Pack absolute query positions in the same `[sequence][gamma]` order.
    pub fn packed_positions(&self) -> Result<Vec<u32>, DsparkBatchInputError> {
        let mut positions = Vec::with_capacity(self.total_rows());
        for sequence in 0..self.batch_len() {
            let base = self.sequence(sequence).absolute_position;
            for query in 0..self.gamma() {
                let position =
                    base.checked_add(query)
                        .ok_or(DsparkBatchInputError::ArithmeticOverflow {
                            operation: "packed position",
                            lhs: base,
                            rhs: query,
                        })?;
                positions.push(u32::try_from(position).map_err(|_| {
                    DsparkBatchInputError::ArithmeticOverflow {
                        operation: "packed position u32",
                        lhs: position,
                        rhs: u32::MAX as usize,
                    }
                })?);
            }
        }
        Ok(positions)
    }

    /// Rows for one query depth across every sequence. This is the batch-wide
    /// execution order for a depth-serial Markov step.
    pub fn rows_at_query(&self, query: usize) -> Result<Vec<usize>, DsparkBatchInputError> {
        if query >= self.gamma() {
            return Err(DsparkBatchInputError::QueryOutOfBounds {
                query,
                gamma: self.gamma(),
            });
        }
        (0..self.batch_len())
            .map(|sequence| self.row_index(sequence, query))
            .collect()
    }

    /// Convert row-major `[anchor, mask1, mask2, mask3]` into Lightning return
    /// order `[mask1, mask2, mask3, anchor]` for every sequence.
    pub fn reorder_sampled_rows(
        &self,
        sampled_rows: &[u32],
    ) -> Result<Vec<Vec<u32>>, DsparkBatchInputError> {
        if sampled_rows.len() != self.total_rows() {
            return Err(DsparkBatchInputError::LengthMismatch {
                field: "sampled_rows",
                expected: self.total_rows(),
                found: sampled_rows.len(),
            });
        }
        let mut output = Vec::with_capacity(self.batch_len());
        for sequence in 0..self.batch_len() {
            let rows = self.sequence_row_range(sequence)?;
            let mut reordered = sampled_rows[rows.start + 1..rows.end].to_vec();
            reordered.push(sampled_rows[rows.start]);
            output.push(reordered);
        }
        Ok(output)
    }
}
