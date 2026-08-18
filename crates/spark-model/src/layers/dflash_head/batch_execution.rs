// SPDX-License-Identifier: AGPL-3.0-only

//! Pure `[B, gamma]` execution-plan projections derived from validated inputs.

use super::batch_inputs::{DsparkBatchInput, DsparkBatchInputError};

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
