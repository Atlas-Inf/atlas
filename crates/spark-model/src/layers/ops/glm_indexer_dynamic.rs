// SPDX-License-Identifier: AGPL-3.0-only

//! Fixed-capacity, device-length GLM C2/C3 semantic-index launch contracts.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const GLM_DYNAMIC_TOPK: u32 = 2048;
pub const GLM_DYNAMIC_WIDTH: u32 = GLM_DYNAMIC_TOPK + 3;
pub const GLM_DYNAMIC_DENSE_OFFSET: usize = GLM_DYNAMIC_WIDTH as usize * 4;
pub const GLM_DYNAMIC_SELECTED_BYTES: usize = GLM_DYNAMIC_DENSE_OFFSET + 4;

/// Process-fixed table geometry, never derived from a capture-time position.
#[derive(Clone, Copy, Debug)]
pub struct GlmDynamicShape {
    capacity_tokens: u32,
    logits_stride: u32,
    block_size: u32,
}

impl GlmDynamicShape {
    pub fn new(max_blocks_per_seq: u32, block_size: u32) -> Result<Self> {
        ensure!(
            max_blocks_per_seq > 0 && block_size > 0 && block_size.is_multiple_of(4),
            "GLM dynamic index requires nonempty cache blocks containing whole four-token pools"
        );
        let capacity_tokens = max_blocks_per_seq
            .checked_mul(block_size)
            .ok_or_else(|| anyhow::anyhow!("GLM dynamic index table capacity overflows u32"))?;
        Ok(Self {
            capacity_tokens,
            logits_stride: capacity_tokens / 4,
            block_size,
        })
    }

    fn score_grid(self) -> [u32; 3] {
        [self.logits_stride.div_ceil(8), 1, 1]
    }

    pub fn score_bytes(self) -> usize {
        self.logits_stride as usize * 4
    }

    pub fn validate_arenas(self, score_bytes: usize, selected_bytes: usize) -> Result<()> {
        ensure!(
            score_bytes >= self.score_bytes(),
            "GLM dynamic score arena cannot hold fixed table capacity"
        );
        ensure!(
            selected_bytes >= GLM_DYNAMIC_SELECTED_BYTES,
            "GLM dynamic selection arena needs 2051 IDs plus dense length"
        );
        Ok(())
    }

    /// Called outside graph capture/replay every step, on both EP ranks.
    pub fn validate_positions(
        self,
        positions: impl IntoIterator<Item = usize>,
        rows: usize,
        model_limit: usize,
    ) -> Result<()> {
        // Rows are indexed one at a time through the shared selection arena,
        // so the width is bounded only by the widest admitted verify block.
        ensure!(
            (2..=crate::speculative::glm_repair_policy::MAX_DFLASH_VERIFY_ROWS).contains(&rows),
            "GLM dynamic index requires 2..=8 verify rows"
        );
        let mut count = 0;
        for position in positions {
            ensure!(
                position < model_limit && position < self.capacity_tokens as usize,
                "GLM dynamic index position exceeds model or fixed table capacity"
            );
            count += 1;
        }
        ensure!(
            count == rows,
            "GLM dynamic index host-position count mismatch"
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_logits_dynamic(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    query: DevicePtr,
    weights: DevicePtr,
    index_cache: DevicePtr,
    logits: DevicePtr,
    block_table: DevicePtr,
    seq_len: DevicePtr,
    shape: GlmDynamicShape,
    index_block_stride_bytes: u64,
    stream: u64,
) -> Result<()> {
    ensure!(!seq_len.is_null(), "GLM dynamic scorer needs device length");
    KernelLaunch::new(gpu, kernel)
        .grid(shape.score_grid())
        .block([256, 1, 1])
        .arg_ptr(query)
        .arg_ptr(weights)
        .arg_ptr(index_cache)
        .arg_ptr(logits)
        .arg_ptr(block_table)
        .arg_u32(1)
        .arg_ptr(seq_len)
        .arg_u32(shape.logits_stride)
        .arg_u32(32)
        .arg_u32(128)
        .arg_u32(4)
        .arg_u32(shape.block_size)
        .arg_u64(index_block_stride_bytes)
        .arg_u32(GLM_DYNAMIC_TOPK)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_topk_expand_dynamic(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    output: DevicePtr,
    seq_len: DevicePtr,
    dense_seq_len: DevicePtr,
    shape: GlmDynamicShape,
    stream: u64,
) -> Result<()> {
    ensure!(
        !seq_len.is_null() && !dense_seq_len.is_null(),
        "GLM dynamic top-k needs device lengths"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .shared_mem(16)
        .arg_ptr(logits)
        .arg_ptr(output)
        .arg_u32(1)
        .arg_ptr(seq_len)
        .arg_u32(shape.logits_stride)
        .arg_u32(GLM_DYNAMIC_TOPK)
        .arg_u32(4)
        .arg_u32(GLM_DYNAMIC_WIDTH)
        .arg_ptr(dense_seq_len)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_sparse_mla_dynamic(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    query: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    token_indices: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    seq_len: DevicePtr,
    num_heads: u32,
    shape: GlmDynamicShape,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        matches!(num_heads, 32 | 64) && !seq_len.is_null(),
        "GLM dynamic sparse attention geometry is unsupported"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, 1, 1])
        .block([256, 1, 1])
        .shared_mem(19 * 4)
        .arg_ptr(query)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(token_indices)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(1)
        .arg_u32(num_heads)
        .arg_u32(512)
        .arg_u32(GLM_DYNAMIC_WIDTH)
        .arg_u32(shape.block_size)
        .arg_f32(inv_sqrt_d)
        .arg_ptr(seq_len)
        .arg_u32(GLM_DYNAMIC_TOPK)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_capacity_covers_table_padding_without_current_length() {
        let shape = GlmDynamicShape::new(1025, 16).unwrap();
        assert_eq!(shape.capacity_tokens, 16400);
        assert_eq!(shape.logits_stride, 4100);
        assert_eq!(shape.score_grid(), [513, 1, 1]);
        assert_eq!(shape.score_bytes(), 16400);
        assert_eq!(GLM_DYNAMIC_DENSE_OFFSET, 8204);
        assert_eq!(GLM_DYNAMIC_SELECTED_BYTES, 8208);
        for positions in [&[0, 2047, 2048][..], &[2050, 16383, 15][..]] {
            shape
                .validate_positions(positions.iter().copied(), 3, 16384)
                .unwrap();
            assert_eq!(shape.score_grid(), [513, 1, 1]);
        }
    }

    #[test]
    fn invalid_capacity_and_replay_positions_fail_before_launch() {
        for (blocks, bs) in [(0, 16), (1, 0), (1, 3), (1, 18), (u32::MAX, 16)] {
            assert!(GlmDynamicShape::new(blocks, bs).is_err());
        }
        let shape = GlmDynamicShape::new(128, 16).unwrap();
        for (positions, rows, limit) in [
            (vec![0], 1, 2048),
            (vec![0, 1], 3, 2048),
            ((0..9).collect(), 9, 2048),
            (vec![2048, 0], 2, 4096),
            (vec![1024, 0], 2, 1024),
            (vec![usize::MAX, 0], 2, usize::MAX),
        ] {
            assert!(shape.validate_positions(positions, rows, limit).is_err());
        }
        // A DFlash verify block of up to eight causal rows is admitted.
        assert!(shape.validate_positions(2040..2048, 8, 4096).is_ok());
    }

    #[test]
    fn fixed_arenas_reject_capture_time_shortcuts() {
        let shape = GlmDynamicShape::new(1025, 16).unwrap();
        assert!(shape.validate_arenas(16400, 8208).is_ok());
        assert!(shape.validate_arenas(16399, 8208).is_err());
        assert!(shape.validate_arenas(16400, 8207).is_err());
        assert!(shape.validate_arenas(2048, 8208).is_err());
    }
}
