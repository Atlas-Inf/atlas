// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_staged_attention(
        &self,
        _layer_idx: usize,
        batch_size: u32,
        max_kv_len: u32,
        serial_block_tables: Option<&[u64]>,
        serial_attention_args: Option<DevicePtr>,
        sinks: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let (Some(tables), Some(args_base)) = (serial_block_tables, serial_attention_args) {
            anyhow::ensure!(
                tables.len() == batch_size as usize,
                "DFlash serial attention table width {} != batch {}",
                tables.len(),
                batch_size
            );
            let q_row_bytes = self
                .gamma
                .checked_mul(self.num_q_heads * self.head_dim)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("DFlash serial attention row offset overflow"))?;
            for sequence in 0..batch_size as usize {
                crate::layers::ops::prefill_attention_paged_dflash_bf16_indirect(
                    ctx.gpu,
                    self.kernels.prefill_attn_dflash_bf16_indirect,
                    self.batch_q.offset(sequence * q_row_bytes),
                    k_pool,
                    v_pool,
                    self.batch_attn_out.offset(sequence * q_row_bytes),
                    DevicePtr(tables[sequence]),
                    self.gamma as u32,
                    args_base.offset(sequence * 12),
                    self.num_q_heads as u32,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    16,
                    self.attn_sliding_window(),
                    self.attn_causal(),
                    1.0 / (self.head_dim as f32).sqrt(),
                    sinks,
                    stream,
                )?;
            }
            return Ok(());
        }
        crate::layers::ops::prefill_attention_paged_batched_sink(
            ctx.gpu,
            self.kernels.prefill_attn_dflash_bf16_batched_sink,
            self.batch_q,
            k_pool,
            v_pool,
            self.batch_attn_out,
            self.batch_block_table_ptrs,
            batch_size,
            self.batch_cu_seqlens,
            self.batch_kv_lens,
            self.gamma as u32,
            max_kv_len,
            0,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            16,
            self.attn_sliding_window(),
            1.0 / (self.head_dim as f32).sqrt(),
            sinks,
            stream,
        )
    }
}
