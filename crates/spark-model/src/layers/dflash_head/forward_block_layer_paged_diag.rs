// SPDX-License-Identifier: AGPL-3.0-only
//! Per-propose paged-attention input dump for the Option B diagnostic
//! toggles. The γ=16 second-sequence fault (job 095) named
//! `prefill_attention_paged_dflash_bf16_indirect` as the failing launch;
//! that kernel bounds-checks every read, so the suspect is its inputs.
//! This dumps the whole block table, the 12-byte indirect
//! `(kv_len, q_offset, q_rope_pos)` triple, and the pool/q_buf pointers.
//! Diagnostic only: every call synchronizes the stream and D2Hs.

use spark_runtime::gpu::DevicePtr;

use super::forward_block_layer_paged::PagedLayerArgs;
use super::{BlockDiffusionDraftHead, DflashScratch};
use crate::layer::ForwardContext;
use anyhow::Result;

impl BlockDiffusionDraftHead {
    /// Read back every input the paged-indirect attention launch will
    /// consume and log them on one line. `block_table_len` bounds the
    /// table read — never read past it.
    pub(super) fn option_b_diag_paged_inputs(
        &self,
        args: &PagedLayerArgs,
        ctx: &ForwardContext,
        scratch: &DflashScratch,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let mut bt_full = vec![0u8; args.block_table_len as usize * 4];
        gpu.copy_d2h(args.block_table_dev, &mut bt_full)?;
        let bt_all: Vec<u32> = bt_full
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut ind_bytes = [0u8; 12];
        gpu.copy_d2h(scratch.option_b_indirect_args_dev, &mut ind_bytes)?;
        let indirect: Vec<u32> = ind_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n_bt = bt_all.len();
        tracing::info!(
            "DFLASH OPTION_B DIAG: ptrs k_pool={:#x} v_pool={:#x} q_buf={:#x} \
             block_table_dev={:#x} bt[0..8]={:?} bt[last4]={:?} n_bt={} \
             indirect(kv_len,q_offset,q_rope_pos)={:?} gamma={} num_kv_heads={} head_dim={}",
            k_pool.0,
            v_pool.0,
            scratch.q_buf.0,
            args.block_table_dev.0,
            &bt_all[..8.min(n_bt)],
            &bt_all[n_bt.saturating_sub(4)..],
            n_bt,
            indirect,
            self.gamma,
            self.num_kv_heads,
            self.head_dim,
        );
        Ok(())
    }
}
