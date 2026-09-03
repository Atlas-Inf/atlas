// SPDX-License-Identifier: AGPL-3.0-only

use super::super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, LayerState, TransformerLayer};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

pub(super) fn nemotron_attention_serial(rope_disabled: bool, value: Option<&str>) -> bool {
    rope_disabled && matches!(value, Some("1"))
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_nemotron_attention_serial<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !nemotron_attention_serial(
            self.rope_disabled,
            std::env::var("ATLAS_LIGHTNING_ATTN_SERIAL").ok().as_deref(),
        ) {
            return Ok(false);
        }
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        for i in 0..num_seqs {
            let mut block_table = block_tables[i].clone();
            let mut disk_blocks = Vec::new();
            let mut disk_offsets = Vec::new();
            self.decode(
                hidden.offset(i * h * bf16),
                residual.offset(i * h * bf16),
                states[i],
                kv_cache,
                seq_lens[i],
                &mut block_table,
                &mut disk_blocks,
                &mut disk_offsets,
                ctx,
                stream,
            )?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::nemotron_attention_serial;

    #[test]
    fn serial_attention_is_exact_opt_in_for_rope_disabled_layers() {
        assert!(!nemotron_attention_serial(false, None));
        assert!(!nemotron_attention_serial(false, Some("1")));
        assert!(!nemotron_attention_serial(true, None));
        assert!(!nemotron_attention_serial(true, Some("0")));
        assert!(!nemotron_attention_serial(true, Some("true")));
        assert!(nemotron_attention_serial(true, Some("1")));
    }
}
