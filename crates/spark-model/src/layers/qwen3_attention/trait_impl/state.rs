// SPDX-License-Identifier: AGPL-3.0-only
//! Attention auxiliary-state release; mechanically extracted from the trait impl.
use super::*;

pub(super) fn free_attention_state(
    layer: &Qwen3AttentionLayer,
    gpu: &dyn GpuBackend,
    state: &mut dyn LayerState,
) -> Result<()> {
    let Some(qsa) = layer.qsa.as_ref() else {
        return Ok(());
    };
    let Some(attn) = state
        .as_any_mut()
        .downcast_mut::<crate::layer::AttnLayerState>()
    else {
        return Ok(());
    };
    if let Some(st) = attn.qsa.as_mut() {
        qsa.free_seq_state(st, gpu)?;
    }
    Ok(())
}
