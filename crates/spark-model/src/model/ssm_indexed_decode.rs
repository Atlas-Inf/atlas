// SPDX-License-Identifier: AGPL-3.0-only

//! Validated per-rank indexed KDA metadata, refreshed before graph lookup.

use anyhow::{Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::{DECODE_META_MAX_ROWS, DECODE_META_MIN_ROWS, DecodeMetaLayout};
use spark_runtime::gpu::DevicePtr;

use super::ssm_pool::SsmStatePool;
use super::types::TransformerModel;
use crate::layer::ssm_batch::{SsmBatchView, SsmPoolView, checked_span};
use crate::layer::{LayerState, SsmLayerState};
use crate::traits::SequenceState;

/// Configuration is already TP-local, exactly as passed to Glm5KdaLayer.
pub(crate) fn runtime_eligible(
    config: &ModelConfig,
    world: usize,
    independent: bool,
    enabled: bool,
    rows: usize,
    padded_rows: usize,
) -> bool {
    runtime_eligible_up_to(config, world, independent, enabled, rows, padded_rows, 4)
}

fn runtime_eligible_up_to(
    config: &ModelConfig,
    world: usize,
    independent: bool,
    enabled: bool,
    rows: usize,
    padded_rows: usize,
    max_rows: usize,
) -> bool {
    enabled
        && independent
        && config.model_type == "glm5_next"
        && world == 2
        && config.tp_world_size == 2
        && config.ep_world_size == 2
        && (2..=max_rows).contains(&rows)
        && padded_rows == rows
        && config.linear_num_key_heads == 32
        && config.linear_num_value_heads == 32
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && config.linear_conv_kernel_dim == 4
}

/// Runs on both head and worker before graph lookup on every eligible step.
/// Unsupported execution scopes keep the old per-row path; malformed eligible
/// state never silently falls back. No stateful kernels are submitted here.
pub(crate) fn prepare_runtime<'a>(
    model: &'a TransformerModel,
    sequences: &[&mut SequenceState],
    rows: usize,
    padded_rows: usize,
    stream: u64,
) -> Result<Option<SsmBatchView<'a>>> {
    let independent_mode = super::glm_independent::enabled(&model.config.model_type)?;
    let selected_scope = independent_mode
        && runtime_eligible_up_to(
            &model.config,
            model.comm.as_ref().map_or(0, |c| c.world_size()),
            model.proposer.is_none() && !model.self_speculative,
            true,
            rows,
            padded_rows,
            8,
        );
    if !selected_scope
        && !runtime_eligible(
            &model.config,
            model.comm.as_ref().map_or(0, |c| c.world_size()),
            model.proposer.is_none() && !model.self_speculative,
            crate::layers::kda_multi_seq_enabled(),
            rows,
            padded_rows,
        )
    {
        return Ok(None);
    }
    ensure!(
        rows == sequences.len(),
        "indexed KDA token/state row count mismatch"
    );
    let scratch = model.buffers.scratch();
    let scratch_bytes = model.buffers.scratch_bytes();
    checked_span(scratch, scratch_bytes)?;
    let metadata_bytes = scratch_bytes
        .checked_sub(32768)
        .ok_or_else(|| anyhow::anyhow!("indexed KDA scratch lacks metadata region"))?;
    let refs: Vec<_> = sequences.iter().map(|seq| &**seq).collect();
    let kinds: Vec<_> = (0..model.layers.len())
        .map(|i| model.config.layer_type(i))
        .collect();
    let prepared = prepare_indexed_decode(
        Some(&model.ssm_pool),
        &refs,
        &kinds,
        scratch.offset(32768),
        metadata_bytes,
        model.buffers.decode_meta(),
    )?
    .ok_or_else(|| anyhow::anyhow!("indexed KDA live pool unexpectedly absent"))?;
    // Unlike the retained-copy API, this permits the local payload to drop
    // immediately after return. Stream ordering puts the upload before replay.
    model.gpu.copy_h2d_async(
        &prepared.slot_bytes,
        prepared.view.layer(0)?.slots(),
        stream,
    )?;
    Ok(Some(prepared.view))
}

/// Host-only upload payload plus stable device view. The caller must upload
/// `slot_bytes` at `view.layer(0)?.slots()` on every rank before future dispatch.
pub(crate) struct PreparedSsmBatch<'a> {
    pub(crate) view: SsmBatchView<'a>,
    pub(crate) slot_bytes: Vec<u8>,
}

pub(crate) fn ssm_ordinal(layer_types: &[LayerType], global_layer: usize) -> Result<usize> {
    ensure!(
        layer_types.get(global_layer) == Some(&LayerType::LinearAttention),
        "indexed SSM requested for a non-SSM layer"
    );
    Ok(layer_types[..global_layer]
        .iter()
        .filter(|&&kind| kind == LayerType::LinearAttention)
        .count())
}

/// `metadata_base` already points to the decode metadata block, not scratch[0].
/// Absence is the existing fallback; any present but inconsistent pool is an error.
pub(crate) fn prepare_indexed_decode<'a>(
    pool: Option<&'a SsmStatePool>,
    sequences: &[&SequenceState],
    layer_types: &[LayerType],
    metadata_base: DevicePtr,
    metadata_bytes: usize,
    layout: DecodeMetaLayout,
) -> Result<Option<PreparedSsmBatch<'a>>> {
    let Some(pool) = pool else { return Ok(None) };
    ensure!(
        pool.h_prefill_stage_pool.is_none(),
        "indexed SSM cannot use staged H storage"
    );
    ensure!(
        pool.num_ssm_layers == pool.h_state_pools.len(),
        "SSM pool layer count mismatch"
    );
    let view = SsmPoolView::new(
        &pool.h_state_pools,
        &pool.conv_state_pools,
        pool.h_bytes,
        pool.h_stored_bytes,
        pool.conv_bytes,
        pool.max_slots,
    )?;
    let rows: Vec<_> = sequences
        .iter()
        .map(|seq| SsmRow {
            slot: seq.ssm_slot_idx(),
            states: &seq.layer_states,
        })
        .collect();
    Ok(Some(prepare_rows(
        view,
        &rows,
        layer_types,
        metadata_base,
        metadata_bytes,
        layout,
    )?))
}

struct SsmRow<'a> {
    slot: Option<usize>,
    states: &'a [Box<dyn LayerState>],
}

fn prepare_rows<'a>(
    pool: SsmPoolView<'a>,
    rows: &[SsmRow<'_>],
    layer_types: &[LayerType],
    metadata_base: DevicePtr,
    metadata_bytes: usize,
    layout: DecodeMetaLayout,
) -> Result<PreparedSsmBatch<'a>> {
    ensure!(
        (1..=8).contains(&rows.len()),
        "indexed SSM supports one to eight rows"
    );
    ensure!(
        layer_types
            .iter()
            .filter(|&&kind| kind == LayerType::LinearAttention)
            .count()
            == pool.layer_count(),
        "SSM model layer map disagrees with pools"
    );
    let mut ids = Vec::with_capacity(rows.len());
    for row in rows {
        let slot = row
            .slot
            .ok_or_else(|| anyhow::anyhow!("live sequence lacks an SSM pool guard"))?;
        ids.push(i32::try_from(slot)?);
        ensure!(
            row.states.len() == layer_types.len(),
            "sequence layer-state count mismatch"
        );
        for (global, kind) in layer_types.iter().enumerate() {
            if *kind != LayerType::LinearAttention {
                continue;
            }
            let state = row.states[global]
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("missing live SSM layer state"))?;
            ensure!(
                !state.h_is_f16 && state.h_prefill_stage.is_none(),
                "indexed SSM requires live FP32 state"
            );
            let (h, conv) = pool.row_pointers(ssm_ordinal(layer_types, global)?, slot)?;
            ensure!(
                state.h_state == h && state.conv_state == conv,
                "live SSM pointers disagree with pool base, layer ordinal, or slot guard"
            );
        }
    }
    let (slots, slot_bytes) = slot_upload(metadata_base, metadata_bytes, layout, &ids)?;
    pool.validate_disjoint(slots, slot_bytes.len())?;
    let view = SsmBatchView::new(pool, slots, &ids)?;
    Ok(PreparedSsmBatch { view, slot_bytes })
}

fn slot_upload(
    metadata_base: DevicePtr,
    metadata_bytes: usize,
    layout: DecodeMetaLayout,
    ids: &[i32],
) -> Result<(DevicePtr, Vec<u8>)> {
    // Check the row ceiling before any offset calculation (the legacy layout
    // constructor intentionally does not impose policy or checked arithmetic).
    ensure!(
        (DECODE_META_MIN_ROWS..=DECODE_META_MAX_ROWS).contains(&layout.rows()),
        "unsupported SSM metadata row capacity"
    );
    ensure!(
        (1..=8).contains(&ids.len()) && ids.len() <= layout.rows(),
        "invalid active SSM row count"
    );
    checked_span(metadata_base, metadata_bytes)?;
    ensure!(
        metadata_bytes >= layout.block_table_off(),
        "SSM metadata region is truncated"
    );
    let slots = metadata_base.offset(layout.ssm_slots_off());
    let mut bytes = vec![255; layout.rows() * 4];
    for (target, id) in bytes.chunks_exact_mut(4).zip(ids) {
        target.copy_from_slice(&id.to_le_bytes());
    }
    Ok((slots, bytes))
}

#[cfg(test)]
#[path = "ssm_indexed_decode_tests.rs"]
mod tests;
