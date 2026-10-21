// SPDX-License-Identifier: AGPL-3.0-only

//! Shared pieces of the FP4-Local MXFP8 layer builders (`gdn_mx8`,
//! `attn_mx8`): the checked weight/scale lookup and the one BF16 prefill
//! scratch every MXFP8 layer expands into.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::ops::Mxfp8Scratch;

/// Inputs per E8M0 scale.
pub(super) const MX8_GROUP: usize = 32;

/// The weight and scale buffers of one MXFP8 module, shapes checked.
pub(super) fn module(
    store: &WeightStore,
    name: &str,
    rows: usize,
    k: usize,
) -> Result<[DevicePtr; 2]> {
    let w = store.get(&format!("{name}.weight"))?;
    let s = store.get(&format!("{name}.weight_scale"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3
            && w.shape == [rows, k]
            && s.dtype == WeightDtype::UInt8
            && s.shape == [rows, k / MX8_GROUP],
        "{name}: expected MXFP8 [{rows}, {k}] + [{rows}, {}], got {:?} {:?} + {:?} {:?}",
        k / MX8_GROUP,
        w.dtype,
        w.shape,
        s.dtype,
        s.shape
    );
    Ok([w.ptr, s.ptr])
}

/// The prefill scratch every MXFP8 layer shares, sized for the largest
/// projection either layer kind holds (GDN `[Q|K|V|Z]` and `out_proj`,
/// attention `q` (with its gate) and `o`) and allocated by the first layer
/// that asks. Prefill runs layers one after another on one stream, so one
/// buffer serves them all.
pub(super) fn scratch_for(
    shared: &mut Option<Mxfp8Scratch>,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Mxfp8Scratch> {
    if let Some(s) = *shared {
        return Ok(s);
    }
    let h = config.hidden_size;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let q_dim = config.num_attention_heads * config.head_dim;
    let q_rows = if config.attn_gated { 2 * q_dim } else { q_dim };
    let elems = [
        config.ssm_qkvz_size() * h,
        h * value_dim,
        q_rows * h,
        h * q_dim,
    ];
    let bytes = elems.into_iter().max().unwrap_or(0) * 2;
    ensure!(bytes > 0, "MXFP8 scratch: the config has no projection");
    let s = Mxfp8Scratch {
        ptr: gpu.alloc(bytes)?,
        bytes,
    };
    tracing::info!(
        "FP4-Local MXFP8: one {:.1} MB BF16 prefill scratch shared by every MXFP8 layer",
        bytes as f64 / 1e6
    );
    *shared = Some(s);
    Ok(s)
}
