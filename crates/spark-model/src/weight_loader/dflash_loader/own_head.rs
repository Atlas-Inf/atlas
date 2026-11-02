// SPDX-License-Identifier: AGPL-3.0-only

//! Own-lm_head policy for DFlash drafters (pure decision — GPU-free).
//! Split out of `dflash_loader.rs` for the 500-LoC file budget.

use anyhow::{Context, Result};
use spark_runtime::weights::{WeightDtype, WeightStore};

/// `has_own_lm_head != Some(true)` → share the target's head (existing
/// z-lab / incoai DFlash and DFlash2 pairings all omit the key). `Some(true)`
/// requires an NVFP4-packed `{prefix}lm_head.weight` (U8) plus `weight_scale`
/// and `weight_scale_2`; a missing tensor or any other dtype bails — serving
/// a shared head on a drafter trained with its own would silently cost
/// acceptance.
pub(crate) fn own_lm_head_present(
    store: &WeightStore,
    prefix: &str,
    has_own_lm_head: Option<bool>,
) -> Result<bool> {
    if has_own_lm_head != Some(true) {
        return Ok(false);
    }
    let weight_name = format!("{prefix}lm_head.weight");
    let meta = store.get(&weight_name).with_context(|| {
        format!("has_own_lm_head=true but {weight_name} is missing from the drafter checkpoint")
    })?;
    anyhow::ensure!(
        meta.dtype == WeightDtype::UInt8,
        "has_own_lm_head=true requires an NVFP4-packed (U8) {weight_name}, found {:?}",
        meta.dtype
    );
    for key in ["weight_scale", "weight_scale_2"] {
        anyhow::ensure!(
            store.contains(&format!("{prefix}lm_head.{key}")),
            "has_own_lm_head=true but {prefix}lm_head.{key} is missing"
        );
    }
    Ok(true)
}
