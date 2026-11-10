// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash2 grouped dynamic convolution weights and their geometry.
//!
//! The geometry comes from the tensors (`base_kernel [2, taps, hidden]` —
//! the input-side and output-side static kernels — and `kernel_projection
//! [2 * taps * num_groups, hidden]`), and the checkpoint's
//! `dflash_config.conv_kernel_size` / `conv_group_size`, when declared, must
//! agree with it: a launch sized for the wrong tap count reads the wrong
//! coefficients without failing.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::DflashConfig;
use crate::weight_map::{DenseWeight, dense_auto};

/// DFlash2 grouped dynamic convolution weights.
#[derive(Clone)]
pub struct DflashConvWeights {
    pub base_kernel: DenseWeight,
    pub kernel_projection: DenseWeight,
    /// Taps (`conv_kernel_size`): 2 on Qwen3.8 / GLM-5.3, 4 on Ripple-65B.
    pub kernel_size: usize,
    /// Channels sharing one dynamic coefficient (`conv_group_size`).
    pub group_size: usize,
}

/// `(kernel_size, group_size)` of a conv whose tensors have these shapes.
pub(crate) fn conv_geometry(
    base: &[usize],
    projection: &[usize],
    hidden: usize,
    config: &DflashConfig,
) -> Result<(usize, usize)> {
    ensure!(
        base.len() == 3 && base[0] == 2 && base[1] > 0 && base[2] == hidden,
        "DFlash2 conv base_kernel shape {base:?} is not [2, taps, {hidden}]"
    );
    let taps = base[1];
    ensure!(
        projection.len() == 2 && projection[1] == hidden && projection[0].is_multiple_of(2 * taps),
        "DFlash2 conv kernel_projection shape {projection:?} is not [2 * {taps} * groups, {hidden}]"
    );
    let groups = projection[0] / (2 * taps);
    ensure!(
        groups > 0 && hidden.is_multiple_of(groups),
        "DFlash2 conv: {groups} groups do not divide hidden {hidden}"
    );
    let group_size = hidden / groups;
    let sub = config.dflash_config.as_ref();
    if let Some(k) = sub.and_then(|c| c.conv_kernel_size) {
        ensure!(
            k == taps,
            "DFlash2 conv_kernel_size {k} != base_kernel taps {taps}"
        );
    }
    if let Some(g) = sub.and_then(|c| c.conv_group_size) {
        ensure!(
            g == group_size,
            "DFlash2 conv_group_size {g} != tensor group size {group_size}"
        );
    }
    Ok((taps, group_size))
}

/// The conv at `{name}.base_kernel` / `{name}.kernel_projection.weight`, or
/// `None` when the layer ships none.
pub(super) fn load_conv(
    store: &WeightStore,
    name: &str,
    config: &DflashConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<DflashConvWeights>> {
    let base = format!("{name}.base_kernel");
    let projection = format!("{name}.kernel_projection.weight");
    if !(store.contains(&base) && store.contains(&projection)) {
        return Ok(None);
    }
    let (kernel_size, group_size) = conv_geometry(
        &store.get(&base)?.shape,
        &store.get(&projection)?.shape,
        config.hidden_size,
        config,
    )
    .with_context(|| format!("DFlash2 conv {name}"))?;
    Ok(Some(DflashConvWeights {
        base_kernel: dense_auto(store, &base, gpu)?,
        kernel_projection: dense_auto(store, &projection, gpu)?,
        kernel_size,
        group_size,
    }))
}

#[cfg(test)]
mod tests {
    use super::conv_geometry;

    fn config(taps: Option<usize>, group: Option<usize>) -> super::DflashConfig {
        let mut v = serde_json::json!({
            "hidden_size": 5120, "num_hidden_layers": 5, "intermediate_size": 40960,
            "num_attention_heads": 80, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 100352,
            "dflash_config": {"mask_token_id": 0, "target_layer_ids": [1]}
        });
        if let Some(t) = taps {
            v["dflash_config"]["conv_kernel_size"] = t.into();
        }
        if let Some(g) = group {
            v["dflash_config"]["conv_group_size"] = g.into();
        }
        super::super::parse_dflash_config(&v.to_string()).unwrap()
    }

    #[test]
    fn ripple_and_qwen38_geometry() {
        // Ripple-65B: base_kernel [2, 4, 5120], kernel_projection [2560, 5120].
        let c = config(Some(4), Some(16));
        assert_eq!(
            conv_geometry(&[2, 4, 5120], &[2560, 5120], 5120, &c).unwrap(),
            (4, 16)
        );
        // Qwen3.8-27B: two taps, 320 groups.
        let c = config(Some(2), Some(16));
        assert_eq!(
            conv_geometry(&[2, 2, 5120], &[1280, 5120], 5120, &c).unwrap(),
            (2, 16)
        );
        let undeclared = config(None, None);
        assert_eq!(
            conv_geometry(&[2, 4, 5120], &[2560, 5120], 5120, &undeclared).unwrap(),
            (4, 16)
        );
    }

    #[test]
    fn declared_geometry_must_match_tensors() {
        let c = config(Some(2), Some(16));
        assert!(conv_geometry(&[2, 4, 5120], &[2560, 5120], 5120, &c).is_err());
        let wide_groups = config(Some(4), Some(32));
        assert!(conv_geometry(&[2, 4, 5120], &[2560, 5120], 5120, &wide_groups).is_err());
        let any = config(None, None);
        assert!(conv_geometry(&[4, 5120], &[2560, 5120], 5120, &any).is_err());
        assert!(conv_geometry(&[2, 4, 5120], &[2561, 5120], 5120, &any).is_err());
    }
}
