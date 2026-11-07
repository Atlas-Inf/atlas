// SPDX-License-Identifier: AGPL-3.0-only

//! Dense Qwen3.6/3.8-27B exact fused GDN decode (`ATLAS_QWEN38_DECODE_FUSE=1`,
//! default off): the GDN group of `qwen4exp_decode_fuse` for the SiLU-gated
//! dense hybrid. A single-token GDN mixer's `dense_gemv_ba_gates`,
//! `causal_conv1d_update_l2norm_f32`, `gated_delta_rule_decode_f32` and
//! `gated_rms_norm_f32_input` run as one launch of `qwen35_gdn_decode_fused`
//! (kernels/gb10/qwen3.6-27b/nvfp4/qwen35_decode_fuse.cu, so the qwen3.6-27b
//! and qwen3.8-27b targets), writing the same bytes: the same IEEE operations
//! in the same order as that tree's four kernels (`gdn_decode_fuse_check`
//! example). Serial decode's arithmetic is unchanged, so the
//! `ATLAS_EXACT_VERIFY` GDN chain stays its exact twin and the two levers
//! combine.
//!
//! Where the kernel is not shipped, or the shape does not fit, the four
//! kernels run unchanged. Exact, and no collective changes, so the ranks need
//! not agree on it (no startup-parity entry).

use anyhow::{Result, anyhow, bail};
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use std::sync::OnceLock;

use super::qwen4exp_decode_fuse::{GdnDecode, launch_gdn_fused};

const LEVER: &str = "ATLAS_QWEN38_DECODE_FUSE";

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => bail!("{LEVER} must be 0 or 1, got {value:?}"),
    }
}

/// Whether the lever is on. The environment is read once; a malformed value
/// fails every launch that asks.
pub fn on() -> Result<bool> {
    static ON: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ON.get_or_init(|| {
        let on = parse(std::env::var(LEVER).ok().as_deref());
        if let Ok(true) = on {
            tracing::info!("{LEVER}: fused GDN decode step (qwen35_gdn_decode_fused)");
        }
        on.map_err(|error| error.to_string())
    })
    .clone()
    .map_err(|error| anyhow!(error))
}

/// `qwen35_gdn_decode_fused` in place of the BA gates, conv, recurrence and
/// SiLU gated-norm launches of one decode token, when the lever is on, the
/// kernel is shipped and the shape fits. Returns whether it launched; on
/// `false` nothing was launched.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode(
    gpu: &dyn GpuBackend,
    b: &GdnDecode,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    d_conv: u32,
    ba_k: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<bool> {
    if !on()? {
        return Ok(false);
    }
    // Looked up only once the lever is on, so other targets leave no failed
    // audit row; resolved once per process.
    static K: OnceLock<KernelHandle> = OnceLock::new();
    let k = *K.get_or_init(|| {
        let k = crate::layers::try_kernel(gpu, "qwen35_decode_fuse", "qwen35_gdn_decode_fused");
        if k.0 == 0 {
            tracing::warn!(
                "{LEVER}=1 ignored: this kernel set ships no qwen35_gdn_decode_fused \
                 (only the qwen3.6-27b tree does: qwen3.6-27b, qwen3.8-27b)"
            );
        }
        k
    });
    launch_gdn_fused(gpu, k, b, nk, nv, kd, vd, d_conv, ba_k, l2_eps, eps, stream)
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn lever_is_off_unless_exactly_one() {
        assert!(!parse(None).unwrap());
        assert!(!parse(Some("0")).unwrap());
        assert!(parse(Some("1")).unwrap());
        for bad in ["", "true", "yes", "2", " 1"] {
            assert!(parse(Some(bad)).is_err(), "{bad:?}");
        }
    }
}
