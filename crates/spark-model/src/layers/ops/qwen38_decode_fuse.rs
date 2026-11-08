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
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

use super::qwen4exp_decode_fuse::{
    GdnDecode, aligned, gdn_geometry_fits, gdn_grid_x_block, launch_gdn_fused,
};

const LEVER: &str = "ATLAS_QWEN38_DECODE_FUSE";
/// The exact verify's GDN rows of one sequence as one launch
/// (`qwen35_gdn_verify_tree`): [`gdn_verify_tree`].
const VERIFY_LEVER: &str = "ATLAS_QWEN38_VERIFY_FUSE";

fn parse(lever: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => bail!("{lever} must be 0 or 1, got {value:?}"),
    }
}

fn read_once(
    cell: &'static OnceLock<std::result::Result<bool, String>>,
    lever: &str,
    what: &str,
) -> Result<bool> {
    cell.get_or_init(|| {
        let on = parse(lever, std::env::var(lever).ok().as_deref());
        if let Ok(true) = on {
            tracing::info!("{lever}: {what}");
        }
        on.map_err(|error| error.to_string())
    })
    .clone()
    .map_err(|error| anyhow!(error))
}

/// Whether the lever is on. The environment is read once; a malformed value
/// fails every launch that asks.
pub fn on() -> Result<bool> {
    static ON: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    read_once(
        &ON,
        LEVER,
        "fused GDN decode step (qwen35_gdn_decode_fused)",
    )
}

/// Whether `ATLAS_QWEN38_VERIFY_FUSE` is on (read once, like [`on`]).
pub fn verify_on() -> Result<bool> {
    static ON: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    read_once(
        &ON,
        VERIFY_LEVER,
        "fused exact GDN verify rows (qwen35_gdn_verify_tree)",
    )
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

/// One sequence's exact-verify GDN rows between its projections: the
/// buffers of `decode_batched_conv_gdn_exact` for `rows` rows.
pub struct GdnVerifyTree {
    pub h_state: DevicePtr,
    pub conv_state: DevicePtr,
    /// Rollback slot of row `t` at `h_snap + t * h_snap_bytes` (rows `0..rows-1`).
    pub h_snap: DevicePtr,
    pub h_snap_bytes: usize,
    pub conv_snap: DevicePtr,
    pub conv_snap_bytes: usize,
    /// `[rows]` i32 parent rows (-1: the state before the verify), rows in
    /// depth-first pre-order; NULL for a chain.
    pub parent: DevicePtr,
    /// `[rows, qkvz_stride]` BF16 `Q | K | V | Z`.
    pub qkvz: DevicePtr,
    pub qkvz_stride: u32,
    pub conv_w: DevicePtr,
    /// `[rows, 2 * nv]` FP32 `gate | beta`.
    pub gates: DevicePtr,
    pub norm_w: DevicePtr,
    /// `[rows, nv * 128]` BF16 normed output.
    pub out: DevicePtr,
    pub rows: u32,
}

/// `qwen35_gdn_verify_tree` in place of the exact verify's per-row conv,
/// recurrence, gated norm and state copies, when `ATLAS_QWEN38_VERIFY_FUSE=1`,
/// the kernel is shipped and the shape fits. Returns whether it launched; on
/// `false` nothing was launched and the caller runs its per-row chain.
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_tree(
    gpu: &dyn GpuBackend,
    b: &GdnVerifyTree,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    d_conv: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<bool> {
    if !verify_on()? {
        return Ok(false);
    }
    static K: OnceLock<KernelHandle> = OnceLock::new();
    let k = *K.get_or_init(|| {
        let k = crate::layers::try_kernel(gpu, "qwen35_decode_fuse", "qwen35_gdn_verify_tree");
        if k.0 == 0 {
            tracing::warn!(
                "{VERIFY_LEVER}=1 ignored: this kernel set ships no qwen35_gdn_verify_tree \
                 (only the qwen3.6-27b tree does: qwen3.6-27b, qwen3.8-27b)"
            );
        }
        k
    });
    let conv_dim = (2 * nk * kd + nv * vd) as usize;
    let fits = k.0 != 0
        && b.rows >= 1
        && gdn_geometry_fits(nk, nv, kd, vd, d_conv, 8)
        && b.h_snap_bytes.is_multiple_of(16)
        && b.conv_snap_bytes.is_multiple_of(16)
        && aligned(&[b.conv_state, b.h_snap, b.conv_snap], 16)
        && aligned(&[b.qkvz.offset(conv_dim * 2), b.norm_w, b.out], 8)
        && (b.qkvz_stride as usize * 2).is_multiple_of(8);
    if !fits {
        return Ok(false);
    }
    let (grid_x, block) = gdn_grid_x_block(nv);
    KernelLaunch::new(gpu, k)
        .grid([grid_x, 1, 1])
        .block([block, 1, 1])
        .arg_ptr(b.h_state)
        .arg_ptr(b.conv_state)
        .arg_ptr(b.h_snap)
        .arg_ptr(b.conv_snap)
        .arg_ptr(b.parent)
        .arg_ptr(b.qkvz)
        .arg_ptr(b.conv_w)
        .arg_ptr(b.gates)
        .arg_ptr(b.norm_w)
        .arg_ptr(b.out)
        .arg_u32(b.rows)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(kd)
        .arg_u32(b.qkvz_stride)
        .arg_u64((b.h_snap_bytes / 4) as u64)
        .arg_u64((b.conv_snap_bytes / 4) as u64)
        .arg_f32(l2_eps)
        .arg_f32(eps)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{LEVER, VERIFY_LEVER, parse};

    #[test]
    fn lever_is_off_unless_exactly_one() {
        for lever in [LEVER, VERIFY_LEVER] {
            assert!(!parse(lever, None).unwrap());
            assert!(!parse(lever, Some("0")).unwrap());
            assert!(parse(lever, Some("1")).unwrap());
            for bad in ["", "true", "yes", "2", " 1"] {
                assert!(parse(lever, Some(bad)).is_err(), "{bad:?}");
            }
        }
    }
}
