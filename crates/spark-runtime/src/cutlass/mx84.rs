// SPDX-License-Identifier: AGPL-3.0-only
//! Grouped MXFP8 x MXFP4 GEMM host wrapper
//! (`cuda/cutlass_mx84_grouped_gemm.cu`): every routed expert's projection of
//! the rows routed to it, in one launch, for a checkpoint that ships its
//! experts as MXFP4 and rounds their inputs to MXFP8.

use anyhow::{Result, bail, ensure};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;

#[cfg(atlas_cutlass)]
use super::*;

/// One group of a launch, as device addresses: `a` E4M3 `[rows, k]`, `sfa`
/// its UE8M0 scales in the block-scaled atom (the tiled layout of
/// `mxfp8_act_quant.cuh`), `b` E2M1 packed `[n, k / 2]` as the checkpoint
/// stores it, `sfb` its scales in the same atom, `d` BF16 `[rows, n]`.
#[derive(Clone, Copy, Debug)]
pub struct Mx84Group {
    pub a: u64,
    pub sfa: u64,
    pub b: u64,
    pub sfb: u64,
    pub d: u64,
    pub rows: usize,
}

/// Device scratch a launch of `groups` groups needs at most: its per-group
/// argument arrays and the scheduler's.
pub fn mx84_workspace_bytes(groups: usize) -> usize {
    groups * 1024 + (1 << 20)
}

/// `d` = `a` times `b` transposed for every group, FP32 accumulation, BF16
/// out. `n` is a multiple of 8 and `k` of 128; `a` is 16-byte aligned and `b`
/// 64. `workspace` is device scratch of `workspace_bytes`
/// ([`mx84_workspace_bytes`]), read in stream order, so a later launch on the
/// same stream may reuse it.
pub fn mx84_grouped(
    groups: &[Mx84Group],
    n: usize,
    k: usize,
    workspace: u64,
    workspace_bytes: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        n > 0 && k > 0 && n.is_multiple_of(8) && k.is_multiple_of(128),
        "CUTLASS mx84 grouped: [{n}, {k}] (n % 8 or k % 128)"
    );
    ensure!(
        groups.iter().all(|g| {
            g.rows > 0
                && g.rows <= i32::MAX as usize
                && ![g.a, g.sfa, g.b, g.sfb, g.d].contains(&0)
                && g.a.is_multiple_of(16)
                && g.b.is_multiple_of(64)
        }),
        "CUTLASS mx84 grouped: a group with no rows, a null operand or a misaligned one"
    );
    ensure!(
        workspace != 0 && workspace_bytes >= mx84_workspace_bytes(groups.len()),
        "CUTLASS mx84 grouped: {workspace_bytes} B of workspace for {} groups",
        groups.len()
    );
    if groups.is_empty() {
        return Ok(());
    }
    #[cfg(atlas_cutlass)]
    {
        let column = |f: fn(&Mx84Group) -> u64| -> Vec<u64> { groups.iter().map(f).collect() };
        let (a, sfa, b) = (column(|g| g.a), column(|g| g.sfa), column(|g| g.b));
        let (sfb, d) = (column(|g| g.sfb), column(|g| g.d));
        let rows: Vec<i32> = groups.iter().map(|g| g.rows as i32).collect();
        let status = drained("mx84_grouped", unsafe {
            atlas_cutlass_mx84_grouped(
                a.as_ptr(),
                sfa.as_ptr(),
                b.as_ptr(),
                sfb.as_ptr(),
                d.as_ptr(),
                rows.as_ptr(),
                groups.len() as i32,
                n as i32,
                k as i32,
                workspace as *mut c_void,
                workspace_bytes,
                stream as *mut c_void,
            )
        });
        if status != 0 {
            bail!(
                "CUTLASS mx84 grouped [{n}, {k}] x {} groups failed: status {status}",
                groups.len()
            );
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = stream;
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

#[cfg(test)]
mod tests {
    use super::{Mx84Group, mx84_grouped, mx84_workspace_bytes};

    fn group(rows: usize) -> Mx84Group {
        Mx84Group {
            a: 0x1000,
            sfa: 0x2000,
            b: 0x3000,
            sfb: 0x4000,
            d: 0x5000,
            rows,
        }
    }

    /// The checks that guard the FFI run before anything is launched, so
    /// they are testable without a GPU: none of these reaches the kernel.
    #[test]
    fn operands_the_kernel_cannot_take_are_refused_before_the_launch() {
        let ws = mx84_workspace_bytes(1);
        let refused = |groups: &[Mx84Group], n, k, workspace, bytes| {
            mx84_grouped(groups, n, k, workspace, bytes, 0).is_err()
        };
        assert!(
            refused(&[group(4)], 576, 5100, 0x9000, ws),
            "k not a multiple of 128"
        );
        assert!(
            refused(&[group(4)], 577, 5120, 0x9000, ws),
            "n not a multiple of 8"
        );
        assert!(
            refused(&[group(0)], 576, 5120, 0x9000, ws),
            "a group with no rows"
        );
        let null = Mx84Group { sfb: 0, ..group(4) };
        assert!(refused(&[null], 576, 5120, 0x9000, ws), "a null operand");
        let odd = Mx84Group {
            b: 0x3010,
            ..group(4)
        };
        assert!(
            refused(&[odd], 576, 5120, 0x9000, ws),
            "a misaligned weight"
        );
        assert!(
            refused(&[group(4)], 576, 5120, 0x9000, ws - 1),
            "a short workspace"
        );
        assert!(refused(&[group(4)], 576, 5120, 0, ws), "no workspace");
    }
}
