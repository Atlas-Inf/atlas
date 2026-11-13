// SPDX-License-Identifier: AGPL-3.0-only

//! The deferred-commit legs of `gdn_verify_tree_check`.

use super::*;

/// The deferred legs' buffers: staging and a commit target.
pub(super) struct Deferred {
    pub(super) stage_qkv: DevicePtr,
    pub(super) stage_gb: DevicePtr,
    pub(super) h: DevicePtr,
    pub(super) conv: DevicePtr,
}

impl Deferred {
    pub(super) fn new(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            stage_qkv: g.alloc(MAX_ROWS * CONV_DIM * 2)?,
            stage_gb: g.alloc(MAX_ROWS * 2 * NV * 4)?,
            h: g.alloc(H_BYTES)?,
            conv: g.alloc(CONV_BYTES)?,
        })
    }

    /// The deferred verify from (h0, conv0) into `f`'s buffers.
    pub(super) fn verify(
        &self,
        g: &dyn GpuBackend,
        c: &Case,
        f: &Fused,
        k: usize,
        table: DevicePtr,
    ) -> Result<()> {
        g.copy_h2d(&c.h0, f.h)?;
        g.copy_h2d(&c.conv0, f.conv)?;
        let b = GdnVerifyTree {
            h_state: f.h,
            conv_state: f.conv,
            h_snap: f.h_snap,
            h_snap_bytes: H_BYTES,
            conv_snap: f.conv_snap,
            conv_snap_bytes: CONV_BYTES,
            parent: table,
            qkvz: c.qkvz,
            qkvz_stride: QKVZ as u32,
            conv_w: c.conv_w,
            gates: c.gates,
            norm_w: c.norm_w,
            out: f.out,
            rows: k as u32,
        };
        let ran = ops::qwen38_decode_fuse::gdn_verify_tree_defer(
            g,
            &b,
            self.stage_qkv,
            self.stage_gb,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            D_CONV as u32,
            L2_EPS,
            EPS,
            0,
        )?;
        ensure!(ran, "gdn_verify_tree_defer declined the 27B shape");
        g.synchronize(0)
    }

    /// Replay `rows` from (h0, conv0); returns (h, conv).
    pub(super) fn commit(
        &self,
        g: &dyn GpuBackend,
        c: &Case,
        rows: &[usize],
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        g.copy_h2d(&c.h0, self.h)?;
        g.copy_h2d(&c.conv0, self.conv)?;
        let layer = GdnCommitLayer {
            h: self.h,
            conv: self.conv,
            stage_qkv: self.stage_qkv,
            stage_gb: self.stage_gb,
            conv_w: c.conv_w,
        };
        ops::qwen38_decode_fuse::gdn_commit_rows(
            g,
            &[layer],
            rows,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            D_CONV as u32,
            L2_EPS,
            0,
        )?;
        g.synchronize(0)?;
        Ok((dn(g, self.h, H_BYTES)?, dn(g, self.conv, CONV_BYTES)?))
    }
}

/// The deferred legs for one case: rows, untouched state, branch slots, and
/// every root path's replay. Returns the failures.
pub(super) fn deferred_leg(
    g: &dyn GpuBackend,
    d: &Deferred,
    c: &Case,
    f: &Fused,
    gold: &Rows,
    parent: &[i32],
    table: bool,
) -> Result<Vec<String>> {
    let k = parent.len();
    let t = f.table(g, table.then_some(parent))?;
    d.verify(g, c, f, k, t)?;
    let mut bad = Vec::new();
    if dn(g, f.out, k * VALUE_DIM * 2)? != gold.out {
        bad.push("normed out".to_string());
    }
    if dn(g, f.h, H_BYTES)? != c.h0 || dn(g, f.conv, CONV_BYTES)? != c.conv0 {
        bad.push("H0 / conv windows moved".to_string());
    }
    let tree = DraftTree {
        tokens: (0..k as u32).collect(),
        parent: parent.to_vec(),
        depth: (0..k)
            .map(|r| {
                let mut d = 0u8;
                let mut x = parent[r];
                while x >= 0 {
                    d += 1;
                    x = parent[x as usize];
                }
                d
            })
            .collect(),
    };
    for r in 0..k {
        // A branch parent keeps its H.
        if table && (r + 2..k).any(|ch| parent[ch] == r as i32) {
            if dn(g, f.h_snap.offset(r * H_BYTES), H_BYTES)? != gold.h[r] {
                bad.push(format!("branch slot {r}"));
            }
        }
        let path = tree.path_to(r);
        let (h, conv) = d.commit(g, c, &path)?;
        if h != gold.h[r] || conv != gold.conv[r] {
            bad.push(format!("replay of {path:?}"));
        }
    }
    Ok(bad)
}
