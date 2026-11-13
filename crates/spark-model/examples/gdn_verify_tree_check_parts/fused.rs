// SPDX-License-Identifier: AGPL-3.0-only

//! The fused tree-verify launch, its byte comparison and today's exact arm.

use super::*;

/// Device buffers of one fused launch.
pub(super) struct Fused {
    pub(super) h: DevicePtr,
    pub(super) conv: DevicePtr,
    pub(super) h_snap: DevicePtr,
    pub(super) conv_snap: DevicePtr,
    pub(super) parent: DevicePtr,
    pub(super) out: DevicePtr,
}

impl Fused {
    pub(super) fn new(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            h: g.alloc(H_BYTES)?,
            conv: g.alloc(CONV_BYTES)?,
            h_snap: g.alloc(MAX_ROWS * H_BYTES)?,
            conv_snap: g.alloc(MAX_ROWS * CONV_BYTES)?,
            parent: g.alloc(MAX_ROWS * 4)?,
            out: g.alloc(MAX_ROWS * VALUE_DIM * 2)?,
        })
    }

    /// The parent table to launch with: uploaded, or NULL for a chain.
    pub(super) fn table(&self, g: &dyn GpuBackend, parent: Option<&[i32]>) -> Result<DevicePtr> {
        Ok(match parent {
            Some(p) => {
                let b: Vec<u8> = p.iter().flat_map(|x| x.to_le_bytes()).collect();
                g.copy_h2d(&b, self.parent)?;
                self.parent
            }
            None => DevicePtr::NULL,
        })
    }

    pub(super) fn launch(
        &self,
        g: &dyn GpuBackend,
        c: &Case,
        rows: usize,
        table: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let b = GdnVerifyTree {
            h_state: self.h,
            conv_state: self.conv,
            h_snap: self.h_snap,
            h_snap_bytes: H_BYTES,
            conv_snap: self.conv_snap,
            conv_snap_bytes: CONV_BYTES,
            parent: table,
            qkvz: c.qkvz,
            qkvz_stride: QKVZ as u32,
            conv_w: c.conv_w,
            gates: c.gates,
            norm_w: c.norm_w,
            out: self.out,
            rows: rows as u32,
        };
        let ran = ops::qwen38_decode_fuse::gdn_verify_tree(
            g,
            &b,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            D_CONV as u32,
            L2_EPS,
            EPS,
            stream,
        )?;
        ensure!(ran, "gdn_verify_tree declined the 27B shape");
        Ok(())
    }

    /// Run from (h0, conv0) and read back: outputs, slots 0..k-2, final state.
    pub(super) fn run(
        &self,
        g: &dyn GpuBackend,
        c: &Case,
        h0: &[u8],
        k: usize,
        parent: Option<&[i32]>,
    ) -> Result<Rows> {
        g.copy_h2d(h0, self.h)?;
        g.copy_h2d(&c.conv0, self.conv)?;
        let table = self.table(g, parent)?;
        self.launch(g, c, k, table, 0)?;
        g.synchronize(0)?;
        let mut h = Vec::new();
        let mut conv = Vec::new();
        for t in 0..k - 1 {
            h.push(dn(g, self.h_snap.offset(t * H_BYTES), H_BYTES)?);
            conv.push(dn(g, self.conv_snap.offset(t * CONV_BYTES), CONV_BYTES)?);
        }
        h.push(dn(g, self.h, H_BYTES)?);
        conv.push(dn(g, self.conv, CONV_BYTES)?);
        Ok(Rows {
            out: dn(g, self.out, k * VALUE_DIM * 2)?,
            h,
            conv,
        })
    }
}

/// Byte mismatches of `got` against `gold` (the fused run's last row's state
/// is its final state; earlier rows' are its slots).
pub(super) fn mismatches(gold: &Rows, got: &Rows) -> Vec<String> {
    let mut bad = Vec::new();
    if gold.out != got.out {
        let rows: Vec<usize> = (0..gold.h.len())
            .filter(|&t| {
                let r = t * VALUE_DIM * 2..(t + 1) * VALUE_DIM * 2;
                gold.out[r.clone()] != got.out[r]
            })
            .collect();
        bad.push(format!("normed out rows {rows:?}"));
    }
    for t in 0..gold.h.len() {
        if gold.h[t] != got.h[t] {
            bad.push(format!("H after row {t}"));
        }
        if gold.conv[t] != got.conv[t] {
            bad.push(format!("conv after row {t}"));
        }
    }
    bad
}

/// Today's exact arm for one layer (timing only): one fused FP32 conv launch
/// with its slots, then per row the recurrence, the gated norm and an H copy.
pub(super) fn exact_arm(
    g: &dyn GpuBackend,
    ks: &Kernels,
    c: &Case,
    f: &Fused,
    k: usize,
    scratch: DevicePtr,
) -> Result<()> {
    let w = |weight| DenseWeight { weight };
    KernelLaunch::new(g, ks.conv_kn)
        .grid([CONV_DIM.div_ceil(256) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(f.conv)
        .arg_ptr(c.qkvz)
        .arg_ptr(c.conv_w)
        .arg_ptr(scratch)
        .arg_ptr(f.conv_snap)
        .arg_u32(k as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32((2 * KEY_DIM) as u32)
        .arg_u32(KD as u32)
        .arg_u32(QKVZ as u32)
        .arg_u32(QKVZ as u32)
        .arg_u32(CONV_ELEMS as u32)
        .arg_f32(L2_EPS)
        .launch(0)?;
    for t in 0..k {
        let row = scratch.offset(t * QKVZ * 4);
        let gdn_out = row.offset(CONV_DIM * 4);
        ops::gdn_decode(
            g,
            ks.gdn,
            f.h,
            row,
            row.offset(KEY_DIM * 4),
            row.offset(2 * KEY_DIM * 4),
            c.gate_row(t),
            c.gate_row(t).offset(NV * 4),
            gdn_out,
            1,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            0,
        )?;
        ops::gated_rms_norm(
            g,
            ks.norm,
            gdn_out,
            c.qkvz_row(t).offset(CONV_DIM * 2),
            &w(c.norm_w),
            f.out.offset(t * VALUE_DIM * 2),
            NV as u32,
            VD as u32,
            VD as u32,
            EPS,
            VD as u32,
            0,
        )?;
        if t + 1 < k {
            g.copy_d2d_async(f.h, f.h_snap.offset(t * H_BYTES), H_BYTES, 0)?;
        }
    }
    Ok(())
}
