// SPDX-License-Identifier: AGPL-3.0-only

//! A visit's k/v convs and per-head norms, between its projections and its
//! KV write (split from `forward.rs` for the 500-line cap; code moved as is).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::batch::Batch;
use super::layer::Inkling2Layer;
use super::ops::{self, HeadNorm};

impl Inkling2Layer {
    /// The k and v convs, then the per-head norms of q, k, v and the
    /// relative-bias queries; `states` is the batch's bound conv states,
    /// `positions` its rows' positions.
    pub(super) fn kv_convs_and_head_norms(
        &self,
        gpu: &dyn GpuBackend,
        batch: &Batch<'_>,
        states: DevicePtr,
        positions: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (s, v, w, d, p) = (&self.shared, self.visit, &self.w, &self.dims, &self.p);
        let (n, seqs) = (batch.rows(), batch.seqs.len());
        let kvs = d.kv_dim * (p.conv_k - 1);
        self.journal_inputs(gpu, batch, [s.k_lin, s.v_lin], 0, stream)?;
        self.conv(
            gpu,
            n,
            seqs,
            s.k_lin,
            ops::ConvOut::Plain(s.kc),
            states,
            0,
            w.k_conv,
            s.dyn_k,
            d.kv_dim,
            p.head_dim,
            stream,
        )?;
        self.conv(
            gpu,
            n,
            seqs,
            s.v_lin,
            ops::ConvOut::Plain(s.vc),
            states,
            kvs,
            w.v_conv,
            s.dyn_v,
            d.kv_dim,
            p.head_dim,
            stream,
        )?;
        let clip = HeadNorm::Clip {
            threshold: p.qk_clip,
        };
        // full layers scale q and the relative-bias queries by log length
        let log_pos = if v.full { positions } else { DevicePtr::NULL };
        self.head_norm(
            gpu, n, s.q, s.qn, w.q_scale, log_pos, p.heads, p.head_dim, clip, stream,
        )?;
        self.head_norm(
            gpu,
            n,
            s.kc,
            s.kn,
            DevicePtr::NULL,
            DevicePtr::NULL,
            p.kv_heads,
            p.head_dim,
            clip,
            stream,
        )?;
        self.head_norm(
            gpu,
            n,
            s.vc,
            s.vn,
            DevicePtr::NULL,
            DevicePtr::NULL,
            p.kv_heads,
            p.head_dim,
            HeadNorm::Rms { eps: p.eps },
            stream,
        )?;
        self.head_norm(
            gpu,
            n,
            s.r,
            s.rn,
            w.rel_scale,
            log_pos,
            p.heads,
            p.d_rel,
            HeadNorm::Clip {
                threshold: p.rel_clip,
            },
            stream,
        )?;
        Ok(())
    }
}
