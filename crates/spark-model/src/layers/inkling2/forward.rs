// SPDX-License-Identifier: AGPL-3.0-only

//! One visit's forward over a batch of rows, in the reference's order and
//! dtypes (transformers PR #51050 `Inkling2Loop{Sliding,Full}{,Loop}DecoderLayer`,
//! Apache-2.0). Visit 0 applies the embedding norm and uploads the batch's
//! metadata for every visit; restart visits of later loop passes stash the
//! previous pass and restore the prelude's state; the last visit applies the
//! final block residual before the shared final norm.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::batch::Batch;
use super::debug;
use super::layer::Inkling2Layer;
use super::ops::{self, DecodeAttn, HeadNorm, MoeDecode};
use super::weights::{InputNorms, KvConv};
use crate::layer::ForwardContext;

impl Inkling2Layer {
    /// This visit over the batch's rows; `conv_states` are the sequences'
    /// conv-state bases, in batch order.
    pub(super) fn forward(
        &self,
        hidden: DevicePtr,
        batch: &Batch<'_>,
        conv_states: &[DevicePtr],
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let (s, v, w, d, p) = (&self.shared, self.visit, &self.w, &self.dims, &self.p);
        ensure!(
            s.is_attached(),
            "inkling2 visit {} has no shared buffers (attach_layer_buffers)",
            self.visit_idx
        );
        let (n, seqs) = (batch.rows(), batch.seqs.len());
        let (h, rb, rf) = (d.hidden, n * d.hidden * 2, n * d.hidden * 4);
        let bstride = d.block_stride() * 2; // bytes between stack blocks
        let bs = kv_cache.block_size();
        let gm = &self.gk;
        let meta = self.meta(gpu, batch, ctx, bs, v.first, stream)?;

        if v.first {
            gm.rmsnorm(
                gpu,
                n,
                DevicePtr::NULL,
                hidden,
                self.g.embed_norm,
                hidden,
                h,
                p.eps,
                stream,
            )?;
            debug::dump(gpu, hidden, batch, h, u32::MAX, stream)?;
        }
        let states = self.bind_states(gpu, conv_states, ctx.graph_capture, stream)?;
        if v.restart {
            // the previous pass: its blocks after the prelude's and its own first push,
            // and its output (this visit's input); then back to the prelude's output
            for j in 0..v.message_blocks {
                let src = s.blocks.offset((v.blocks_before + 1 + j) * bstride);
                gpu.copy_d2d_async(src, s.msg_blocks.offset(j * bstride), rb, stream)?;
            }
            gpu.copy_d2d_async(hidden, s.msg_h, rb, stream)?;
            gpu.copy_d2d_async(s.prelude_h, hidden, rb, stream)?;
        }
        let x = hidden;
        let res = |i: usize| s.res3.offset(i * rf);
        let normed = |i: usize| s.normed3.offset(i * rb);

        // attention inputs (fp32 in res3 unless layer 0)
        let n_in = if v.full { 3 } else { 1 };
        let has_res = !w.attn_query.is_null();
        if has_res {
            self.block_res(
                gpu,
                n,
                s.blocks,
                v.blocks_before,
                DevicePtr::NULL,
                x,
                w.attn_query,
                n_in,
                s.res3,
                stream,
            )?;
        }
        let gated_inputs = v.pass > 0;
        if gated_inputs {
            let ll = v
                .loop_layer
                .context("inkling2: later pass outside the loop")?;
            let q = self.msg_queries.offset(ll * 4 * h * 4);
            self.block_res(
                gpu,
                n,
                s.msg_blocks,
                v.message_blocks,
                DevicePtr::NULL,
                s.msg_h,
                q,
                4,
                s.msg4,
                stream,
            )?;
            // q, k, v inputs gated by their messages in one pass; SWA layers fan
            // their one input out to the three
            let row_elems = n * h;
            self.msg_gates(
                gpu,
                n,
                0..3,
                (res(0), if v.full { row_elems } else { 0 }),
                (s.msg4, row_elems),
                (res(0), row_elems),
                stream,
            )?;
        }
        // input norms -> normed3
        let inputs_split = v.full || gated_inputs;
        let src_f32 = |i: usize| if has_res { res(i) } else { DevicePtr::NULL };
        let src_bf16 = if has_res { DevicePtr::NULL } else { x };
        match w.input_norms {
            InputNorms::Shared(nw) if !inputs_split => {
                gm.rmsnorm(
                    gpu,
                    n,
                    src_f32(0),
                    src_bf16,
                    nw,
                    normed(0),
                    h,
                    p.eps,
                    stream,
                )?;
            }
            InputNorms::Shared(nw) => {
                for i in 0..3 {
                    gm.rmsnorm(
                        gpu,
                        n,
                        src_f32(i),
                        src_bf16,
                        nw,
                        normed(i),
                        h,
                        p.eps,
                        stream,
                    )?;
                }
            }
            InputNorms::Separate { q, k, v: vn } => {
                for (i, nw) in [q, k, vn].into_iter().enumerate() {
                    gm.rmsnorm(
                        gpu,
                        n,
                        src_f32(i),
                        src_bf16,
                        nw,
                        normed(i),
                        h,
                        p.eps,
                        stream,
                    )?;
                }
            }
        }
        let (aq, ak, av) = if inputs_split {
            (normed(0), normed(1), normed(2))
        } else {
            (normed(0), normed(0), normed(0))
        };

        // projections
        gm.proj(gpu, n, aq, w.wq, s.q, d.q_dim, h, stream)?;
        gm.proj(gpu, n, ak, w.wk, s.k_lin, d.kv_dim, h, stream)?;
        gm.proj(gpu, n, av, w.wv, s.v_lin, d.kv_dim, h, stream)?;
        gm.proj(gpu, n, aq, w.wr, s.r, d.rel_dim, h, stream)?;
        gm.proj(gpu, n, aq, w.gate, s.gate, d.q_dim, h, stream)?;
        gm.proj(gpu, n, aq, w.sink, s.sink, p.heads, h, stream)?;
        if let KvConv::Dynamic { proj, .. } = w.k_conv {
            gm.proj(gpu, n, ak, proj, s.dyn_k, d.dyn_dim, h, stream)?;
        }
        if let KvConv::Dynamic { proj, .. } = w.v_conv {
            gm.proj(gpu, n, av, proj, s.dyn_v, d.dyn_dim, h, stream)?;
        }

        // k/v convs, then the per-head epilogues
        // each state: k, v, post-attention, post-MLP convs (fp32 elements)
        let kvs = d.kv_dim * (p.conv_k - 1);
        let hs = h * (p.conv_k - 1);
        self.conv(
            gpu, n, seqs, s.k_lin, s.kc, states, 0, w.k_conv, s.dyn_k, d.kv_dim, p.head_dim, stream,
        )?;
        self.conv(
            gpu, n, seqs, s.v_lin, s.vc, states, kvs, w.v_conv, s.dyn_v, d.kv_dim, p.head_dim,
            stream,
        )?;
        let clip = HeadNorm::Clip {
            threshold: p.qk_clip,
        };
        // full layers scale q and the relative-bias queries by log length
        let log_pos = if v.full {
            meta.positions
        } else {
            DevicePtr::NULL
        };
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

        // KV of every row first, then each row attends over its own prefix
        let (k_pool, v_pool) = (
            kv_cache.k_pool_ptr(self.visit_idx),
            kv_cache.v_pool_ptr(self.visit_idx),
        );
        gm.kv_write(
            gpu, n, s.kn, s.vn, k_pool, v_pool, meta.slots, p.kv_heads, p.head_dim, bs, stream,
        )?;
        ops::decode_attn(
            gpu,
            self.k.decode_attn,
            &DecodeAttn {
                q: s.qn,
                k_cache: k_pool,
                v_cache: v_pool,
                out: s.attn,
                block_tables: meta.block_table,
                positions: meta.positions,
                rel: s.rn,
                rel_proj: w.rel_proj,
                sinks: s.sink,
                // 0 = every row reads one sequence's table
                max_blocks_per_seq: meta.table_stride,
                num_seqs: n,
                heads: p.heads,
                kv_heads: p.kv_heads,
                head_dim: p.head_dim,
                block_size: bs,
                q_stride: d.q_dim,
                sliding_window: if v.full { 0 } else { p.sliding_window },
                d_rel: p.d_rel,
                rel_extent: p.rel_extent,
            },
            stream,
        )?;
        gm.gate_mul(gpu, s.attn, s.gate, s.gated, n * d.q_dim, stream)?;
        gm.proj(gpu, n, s.gated, w.wo, s.proj, h, d.q_dim, stream)?;
        self.conv(
            gpu,
            n,
            seqs,
            s.proj,
            s.conv_out,
            states,
            2 * kvs,
            KvConv::Static {
                weight: w.attn_conv,
            },
            DevicePtr::NULL,
            h,
            h,
            stream,
        )?;

        // block push + residual, MLP input
        if v.boundary {
            gpu.copy_d2d_async(x, s.blocks.offset(v.blocks_before * bstride), rb, stream)?;
        }
        gm.residual(gpu, x, s.conv_out, s.resid, n * h, v.boundary, stream)?;
        let nb_mlp = v.blocks_before + usize::from(v.boundary);
        self.block_res(
            gpu,
            n,
            s.blocks,
            nb_mlp,
            s.resid,
            DevicePtr::NULL,
            w.mlp_query,
            1,
            s.mlp_in,
            stream,
        )?;
        if gated_inputs {
            let mlp_msg = s.msg4.offset(3 * rf);
            self.msg_gates(
                gpu,
                n,
                3..4,
                (s.mlp_in, 0),
                (mlp_msg, 0),
                (s.mlp_in, 0),
                stream,
            )?;
        }
        gm.rmsnorm(
            gpu,
            n,
            s.mlp_in,
            DevicePtr::NULL,
            w.mlp_norm,
            s.mlp_normed,
            h,
            p.eps,
            stream,
        )?;

        // router, experts, post-MLP conv, layer output
        gm.proj_f32(
            gpu,
            n,
            s.mlp_normed,
            w.router,
            s.logits,
            d.experts,
            h,
            stream,
        )?;
        gm.route(
            gpu,
            n,
            s.logits,
            w.router_bias,
            s.ids,
            s.weights,
            d.experts,
            d.top_k,
            p.route_scale,
            stream,
        )?;
        ops::moe_decode(
            gpu,
            &self.k,
            &MoeDecode {
                x: s.mlp_normed,
                experts: w.experts,
                expert_ids: s.ids,
                weights: s.weights,
                act: s.act,
                y: s.y,
                out: s.moe_out,
                rows: n,
                top_k: d.top_k,
                hidden: h,
                inter: d.inter,
                out_stride: h,
                gate_softcap: p.gate_softcap,
                up_softcap: p.up_softcap,
            },
            stream,
        )?;
        self.conv(
            gpu,
            n,
            seqs,
            s.moe_out,
            s.mlp_conv,
            states,
            2 * kvs + hs,
            KvConv::Static { weight: w.mlp_conv },
            DevicePtr::NULL,
            h,
            h,
            stream,
        )?;
        gm.layer_out(gpu, s.resid, s.mlp_conv, hidden, n * h, stream)?;
        debug::dump(gpu, hidden, batch, h, self.visit_idx as u32, stream)?;
        if debug::tap_visit() == Some(self.visit_idx) {
            self.debug_taps(gpu, batch, stream)?;
        }

        if v.save_prelude {
            gpu.copy_d2d_async(hidden, s.prelude_h, rb, stream)?;
        }
        if v.last {
            self.block_res(
                gpu,
                n,
                s.blocks,
                nb_mlp,
                DevicePtr::NULL,
                hidden,
                self.g.output_query,
                1,
                s.final_out,
                stream,
            )?;
            gm.cast_bf16(gpu, s.final_out, hidden, n * h, stream)?;
        }
        Ok(())
    }
}
