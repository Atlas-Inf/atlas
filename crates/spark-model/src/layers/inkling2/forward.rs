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
use super::layer::Inkling2Layer;
use super::ops::{self, DecodeAttn};
use super::weights::{InputNorms, KvConv};
use crate::layer::ForwardContext;

/// Planes of `Shared::res_sq`: res3's three, msg4's four, then mlp_in.
const SQ_RES3: usize = 0;
const SQ_MSG4: usize = 3;
const SQ_MLP: usize = 7;

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
        // A verify run (journaled rows) takes decode's kernels at any row
        // count, so each verified token is the one serial decode picks; prefill
        // rows take the prefill kernels and the checkpoint's declared numerics.
        let exact = !batch.journals.is_empty();
        let (h, rb, rf) = (d.hidden, n * d.hidden * 2, n * d.hidden * 4);
        let bstride = d.block_stride() * 2; // bytes between stack blocks
        let bs = kv_cache.block_size();
        let gm = &self.gk;
        let meta = self.meta(gpu, batch, ctx, bs, v.first, stream)?;
        // window-bounded KV: the sliding-window visits' rows sit at local
        // blocks (`kv_cache::window`); visit 0 maps the rows' slots and tables
        let window = kv_cache.window_map_ptr();
        if v.first && !window.is_null() {
            let local = (s.local_slots, s.local_table);
            gm.window_meta(gpu, &meta, window, local, n, bs, p.sliding_window, stream)?;
        }
        let (kv_slots, kv_table) = if v.full || window.is_null() {
            (meta.slots, meta.block_table)
        } else {
            (s.local_slots, s.local_table)
        };

        // `hidden` arrives normed: the embedding table is (`load_embedding`),
        // and image rows never pass through `embed_norm`.
        let states = self.bind_states(gpu, conv_states, ctx.graph_capture, stream)?;
        for (j, state) in batch.journals.iter().zip(conv_states) {
            gpu.copy_d2d_async(*state, *j, self.conv_state_bytes(), stream)?;
        }
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
        // decode and verify rows read keys split over the GPU; a prefill chunk of one
        // sequence takes the prefill kernels
        let body = ops::attn_body(
            batch.seqs.len(),
            batch.rows_per_seq,
            super::layer::MAX_VERIFY_ROWS,
            ops::attn_slots(d.max_rows, d.max_seqs),
        );
        let prefill = body == ops::AttnBody::Prefill;
        // later passes gate their inputs by messages; a prefill chunk's fused gate reads
        // its rows' sums of squares, which their block residuals write (planes SQ_* of
        // res_sq), and (gb10) rebuilds the messages from their sources with the mix
        // weights the message block residual writes instead of the fp32 messages
        let gated_inputs = v.pass > 0;
        let fused_gate = gated_inputs && prefill && ops::fused_takes(n, false, self.p.gate_rank, h);
        let rebuilt_msgs = fused_gate && !cfg!(atlas_scale);
        let sq = |plane: usize| {
            if fused_gate {
                s.res_sq.offset(plane * n * 4)
            } else {
                DevicePtr::NULL
            }
        };
        let msg = |first: usize| {
            rebuilt_msgs.then(|| ops::MsgSources {
                blocks: s.msg_blocks,
                block_stride: d.block_stride(),
                num_blocks: v.message_blocks,
                partial: s.msg_h,
                p: s.msg_p,
                first,
            })
        };

        // attention inputs (fp32 in res3 unless layer 0), then their norms into normed3
        let n_in = if v.full { 3 } else { 1 };
        let has_res = !w.attn_query.is_null();
        let ws = match w.input_norms {
            InputNorms::Shared(nw) if !self.inputs_split() => vec![nw],
            InputNorms::Shared(nw) => vec![nw; 3],
            InputNorms::Separate { q, k, v: vn } => vec![q, k, vn],
        };
        // a prefill chunk's rows no message gate rewrites: the block residual norms
        // its own outputs (the fp32 rows never leave the kernel). Decode rows keep
        // verify's kernels, so text with drafts stays the text without.
        let res_norm = |ws: &[DevicePtr], y: DevicePtr| {
            let fused = prefill && !gated_inputs;
            let wq = |q: usize| ws[q.min(ws.len() - 1)];
            fused.then(|| ops::ResNorm {
                w: [wq(0), wq(1), wq(2)],
                y,
                eps: p.eps,
            })
        };
        let in_norm = if has_res && ws.len() == n_in {
            res_norm(&ws, normed(0))
        } else {
            None
        };
        if has_res {
            let out = if in_norm.is_some() {
                DevicePtr::NULL
            } else {
                s.res3
            };
            self.block_res(
                gpu,
                n,
                s.blocks,
                v.blocks_before,
                DevicePtr::NULL,
                x,
                w.attn_query,
                n_in,
                (out, sq(SQ_RES3), in_norm, DevicePtr::NULL),
                stream,
            )?;
        }
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
                if rebuilt_msgs {
                    (DevicePtr::NULL, sq(SQ_MSG4), None, s.msg_p)
                } else {
                    (s.msg4, sq(SQ_MSG4), None, DevicePtr::NULL)
                },
                stream,
            )?;
            // q, k, v inputs gated by their messages in one pass; SWA layers fan
            // their one input out to the three
            let row_elems = n * h;
            self.msg_gates(
                gpu,
                (n, prefill),
                0..3,
                (res(0), if v.full { row_elems } else { 0 }),
                (s.msg4, row_elems),
                (res(0), row_elems),
                (sq(SQ_RES3), sq(SQ_MSG4)),
                msg(0),
                stream,
            )?;
        }
        // input norms -> normed3, one launch; layer 0 reads x for all three
        if in_norm.is_none() {
            let src_f32 = if has_res { res(0) } else { DevicePtr::NULL };
            let src_bf16 = if has_res { DevicePtr::NULL } else { x };
            let in_plane = if has_res { n * h } else { 0 };
            gm.rmsnorms(
                gpu,
                n,
                src_f32,
                src_bf16,
                in_plane,
                &ws,
                normed(0),
                h,
                p.eps,
                stream,
            )?;
        }
        self.project(gpu, n, exact, stream)?;

        // each state: k, v, post-attention, post-MLP convs (fp32 elements)
        let kvs = d.kv_dim * (p.conv_k - 1);
        let hs = h * (p.conv_k - 1);
        self.kv_convs_and_head_norms(gpu, batch, states, meta.positions, stream)?;

        // KV of every row first, then each row attends over its own prefix
        let (k_pool, v_pool) = (
            kv_cache.k_pool_ptr(self.visit_idx),
            kv_cache.v_pool_ptr(self.visit_idx),
        );
        gm.kv_write(
            gpu, n, s.kn, s.vn, k_pool, v_pool, kv_slots, p.kv_heads, p.head_dim, bs, stream,
        )?;
        let gated = if body == ops::AttnBody::Prefill {
            // a prefill chunk of one sequence: tensor cores
            ops::prefill_attn(
                gpu,
                &self.k,
                &ops::PrefillAttn {
                    q: s.qn,
                    k_cache: k_pool,
                    v_cache: v_pool,
                    out: s.attn,
                    gate: s.gate,
                    gated: s.gated,
                    block_table: kv_table,
                    rel: s.rn,
                    rel_proj: w.rel_proj,
                    rel_proj_t: w.rel_proj_t,
                    bias: s.rel_table,
                    sinks: s.sink,
                    q_len: n,
                    q_offset: batch.seqs[0].0,
                    heads: p.heads,
                    kv_heads: p.kv_heads,
                    head_dim: p.head_dim,
                    block_size: bs,
                    sliding_window: if v.full { 0 } else { p.sliding_window },
                    d_rel: p.d_rel,
                    rel_extent: p.rel_extent,
                },
                stream,
            )?
        } else {
            // decode and verify rows split their keys over the GPU; prefill
            // rows have rows enough
            let splits = match body {
                ops::AttnBody::Decode { splits } => splits,
                ops::AttnBody::Prefill => 1,
            };
            ops::decode_attn(
                gpu,
                &self.k,
                &DecodeAttn {
                    q: s.qn,
                    k_cache: k_pool,
                    v_cache: v_pool,
                    partials: s.attn_partials,
                    out: s.attn,
                    block_tables: kv_table,
                    positions: meta.positions,
                    rel: s.rn,
                    rel_proj: w.rel_proj,
                    sinks: s.sink,
                    // 0 = every row reads one sequence's table
                    max_blocks_per_seq: meta.table_stride,
                    num_seqs: n,
                    splits,
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
            false
        };
        if !gated {
            gm.gate_mul(gpu, s.attn, s.gate, s.gated, n * d.q_dim, stream)?;
        }
        gm.proj(gpu, (n, exact), s.gated, w.wo, s.proj, h, d.q_dim, stream)?;
        self.journal_inputs(gpu, batch, [s.proj], 2, stream)?;
        // the post-attention conv adds itself to the residual stream (fp32 resid)
        self.conv(
            gpu,
            n,
            seqs,
            s.proj,
            ops::ConvOut::AddTo {
                x,
                restart: v.boundary,
                r: s.resid,
            },
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

        // block push, MLP input
        if v.boundary {
            gpu.copy_d2d_async(x, s.blocks.offset(v.blocks_before * bstride), rb, stream)?;
        }
        let nb_mlp = v.blocks_before + usize::from(v.boundary);
        let mlp_norm = res_norm(&[w.mlp_norm], s.mlp_normed);
        self.block_res(
            gpu,
            n,
            s.blocks,
            nb_mlp,
            s.resid,
            DevicePtr::NULL,
            w.mlp_query,
            1,
            match mlp_norm {
                Some(_) => (DevicePtr::NULL, DevicePtr::NULL, mlp_norm, DevicePtr::NULL),
                None => (s.mlp_in, sq(SQ_MLP), None, DevicePtr::NULL),
            },
            stream,
        )?;
        if gated_inputs {
            let mlp_msg = s.msg4.offset(3 * rf);
            self.msg_gates(
                gpu,
                (n, prefill),
                3..4,
                (s.mlp_in, 0),
                (mlp_msg, 0),
                (s.mlp_in, 0),
                (sq(SQ_MLP), sq(SQ_MSG4 + 3)),
                msg(3),
                stream,
            )?;
        }
        if mlp_norm.is_none() {
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
        }

        // router, experts, post-MLP conv, layer output
        gm.proj_f32(
            gpu,
            (n, exact),
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
        self.routed_experts(gpu, n, exact, stream)?;
        self.journal_inputs(gpu, batch, [s.moe_out], 3, stream)?;
        self.conv(
            gpu,
            n,
            seqs,
            s.moe_out,
            ops::ConvOut::Finish {
                r: s.resid,
                out: hidden,
            },
            states,
            2 * kvs + hs,
            KvConv::Static { weight: w.mlp_conv },
            DevicePtr::NULL,
            h,
            h,
            stream,
        )?;

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
                (s.final_out, DevicePtr::NULL, None, DevicePtr::NULL),
                stream,
            )?;
            gm.cast_bf16(gpu, s.final_out, hidden, n * h, stream)?;
        }
        Ok(())
    }
}
