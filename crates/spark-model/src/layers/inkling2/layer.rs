// SPDX-License-Identifier: AGPL-3.0-only

//! One visit of an Inkling 2 physical layer: single-token decode in the
//! reference's order and dtypes (transformers PR #51050
//! `Inkling2Loop{Sliding,Full}{,Loop}DecoderLayer`, Apache-2.0). The prefill
//! fallback runs this per token. Visit 0 also applies the embedding norm and
//! uploads the token's metadata for every visit; restart visits of later loop
//! passes stash the previous pass and restore the prelude's state; the last
//! visit applies the final block residual before the shared final norm.

use std::any::Any;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::glue::GlueKernels;
use super::ops::{
    self, BlockRes, DecodeAttn, HeadNorm, HeadNormArgs, Inkling2Kernels, MoeDecode, MsgGate,
    ShortConv,
};
use super::schedule::Visit;
use super::shared::{Dims, Shared};
use super::weights::{GlobalWeights, InputNorms, KvConv, PhysicalWeights};
use crate::layer::{ForwardContext, LayerState, TransformerLayer};

/// Numerics shared by every visit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_n_floor: f32,
    pub log_alpha: f32,
    pub conv_k: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub sliding_window: usize,
    pub gate_rank: usize,
    pub gate_softcap: f32,
    pub up_softcap: f32,
    pub route_scale: f32,
}

/// Per-sequence conv state of one visit: k, v, post-attention and post-MLP
/// convs, `[channels, k - 1]` fp32 each.
pub(crate) struct Inkling2SeqState {
    pub conv: DevicePtr,
}

impl LayerState for Inkling2SeqState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub(crate) struct Inkling2Layer {
    pub visit_idx: usize,
    pub visit: Visit,
    pub w: PhysicalWeights,
    pub g: GlobalWeights,
    /// `[core, 4, H]` fp32: per loop layer the q, k, v, mlp message queries
    pub msg_queries: DevicePtr,
    pub k: Inkling2Kernels,
    pub gk: GlueKernels,
    pub dims: Dims,
    pub p: Params,
    pub shared: Shared,
}

impl Inkling2Layer {
    fn conv_state_bytes(&self) -> usize {
        (2 * self.dims.kv_dim + 2 * self.dims.hidden) * (self.p.conv_k - 1) * 4
    }

    /// Upload this token's position, slot, length and block table (visit 0).
    fn upload_meta(
        &self,
        gpu: &dyn GpuBackend,
        seq_len: usize,
        block_table: &[u32],
        bs: usize,
        stream: u64,
    ) -> Result<()> {
        let s = &self.shared;
        let pos = seq_len;
        let blk = *block_table
            .get(pos / bs)
            .with_context(|| format!("inkling2: no KV block for position {pos}"))?;
        ensure!(
            block_table.len() <= self.dims.max_blocks_per_seq,
            "inkling2: block table over its buffer"
        );
        // earlier kernels of this stream may still read the previous token's metadata
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&(pos as i32).to_le_bytes(), s.position)?;
        gpu.copy_h2d(
            &((blk as i64) * bs as i64 + (pos % bs) as i64).to_le_bytes(),
            s.slot,
        )?;
        gpu.copy_h2d(&((pos + 1) as i32).to_le_bytes(), s.seq_len)?;
        let table: Vec<u8> = block_table.iter().flat_map(|b| b.to_le_bytes()).collect();
        gpu.copy_h2d(&table, s.block_table)?;
        Ok(())
    }

    fn conv(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        state: DevicePtr,
        conv: KvConv,
        dyn_taps: DevicePtr,
        channels: usize,
        head_dim: usize,
        stream: u64,
    ) -> Result<()> {
        let s = &self.shared;
        let (weight, dyn_taps) = match conv {
            KvConv::Static { weight } => (weight, DevicePtr::NULL),
            KvConv::Dynamic { static_w, .. } => (static_w, dyn_taps),
        };
        ops::short_conv(
            gpu,
            &self.k,
            &ShortConv {
                x,
                out,
                state,
                row_seq: s.row_seq,
                seq_start: s.seq_start,
                slots: s.conv_slot,
                weight,
                dyn_taps,
                rows: 1,
                num_seqs: 1,
                channels,
                k: self.p.conv_k,
                x_stride: channels,
                out_stride: channels,
                dyn_stride: self.p.kv_heads * self.p.conv_k,
                head_dim,
            },
            stream,
        )
    }

    fn block_res(
        &self,
        gpu: &dyn GpuBackend,
        blocks: DevicePtr,
        nb: usize,
        partial_f32: DevicePtr,
        partial_bf16: DevicePtr,
        queries: DevicePtr,
        nq: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.dims.hidden;
        ops::block_res(
            gpu,
            self.k.block_res,
            &BlockRes {
                blocks,
                partial_f32,
                partial_bf16,
                queries,
                out,
                rows: 1,
                hidden: h,
                num_blocks: nb,
                block_stride: self.dims.stack_blocks * h,
                num_queries: nq,
                eps: 1e-6,
            },
            stream,
        )
    }

    fn msg_gate(
        &self,
        gpu: &dyn GpuBackend,
        which: usize,
        h: DevicePtr,
        m: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let g = self
            .w
            .messages
            .context("inkling2: message gate on a layer outside the loop")?[which];
        ops::msg_gate(
            gpu,
            self.k.msg_gate,
            &MsgGate {
                h,
                m,
                down: g.down,
                forget: g.forget,
                gain: g.gain,
                out,
                rows: 1,
                hidden: self.dims.hidden,
                rank: self.p.gate_rank,
                eps: self.p.eps,
            },
            stream,
        )
    }

    fn head_norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        y: DevicePtr,
        scale: DevicePtr,
        log_scaled: bool,
        heads: usize,
        head_dim: usize,
        norm: HeadNorm,
        stream: u64,
    ) -> Result<()> {
        ops::head_norm(
            gpu,
            self.k.head_norm,
            &HeadNormArgs {
                x,
                y,
                scale,
                positions: if log_scaled {
                    self.shared.position
                } else {
                    DevicePtr::NULL
                },
                rows: 1,
                heads,
                head_dim,
                in_stride: heads * head_dim,
                out_stride: heads * head_dim,
                norm,
                log_n_floor: self.p.log_n_floor,
                log_alpha: self.p.log_alpha,
            },
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &PagedKvCache,
        seq_len: usize,
        block_table: &[u32],
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
        let (h, hb, hf) = (d.hidden, d.hidden * 2, d.hidden * 4);
        let conv_state = state
            .as_any_mut()
            .downcast_mut::<Inkling2SeqState>()
            .context("inkling2 layer state")?
            .conv;
        let bs = kv_cache.block_size();

        if v.first {
            self.upload_meta(gpu, seq_len, block_table, bs, stream)?;
            self.gk.rmsnorm(
                gpu,
                DevicePtr::NULL,
                hidden,
                self.g.embed_norm,
                hidden,
                h,
                p.eps,
                stream,
            )?;
        }
        if v.restart {
            // the previous pass: its blocks after the prelude's and its own first push,
            // and its output (this visit's input); then back to the prelude's output
            let skip = (v.blocks_before + 1) * hb;
            gpu.copy_d2d_async(
                s.blocks.offset(skip),
                s.msg_blocks,
                v.message_blocks * hb,
                stream,
            )?;
            gpu.copy_d2d_async(hidden, s.msg_h, hb, stream)?;
            gpu.copy_d2d_async(s.prelude_h, hidden, hb, stream)?;
        }
        let x = hidden;
        let res = |i: usize| s.res3.offset(i * hf);
        let normed = |i: usize| s.normed3.offset(i * hb);

        // attention inputs (fp32 in res3 unless layer 0)
        let n_in = if v.full { 3 } else { 1 };
        let has_res = !w.attn_query.is_null();
        if has_res {
            self.block_res(
                gpu,
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
            let q = self.msg_queries.offset(ll * 4 * hf);
            self.block_res(
                gpu,
                s.msg_blocks,
                v.message_blocks,
                DevicePtr::NULL,
                s.msg_h,
                q,
                4,
                s.msg4,
                stream,
            )?;
            // q, k, v inputs gated by their messages; SWA layers fan one input out
            // to three, so write slots 2 and 1 before slot 0 is overwritten
            for i in [2usize, 1, 0] {
                let src = if v.full { res(i) } else { res(0) };
                self.msg_gate(gpu, i, src, s.msg4.offset(i * hf), res(i), stream)?;
            }
        }
        // input norms -> normed3; aq/ak/av point at the right slot
        let inputs_split = v.full || gated_inputs;
        let src_f32 = |i: usize| if has_res { res(i) } else { DevicePtr::NULL };
        let src_bf16 = if has_res { DevicePtr::NULL } else { x };
        match w.input_norms {
            InputNorms::Shared(nw) if !inputs_split => {
                self.gk
                    .rmsnorm(gpu, src_f32(0), src_bf16, nw, normed(0), h, p.eps, stream)?;
            }
            InputNorms::Shared(nw) => {
                for i in 0..3 {
                    self.gk
                        .rmsnorm(gpu, src_f32(i), src_bf16, nw, normed(i), h, p.eps, stream)?;
                }
            }
            InputNorms::Separate { q, k, v: vn } => {
                for (i, nw) in [q, k, vn].into_iter().enumerate() {
                    self.gk
                        .rmsnorm(gpu, src_f32(i), src_bf16, nw, normed(i), h, p.eps, stream)?;
                }
            }
        }
        let (aq, ak, av) = if inputs_split {
            (normed(0), normed(1), normed(2))
        } else {
            (normed(0), normed(0), normed(0))
        };

        // projections
        let gm = &self.gk;
        gm.gemv(gpu, aq, w.wq, s.q, d.q_dim, h, stream)?;
        gm.gemv(gpu, ak, w.wk, s.k_lin, d.kv_dim, h, stream)?;
        gm.gemv(gpu, av, w.wv, s.v_lin, d.kv_dim, h, stream)?;
        gm.gemv(gpu, aq, w.wr, s.r, d.rel_dim, h, stream)?;
        gm.gemv(gpu, aq, w.gate, s.gate, d.q_dim, h, stream)?;
        gm.gemv(gpu, aq, w.sink, s.sink, p.heads, h, stream)?;
        if let KvConv::Dynamic { proj, .. } = w.k_conv {
            gm.gemv(gpu, ak, proj, s.dyn_k, d.dyn_dim, h, stream)?;
        }
        if let KvConv::Dynamic { proj, .. } = w.v_conv {
            gm.gemv(gpu, av, proj, s.dyn_v, d.dyn_dim, h, stream)?;
        }

        // k/v convs, then the per-head epilogues
        let kvs = d.kv_dim * (p.conv_k - 1) * 4;
        let hs = h * (p.conv_k - 1) * 4;
        self.conv(
            gpu, s.k_lin, s.kc, conv_state, w.k_conv, s.dyn_k, d.kv_dim, p.head_dim, stream,
        )?;
        self.conv(
            gpu,
            s.v_lin,
            s.vc,
            conv_state.offset(kvs),
            w.v_conv,
            s.dyn_v,
            d.kv_dim,
            p.head_dim,
            stream,
        )?;
        let clip = HeadNorm::Clip {
            threshold: p.qk_clip,
        };
        self.head_norm(
            gpu, s.q, s.qn, w.q_scale, v.full, p.heads, p.head_dim, clip, stream,
        )?;
        self.head_norm(
            gpu,
            s.kc,
            s.kn,
            DevicePtr::NULL,
            false,
            p.kv_heads,
            p.head_dim,
            clip,
            stream,
        )?;
        self.head_norm(
            gpu,
            s.vc,
            s.vn,
            DevicePtr::NULL,
            false,
            p.kv_heads,
            p.head_dim,
            HeadNorm::Rms { eps: p.eps },
            stream,
        )?;
        self.head_norm(
            gpu,
            s.r,
            s.rn,
            w.rel_scale,
            v.full,
            p.heads,
            p.d_rel,
            HeadNorm::Clip {
                threshold: p.rel_clip,
            },
            stream,
        )?;

        // KV, attention, gate, output projection, post-attention conv
        let (k_pool, v_pool) = (
            kv_cache.k_pool_ptr(self.visit_idx),
            kv_cache.v_pool_ptr(self.visit_idx),
        );
        gm.kv_write(
            gpu, s.kn, s.vn, k_pool, v_pool, s.slot, p.kv_heads, p.head_dim, bs, stream,
        )?;
        ops::decode_attn(
            gpu,
            self.k.decode_attn,
            &DecodeAttn {
                q: s.qn,
                k_cache: k_pool,
                v_cache: v_pool,
                out: s.attn,
                block_tables: s.block_table,
                seq_lens: s.seq_len,
                rel: s.rn,
                rel_proj: w.rel_proj,
                sinks: s.sink,
                max_blocks_per_seq: d.max_blocks_per_seq,
                num_seqs: 1,
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
        gm.gate_mul(gpu, s.attn, s.gate, s.gated, d.q_dim, stream)?;
        gm.gemv(gpu, s.gated, w.wo, s.proj, h, d.q_dim, stream)?;
        let attn_state = conv_state.offset(2 * kvs);
        self.conv(
            gpu,
            s.proj,
            s.conv_out,
            attn_state,
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
            gpu.copy_d2d_async(x, s.blocks.offset(v.blocks_before * hb), hb, stream)?;
        }
        gm.residual(gpu, x, s.conv_out, s.resid, h, v.boundary, stream)?;
        let nb_mlp = v.blocks_before + usize::from(v.boundary);
        self.block_res(
            gpu,
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
            self.msg_gate(gpu, 3, s.mlp_in, s.msg4.offset(3 * hf), s.mlp_in, stream)?;
        }
        gm.rmsnorm(
            gpu,
            s.mlp_in,
            DevicePtr::NULL,
            w.mlp_norm,
            s.mlp_normed,
            h,
            p.eps,
            stream,
        )?;

        // router, experts, post-MLP conv, layer output
        gm.gemv_f32(gpu, s.mlp_normed, w.router, s.logits, d.experts, h, stream)?;
        gm.route(
            gpu,
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
                rows: 1,
                top_k: d.top_k,
                hidden: h,
                inter: d.inter,
                out_stride: h,
                gate_softcap: p.gate_softcap,
                up_softcap: p.up_softcap,
            },
            stream,
        )?;
        let mlp_state = conv_state.offset(2 * kvs + hs);
        self.conv(
            gpu,
            s.moe_out,
            s.mlp_conv,
            mlp_state,
            KvConv::Static { weight: w.mlp_conv },
            DevicePtr::NULL,
            h,
            h,
            stream,
        )?;
        gm.layer_out(gpu, s.resid, s.mlp_conv, hidden, h, stream)?;

        if v.save_prelude {
            gpu.copy_d2d_async(hidden, s.prelude_h, hb, stream)?;
        }
        if v.last {
            self.block_res(
                gpu,
                s.blocks,
                nb_mlp,
                DevicePtr::NULL,
                hidden,
                self.g.output_query,
                1,
                s.final_out,
                stream,
            )?;
            gm.cast_bf16(gpu, s.final_out, hidden, h, stream)?;
        }
        Ok(())
    }
}

impl TransformerLayer for Inkling2Layer {
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward(hidden, state, kv_cache, seq_len, block_table, ctx, stream)
    }

    /// Visit 0 uploads the token's metadata from the host every step.
    fn decode_graph_unsupported(&self) -> bool {
        true
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let bytes = self.conv_state_bytes();
        let conv = gpu.alloc(bytes).context("allocate inkling2 conv state")?;
        gpu.copy_h2d(&vec![0u8; bytes], conv)?;
        Ok(Box::new(Inkling2SeqState { conv }))
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn LayerState) -> Result<()> {
        if let Some(st) = state.as_any_mut().downcast_mut::<Inkling2SeqState>()
            && !st.conv.is_null()
        {
            gpu.free(st.conv)?;
            st.conv = DevicePtr::NULL;
        }
        Ok(())
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}
