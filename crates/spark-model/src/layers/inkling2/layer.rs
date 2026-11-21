// SPDX-License-Identifier: AGPL-3.0-only

//! One visit of an Inkling 2 physical layer over `rows` consecutive tokens of
//! one sequence (a decode step is one row, a prefill chunk many), in the
//! reference's order and dtypes (transformers PR #51050
//! `Inkling2Loop{Sliding,Full}{,Loop}DecoderLayer`, Apache-2.0). The model
//! runs prefill layer-major, so every per-token state (block stack, prelude
//! save, messages) is kept per row. Visit 0 also applies the embedding norm
//! and uploads the rows' metadata for every visit; restart visits of later
//! loop passes stash the previous pass and restore the prelude's state; the
//! last visit applies the final block residual before the shared final norm.

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

/// One forward's rows.
#[derive(Clone, Copy)]
struct Rows {
    n: usize,
    start: usize,
}

impl Inkling2Layer {
    fn conv_state_bytes(&self) -> usize {
        (2 * self.dims.kv_dim + 2 * self.dims.hidden) * (self.p.conv_k - 1) * 4
    }

    /// Upload the rows' positions, slots, lengths and the block table (visit 0).
    fn upload_meta(
        &self,
        gpu: &dyn GpuBackend,
        rows: Rows,
        block_table: &[u32],
        bs: usize,
        stream: u64,
    ) -> Result<()> {
        let s = &self.shared;
        ensure!(
            rows.n <= self.dims.max_rows,
            "inkling2: {} rows over the {}-row buffers",
            rows.n,
            self.dims.max_rows
        );
        ensure!(
            block_table.len() <= self.dims.max_blocks_per_seq,
            "inkling2: block table over its buffer"
        );
        let (mut pos, mut slot, mut len) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..rows.n {
            let p = rows.start + r;
            let blk = *block_table
                .get(p / bs)
                .with_context(|| format!("inkling2: no KV block for position {p}"))?;
            pos.extend((p as i32).to_le_bytes());
            slot.extend((i64::from(blk) * bs as i64 + (p % bs) as i64).to_le_bytes());
            len.extend(((p + 1) as i32).to_le_bytes());
        }
        // earlier kernels of this stream may still read the previous forward's metadata
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&pos, s.positions)?;
        gpu.copy_h2d(&slot, s.slots)?;
        gpu.copy_h2d(&len, s.seq_lens)?;
        gpu.copy_h2d(
            &[0i32.to_le_bytes(), (rows.n as i32).to_le_bytes()].concat(),
            s.seq_start,
        )?;
        let table: Vec<u8> = block_table.iter().flat_map(|b| b.to_le_bytes()).collect();
        gpu.copy_h2d(&table, s.block_table)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        gpu: &dyn GpuBackend,
        rows: Rows,
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
                rows: rows.n,
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

    #[allow(clippy::too_many_arguments)]
    fn block_res(
        &self,
        gpu: &dyn GpuBackend,
        rows: Rows,
        blocks: DevicePtr,
        nb: usize,
        partial_f32: DevicePtr,
        partial_bf16: DevicePtr,
        queries: DevicePtr,
        nq: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ops::block_res(
            gpu,
            self.k.block_res,
            &BlockRes {
                blocks,
                partial_f32,
                partial_bf16,
                queries,
                out,
                rows: rows.n,
                hidden: self.dims.hidden,
                num_blocks: nb,
                row_stride: self.dims.hidden,
                block_stride: self.dims.block_stride(),
                num_queries: nq,
                eps: 1e-6,
            },
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn msg_gate(
        &self,
        gpu: &dyn GpuBackend,
        rows: Rows,
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
                rows: rows.n,
                hidden: self.dims.hidden,
                rank: self.p.gate_rank,
                eps: self.p.eps,
            },
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn head_norm(
        &self,
        gpu: &dyn GpuBackend,
        rows: Rows,
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
                    self.shared.positions
                } else {
                    DevicePtr::NULL
                },
                rows: rows.n,
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
        rows: Rows,
        state: &mut dyn LayerState,
        kv_cache: &PagedKvCache,
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
        let n = rows.n;
        let (h, rb, rf) = (d.hidden, n * d.hidden * 2, n * d.hidden * 4);
        let bstride = d.block_stride() * 2; // bytes between stack blocks
        let conv_state = state
            .as_any_mut()
            .downcast_mut::<Inkling2SeqState>()
            .context("inkling2 layer state")?
            .conv;
        let bs = kv_cache.block_size();
        let gm = &self.gk;

        if v.first {
            self.upload_meta(gpu, rows, block_table, bs, stream)?;
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
            debug_dump(gpu, hidden, rows, h, u32::MAX, stream)?;
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

        // attention inputs (fp32 in res3 unless layer 0)
        let n_in = if v.full { 3 } else { 1 };
        let has_res = !w.attn_query.is_null();
        if has_res {
            self.block_res(
                gpu,
                rows,
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
                rows,
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
                self.msg_gate(gpu, rows, i, src, s.msg4.offset(i * rf), res(i), stream)?;
            }
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
        let kvs = d.kv_dim * (p.conv_k - 1) * 4;
        let hs = h * (p.conv_k - 1) * 4;
        self.conv(
            gpu, rows, s.k_lin, s.kc, conv_state, w.k_conv, s.dyn_k, d.kv_dim, p.head_dim, stream,
        )?;
        self.conv(
            gpu,
            rows,
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
            gpu, rows, s.q, s.qn, w.q_scale, v.full, p.heads, p.head_dim, clip, stream,
        )?;
        self.head_norm(
            gpu,
            rows,
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
            rows,
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
            rows,
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

        // KV of every row first, then each row attends over its own prefix
        let (k_pool, v_pool) = (
            kv_cache.k_pool_ptr(self.visit_idx),
            kv_cache.v_pool_ptr(self.visit_idx),
        );
        gm.kv_write(
            gpu, n, s.kn, s.vn, k_pool, v_pool, s.slots, p.kv_heads, p.head_dim, bs, stream,
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
                seq_lens: s.seq_lens,
                rel: s.rn,
                rel_proj: w.rel_proj,
                sinks: s.sink,
                // every row shares the sequence's one block table
                max_blocks_per_seq: 0,
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
        let attn_state = conv_state.offset(2 * kvs);
        self.conv(
            gpu,
            rows,
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
            gpu.copy_d2d_async(x, s.blocks.offset(v.blocks_before * bstride), rb, stream)?;
        }
        gm.residual(gpu, x, s.conv_out, s.resid, n * h, v.boundary, stream)?;
        let nb_mlp = v.blocks_before + usize::from(v.boundary);
        self.block_res(
            gpu,
            rows,
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
            self.msg_gate(
                gpu,
                rows,
                3,
                s.mlp_in,
                s.msg4.offset(3 * rf),
                s.mlp_in,
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
        let mlp_state = conv_state.offset(2 * kvs + hs);
        self.conv(
            gpu,
            rows,
            s.moe_out,
            s.mlp_conv,
            mlp_state,
            KvConv::Static { weight: w.mlp_conv },
            DevicePtr::NULL,
            h,
            h,
            stream,
        )?;
        gm.layer_out(gpu, s.resid, s.mlp_conv, hidden, n * h, stream)?;
        debug_dump(gpu, hidden, rows, h, self.visit_idx as u32, stream)?;
        if debug_tap_visit() == Some(self.visit_idx) {
            self.debug_taps(gpu, rows, stream)?;
        }

        if v.save_prelude {
            gpu.copy_d2d_async(hidden, s.prelude_h, rb, stream)?;
        }
        if v.last {
            self.block_res(
                gpu,
                rows,
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

    fn debug_taps(&self, gpu: &dyn GpuBackend, rows: Rows, stream: u64) -> Result<()> {
        let (s, d, p) = (&self.shared, &self.dims, &self.p);
        let h = d.hidden;
        let taps: [(u32, DevicePtr, usize); 16] = [
            (1, s.res3, h * 4),
            (2, s.normed3, h * 2),
            (3, s.qn, d.q_dim * 2),
            (4, s.kn, d.kv_dim * 2),
            (5, s.vn, d.kv_dim * 2),
            (6, s.rn, d.rel_dim * 2),
            (7, s.sink, p.heads * 2),
            (8, s.attn, d.q_dim * 2),
            (9, s.proj, h * 2),
            (10, s.conv_out, h * 2),
            (11, s.resid, h * 4),
            (12, s.mlp_in, h * 4),
            (13, s.mlp_normed, h * 2),
            (14, s.ids, d.top_k * 4),
            (16, s.moe_out, h * 2),
            (17, s.mlp_conv, h * 2),
        ];
        for r in 0..rows.n {
            for (tag, ptr, row_bytes) in taps {
                debug_tap(
                    gpu,
                    ptr.offset(r * row_bytes),
                    row_bytes,
                    tag,
                    rows.start + r,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

/// Debug: with `ATLAS_INKLING2_DUMP=<file>`, append `(visit u32, position u32,
/// BF16 hidden row)` records (visit `u32::MAX` = the normed embedding), for
/// comparison with the transformers reference's per-visit hidden states.
fn debug_dump(
    gpu: &dyn GpuBackend,
    hidden: DevicePtr,
    rows: Rows,
    h: usize,
    visit: u32,
    stream: u64,
) -> Result<()> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var("ATLAS_INKLING2_DUMP").ok()) else {
        return Ok(());
    };
    use std::io::Write;
    gpu.synchronize(stream)?;
    let mut all = vec![0u8; rows.n * h * 2];
    gpu.copy_d2h(hidden, &mut all)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for (r, row) in all.chunks_exact(h * 2).enumerate() {
        f.write_all(&visit.to_le_bytes())?;
        f.write_all(&((rows.start + r) as u32).to_le_bytes())?;
        f.write_all(row)?;
    }
    Ok(())
}

/// Debug: the visit whose intermediates `ATLAS_INKLING2_TAP=<file>` records.
fn debug_tap_visit() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_INKLING2_TAP")
            .ok()
            .and(std::env::var("ATLAS_INKLING2_TAP_VISIT").ok())
            .and_then(|v| v.parse().ok())
    })
}

/// Append `(tag u32, position u32, len u32, raw bytes)` to `ATLAS_INKLING2_TAP`.
fn debug_tap(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    bytes: usize,
    tag: u32,
    pos: usize,
    stream: u64,
) -> Result<()> {
    use std::io::Write;
    let path = std::env::var("ATLAS_INKLING2_TAP")?;
    gpu.synchronize(stream)?;
    let mut buf = vec![0u8; bytes];
    gpu.copy_d2h(ptr, &mut buf)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(&tag.to_le_bytes())?;
    f.write_all(&(pos as u32).to_le_bytes())?;
    f.write_all(&(bytes as u32).to_le_bytes())?;
    f.write_all(&buf)?;
    Ok(())
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
        let rows = Rows {
            n: 1,
            start: seq_len,
        };
        self.forward(hidden, rows, state, kv_cache, block_table, ctx, stream)
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rows = Rows {
            n: num_tokens,
            start: seq_len_start,
        };
        self.forward(hidden, rows, state, kv_cache, block_table, ctx, stream)
    }

    /// Visit 0 uploads the rows' metadata from the host every forward.
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
