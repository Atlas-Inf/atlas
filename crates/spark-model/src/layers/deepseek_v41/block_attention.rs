// SPDX-License-Identifier: AGPL-3.0-only

//! The attention sublayer of a DeepSeek-V4.1 block on one token.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::block::Block;
use super::block_index::Chosen;
use super::cache_recipe::{WINDOW, visible_groups, window_row, window_slot};
use super::gemm::{GEMM_ROWS, Input, Sent};
use super::index;
use super::ops::{self, Coded, HEAD_DIM, MAX_KEYS, Turn};
use super::owned::{self, Owned};
use super::parts::{self, PARTS};
use super::profile;
use super::shape::Scratch;
use super::state::{KEPT_ROWS, SeqState};

impl Block {
    /// The device word holding `position`, for a rotation to read.
    pub(super) fn at(&self, position: usize) -> Result<DevicePtr> {
        let sc = &self.shared.scratch;
        ensure!(
            position < sc.max_groups,
            "layer {}: position {position} of at most {}",
            self.index,
            sc.max_groups
        );
        Ok(sc.positions.offset(position * 4))
    }

    /// The window's keys for a batch of more than one row from `position`,
    /// in `scratch.chunk_keys`: the ring unrolled into position order, ahead
    /// of the batch's own, which `front` has written there and which go on
    /// into the ring (`dsv41_window_unroll`). A small batch may be a verify
    /// whose rows are taken back, and leaves the keys its rows displace.
    fn unroll_window(
        &self,
        gpu: &dyn GpuBackend,
        state: &SeqState,
        position: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let (k, sc) = (&self.shared.kernels.cache, &self.shared.scratch);
        let keep = if rows <= KEPT_ROWS { rows } else { 0 };
        KernelLaunch::new(gpu, k.window_unroll)
            .grid([(WINDOW * HEAD_DIM / 8).div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.window)
            .arg_ptr(sc.chunk_keys)
            .arg_ptr(state.kept_window)
            .arg_u32(position as u32)
            .arg_u32(rows as u32)
            .arg_u32(keep as u32)
            .launch(stream)
    }

    /// Attention of the `rows` inputs in `scratch.normed`, and in FP8 in
    /// `scratch.quantized`, into the parts of [`Block::attention_parts`].
    /// The heads, and the output projections behind them, are in four parts;
    /// `comm` reaches the ranks that hold the others.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        rows: usize,
        state: &mut SeqState,
        position: usize,
        comm: Option<&dyn CommBackend>,
        own: Option<Owned>,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights.attn,
        );
        let began = profile::begin(gpu, stream)?;
        // Before the ring is written or read: the keys a verify's refused
        // rows displaced.
        state.restore_window(gpu, k.cache.window_restore, stream)?;
        // The low-rank query and the window key, both from the input's FP8
        // form; then the query normed, and the key normed, turned to the
        // token's position and rounded into the window.
        let gemm = self.gemm.as_ref().filter(|_| rows >= GEMM_ROWS);
        match gemm {
            Some(g) => {
                // Of the rows whose input is here: every row's, or the rows
                // this rank keeps, which the ranks then gather.
                let (at, kept) = own.map_or((0, rows), |own| (own.first, own.count));
                if kept > 0 {
                    let x = Input::rows(sc.quantized.offset(at * s.hidden * 2), s.hidden);
                    let act = sc.gemm.quantize(gpu, x, kept, s.hidden, stream)?;
                    g.wq_a
                        .gemm(act, sc.q_low.offset(at * s.q_rank * 2), kept, stream)?;
                    g.wkv
                        .gemm(act, sc.key.offset(at * HEAD_DIM * 2), kept, stream)?;
                }
                // The query and the key are gathered as the launch below
                // leaves them. A key source and an indexer read every row's
                // input.
                let reads_input =
                    self.weights.compressor.is_some() || self.weights.indexer.is_some();
                if let Some(own) = own.filter(|_| reads_input) {
                    let (held, recv, bytes) = (self.shared.held, sc.gather_recv, s.hidden * 2);
                    owned::gather_rows(gpu, comm, held, own, sc.normed, recv, bytes, stream)?;
                }
            }
            None => parts::project_two(
                gpu,
                k.parts.fp8_two,
                sc.quantized,
                (w.wq_a.rows_from(0), w.wq_a.n, sc.q_low),
                (w.wkv.rows_from(0), w.wkv.n, sc.key),
                s.hidden,
                rows,
                stream,
            )?,
        }
        self.at(position + rows - 1)?;
        let turn = Turn {
            positions: self.at(position)?,
            freqs: self.freqs,
            rope: s.rope,
        };
        // One row writes its key into the ring and reads the ring. More read
        // the window unrolled, with their own keys after it.
        let (window, slot) = match rows {
            1 => (
                state.window,
                state.window.offset(window_slot(position) * HEAD_DIM * 2),
            ),
            _ => (sc.chunk_keys, sc.chunk_keys.offset(WINDOW * HEAD_DIM * 2)),
        };
        // Of the rows whose query and key are here: every row's, or the rows
        // this rank keeps, whose normed query and rounded key the ranks then
        // gather.
        let (at, kept) = own.map_or((0, rows), |own| (own.first, own.count));
        if kept > 0 {
            let turn = Turn {
                positions: self.at(position + at)?,
                ..turn
            };
            ops::front(
                gpu,
                k.cache.front,
                sc.q_low.offset(at * s.q_rank * 2),
                w.q_norm,
                sc.q_low_normed.offset(at * s.q_rank * 2),
                sc.key.offset(at * HEAD_DIM * 2),
                w.kv_norm,
                slot.offset(at * HEAD_DIM * 2),
                turn,
                s.q_rank,
                kept,
                s.norm_eps,
                stream,
            )?;
        }
        if let Some(own) = own {
            let (held, recv) = (self.shared.held, sc.gather_recv);
            for (base, row_bytes) in [(sc.q_low_normed, s.q_rank * 2), (slot, HEAD_DIM * 2)] {
                owned::gather_rows(gpu, comm, held, own, base, recv, row_bytes, stream)?;
            }
        }
        if rows > 1 {
            self.unroll_window(gpu, state, position, rows, stream)?;
        }
        state.kept = (rows > 1 && rows <= KEPT_ROWS).then_some((position, rows));
        profile::end(gpu, stream, "query and window key", began)?;
        let began = profile::begin(gpu, stream)?;
        // Compressed keys: a source writes its own and chooses which a query
        // reads; the layers after it read the same keys at the same choice.
        let mut compressed = state.window;
        let mut chosen = match rows {
            1 => Chosen::Host(Vec::new()),
            _ => Chosen::Table(DevicePtr::NULL),
        };
        if self.ratio > 0 {
            if self.weights.compressor.is_some() {
                self.compress(gpu, state, position, rows, stream)?;
            }
            (compressed, chosen) =
                self.compressed_rows(gpu, row, rows, state, position, comm, own, stream)?;
        }
        // The keys each query reads: its window, then its chosen.
        let keys = match chosen {
            // A row alone reads the ring.
            Chosen::Host(positions) => {
                let mut idx = window_row(position);
                idx.extend(positions);
                let bytes: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();
                // In stream order, behind the kernels of the batch before.
                gpu.copy_h2d_async(&bytes, sc.idx, stream)?;
                idx.len()
            }
            // A batch reads the window unrolled, and the shorter rows end in
            // none; a row alone whose choice stayed on the device, the ring.
            Chosen::Table(table) => {
                let most = match self.ratio {
                    0 => 0,
                    ratio => visible_groups(position + rows - 1, ratio).min(s.index_topk),
                };
                let (keys, topk) = (WINDOW + most, s.index_topk);
                index::attn_index(
                    gpu,
                    k.index.attn_index,
                    table,
                    sc.idx,
                    keys,
                    topk,
                    position,
                    rows,
                    rows == 1,
                    stream,
                )?;
                keys
            }
        };
        profile::end(gpu, stream, "compress and choose", began)?;
        self.attend_keys(gpu, rows, window, compressed, keys, turn, comm, own, stream)
    }

    /// The rest of the attention, from the keys: each of the `rows` queries
    /// in `scratch.q_low_normed` reads the `keys` keys its row of
    /// `scratch.idx` names, the window's of `window` and then those of
    /// `others`, and the heads' output goes through the output projections
    /// into the parts of [`Block::attention_parts`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attend_keys(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        window: DevicePtr,
        others: DevicePtr,
        keys: usize,
        turn: Turn,
        comm: Option<&dyn CommBackend>,
        own: Option<Owned>,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights.attn,
        );
        let held = self.shared.held;
        let gemm = self.gemm.as_ref().filter(|_| rows >= GEMM_ROWS);
        ensure!(
            keys <= MAX_KEYS,
            "layer {}: {keys} keys for one query",
            self.index
        );
        let (heads, groups) = (s.heads / PARTS, s.o_groups / PARTS);
        let group_in = s.heads / s.o_groups * HEAD_DIM;
        ensure!(
            w.wo_a.k == group_in
                && w.wo_a.n == s.o_groups * s.o_rank
                && w.wq_b.n == s.heads * HEAD_DIM,
            "wo_a [{}, {}], wq_b [{}, {}]",
            w.wo_a.n,
            w.wo_a.k,
            w.wq_b.n,
            w.wq_b.k
        );
        for part in held.range() {
            let began = profile::begin(gpu, stream)?;
            // This part's heads: projected from the query's FP8 form, and
            // attended, turned to the position and back inside the kernel.
            let of_held = (part - held.first) * sc.rows * heads * HEAD_DIM * 2;
            let (q, attended) = (sc.q.offset(of_held), sc.attended.offset(of_held));
            match gemm {
                Some(g) => sc.gemm.project(
                    gpu,
                    Input::rows(sc.q_low_normed, s.q_rank),
                    g.wq_b[part].as_ref().context("a held part of wq_b")?,
                    q,
                    rows,
                    stream,
                )?,
                None => {
                    let wq = w.wq_b.rows_from(part * heads * HEAD_DIM);
                    parts::project_rows(
                        gpu,
                        k.parts.fp8,
                        sc.q_low_normed,
                        wq.data,
                        wq.scales,
                        q,
                        2,
                        heads * HEAD_DIM,
                        rows,
                        heads * HEAD_DIM,
                        w.wq_b.k,
                        stream,
                    )?
                }
            }
            // A batch's heads leave the attention in the codes wo_a reads,
            // a group of heads an operand, where the values were.
            let of_scales = (part - held.first) * groups * Scratch::attn_scale_bytes(s, sc.rows);
            let coded = gemm
                .filter(|_| ops::in_one_launch(rows, self.tensor_cores))
                .map(|_| Coded {
                    codes: attended,
                    code_stride: rows * group_in,
                    scales: sc.attn_scales.offset(of_scales),
                    scale_stride: Scratch::attn_scale_bytes(s, sc.rows),
                    group_heads: s.heads / s.o_groups,
                });
            ops::sparse_attn(
                gpu,
                &k.cache,
                q,
                window,
                others,
                sc.idx,
                w.sink.offset(part * heads * 4),
                attended,
                sc.attn_records,
                rows,
                heads,
                keys,
                WINDOW,
                Some(turn),
                self.tensor_cores,
                coded,
                stream,
            )?;
            profile::end(gpu, stream, "sparse attention", began)?;
            let began = profile::begin(gpu, stream)?;
            // Project out, rounding the input to FP8: this part's groups of
            // heads through their blocks of wo_a. A part of the low-rank
            // output follows the last, with the batch's rows and no more.
            let (low, width) = (part * groups * s.o_rank, groups * s.o_rank);
            let o_low = sc.o_low.offset(part * rows * width * 2);
            match gemm {
                // A group a GEMM, and a group's rows together: left as the
                // codes wo_b reads, which are half the bytes to gather.
                Some(g) => {
                    let sent = Sent::new(rows, groups, s.o_rank);
                    let (to, values) = (sc.o_low.offset(part * sent.bytes), sc.gather_recv);
                    for group in 0..groups {
                        let block = g.wo_a[part * groups + group]
                            .as_ref()
                            .context("a held group of wo_a")?;
                        match coded {
                            Some(coded) => {
                                block.gemm(coded.operand(group), values, rows, stream)?
                            }
                            None => {
                                let x = attended.offset(group * group_in * 2);
                                let x = Input::rows(x, groups * group_in);
                                sc.gemm.project(gpu, x, block, values, rows, stream)?
                            }
                        }
                        let at = group * rows * s.o_rank;
                        let scales = to.offset(sent.scales + at / 32);
                        sc.gemm.codes(
                            gpu,
                            values,
                            to.offset(at),
                            scales,
                            rows,
                            s.o_rank,
                            stream,
                        )?;
                    }
                }
                // The heads' output rounded to FP8 once, where it lies.
                None => {
                    let (quant, values) = (k.cache.fp8_quant_dequant, rows * groups * group_in);
                    ops::fp8_quant_dequant(gpu, quant, attended, attended, values, stream)?;
                    parts::project_groups(
                        gpu,
                        k.parts.fp8_groups,
                        attended,
                        w.wo_a.rows_from(low),
                        o_low,
                        rows,
                        groups,
                        s.o_rank,
                        group_in,
                        stream,
                    )?
                }
            }
            profile::end(gpu, stream, "output projections", began)?;
        }
        // A few rows: each part straight on through its columns of wo_b.
        let Some(g) = gemm else {
            return self.output_few(gpu, rows, comm, stream);
        };
        // A batch's wo_b reads every group, so the ranks gather their
        // low-rank outputs, the codes the GEMMs left, and a part computes a
        // quarter of wo_b's rows from them all.
        let (width, part_n) = (groups * s.o_rank, s.hidden / PARTS);
        ensure!(
            w.wo_b.n == s.hidden && w.wo_b.k == PARTS * width,
            "wo_b [{}, {}]",
            w.wo_b.n,
            w.wo_b.k
        );
        let sent = Sent::new(rows, groups, s.o_rank);
        // With the rows shared out and wo_b whole here, a rank takes the rows
        // it keeps through every part of wo_b: it needs the other ranks'
        // codes of those rows alone, a quarter of the gather's traffic, and
        // its output goes nowhere.
        let whole = g.wo_b.iter().all(Option::is_some);
        if let Some(own) = own.filter(|own| whole && held.count == 1 && own.each * PARTS == rows)
            && self.codes_home(comm, own, sent, rows, stream)?
        {
            let began = profile::begin(gpu, stream)?;
            let mine = own.first..own.first + own.count;
            let act = sc
                .gemm
                .gathered_rows(gpu, sc.o_low, sent, rows, mine, stream)?;
            for (part, block) in g.wo_b.iter().flatten().enumerate() {
                let out = sc.attn_out.offset((part * rows + own.first) * part_n * 2);
                block.gemm(act, out, own.count, stream)?;
            }
            return profile::end(gpu, stream, "output projections", began);
        }
        profile::stage(gpu, stream, "gather", || {
            let recv = sc.gather_recv;
            parts::gather(
                gpu, comm, held, sc.o_low, recv, sent.bytes, sent.bytes, stream,
            )
        })?;
        let began = profile::begin(gpu, stream)?;
        for part in held.range() {
            let out = sc.attn_out.offset(part * rows * part_n * 2);
            // Every rank's codes, a row's together.
            let act = sc.gemm.gathered(gpu, sc.o_low, sent, rows, stream)?;
            let block = g.wo_b[part].as_ref().context("a held part of wo_b")?;
            block.gemm(act, out, rows, stream)?;
        }
        profile::end(gpu, stream, "output projections", began)?;
        profile::stage(gpu, stream, "gather", || {
            let (recv, bytes) = (sc.gather_recv, part_n * 2);
            match own {
                // Only the rank that keeps a row reads its output.
                Some(own) => {
                    owned::to_owners(gpu, comm, held, own, sc.attn_out, recv, rows, bytes, stream)
                }
                None => parts::gather(
                    gpu,
                    comm,
                    held,
                    sc.attn_out,
                    recv,
                    rows * bytes,
                    rows * bytes,
                    stream,
                ),
            }
        })
    }
}
