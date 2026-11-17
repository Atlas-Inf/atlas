// SPDX-License-Identifier: AGPL-3.0-only

//! The release's draft head (DSpark: `mtp.*`, `forward_spec` in its
//! `inference/model.py`), on every rank of the model.
//!
//! After the model has chosen a token, a pass proposes the five after it:
//! a block of rows (the token, then noise tokens) goes through three stages,
//! each a block of the model's kind that attends over the main model's last
//! 128 positions and over the block itself, and the stages' output through
//! the model's own head and a Markov head that chains the rows. The model
//! then verifies the proposals, so a pass decides nothing: it only has to
//! be cheap and often right.
//!
//! A pass costs what a block costs, which is reading weights, so the ranks
//! split it as they split a layer: each holds its parts of the stages
//! (`BlockWeights`), a quarter of the rows of the projection in front, and
//! a quarter of the head. The rank that serves the request starts a pass
//! and tells the others (`draft_propose.rs`); all then run [`Draft::pass`].

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::block::{Block, BlockKernels, Candidates, Handoff, Scratch, Shape, Shared};
use super::cache_recipe::WINDOW;
use super::draft_head::{HostHead, KEEP, Kept};
use super::draft_ops::{self, DraftKernels};
use super::draft_seq::{DraftSeq, Plan, slot_runs};
use super::hc;
use super::ops::{self, HEAD_DIM};
use super::parts::{self, Held, PARTS};
use super::shape::ladder;
use super::weights::{BlockWeights, Fp8Proj, fp8_held, row_part, tensor};
use crate::layers::ops::{mxfp8_quantize, rms_norm};
use crate::weight_map::{DenseWeight, Mxfp8Weight};

/// Rows a stage's scratch has room for: a block is five.
const BLOCK_ROOM: usize = 8;

/// The buffers of a pass.
struct Buffers {
    /// What the rank that serves the request tells the others of a pass.
    word: DevicePtr,
    /// For the positions a pass makes keys for, up to a window of them: the
    /// target layers' means in FP8 (`[128, targets * hidden]` BF16), the
    /// projection of them in parts and in rows, its norm, and a stage's keys
    /// before and after theirs.
    means: DevicePtr,
    projected: DevicePtr,
    main: DevicePtr,
    main_normed: DevicePtr,
    keys: DevicePtr,
    keys_normed: DevicePtr,
    recv: DevicePtr,
    /// The block: its rows' embeddings, stream and mix, what the stages
    /// leave of them, the norm of that, one part's logits (`[rows, vocab /
    /// 4]` FP32) and every part's largest (`[4, rows, KEEP]` of a value and
    /// a token).
    embeds: DevicePtr,
    streams: DevicePtr,
    pre: DevicePtr,
    x: DevicePtr,
    normed: DevicePtr,
    logits: DevicePtr,
    best: DevicePtr,
}

pub struct Draft {
    shape: Shape,
    pub(super) held: Held,
    kernels: BlockKernels,
    own: DraftKernels,
    stages: Vec<Block>,
    /// `[hidden, targets * hidden]`: the target layers' means into one row.
    main_proj: Fp8Proj,
    main_norm: DevicePtr,
    norm: DevicePtr,
    /// The model's embedding table, `[vocab, hidden]` BF16; and the parts of
    /// its head this rank holds, in FP8: a draft is checked, not kept, and
    /// the head is a third of what a pass reads.
    embed: DevicePtr,
    head: Mxfp8Weight,
    vocab: usize,
    noise: u32,
    /// Rows of a block.
    pub(super) rows: usize,
    targets: usize,
    buffers: Buffers,
    /// On the rank that serves requests.
    pub(super) host: Option<HostHead>,
}

impl Draft {
    /// This rank's part of the draft head, from `store`. `embed` and `head`
    /// are the model's own.
    pub fn load(
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        embed: DevicePtr,
        head: DevicePtr,
    ) -> Result<Self> {
        let main = Shape::from_config(config)?;
        let shape = Shape {
            experts: match config.dspark_n_routed_experts {
                0 => main.experts,
                experts => experts,
            },
            top_k: match config.dspark_num_experts_per_tok {
                0 => main.top_k,
                top_k => top_k,
            },
            ..main
        };
        let (stages, rows, targets) = (
            config.num_mtp_modules,
            config.dspark_block_size,
            config.dspark_target_layer_ids.len(),
        );
        let (hidden, vocab) = (shape.hidden, config.vocab_size);
        ensure!(
            stages > 0 && (1..=BLOCK_ROOM).contains(&rows) && targets > 0,
            "deepseek_v41: a draft head of {stages} stages, blocks of {rows} and {targets} target layers"
        );
        ensure!(
            vocab.is_multiple_of(PARTS) && hidden.is_multiple_of(PARTS * 32),
            "deepseek_v41: a head of [{vocab}, {hidden}] does not split in {PARTS} parts"
        );
        let held = Held::of_rank(config.ep_rank, config.ep_world_size.max(1))?;
        let stream = gpu.default_stream();
        let kernels = BlockKernels::load(gpu)?;
        let positions = match config.max_seq_len {
            0 => config.max_position_embeddings,
            served => served.min(config.max_position_embeddings),
        };
        let shared = Shared {
            shape,
            held,
            kernels,
            scratch: Arc::new(Scratch::alloc_rows(
                gpu,
                &shape,
                positions + BLOCK_ROOM,
                BLOCK_ROOM,
                held,
            )?),
            handoff: Arc::new(std::sync::Mutex::new(Handoff::default())),
        };
        // A stage attends over a window only, which is not YaRN-scaled.
        let freqs = ladder(gpu, config, shape.rope, 0, config.window_rope_theta)?;
        let blocks = (0..stages)
            .map(|stage| {
                let name = format!("mtp.{stage}");
                let weights = BlockWeights::load_named(
                    store,
                    gpu,
                    &kernels.parts,
                    &shape,
                    &name,
                    0,
                    false,
                    held,
                    true,
                    stream,
                )
                .with_context(|| name.clone())?;
                Ok(Block {
                    index: config.num_hidden_layers + stage,
                    ratio: 0,
                    candidates: Candidates::None,
                    freqs,
                    weights,
                    gemm: None,
                    // A draft is checked, not kept: the fast form.
                    tensor_cores: true,
                    shared: shared.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let main_proj = fp8_held(store, gpu, "mtp.0.main_proj", held)?;
        ensure!(
            main_proj.n == hidden && main_proj.k == targets * hidden,
            "mtp.0.main_proj [{}, {}]",
            main_proj.n,
            main_proj.k
        );
        let last = format!("mtp.{}", stages - 1);
        let bf16 = |values: usize| gpu.alloc(values * 2);
        let copies = shape.hc.copies;
        let buffers = Buffers {
            word: gpu.alloc(64)?,
            means: bf16(WINDOW * targets * hidden)?,
            projected: bf16(WINDOW * hidden)?,
            main: bf16(WINDOW * hidden)?,
            main_normed: bf16(WINDOW * hidden)?,
            keys: bf16(WINDOW * HEAD_DIM)?,
            keys_normed: bf16(WINDOW * HEAD_DIM)?,
            recv: bf16(WINDOW * hidden)?,
            embeds: bf16(rows * hidden)?,
            streams: bf16(rows * copies * hidden)?,
            pre: gpu.alloc(rows * copies * 4)?,
            x: bf16(rows * hidden)?,
            normed: bf16(rows * hidden)?,
            logits: gpu.alloc(rows * vocab / PARTS * 4)?,
            best: gpu.alloc(PARTS * rows * KEEP * 8)?,
        };
        // The Markov and confidence heads are the host's, where requests
        // are served; the other ranks have no use for them.
        let host = HostHead::load(store, gpu, &last, hidden, vocab, config.dspark_markov_rank)?;
        Ok(Self {
            shape,
            held,
            kernels,
            own: DraftKernels::load(gpu)?,
            stages: blocks,
            main_proj,
            main_norm: tensor(store, "mtp.0.main_norm.weight", WeightDtype::BF16)?,
            norm: tensor(store, &format!("{last}.norm.weight"), WeightDtype::BF16)?,
            embed,
            head: {
                let (first, rows) = (held.first * vocab / PARTS, held.count * vocab / PARTS);
                let quantize = gpu.kernel("mxfp8_gemv", "mxfp8_quantize_bf16")?;
                let fp8 = Mxfp8Weight::alloc(gpu, rows, hidden)?;
                let bf16 = head.offset(first * hidden * 2);
                mxfp8_quantize(
                    gpu, quantize, bf16, fp8.data, fp8.scales, rows, hidden, stream,
                )?;
                gpu.synchronize(stream)?;
                fp8
            },
            vocab,
            noise: config.dspark_noise_token_id,
            rows,
            targets,
            buffers,
            host: (config.ep_rank == 0).then_some(host),
        })
    }

    /// Whether `token` is one of the model's.
    pub(super) fn holds(&self, token: u32) -> bool {
        (token as usize) < self.vocab
    }

    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        weight: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let (kernel, weight) = (self.kernels.rms_norm, DenseWeight { weight });
        let (rows, hidden) = (rows as u32, self.shape.hidden as u32);
        rms_norm(
            gpu,
            kernel,
            x,
            &weight,
            out,
            rows,
            hidden,
            self.shape.norm_eps,
            stream,
        )
    }

    /// The word of a pass, which the rank that serves the request says and
    /// the others hear: the token the block starts from and the sequence's
    /// length, or that there is no pass.
    pub(super) fn agree(
        &self,
        gpu: &dyn GpuBackend,
        comm: Option<&dyn CommBackend>,
        said: Option<Option<(u32, usize)>>,
        stream: u64,
    ) -> Result<Option<(u32, usize)>> {
        let Some(comm) = comm.filter(|_| self.held.count < PARTS) else {
            return Ok(said.flatten());
        };
        let word = self.buffers.word;
        let mut bytes = [0u8; 16];
        gpu.synchronize(stream)?;
        if let Some(said) = said {
            let (go, token, len) = match said {
                Some((token, len)) => (1u32, token, len as u32),
                None => (0, 0, 0),
            };
            for (at, value) in [go, token, len].into_iter().enumerate() {
                bytes[at * 4..at * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            gpu.copy_h2d(&bytes, word)?;
        }
        comm.broadcast(word.0, bytes.len(), 0)?;
        gpu.copy_d2h(word, &mut bytes)?;
        let value = |at: usize| u32::from_le_bytes([0, 1, 2, 3].map(|b| bytes[at * 4 + b]));
        Ok((value(0) != 0).then(|| (value(1), value(2) as usize)))
    }

    /// Window keys, in every stage, for the positions of `plan` the stages
    /// have none for: the target layers' means through the projection in
    /// front, which the ranks compute a quarter each, and through each
    /// stage's key projection.
    fn ingest(
        &self,
        gpu: &dyn GpuBackend,
        comm: Option<&dyn CommBackend>,
        seq: &DraftSeq,
        plan: &Plan,
        stream: u64,
    ) -> Result<()> {
        let (b, k) = (&self.buffers, &self.kernels);
        let rows = plan.ingest.len();
        if rows == 0 {
            return Ok(());
        }
        let (hidden, wide) = (self.shape.hidden, self.targets * self.shape.hidden);
        let quant = k.cache.fp8_quant_dequant;
        let mut done = 0;
        for (slot, count) in slot_runs(&plan.ingest) {
            let from = seq.means.offset(slot * wide * 2);
            let to = b.means.offset(done * wide * 2);
            ops::fp8_quant_dequant(gpu, quant, from, to, count * wide, stream)?;
            done += count;
        }
        let part_n = hidden / PARTS;
        for part in self.held.range() {
            let w = row_part(&self.main_proj, part);
            let out = b.projected.offset(part * rows * part_n * 2);
            parts::project_rows(
                gpu,
                k.parts.fp8,
                b.means,
                w.data,
                w.scales,
                out,
                2,
                part_n,
                rows,
                part_n,
                wide,
                stream,
            )?;
        }
        let bytes = rows * part_n * 2;
        parts::gather(
            gpu,
            comm,
            self.held,
            b.projected,
            b.recv,
            bytes,
            bytes,
            stream,
        )?;
        draft_ops::rows_of_parts(gpu, &self.own, b.projected, b.main, rows, part_n, stream)?;
        self.norm(gpu, b.main, self.main_norm, b.main_normed, rows, stream)?;
        ops::fp8_quant_dequant(
            gpu,
            quant,
            b.main_normed,
            b.main_normed,
            rows * hidden,
            stream,
        )?;
        for (stage, window) in self.stages.iter().zip(&seq.windows) {
            stage.main_keys(
                gpu,
                b.main_normed,
                rows,
                *window,
                plan.ingest.start,
                b.keys,
                b.keys_normed,
                stream,
            )?;
        }
        Ok(())
    }

    /// This rank's part of a pass for a sequence of `len` tokens whose next
    /// is `token`: the keys the stages lack, the block through the stages,
    /// and the largest logits of this rank's parts of the head, which the
    /// ranks gather.
    pub(super) fn pass(
        &self,
        gpu: &dyn GpuBackend,
        comm: Option<&dyn CommBackend>,
        token: u32,
        len: usize,
        seq: &mut DraftSeq,
        plan: &Plan,
        stream: u64,
    ) -> Result<()> {
        let (b, k, s) = (&self.buffers, &self.kernels, &self.shape);
        let (rows, hidden) = (self.rows, s.hidden);
        ensure!(
            self.holds(token) && seq.windows.len() == self.stages.len(),
            "deepseek_v41: a draft from token {token} over {} windows",
            seq.windows.len()
        );
        self.ingest(gpu, comm, seq, plan, stream)?;
        // The block: the token, then noise.
        for row in 0..rows {
            let id = if row == 0 { token } else { self.noise } as usize;
            let (from, to) = (
                self.embed.offset(id * hidden * 2),
                b.embeds.offset(row * hidden * 2),
            );
            gpu.copy_d2d_async(from, to, hidden * 2, stream)?;
        }
        hc::start(gpu, &k.hc, &s.hc, b.embeds, b.streams, b.pre, rows, stream)?;
        let seen = plan.slots();
        for (stage, window) in self.stages.iter().zip(&seq.windows) {
            stage.forward_drafts(
                gpu, rows, b.streams, b.pre, *window, &seen, len, comm, stream,
            )?;
        }
        seq.ran(plan);
        hc::collapse(
            gpu,
            k.hc.collapse,
            &s.hc,
            b.streams,
            b.pre,
            b.x,
            rows,
            stream,
        )?;
        self.norm(gpu, b.x, self.norm, b.normed, rows, stream)?;
        // The head a part at a time: its logits, and the largest of them.
        let part_n = self.vocab / PARTS;
        let bytes = rows * KEEP * 8;
        for part in self.held.range() {
            let at = (part - self.held.first) * part_n * hidden;
            let (weights, scales) = (self.head.data.offset(at), self.head.scales.offset(at / 32));
            parts::project_rows(
                gpu,
                k.parts.fp8_f32,
                b.normed,
                weights,
                scales,
                b.logits,
                4,
                part_n,
                rows,
                part_n,
                hidden,
                stream,
            )?;
            let (best, first) = (b.best.offset(part * bytes), part * part_n);
            draft_ops::largest(gpu, &self.own, b.logits, best, rows, part_n, first, stream)?;
        }
        parts::gather(gpu, comm, self.held, b.best, b.recv, bytes, bytes, stream)
    }

    /// What a pass left for the host: each row's largest logits from every
    /// part of the head, and the rows the head read before their norm
    /// (`[rows, hidden]` BF16).
    pub(super) fn read(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<(Vec<Vec<Kept>>, Vec<u16>)> {
        let (b, rows) = (&self.buffers, self.rows);
        gpu.synchronize(stream)?;
        let mut best = vec![0u8; PARTS * rows * KEEP * 8];
        gpu.copy_d2h(b.best, &mut best)?;
        let mut x = vec![0u8; rows * self.shape.hidden * 2];
        gpu.copy_d2h(b.x, &mut x)?;
        let x = x
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok((draft_ops::kept(&best, rows), x))
    }
}
