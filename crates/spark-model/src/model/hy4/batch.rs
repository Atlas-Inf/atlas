// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::hy4::batch::{self, MAX_ROWS};
use crate::layers::hy4::timing;
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::atomic::Ordering;

pub(super) const BATCH: u32 = 4;
pub(super) const LAST: u32 = 5;
pub(super) const VERIFY: u32 = 6;

pub(super) struct Workspace<'a> {
    _memory: Owned<'a>,
    pub capacity: usize,
    pub tokens: DevicePtr,
    highway: DevicePtr,
    pub normed: DevicePtr,
    output: DevicePtr,
    pub post: DevicePtr,
    pub logits: DevicePtr,
}
impl<'a> Workspace<'a> {
    pub fn new(s: &'a Shared) -> Result<Self> {
        let capacity = MAX_ROWS.min(s.max_seq);
        let h = s.config.hidden_size;
        let mut m = Owned::new(s.gpu.as_ref());
        Ok(Self {
            capacity,
            tokens: m.alloc(capacity * 4)?,
            highway: m.alloc(capacity * 4 * h * 2)?,
            normed: m.alloc(capacity * h * 2)?,
            output: m.alloc(capacity * h * 2)?,
            post: m.alloc(capacity * 4 * 4)?,
            logits: m.alloc(capacity * s.config.vocab_size * 4)?,
            _memory: m,
        })
    }
}
impl Runtime<'_> {
    pub(super) fn rows(
        &self,
        s: &Shared,
        slot: usize,
        tokens: &[u32],
        mode: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= self.multi.capacity
                && matches!(mode, BATCH | LAST | VERIFY),
            "Hy4 invalid batch geometry/logit mode"
        );
        let mut requests = self.requests.lock();
        let cell = requests
            .get_mut(&slot)
            .context("Hy4 batch request missing")?;
        cell.with_dependent_mut(|owner, data| {
            ensure!(
                !data.poisoned
                    && data
                        .history
                        .len()
                        .checked_add(tokens.len())
                        .is_some_and(|n| n <= s.max_seq)
                    && tokens.iter().all(|t| (*t as usize) < s.config.vocab_size),
                "Hy4 batch state/token/capacity"
            );
            let result = (|| {
                let gpu = s.gpu.as_ref();
                let h = s.config.hidden_size;
                let w = &self.multi;
                let n = tokens.len();
                let start = data.history.len();
                let timer = Some(timing::Token::begin_batch(
                    gpu,
                    stream,
                    s.config.tp_rank,
                    start,
                    n,
                ));
                let row_span = timing::span("other", stream)?;
                let mut history = data.history.clone();
                history.extend_from_slice(tokens);
                let positions = (start..start + n)
                    .map(|p| [p as i64; 3])
                    .collect::<Vec<_>>();
                KernelLaunch::new(
                    gpu,
                    gpu.op_cache()
                        .kernel(gpu, "hy4_model", "hy4_model_batch_embed")?,
                )
                .grid([(4 * h).div_ceil(256) as u32, n as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(self.embed)
                .arg_ptr(w.tokens)
                .arg_ptr(w.highway)
                .arg_u32(h as u32)
                .launch(stream)?;
                for (cell, state) in owner.layers.iter().zip(&mut data.layers) {
                    let b = cell.borrow_dependent();
                    if let Some(e) = &b.engram {
                        let _engram = timing::span("engram", stream)?;
                        e.forward_batch(
                            w.highway,
                            &history,
                            start,
                            state.engram.as_mut().context("Hy4 batch Engram scratch")?,
                            s.comm.as_deref(),
                            stream,
                        )?;
                    }
                    let hc = timing::span("ihc_norms", stream)?;
                    b.hc_attn.collapse_norm(
                        gpu,
                        w.highway,
                        b.input_norm,
                        w.normed,
                        w.post,
                        n as u32,
                        stream,
                    )?;
                    drop(hc);
                    let attn = timing::span("attention", stream)?;
                    if mode == VERIFY && data.checkpoint.is_some() {
                        state.attn.begin_prefix(n, stream)?;
                    }
                    state.attn.forward(
                        w.normed,
                        w.output,
                        &positions,
                        s.comm.as_deref(),
                        stream,
                    )?;
                    drop(attn);
                    let hc = timing::span("ihc_norms", stream)?;
                    if n <= 8 && h <= 4096 {
                        b.hc_mlp.scatter_collapse_norm(
                            gpu, w.output, w.highway, w.post, b.mlp_norm, w.normed, n as u32,
                            stream,
                        )?;
                    } else {
                        b.hc_attn.scatter(
                            gpu,
                            [w.output, w.highway, w.post, w.highway],
                            n as u32,
                            stream,
                        )?;
                        b.hc_mlp.collapse_norm(
                            gpu, w.highway, b.mlp_norm, w.normed, w.post, n as u32, stream,
                        )?;
                    }
                    drop(hc);
                    b.ffn.forward(
                        w.normed,
                        w.output,
                        n as u32,
                        &mut state.ffn,
                        s.comm.as_deref(),
                        stream,
                    )?;
                    let _hc = timing::span("ihc_norms", stream)?;
                    b.hc_mlp.scatter(
                        gpu,
                        [w.output, w.highway, w.post, w.highway],
                        n as u32,
                        stream,
                    )?;
                }
                let logits = if mode == BATCH {
                    DevicePtr::NULL
                } else {
                    let (input, rows) = if mode == LAST {
                        (w.highway.offset((n - 1) * 4 * h * 2), 1)
                    } else {
                        (w.highway, n)
                    };
                    let hc = timing::span("ihc_norms", stream)?;
                    self.head.collapse_norm(
                        gpu,
                        input,
                        self.final_norm,
                        w.normed,
                        DevicePtr::NULL,
                        rows as u32,
                        stream,
                    )?;
                    drop(hc);
                    if mode == LAST {
                        let head = timing::span("lm_head", stream)?;
                        KernelLaunch::new(gpu, self.kernels[2])
                            .grid([self.vocab.rows.div_ceil(8) as u32, 1, 1])
                            .block([256, 1, 1])
                            .arg_ptr(w.normed)
                            .arg_ptr(self.vocab.weight)
                            .arg_ptr(self.logits.offset(self.vocab.start * 4))
                            .arg_u32(self.vocab.rows as u32)
                            .arg_u32(h as u32)
                            .arg_u32(0)
                            .launch(stream)?;
                        drop(head);
                        let _gather = timing::span("logits_gather", stream)?;
                        self.vocab.gather(s, self.logits, stream)?;
                        self.logits
                    } else {
                        let head = timing::span("lm_head", stream)?;
                        batch::gemm(
                            gpu,
                            [
                                w.normed,
                                self.vocab.weight,
                                DevicePtr::NULL,
                                w.logits.offset(self.vocab.start * 4),
                            ],
                            [self.vocab.rows, h, rows, 1, 0, 0, h, s.config.vocab_size, h],
                            [0, 1],
                            stream,
                        )?;
                        drop(head);
                        let _gather = timing::span("logits_gather", stream)?;
                        for row in 0..rows {
                            self.vocab.gather(
                                s,
                                w.logits.offset(row * s.config.vocab_size * 4),
                                stream,
                            )?;
                        }
                        w.logits
                    }
                };
                if data.draft.is_some() {
                    if mode == BATCH {
                        self.head.collapse_norm(
                            gpu,
                            w.highway.offset((n - 1) * 4 * h * 2),
                            self.final_norm,
                            w.normed,
                            DevicePtr::NULL,
                            1,
                            stream,
                        )?;
                    }
                    let hidden = w
                        .normed
                        .offset(if mode == VERIFY { (n - 1) * h * 2 } else { 0 });
                    self.capture_draft(data, hidden, stream)?;
                    if mode == VERIFY && data.checkpoint.is_some() {
                        self.capture_draft_prefix(data, w.normed, start, n, stream)?;
                    }
                }
                drop(row_span);
                // BATCH chunks produce no logits and the host never reads GPU
                // state mid-prompt, so they must not drain the pipeline; only
                // LAST/VERIFY need the sync (logits readback / draft capture).
                if mode != BATCH {
                    gpu.synchronize(stream)?;
                }
                if let Some(timer) = timer {
                    timer.finish()?;
                }
                Ok(logits)
            })();
            if result.is_ok() {
                data.history.extend_from_slice(tokens);
                self.verify_logits.store(mode == VERIFY, Ordering::Relaxed);
                self.head_rows.store(
                    if mode == LAST {
                        1
                    } else if mode == VERIFY {
                        tokens.len()
                    } else {
                        0
                    },
                    Ordering::Relaxed,
                );
            } else {
                data.poisoned = true;
            }
            result
        })
    }
}

impl Hy4Model {
    pub(super) fn forward_rows(
        &self,
        tokens: &[u32],
        seq: &mut crate::traits::SequenceState,
        mode: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let s = self.borrow_owner();
        let r = self.borrow_dependent();
        let stream = self.eager_stream(stream)?;
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= r.multi.capacity
                && matches!(mode, BATCH | LAST | VERIFY)
                && seq
                    .seq_len
                    .checked_add(tokens.len())
                    .is_some_and(|n| n <= s.max_seq)
                && tokens.iter().all(|t| (*t as usize) < s.config.vocab_size),
            "Hy4 batch input bounds"
        );
        {
            let requests = r.requests.lock();
            let request = requests
                .get(&seq.slot_idx)
                .context("Hy4 batch slot not allocated")?;
            ensure!(
                !request.borrow_dependent().poisoned
                    && request.borrow_dependent().history.len() == seq.seq_len,
                "Hy4 batch position mismatch"
            );
        }
        s.gpu.copy_h2d(
            &tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<_>>(),
            r.multi.tokens,
        )?;
        self.send([
            mode,
            u32::try_from(seq.slot_idx)?,
            tokens.len() as u32,
            u32::try_from(seq.seq_len)?,
        ])?;
        if let Some(c) = s.comm.as_deref().filter(|c| c.world_size() > 1) {
            c.broadcast(r.multi.tokens.0, tokens.len() * 4, 0)?;
        }
        let ptr = r.rows(s, seq.slot_idx, tokens, mode, stream)?;
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len += tokens.len();
        Ok(ptr)
    }
}
