// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2's DSpark drafter on the DFlash head (`thinkingmachines/
//! Inkling-2-Mini` `mtp.safetensors`; reference vLLM PR #69028
//! `models/inkling2/nvidia/dspark.py`, Apache-2.0).
//!
//! The block body is the DFlash head's: 3 bidirectional layers (local layers
//! 0 and 2 on a 1,032-key window), q/k norms, neox RoPE, SwiGLU. Inkling adds:
//!
//! - context: each draft layer mixes the four taps (target visits 5, 23, 41
//!   inputs and the final-norm output) with its own `softmax(fuse_weight)`
//!   row and RMSNorm (`dflare_norm`), then projects its own context K/V
//!   (`k_proj_ctx`, `v_proj_ctx`, `k_norm_ctx`);
//! - block input: the anchor's normalized target embedding, then `mask_embed`;
//! - tail: a recurrent (`rnn_hidden`) Markov head over the drafter's normed
//!   rows and `f_rnn` (the Markov mix of the last committed token's taps),
//!   whose latent goes through `hidden_out` and the target's head, folded at
//!   load into one `[vocab, 264]` table (bias in column 256).
//!
//! Kernels: `kernels/gb10/inkling-2-mini/nvfp4/inkling2_dspark.cu`. Drafts
//! only steer what the target verifies; the verify decides every token.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{BlockDiffusionDraftHead, DflashScratch};
use crate::layers::ops;
use crate::weight_loader::dflash_loader::{INKLING2_DSPARK_TAPS, Inkling2DsparkWeights};
use crate::weight_map::DenseWeight;

/// Fold table width: the Markov rank, the bias column, zero padding to 8.
const FOLD_K: usize = 264;
const RANK: usize = 256;

struct Kernels {
    mix: KernelHandle,
    block_input: KernelHandle,
    rnn_ctx: KernelHandle,
    rnn_raw: KernelHandle,
    rnn_gate: KernelHandle,
    pick: KernelHandle,
}

impl Kernels {
    fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |f: &str| {
            gpu.kernel("inkling2_dspark", f)
                .with_context(|| format!("Inkling 2 DSpark kernel inkling2_dspark::{f}"))
        };
        Ok(Self {
            mix: k("inkling2_dspark_mix")?,
            block_input: k("inkling2_dspark_block_input")?,
            rnn_ctx: k("inkling2_dspark_rnn_ctx")?,
            rnn_raw: k("inkling2_dspark_rnn_raw")?,
            rnn_gate: k("inkling2_dspark_rnn_gate")?,
            pick: k("inkling2_dspark_pick")?,
        })
    }
}

/// The variant's device state (weights and kernels), on the head.
pub struct InklingDspark {
    w: Inkling2DsparkWeights,
    fold: DenseWeight,
    k: Kernels,
    /// A step's shared head pass (`inkling_group.rs`).
    group: group::TailGroup,
    /// The staged batch's per-sequence tail buffers (`inkling_batch.rs`).
    batch: batch::BatchScratch,
    /// Report each draft's confidence after the block's tokens
    /// (`draft_conf.rs`): the log-probability of the pick under the
    /// softcapped logits, for the scheduler's verify width.
    conf: bool,
}

/// The checkpoint's `final_logit_softcapping`, which the reference applies
/// to a draft row's logits (`markov_logits`). It does not move the argmax;
/// the confidence is taken under it.
const LOGIT_SOFTCAP: f32 = 30.0;

/// One propose lane's buffers.
pub struct InklingScratch {
    f_rnn: DevicePtr,
    ctx: DevicePtr,
    raw: DevicePtr,
    state: DevicePtr,
    latent: DevicePtr,
    bias: DevicePtr,
    /// `[ctx rows, 2 * kv_dim]` one layer's context K/V before the scatter.
    kv_tmp: DevicePtr,
    /// Draft steps the next tail runs; 0 is the whole block. A step that
    /// verifies fewer drafts than the block holds (two or more live
    /// sequences) sets it, so the tail stops there: each step is a pass over
    /// the folded head (about 0.4 ms on one GB10).
    tail_steps: std::sync::atomic::AtomicUsize,
    /// The next tail runs here whatever the group holds (a graph capture).
    inline_tail: std::sync::atomic::AtomicBool,
    /// The last tail was held for the group's head pass: no drafts yet.
    held: std::sync::atomic::AtomicBool,
}

/// The stored step count for a tail of `drafts` drafts in a `gamma`-row
/// block: 0 for the whole block (`gamma - 1` drafts or more), else the
/// drafts, at least one.
fn tail_steps_for(drafts: usize, gamma: usize) -> usize {
    if drafts + 1 < gamma { drafts.max(1) } else { 0 }
}

impl InklingScratch {
    pub fn alloc(
        gpu: &dyn GpuBackend,
        hidden: usize,
        gamma: usize,
        vocab: usize,
        ctx_rows: usize,
        kv_dim: usize,
    ) -> Result<Self> {
        let bf16 = 2;
        let a = |bytes: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(bytes).context("Inkling 2 DSpark scratch")?;
            gpu.copy_h2d(&vec![0u8; bytes], p)?;
            Ok(p)
        };
        Ok(Self {
            f_rnn: a(hidden * bf16)?,
            ctx: a(gamma * 3 * RANK * 4)?,
            raw: a(3 * RANK * 4)?,
            state: a(RANK * bf16)?,
            latent: a(FOLD_K * bf16)?,
            bias: a(vocab * bf16)?,
            kv_tmp: a(ctx_rows.max(1) * 2 * kv_dim * bf16)?,
            tail_steps: Default::default(),
            inline_tail: Default::default(),
            held: Default::default(),
        })
    }

    /// Keep the next tail out of the group (`true`) or let it join.
    pub(super) fn set_inline_tail(&self, inline: bool) {
        self.inline_tail
            .store(inline, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the last tail was held for the group, clearing the mark.
    pub(super) fn take_held(&self) -> bool {
        self.held.swap(false, std::sync::atomic::Ordering::Relaxed)
    }

    /// Run the next tail for `drafts` drafts of a `gamma`-row block.
    pub(super) fn set_tail_steps(&self, drafts: usize, gamma: usize) {
        self.tail_steps.store(
            tail_steps_for(drafts, gamma),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// The next tail's step count, if shorter than the block's.
    pub(super) fn short_tail(&self) -> Option<usize> {
        match self.tail_steps.load(std::sync::atomic::Ordering::Relaxed) {
            0 => None,
            steps => Some(steps),
        }
    }
}

impl InklingDspark {
    /// Resolve the kernels and fold `hidden_out` into the target's BF16 head
    /// (`lm_head [vocab, H]`): `fold[v] = lm_head[v] . [hidden_out^T; b; 0]`.
    pub fn build(
        w: Inkling2DsparkWeights,
        lm_head: DevicePtr,
        vocab: usize,
        hidden: usize,
        gamma: usize,
        batch_capacity: usize,
        dense_gemm: KernelHandle,
        conf: bool,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            !lm_head.is_null() && w.markov_rank == RANK,
            "Inkling 2 DSpark needs the target's BF16 head and Markov rank {RANK}"
        );
        ensure!(
            !w.target_final_norm.is_null(),
            "Inkling 2 DSpark: target norms not attached"
        );
        let bf16 = 2;
        let mut out_w = vec![0u8; hidden * RANK * bf16];
        gpu.copy_d2h(w.hidden_out, &mut out_w)?;
        let mut out_b = vec![0u8; hidden * bf16];
        gpu.copy_d2h(w.hidden_out_bias, &mut out_b)?;
        // [FOLD_K, H]: rows r < RANK = hidden_out[:, r], row RANK = bias, rest 0
        let mut b = vec![0u8; FOLD_K * hidden * bf16];
        for r in 0..RANK {
            for k in 0..hidden {
                let src = (k * RANK + r) * bf16;
                let dst = (r * hidden + k) * bf16;
                b[dst..dst + 2].copy_from_slice(&out_w[src..src + 2]);
            }
        }
        b[RANK * hidden * bf16..(RANK + 1) * hidden * bf16].copy_from_slice(&out_b);
        let rhs = gpu.alloc(b.len())?;
        gpu.copy_h2d(&b, rhs)?;
        let fold = gpu
            .alloc(vocab * FOLD_K * bf16)
            .context("Inkling 2 DSpark fold table")?;
        let stream = gpu.default_stream();
        ops::dense_gemm(
            gpu,
            dense_gemm,
            lm_head,
            &DenseWeight { weight: rhs },
            fold,
            vocab as u32,
            FOLD_K as u32,
            hidden as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        gpu.free(rhs)?;
        tracing::info!(
            "Inkling 2 DSpark: fold table [{vocab}, {FOLD_K}] BF16 ({:.1} MB) from the target head",
            (vocab * FOLD_K * bf16) as f64 / 1e6
        );
        Ok(Self {
            w,
            fold: DenseWeight { weight: fold },
            k: Kernels::resolve(gpu)?,
            group: group::TailGroup::alloc(gpu, hidden, gamma, vocab)?,
            batch: batch::BatchScratch::alloc(gpu, batch_capacity.max(1), hidden, gamma, vocab)?,
            conf,
        })
    }

    /// The context key norm of draft layer `l`.
    pub fn ctx_k_norm(&self, l: usize) -> &DenseWeight {
        &self.w.k_norm_ctx[l]
    }

    /// `out [rows, H] = mix_row(row)` of `rows` accumulator rows at `src`.
    #[allow(clippy::too_many_arguments)]
    fn mix(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        rows: u32,
        hidden: u32,
        row: usize,
        norm: DevicePtr,
        eps: f32,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let a = self.w.alpha[row];
        KernelLaunch::new(gpu, self.k.mix)
            .grid([rows, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_u32(INKLING2_DSPARK_TAPS as u32 * hidden)
            .arg_f32(a[0])
            .arg_f32(a[1])
            .arg_f32(a[2])
            .arg_f32(a[3])
            .arg_ptr(norm)
            .arg_ptr(self.w.target_final_norm)
            .arg_f32(eps)
            .arg_f32(self.w.target_eps)
            .arg_ptr(out)
            .arg_u32(hidden)
            .launch(stream)
    }
}

impl BlockDiffusionDraftHead {
    /// Context K/V for `n` accumulator rows at `src`: per draft layer, its
    /// tap mix into `fc_dst`, its `[k_proj_ctx; v_proj_ctx]` GEMM, and a
    /// strided copy into `fused_dst`'s `[K_l | V_l]` columns — the layout
    /// `ctx_kv_scatter` reads (with `k_norm_ctx`, see [`Self::ctx_k_norm`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inkling_ctx_kv_project(
        &self,
        ink: &InklingDspark,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        n: u32,
        fc_dst: DevicePtr,
        fused_dst: DevicePtr,
        scratch: &DflashScratch,
        stream: u64,
    ) -> Result<()> {
        let sc = scratch
            .inkling
            .as_ref()
            .context("Inkling 2 DSpark lane scratch")?;
        let h = self.hidden_size as u32;
        let kv2 = 2 * self.num_kv_heads * self.head_dim;
        let bf16 = 2usize;
        for l in 0..self.num_layers {
            ink.mix(
                gpu,
                src,
                n,
                h,
                l,
                ink.w.dflare_norms[l].weight,
                self.rms_norm_eps,
                fc_dst,
                stream,
            )?;
            self.drafter_dense_gemm(
                gpu,
                fc_dst,
                &ink.w.kv_proj_ctx[l],
                sc.kv_tmp,
                n,
                kv2 as u32,
                h,
                stream,
            )?;
            gpu.copy_d2d_2d_async(
                sc.kv_tmp,
                kv2 * bf16,
                fused_dst.offset(l * kv2 * bf16),
                self.num_layers * kv2 * bf16,
                kv2 * bf16,
                n as usize,
                stream,
            )?;
        }
        Ok(())
    }

    /// The key norm `ctx_kv_scatter` applies to draft layer `l`'s context K.
    pub(super) fn ctx_k_norm(&self, l: usize) -> &DenseWeight {
        match self.inkling.as_ref() {
            Some(ink) => ink.ctx_k_norm(l),
            None => &self.layers[l].k_norm,
        }
    }

    /// Before the propose graphs: the block input over `stream_buf`'s γ rows
    /// (embedded anchor, then masks) and `f_rnn` from the newest of the
    /// `ctx_rows` accumulator rows at `acc` (zero without context).
    pub(super) fn inkling_prepare(
        &self,
        gpu: &dyn GpuBackend,
        scratch: &DflashScratch,
        acc: Option<(DevicePtr, usize)>,
        stream: u64,
    ) -> Result<()> {
        let Some(ink) = self.inkling.as_ref() else {
            return Ok(());
        };
        let sc = scratch
            .inkling
            .as_ref()
            .context("Inkling 2 DSpark lane scratch")?;
        let h = self.hidden_size;
        KernelLaunch::new(gpu, ink.k.block_input)
            .grid([self.gamma as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(scratch.stream_buf)
            .arg_ptr(ink.w.target_embed_norm)
            .arg_ptr(ink.w.mask_embed)
            .arg_f32(ink.w.target_eps)
            .arg_u32(h as u32)
            .launch(stream)?;
        match acc {
            Some((base, rows)) if rows > 0 => {
                let row_bytes = INKLING2_DSPARK_TAPS * h * 2;
                let last = base.offset((rows - 1) * row_bytes);
                let markov = self.num_layers;
                ink.mix(
                    gpu,
                    last,
                    1,
                    h as u32,
                    markov,
                    ink.w.markov_norm.weight,
                    self.rms_norm_eps,
                    sc.f_rnn,
                    stream,
                )
            }
            _ => gpu.memset_async(sc.f_rnn, 0, h * 2, stream),
        }
    }
}

/// One `inkling2_dspark_pick` launch's rows (`sequences` of them, each at
/// its index times the strides; a null `bias` or `conf` is none).
pub(super) struct Pick {
    logits: DevicePtr,
    /// BF16 elements between two sequences' rows.
    logits_stride: u64,
    bias: DevicePtr,
    bias_stride: u32,
    token: DevicePtr,
    /// `u32` words between two sequences' tokens (and confidences).
    token_stride: u32,
    conf: DevicePtr,
    conf_stride: u32,
    vocab: usize,
}

/// The pick and, with `p.conf`, the drafter's confidence in it, for
/// `sequences` rows (see the kernel).
pub(super) fn pick(
    gpu: &dyn GpuBackend,
    ink: &InklingDspark,
    p: &Pick,
    sequences: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, ink.k.pick)
        .grid([sequences as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(p.logits)
        .arg_u64(p.logits_stride)
        .arg_ptr(p.bias)
        .arg_u32(p.bias_stride)
        .arg_ptr(p.token)
        .arg_u32(p.token_stride)
        .arg_ptr(p.conf)
        .arg_u32(p.conf_stride)
        .arg_u32(p.vocab as u32)
        .arg_f32(LOGIT_SOFTCAP)
        .launch(stream)
}

#[path = "inkling_batch.rs"]
mod batch;
#[path = "inkling_group.rs"]
mod group;
#[path = "inkling_tail.rs"]
mod tail;

#[cfg(test)]
mod tail_steps_tests {
    use super::tail_steps_for;

    #[test]
    fn a_tail_runs_the_steps_its_verify_will_read() {
        // γ = 8: seven drafts is the whole block (the captured graph).
        assert_eq!(tail_steps_for(7, 8), 0);
        assert_eq!(tail_steps_for(9, 8), 0);
        // Fewer drafts stop the tail early, never before the first.
        assert_eq!(tail_steps_for(3, 8), 3);
        assert_eq!(tail_steps_for(1, 8), 1);
        assert_eq!(tail_steps_for(0, 8), 1);
    }
}
