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
        })
    }
}

/// The variant's device state (weights and kernels), on the head.
pub struct InklingDspark {
    w: Inkling2DsparkWeights,
    fold: DenseWeight,
    k: Kernels,
}

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
        })
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
        dense_gemm: KernelHandle,
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

    /// The draft tokens of rows 1..γ from the block's logits (rows 1..γ of
    /// `logits` hold the base logits of the drafter's normed rows `normed`):
    /// per step, the recurrence from the previous token, its fold over the
    /// head added to that row's logits, then the argmax. Row 0 (the anchor)
    /// is argmaxed as is; it is not a draft.
    pub(super) fn inkling_tail(
        &self,
        ink: &InklingDspark,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        scratch: &DflashScratch,
        stream: u64,
    ) -> Result<()> {
        let sc = scratch
            .inkling
            .as_ref()
            .context("Inkling 2 DSpark lane scratch")?;
        let (h, v) = (self.hidden_size as u32, self.vocab_size);
        let steps = sc
            .short_tail()
            .unwrap_or(self.gamma - 1)
            .min(self.gamma - 1);
        let bf16 = 2usize;
        let row = |i: usize| scratch.logits.offset(i * v * bf16);
        ops::argmax_bf16(
            gpu,
            self.kernels.argmax,
            row(0),
            scratch.draft_tokens_dev,
            v as u32,
            stream,
        )?;
        KernelLaunch::new(gpu, ink.k.rnn_ctx)
            .grid([(3 * RANK / 8) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ink.w.rnn_proj.weight)
            .arg_ptr(ink.w.rnn_bias)
            .arg_ptr(normed)
            .arg_ptr(sc.f_rnn)
            .arg_ptr(sc.ctx)
            .arg_u32(steps as u32)
            .arg_u32(h)
            .launch(stream)?;
        gpu.memset_async(sc.state, 0, RANK * bf16, stream)?;
        for i in 0..steps {
            // step i drafts row i + 1 from the token before it (the anchor first)
            let prev = if i == 0 {
                scratch.markov_prev_dev
            } else {
                scratch.draft_tokens_dev.offset(i * 4)
            };
            KernelLaunch::new(gpu, ink.k.rnn_raw)
                .grid([(3 * RANK / 8) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(ink.w.rnn_proj.weight)
                .arg_ptr(sc.ctx.offset(i * 3 * RANK * 4))
                .arg_ptr(sc.state)
                .arg_ptr(ink.w.markov_w1.weight)
                .arg_ptr(prev)
                .arg_ptr(sc.raw)
                .arg_u32(h)
                .launch(stream)?;
            KernelLaunch::new(gpu, ink.k.rnn_gate)
                .grid([1, 1, 1])
                .block([RANK as u32, 1, 1])
                .arg_ptr(sc.raw)
                .arg_ptr(sc.state)
                .arg_ptr(sc.latent)
                .arg_u32(FOLD_K as u32)
                .launch(stream)?;
            ops::dense_gemv(
                gpu,
                self.kernels.dense_gemv,
                sc.latent,
                &ink.fold,
                sc.bias,
                v as u32,
                FOLD_K as u32,
                stream,
            )?;
            ops::residual_add(
                gpu,
                self.kernels.residual_add,
                row(i + 1),
                sc.bias,
                v as u32,
                stream,
            )?;
            ops::argmax_bf16(
                gpu,
                self.kernels.argmax,
                row(i + 1),
                scratch.draft_tokens_dev.offset((i + 1) * 4),
                v as u32,
                stream,
            )?;
        }
        Ok(())
    }
}

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
