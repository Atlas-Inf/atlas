// SPDX-License-Identifier: AGPL-3.0-only

//! MiMo-V2 attention path (Q/K head dim 192, V head dim 128).
//!
//! The shared paged-attention chain assumes one head dim for Q, K and V, so
//! MiMo layers short-circuit `attention_forward` / paged prefill / multi-seq
//! decode into this module:
//!
//!   fused QKV GEMV/GEMM (BF16, checkpoint TP-chunk row order)
//!   → `mimo_qkv_rope_cache` (split, partial RoPE, paged K/V write)
//!   → `mimo_paged_decode_attn` (+ `_reduce`) or `mimo_prefill_attn`
//!   → O projection (BF16; `attention_value_scale` is applied to the
//!     attention output inside the kernels).
//!
//! Kernels live in `kernels/gb10/mimo-v2.6-flash/nvfp4/mimo_attn.cu`.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8BlockWeight};

pub(crate) const MIMO_QK_DIM: u32 = 192;
pub(crate) const MIMO_V_DIM: u32 = 128;
const MIMO_PREFILL_Q: u32 = 16;
/// Tensor-core paged prefill (`mimo_prefill_tc`): BR=32 queries/block, 4 warps.
const MIMO_TC_BR: u32 = 32;
/// Split-K partitions for full-attention decode: one per 512 cached tokens.
const MIMO_SPLIT_TOKENS: u32 = 512;
const MIMO_MAX_SPLITS: u32 = 16;

/// Bytes of split-K scratch the loader allocates (shared by all layers):
/// `max_seqs × nq × MIMO_MAX_SPLITS × (128 acc + 2 m/l) × f32`.
pub(crate) fn mimo_workspace_bytes(max_seqs: usize, nq: usize) -> usize {
    max_seqs * nq * MIMO_MAX_SPLITS as usize * (MIMO_V_DIM as usize + 2) * 4
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MimoKernels {
    pub qkv_rope_cache: KernelHandle,
    pub decode: KernelHandle,
    pub decode_reduce: KernelHandle,
    pub prefill: KernelHandle,
    /// Tensor-core paged prefill; null when the kernel build predates it.
    pub prefill_tc: KernelHandle,
    /// FP8 tensor-core paged prefill; null when the kernel build predates it.
    pub prefill_tc_fp8: KernelHandle,
}

impl MimoKernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| {
            gpu.kernel("mimo_attn", name).with_context(|| {
                format!("mimo_attn::{name} missing — build with ATLAS_TARGET_MODEL=mimo-v2.6-flash")
            })
        };
        Ok(Self {
            qkv_rope_cache: k("mimo_qkv_rope_cache")?,
            decode: k("mimo_paged_decode_attn")?,
            decode_reduce: k("mimo_paged_decode_reduce")?,
            prefill: k("mimo_prefill_attn")?,
            prefill_tc: crate::layers::try_kernel(gpu, "mimo_prefill_tc", "mimo_prefill_tc"),
            prefill_tc_fp8: crate::layers::try_kernel(
                gpu,
                "mimo_prefill_tc_fp8",
                "mimo_prefill_tc_fp8",
            ),
        })
    }
}

/// The fused QKV projection's weight form: BF16 (dequantized at load) or
/// the checkpoint's native FP8 block bytes (`ATLAS_MIMO_FP8_QKV=1`).
#[derive(Debug, Clone)]
pub(crate) enum MimoQkv {
    Bf16(DenseWeight),
    Fp8b(Fp8BlockWeight),
}

/// Per-layer MiMo attention weights and geometry.
#[derive(Debug, Clone)]
pub(crate) struct MimoAttn {
    /// Fused QKV, rows in the checkpoint's chunk order `[Q_c | K_c | V_c] ×
    /// tp_chunks`. BF16 is `[tp_chunks * chunk_rows, hidden]`; FP8 blocks pad
    /// full-attention chunks to the 128-row grid, so `chunk_rows` is the
    /// stride the rope kernel uses either way.
    pub qkv: MimoQkv,
    /// O projection, BF16 `[hidden, nq * 128]` (checkpoint tensor).
    pub o_proj: DenseWeight,
    /// Per-head sink logits `[nq]` BF16, or NULL.
    pub sinks: DevicePtr,
    pub nq: u32,
    pub nkv: u32,
    pub tp_chunks: u32,
    pub chunk_rows: u32,
    /// Sliding window in tokens; 0 = full attention.
    pub window: u32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub scale: f32,
    /// `attention_value_scale`, applied to the attention output.
    pub out_scale: f32,
    pub workspace: DevicePtr,
    pub workspace_bytes: usize,
    pub kernels: MimoKernels,
}

impl MimoAttn {
    pub(crate) fn qkv_rows(&self) -> u32 {
        self.tp_chunks * self.chunk_rows
    }

    fn num_splits(&self, max_ctx: u32, num_seqs: u32) -> u32 {
        if self.window > 0 {
            return 1;
        }
        let per_split = num_seqs as usize * self.nq as usize * (MIMO_V_DIM as usize + 2) * 4;
        let fit = (self.workspace_bytes / per_split.max(1)) as u32;
        max_ctx
            .div_ceil(MIMO_SPLIT_TOKENS)
            .clamp(1, MIMO_MAX_SPLITS)
            .min(fit)
            .max(1)
    }
}

impl Qwen3AttentionLayer {
    pub(crate) fn set_mimo(&mut self, mimo: MimoAttn) {
        self.mimo = Some(mimo);
    }

    fn mimo(&self) -> &MimoAttn {
        self.mimo
            .as_ref()
            .expect("MiMo path entered without MimoAttn")
    }

    /// FP8 KV args for the MiMo kernels: `(kv_fp8, k_scale, v_scale)`. BF16 KV
    /// passes `kv_fp8 = 0` (scales unused); FP8 KV dequantizes with these.
    fn mimo_kv_fp8(&self) -> (u32, f32, f32) {
        if self.kv_dtype == KvCacheDtype::Fp8 {
            let (k, v) = self.effective_fp8_scales();
            (1, k, v)
        } else {
            (0, 1.0, 1.0)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mimo_gemm(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        w: &DenseWeight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        static BATCHM: std::sync::OnceLock<KernelHandle> = std::sync::OnceLock::new();
        let batchm = *BATCHM.get_or_init(|| {
            crate::layers::try_kernel(ctx.gpu, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
        });
        if m == 1 {
            ops::dense_gemv(ctx.gpu, self.dense_gemv_k, input, w, out, n, k, stream)
        } else if batchm.0 != 0 && m <= ops::DENSE_GEMV_BATCHM_MAX_M {
            // Speculative verify / small multi-seq batches: one weight pass for
            // all rows instead of a large-M tiled GEMM.
            ops::dense_gemv_batchm(ctx.gpu, batchm, input, w, out, m, n, k, n, stream)
        } else if ctx.dispatch.cublas_gemm {
            ops::cublas_bf16_proj_dense(input, w.weight, out, m, n, k, stream)
        } else if self.dense_gemm_pipelined_k.0 != 0 {
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_pipelined_k,
                input,
                w,
                out,
                m,
                n,
                k,
                stream,
            )
        } else {
            ops::dense_gemm(ctx.gpu, self.dense_gemm_k, input, w, out, m, n, k, stream)
        }
    }

    /// The QKV projection on either weight form. FP8 blocks run the native
    /// `fp8b` tiers: the row-invariant GEMV tiers through GLM's decode-row
    /// ceiling, the exact W8A16 prefill GEMM beyond it.
    fn mimo_qkv_gemm(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let w = match &self.mimo().qkv {
            MimoQkv::Bf16(w) => return self.mimo_gemm(ctx, input, w, out, m, n, k, stream),
            MimoQkv::Fp8b(w) => w,
        };
        let kernels = ops::native_kernels(ctx.gpu);
        ensure!(
            kernels.fp8b.complete(false) && kernels.w8a16_gemm.0 != 0,
            "mimo_v2: FP8 block QKV needs the fp8b GEMV tiers and w8a16_gemm_pipelined \
             (rebuild the kernels)"
        );
        let (m, n, k) = (m as usize, n as usize, k as usize);
        // GLM's ceiling: decode/verify batches stay on the GEMV tiers.
        const GEMV_MAX_ROWS: usize = 2 * ops::MXFP8_MAX_ROWS;
        if m <= GEMV_MAX_ROWS {
            ops::fp8b_gemv_rows(ctx.gpu, &kernels.fp8b, w, input, out, m, n, k, n, stream)
        } else {
            ops::fp8b_prefill_gemm(
                ctx.gpu,
                &kernels,
                ops::Fp8ActScratch::of(ctx.buffers),
                ops::Fp8bPrefill::W8a16,
                w,
                input,
                out,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// QKV projection + RoPE + paged K/V write. Leaves Q `[n, nq*192]` in
    /// `ssm_deinterleaved`. With `contiguous`, roped K `[n, nkv, 192]` and V
    /// `[n, nkv, 128]` are also written back-to-back into `ssm_qkvz`.
    #[allow(clippy::too_many_arguments)]
    fn mimo_project_qkv(
        &self,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        normed: DevicePtr,
        n: u32,
        meta: &AttnMetadataDev,
        kv_write_floor: u32,
        contiguous: bool,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        let m = self.mimo();
        let h = ctx.config.hidden_size as u32;
        let (k_contig, v_contig) = if contiguous {
            let k = ctx.buffers.ssm_qkvz();
            (k, k.offset(n as usize * (m.nkv * MIMO_QK_DIM) as usize * 2))
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let qkv_out = ctx.buffers.qkv_output();
        let q = ctx.buffers.ssm_deinterleaved();
        self.mimo_qkv_gemm(ctx, normed, qkv_out, n, m.qkv_rows(), h, stream)?;
        let (kv_fp8, k_scale, v_scale) = self.mimo_kv_fp8();
        let (k_inv_scale, v_inv_scale) = if kv_fp8 == 1 {
            (1.0 / k_scale, 1.0 / v_scale)
        } else {
            (1.0, 1.0)
        };
        KernelLaunch::new(ctx.gpu, m.kernels.qkv_rope_cache)
            .grid([n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(qkv_out)
            .arg_ptr(q)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(kv_cache.v_pool_ptr(self.attn_layer_idx))
            .arg_ptr(meta.positions)
            .arg_ptr(meta.slot)
            .arg_ptr(k_contig)
            .arg_ptr(v_contig)
            .arg_u32(m.nq)
            .arg_u32(m.nkv)
            .arg_u32(m.tp_chunks)
            .arg_u32(m.chunk_rows)
            .arg_u32(m.rotary_dim)
            .arg_f32(m.rope_theta)
            .arg_u32(kv_cache.block_size() as u32)
            .arg_u32(kv_write_floor)
            .arg_ptr(kv_cache.swa_slot_map_ptr(self.attn_layer_idx))
            .arg_u32(kv_fp8)
            .arg_f32(k_inv_scale)
            .arg_f32(v_inv_scale)
            .launch(stream)?;
        Ok((q, k_contig, v_contig))
    }

    /// Decode for `meta.num_seqs` sequences, one query token each. Writes the
    /// O projection to `o_out` and returns it.
    pub(super) fn mimo_attend_decode(
        &self,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        normed: DevicePtr,
        meta: &AttnMetadataDev,
        o_out: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let m = self.mimo();
        let num_seqs = meta.num_seqs;
        ensure!(
            m.nq / m.nkv <= 32 && m.nq.is_multiple_of(m.nkv),
            "MiMo decode: GQA group {} / {} unsupported",
            m.nq,
            m.nkv
        );
        let (q, _, _) =
            self.mimo_project_qkv(ctx, kv_cache, normed, num_seqs, meta, 0, false, stream)?;
        let attn = ctx.buffers.attn_output();
        let bs = kv_cache.block_size() as u32;
        let splits = m.num_splits(meta.max_blocks_per_seq * bs, num_seqs);
        let ws_ml = m.workspace;
        let ws_acc = m
            .workspace
            .offset(num_seqs as usize * m.nq as usize * splits as usize * 2 * 4);
        let (kv_fp8, k_scale, v_scale) = self.mimo_kv_fp8();
        KernelLaunch::new(ctx.gpu, m.kernels.decode)
            .grid([m.nkv, splits, num_seqs])
            .block([32 * (m.nq / m.nkv), 1, 1])
            .arg_ptr(q)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(kv_cache.v_pool_ptr(self.attn_layer_idx))
            .arg_ptr(meta.block_table)
            .arg_ptr(meta.seq_len)
            .arg_ptr(m.sinks)
            .arg_ptr(attn)
            .arg_ptr(ws_ml)
            .arg_ptr(ws_acc)
            .arg_u32(m.nq)
            .arg_u32(m.nkv)
            .arg_u32(bs)
            .arg_u32(meta.max_blocks_per_seq)
            .arg_u32(m.window)
            .arg_f32(m.scale)
            .arg_f32(m.out_scale)
            .arg_u32(m.nq * MIMO_QK_DIM)
            .arg_u32(m.nq * MIMO_V_DIM)
            .arg_ptr(kv_cache.swa_slot_map_ptr(self.attn_layer_idx))
            .arg_u32(kv_fp8)
            .arg_f32(k_scale)
            .arg_f32(v_scale)
            .launch(stream)?;
        if splits > 1 {
            KernelLaunch::new(ctx.gpu, m.kernels.decode_reduce)
                .grid([m.nq, num_seqs, 1])
                .block([MIMO_V_DIM, 1, 1])
                .arg_ptr(ws_ml)
                .arg_ptr(ws_acc)
                .arg_ptr(m.sinks)
                .arg_ptr(attn)
                .arg_ptr(meta.seq_len)
                .arg_u32(m.nq)
                .arg_u32(splits)
                .arg_f32(m.out_scale)
                .arg_u32(m.nq * MIMO_V_DIM)
                .launch(stream)?;
        }
        let h = ctx.config.hidden_size as u32;
        self.mimo_gemm(
            ctx,
            attn,
            &m.o_proj,
            o_out,
            num_seqs,
            h,
            m.nq * MIMO_V_DIM,
            stream,
        )?;
        Ok(o_out)
    }

    /// Single-sequence prefill chunk at absolute positions
    /// `[seq_len_start, seq_len_start + num_tokens)`. Returns the O projection
    /// (`norm_output`). `contiguous` (chunk 0 only) attends over the chunk's
    /// own K/V instead of the paged cache.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mimo_attend_prefill(
        &self,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        normed: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        kv_write_floor: usize,
        contiguous: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let m = self.mimo();
        ensure!(
            !contiguous || seq_len_start == 0,
            "MiMo contiguous prefill must start at position 0"
        );
        let meta = ctx
            .attn_metadata
            .context("MiMo prefill requires pre-uploaded attention metadata")?;
        let n = num_tokens as u32;
        let (q, k_contig, v_contig) = self.mimo_project_qkv(
            ctx,
            kv_cache,
            normed,
            n,
            &meta,
            kv_write_floor as u32,
            contiguous,
            stream,
        )?;
        let (k_src, v_src, block_table) = if contiguous {
            (k_contig, v_contig, DevicePtr::NULL)
        } else {
            (
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta.block_table,
            )
        };
        let attn = ctx.buffers.attn_output();
        let (kv_fp8, k_scale, v_scale) = self.mimo_kv_fp8();
        // Chunked (paged) prefill uses a tensor-core kernel when built:
        // prefill_tc for BF16 KV, prefill_tc_fp8 for FP8 KV. Chunk-0
        // contiguous prefill and stale kernel builds keep the scalar path.
        if !contiguous && kv_fp8 == 0 && m.kernels.prefill_tc.0 != 0 {
            let kv_len = (seq_len_start + num_tokens) as u32;
            KernelLaunch::new(ctx.gpu, m.kernels.prefill_tc)
                .grid([m.nq, div_ceil(n, MIMO_TC_BR), 1])
                .block([128, 1, 1])
                .arg_ptr(q)
                .arg_ptr(k_src)
                .arg_ptr(v_src)
                .arg_ptr(attn)
                .arg_ptr(block_table)
                .arg_u32(n)
                .arg_u32(kv_len)
                .arg_u32(seq_len_start as u32)
                .arg_u32(m.nq)
                .arg_u32(m.nkv)
                .arg_u32(kv_cache.block_size() as u32)
                .arg_ptr(kv_cache.swa_slot_map_ptr(self.attn_layer_idx))
                .arg_u32(m.window)
                .arg_u32(1)
                .arg_f32(m.scale)
                .arg_f32(m.out_scale)
                .arg_ptr(m.sinks)
                .launch(stream)?;
        } else if !contiguous && kv_fp8 != 0 && m.kernels.prefill_tc_fp8.0 != 0 {
            let kv_len = (seq_len_start + num_tokens) as u32;
            KernelLaunch::new(ctx.gpu, m.kernels.prefill_tc_fp8)
                .grid([m.nq, div_ceil(n, MIMO_TC_BR), 1])
                .block([128, 1, 1])
                .arg_ptr(q)
                .arg_ptr(k_src)
                .arg_ptr(v_src)
                .arg_ptr(attn)
                .arg_ptr(block_table)
                .arg_u32(n)
                .arg_u32(kv_len)
                .arg_u32(seq_len_start as u32)
                .arg_u32(m.nq)
                .arg_u32(m.nkv)
                .arg_u32(kv_cache.block_size() as u32)
                .arg_ptr(kv_cache.swa_slot_map_ptr(self.attn_layer_idx))
                .arg_u32(m.window)
                .arg_u32(1)
                .arg_f32(m.scale)
                .arg_f32(k_scale)
                .arg_f32(v_scale)
                .arg_f32(m.out_scale)
                .arg_ptr(m.sinks)
                .launch(stream)?;
        } else {
            KernelLaunch::new(ctx.gpu, m.kernels.prefill)
                .grid([div_ceil(n, MIMO_PREFILL_Q), m.nq, 1])
                .block([32 * MIMO_PREFILL_Q, 1, 1])
                .arg_ptr(q)
                .arg_ptr(k_src)
                .arg_ptr(v_src)
                .arg_ptr(block_table)
                .arg_ptr(m.sinks)
                .arg_ptr(attn)
                .arg_u32(n)
                .arg_u32(seq_len_start as u32)
                .arg_u32(m.nq)
                .arg_u32(m.nkv)
                .arg_u32(kv_cache.block_size() as u32)
                .arg_u32(m.window)
                .arg_f32(m.scale)
                .arg_f32(m.out_scale)
                .arg_u32(m.nq * MIMO_QK_DIM)
                .arg_u32(m.nq * MIMO_V_DIM)
                .arg_ptr(kv_cache.swa_slot_map_ptr(self.attn_layer_idx))
                .arg_u32(kv_fp8)
                .arg_f32(k_scale)
                .arg_f32(v_scale)
                .launch(stream)?;
        }
        let h = ctx.config.hidden_size as u32;
        let o_out = ctx.buffers.norm_output();
        self.mimo_gemm(ctx, attn, &m.o_proj, o_out, n, h, m.nq * MIMO_V_DIM, stream)?;
        Ok(o_out)
    }
}
