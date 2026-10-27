// SPDX-License-Identifier: AGPL-3.0-only

//! The row-wise front of a multi-row drafter forward, shared by the batched
//! propose (`qwen4exp_mtp_batch.rs`, one row a sequence) and the absorb
//! (`qwen4exp_mtp_absorb.rs`, verified rows of one or more sequences).

use super::*;

/// Device scratch of the row-wise front: `rows` rows of each buffer.
#[derive(Clone, Copy)]
pub(super) struct RowScratch {
    pub normed_h: DevicePtr,
    pub h_streams: DevicePtr,
    pub embed_buf: DevicePtr,
    pub normed_e: DevicePtr,
    pub e_branch: DevicePtr,
    /// `dense_gemm_bf16_pipelined` for the projections, 0 = the GEMV tiers
    /// (the propose's own scratch: its bytes as before).
    pub gemm: KernelHandle,
}

impl RowScratch {
    /// Allocate `rows` rows of each buffer.
    pub(super) fn alloc(gpu: &dyn GpuBackend, rows: usize, h: usize, hc: usize) -> Result<Self> {
        let bf16 = |n: usize| -> Result<DevicePtr> { gpu.alloc(rows * n * 2) };
        Ok(Self {
            normed_h: bf16(hc * h)?,
            h_streams: bf16(hc * h)?,
            embed_buf: bf16(h)?,
            normed_e: bf16(h)?,
            e_branch: bf16(h)?,
            gemm: crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined"),
        })
    }
}

impl Qwen4ExpMtpHead {
    /// The propose's own scratch (`PROPOSE_BATCH_MAX` rows).
    pub(super) fn own_scratch(&self) -> RowScratch {
        RowScratch {
            normed_h: self.normed_h,
            h_streams: self.h_streams,
            embed_buf: self.embed_buf,
            normed_e: self.normed_e,
            e_branch: self.e_branch,
            gemm: KernelHandle(0),
        }
    }

    /// `[m, hidden] x weight^T` into `out`: `s.gemm` when set, else the
    /// GEMV row tiers.
    fn project_rows(
        &self,
        s: &RowScratch,
        input: DevicePtr,
        weight: &DenseWeight,
        out: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        if s.gemm.0 != 0 {
            return ops::dense_gemm_bf16_pipelined(
                ctx.gpu, s.gemm, input, weight, out, m, h, h, stream,
            );
        }
        self.wide_rows.dense_rows(
            ctx.gpu,
            self.dense_gemv_batchm_k,
            input,
            weight,
            out,
            (m, h, h, h),
            stream,
        )
    }

    /// `forward_one`'s steps 1-3 at M = n: rows `0..n` of the stream highway
    /// (each row's incoming residual) and the n token ids at `tokens` (device
    /// u32) become the body's input streams, in place, through scratch `s`.
    pub(super) fn stage_rows_input(
        &self,
        s: &RowScratch,
        tokens: DevicePtr,
        n: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let hc = ctx.config.hc_mult.max(1);
        let eps = ctx.config.rms_norm_eps as f32;
        let streams = ctx.buffers.hc_streams();
        let (h32, rows32, n32) = (h as u32, (n * hc) as u32, n as u32);

        // ── 1. Per-stream grouped norm of each row's incoming residual ──
        self.stream_norms(ctx, streams, n * hc, s.normed_h, stream)?;
        // ── 2. Per-stream projection, every (row, stream) in one weight pass ──
        let fc_hidden = &self.module.fc_hidden;
        self.project_rows(s, s.normed_h, fc_hidden, s.h_streams, rows32, ctx, stream)?;
        // ── 3. Embedding branch from the device token row ──
        ops::batched_embed(
            gpu,
            self.batched_embed_k,
            tokens,
            self.embed_tokens.weight,
            s.embed_buf,
            n32,
            h32,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            s.embed_buf,
            &self.module.pre_fc_norm_embedding,
            s.normed_e,
            n32,
            h32,
            eps,
            stream,
        )?;
        let fc_e = &self.module.fc_embedding;
        self.project_rows(s, s.normed_e, fc_e, s.e_branch, n32, ctx, stream)?;
        ops::hc_expand(
            gpu,
            self.hc_expand_k,
            s.e_branch,
            streams,
            n32,
            h32,
            hc as u32,
            stream,
        )?;
        self.f32_add_bf16(gpu, streams, s.h_streams, rows32 * h32, stream)
    }

    /// The body's context for a multi-row forward with metadata `meta`.
    pub(super) fn body_ctx<'a, 'c: 'a>(
        &'a self,
        ctx: &ForwardContext<'c>,
        meta: AttnMetadataDev,
    ) -> ForwardContext<'a> {
        ForwardContext {
            ssm_batch: None,
            buffers: ctx.buffers,
            gpu: ctx.gpu,
            // The body's own (TP=1) config, as in `forward_one`.
            config: &self.module.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(meta),
            profile: ctx.profile,
            // Rank 0 only, every head and expert local: no collective.
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        }
    }
}
