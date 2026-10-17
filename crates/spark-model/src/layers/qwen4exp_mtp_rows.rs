// SPDX-License-Identifier: AGPL-3.0-only

//! The row-wise front of a multi-row drafter forward, shared by the batched
//! propose (`qwen4exp_mtp_batch.rs`, one row a sequence) and the absorb
//! (`qwen4exp_mtp_absorb.rs`, verified rows of one or more sequences).

use super::*;

impl Qwen4ExpMtpHead {
    /// `forward_one`'s steps 1-3 at M = n: rows `0..n` of the stream highway
    /// (each row's incoming residual) and the n token ids at `tokens` (device
    /// u32) become the body's input streams, in place.
    pub(super) fn stage_rows_input(
        &self,
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
        self.stream_norms(ctx, streams, n * hc, stream)?;
        // ── 2. Per-stream projection, every (row, stream) in one weight pass ──
        self.wide_rows.dense_rows(
            gpu,
            self.dense_gemv_batchm_k,
            self.normed_h,
            &self.module.fc_hidden,
            self.h_streams,
            (rows32, h32, h32, h32),
            stream,
        )?;
        // ── 3. Embedding branch from the device token row ──
        ops::batched_embed(
            gpu,
            self.batched_embed_k,
            tokens,
            self.embed_tokens.weight,
            self.embed_buf,
            n32,
            h32,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            self.embed_buf,
            &self.module.pre_fc_norm_embedding,
            self.normed_e,
            n32,
            h32,
            eps,
            stream,
        )?;
        self.wide_rows.dense_rows(
            gpu,
            self.dense_gemv_batchm_k,
            self.normed_e,
            &self.module.fc_embedding,
            self.e_branch,
            (n32, h32, h32, h32),
            stream,
        )?;
        ops::hc_expand(
            gpu,
            self.hc_expand_k,
            self.e_branch,
            streams,
            n32,
            h32,
            hc as u32,
            stream,
        )?;
        self.f32_add_bf16(gpu, streams, self.h_streams, rows32 * h32, stream)
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
