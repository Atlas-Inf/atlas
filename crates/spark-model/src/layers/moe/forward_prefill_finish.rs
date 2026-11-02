// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_prefill_finish — the tail of `forward_prefill`
//! (routed-expert LoRA fold → unpermute+weighted reduce → shared-expert
//! blend → EP all-reduce). Split out under the 500-LoC cap; pure move.

use super::*;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    #[allow(unused_assignments)]
    pub(super) fn forward_prefill_finish(
        &self,
        input: DevicePtr,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        total_expanded: u32,
        token_to_perm: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        h: u32,
        n: u32,
        top_k: u32,
        num_tokens: usize,
        has_shared: bool,
        use_overlap: bool,
        sp: Option<crate::layers::glm_sp::SpRows>,
        split: bool,
        overlap_shared_reduce: bool,
        shared_in: DevicePtr,
        shared_n: u32,
        defer_shared_hc: bool,
        // The E4M3 down rows' scales (`prequant_fp4_down`), or BF16 rows.
        e4m3_scales: Option<DevicePtr>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        macro_rules! prof_step {
            ($label:expr) => {
                if let Some(t) = t0.take() {
                    ctx.gpu.synchronize(stream)?;
                    let elapsed = t.elapsed().as_micros();
                    tracing::info!("  MoE prefill [{}] N={}: {}µs", $label, num_tokens, elapsed);
                    t0 = Some(std::time::Instant::now());
                }
            };
        }

        // Feature-1: fold the routed-expert down_proj LoRA deltas onto the sorted
        // `expert_down_out` BEFORE the unpermute + weighted reduce, so the router
        // weight multiplies base+delta (PEFT semantics). x = the post-SiLU sorted
        // activations. No-op unless routed-expert deltas are installed.
        self.apply_expert_lora_prefill_down(
            ctx.buffers.expert_gate_out(),
            expert_down_out,
            expert_offsets,
            sorted_token_ids,
            total_expanded,
            ctx,
            stream,
        )?;

        // 7. Unpermute + weighted reduce: scatter sorted outputs to token order
        let output = ctx.buffers.moe_output();
        if let Some((sp, blocks)) = self.output_seam_blocks(
            sp,
            (split, use_overlap, defer_shared_hc),
            has_shared.then_some(shared_n),
            h,
            ctx,
        ) {
            // ATLAS_GLM_SP_GATHER_OVERLAP on the chain: unpermute a row
            // block of every part at a time, its reduce-scatter behind it,
            // and leave the blend and the post to the next attention seam
            // (`layers::glm_sp_moe_seam`).
            if overlap_shared_reduce {
                self.run_shared_expert_prefill(
                    shared_in,
                    shared_n,
                    h,
                    ctx.config.shared_expert_intermediate_size as u32,
                    self.prefill_stream,
                    stream,
                    true,
                    ctx,
                )?;
            }
            let blend = has_shared.then_some(crate::layers::glm_sp_moe_seam::Blend {
                kernel: self.moe_batched_blend,
                shared: ctx.buffers.attn_output(),
                input: shared_in,
                gate: self.weights.shared_expert_gate.weight,
                join: overlap_shared_reduce.then_some(self.event_b),
            });
            let (start, end) = ctx.config.local_expert_range();
            let (row, route) = (h as usize * 2, top_k as usize * 4);
            // Only the `_vec8` kernel strides a short grid over its tokens.
            let vec8 = ctx
                .gpu
                .op_cache()
                .kernel(ctx.gpu, "moe", "moe_unpermute_reduce_indexed_ep_vec8")
                .is_ok_and(|k| k.0 == self.moe_unpermute_reduce_ep.0);
            let ctas = vec8.then_some(crate::layers::glm_sp_moe_seam::UNPERMUTE_CTAS);
            let e4m3 = e4m3_scales.filter(|_| self.down_e4m3.is_some());
            // The unpermute kernels' producers of a fused reduce-scatter.
            let fused_kernel = (vec8 || e4m3.is_some())
                .then(|| {
                    let name = match e4m3 {
                        Some(_) => "moe_unpermute_rs_fused_vec8_e4m3",
                        None => "moe_unpermute_rs_fused_vec8",
                    };
                    ctx.gpu
                        .op_cache()
                        .kernel(ctx.gpu, "moe_unpermute_fused", name)
                        .ok()
                })
                .flatten();
            use crate::layers::glm_sp_moe_seam::Produce;
            return crate::layers::glm_sp_moe_seam::begin(
                sp,
                &blocks,
                output,
                blend,
                fused_kernel.is_some(),
                ctx,
                stream,
                |produce| match produce {
                    Produce::Fused(plan) => crate::layers::glm_sp_moe_seam::unpermute_rs_fused(
                        ctx.gpu,
                        fused_kernel.expect("a fused seam without its producer"),
                        expert_down_out,
                        output,
                        (token_to_perm, indices_dev, weights_dev),
                        h,
                        top_k,
                        (start as u32, end as u32),
                        // A short grid, as the blocked seam's (job 51's best).
                        ctas.unwrap_or(32),
                        e4m3,
                        plan.words(),
                        stream,
                    ),
                    Produce::Block((r0, rows)) => {
                        for part in 0..sp.ranks {
                            let t0 = part * sp.rows + r0;
                            ops::moe_unpermute_reduce_indexed_ep_grid(
                                ctx.gpu,
                                self.ep_reduce_kernel(e4m3_scales),
                                expert_down_out,
                                output.offset(t0 * row),
                                token_to_perm.offset(t0 * route),
                                indices_dev.offset(t0 * route),
                                weights_dev.offset(t0 * route),
                                h,
                                rows as u32,
                                top_k,
                                start as u32,
                                end as u32,
                                ctas.unwrap_or(rows as u32),
                                e4m3_scales,
                                stream,
                            )?;
                        }
                        Ok(())
                    }
                },
            );
        }
        // ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE: under SP, the unpermute of the
        // rows the peer needs first, their reduce-scatter on the wire while
        // the rest unpermute (`layers::qwen4exp_sp_pipe`).
        // With `_SP_SHARED`, the shared expert runs on the wire too (it was
        // deferred by `forward_prefill`, `defer_shared_to_rs`).
        let defer_shared =
            !split && self.defer_shared_to_rs(sp, has_shared, overlap_shared_reduce, h, ctx);
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let run_shared = |s: u64| {
            self.run_shared_expert_prefill(shared_in, shared_n, h, shared_inter, s, s, false, ctx)
        };
        // An MXFP8 shared expert's partial is blended into each row range
        // before it goes on the wire (`shared_mxfp8.rs`).
        let mx_split = split && self.shared_joins_reduce();
        let rs_done = match sp {
            Some(sp) if (!split || mx_split) && !overlap_shared_reduce && !use_overlap => self
                .q38_unpermute_reduce_scatter(
                    sp,
                    [
                        expert_down_out,
                        output,
                        token_to_perm,
                        indices_dev,
                        weights_dev,
                    ],
                    [h, top_k],
                    mx_split.then_some(input),
                    ctx,
                    stream,
                    |s| if defer_shared { run_shared(s) } else { Ok(()) },
                )?,
            _ => false,
        };
        if defer_shared && !rs_done {
            run_shared(stream)?;
        }
        let blended = !rs_done
            && self.unpermute_ep_prefill(
                expert_down_out,
                e4m3_scales,
                output,
                token_to_perm,
                indices_dev,
                weights_dev,
                h,
                n,
                top_k,
                split.then_some(input),
                ctx,
                stream,
            )?;

        // 8. Blend shared expert: output += sigmoid(dot(input, gate)) * shared
        // Skip when has_shared == false (no shared expert in this model config).
        // EP fix: defer shared expert blend until AFTER all-reduce to avoid doubling.
        let is_ep_prefill = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        if has_shared && !is_ep_prefill {
            let shared_down_out = ctx.buffers.attn_output();
            if use_overlap {
                ctx.gpu.stream_wait_event(stream, self.event_b)?;
            }
            super::dump::dump_routed_only(ctx.gpu, stream, output, n, h)?;
            super::dump::dump_shared_out(ctx.gpu, stream, shared_down_out, n, h)?;
            super::dump::dump_shared_gate(
                ctx.gpu,
                stream,
                input,
                self.weights.shared_expert_gate.weight,
                n,
                h,
            )?;
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                shared_down_out,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                stream,
            )?;
        }
        super::dump::dump_moe_out(ctx.gpu, stream, output, n, h)?;
        prof_step!("unpermute_blend");

        // The routed result is now complete. Starting the shared expert here
        // lets its GEMMs run beside the EP collective instead of beside the
        // bandwidth-heavy routed GEMMs. event_a makes the auxiliary stream wait
        // for this point; event_b is joined immediately before the shared blend.
        if overlap_shared_reduce {
            self.run_shared_expert_prefill(
                shared_in,
                shared_n,
                h,
                ctx.config.shared_expert_intermediate_size as u32,
                self.prefill_stream,
                stream,
                true,
                ctx,
            )?;
        }

        // `ATLAS_GLM_SHARED_TP_SPLIT=1` (or an MXFP8 shared expert's slice):
        // blend this rank's partial shared-expert columns before the reduce,
        // unless the unpermute (or the reduce-scatter pipe) already did.
        if split && !blended && !rs_done {
            ops::moe_batched_blend(
                ctx.gpu,
                self.moe_batched_blend,
                output,
                ctx.buffers.attn_output(),
                input,
                self.weights.shared_expert_gate.weight,
                h,
                n,
                stream,
            )?;
        }

        // ATLAS_GLM_DET_TRACE stages (`det_rows`: the rows the reduce leaves here).
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        let det_rows = sp.map_or((0, num_tokens), |sp| (sp.row0, sp.rows));
        let row = h as usize * 2;
        det.tap("moe_local", output, (0, num_tokens), row);
        // EP all-reduce
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            let _t0 = if ctx.profile {
                ctx.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            if rs_done {
                // Reduce-scattered with the unpermute above.
            } else if let Some(sp) = sp {
                sp.reduce_scatter(output, h as usize, ctx, stream)?;
            } else if ctx.graph_capture {
                comm.all_reduce(output.0, num_tokens * h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, num_tokens * h as usize * 2, stream)?;
            }
            if let Some(t0) = _t0 {
                ctx.gpu.synchronize(stream)?;
                tracing::info!(
                    "  EP allreduce (moe out) N={}: {}µs",
                    num_tokens,
                    t0.elapsed().as_micros(),
                );
            }
            let reduced = sp.map_or(output, |sp| sp.local(output, h as usize));
            det.tap("moe_red", reduced, det_rows, row);
            // Add shared expert ONCE after all-reduce (prevents EP doubling)
            if has_shared && !defer_shared_hc && !split {
                // A `full_shared` split ran the shared expert over every row;
                // the blend takes this rank's rows of it.
                let (shared_down_out, shared_in, shared_n) = match sp {
                    Some(sp) if sp.full_shared => (
                        sp.local(ctx.buffers.attn_output(), h as usize),
                        sp.local(input, h as usize),
                        sp.rows as u32,
                    ),
                    _ => (ctx.buffers.attn_output(), shared_in, shared_n),
                };
                if use_overlap || overlap_shared_reduce {
                    ctx.gpu.stream_wait_event(stream, self.event_b)?;
                }
                det.tap("moe_sh", shared_down_out, det_rows, row);
                ops::moe_batched_blend(
                    ctx.gpu,
                    self.moe_batched_blend,
                    reduced,
                    shared_down_out,
                    shared_in,
                    self.weights.shared_expert_gate.weight,
                    h,
                    shared_n,
                    stream,
                )?;
                det.tap("moe", reduced, det_rows, row);
            }
        }

        Ok(())
    }
}
