// SPDX-License-Identifier: AGPL-3.0-only

//! Steps 1-3 of `MoeLayer::forward_prefill` (router GEMM, top-k, sort by
//! expert), and `ATLAS_GLM_MOE_ROUTE_FUSE=1` (default off), which runs them
//! for a canonical verify block in two launches. Router logits are FP32 under
//! `MoeLayer::router_f32` (GLM-5) on every path.
//!
//! A GLM verify block of 1..=32 rows reaches its routed gate/up through five
//! dependent launches per MoE layer: the canonical router GEMV, the sigmoid
//! top-k, the sort, the M16 worklist build and the NVFP4 quantization of the
//! MoE input, each a few microseconds of work on one to eighteen CTAs (about
//! 38 us from the shared expert's end to the gate/up's start, 42 MoE layers
//! a step). With the switch, `glm_moe_router_quant` runs the router (its
//! weight asked into L2 while the launch waits on the shared expert) beside
//! the quantization, and `glm_moe_route_select` the top-k, the sort and the
//! worklist in one CTA (kernels/gb10/glm-5.3-flash/nvfp4/glm_moe_route.cu).
//! Every word the five launches leave for the routed FFN is the same
//! (`route_fuse_test_gpu.rs`), so the output bits are.
//!
//! It serves only the batches whose routed FFN runs the M16 stream twins
//! through the prequant path ([`MoeLayer::stream_m16_prequant`]): their
//! gate/up reads the worklist and the quantized input this writes, and the
//! routed FFN then skips its own worklist build and quantization
//! (`prerouted`). Every other batch runs the five launches.

use super::*;
#[cfg(test)]
use spark_runtime::kernel_args::KernelLaunch;

mod kernels;

/// Widest batch the switch takes: the select's top-k runs a warp per row on
/// eight warps, and with each lane's scores in registers it beats the
/// per-row top-k CTAs to 32 rows (glm_moe_route_bench: 12.8 us a layer saved
/// at 16 rows, 5.8 at 32). Wider batches run the five launches, same words.
const MAX_ROWS: u32 = 32;

/// `ATLAS_GLM_MOE_ROUTE_FUSE` as set: `0` (or unset) or `1`. Read once.
pub(crate) fn route_fuse_requested() -> Result<bool> {
    static ON: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(
        || match std::env::var("ATLAS_GLM_MOE_ROUTE_FUSE").as_deref() {
            Err(_) | Ok("0") => Ok(false),
            Ok("1") => Ok(true),
            Ok(value) => Err(format!(
                "ATLAS_GLM_MOE_ROUTE_FUSE requires 0 or 1, got {value:?}"
            )),
        },
    )
    .clone()
    .map_err(anyhow::Error::msg)
}

/// The two routing kernels, both loaded or both null, and the one-launch
/// pair for batches of at most [`ONE_MAX_ROWS`] rows (null when the target
/// lacks it) with its completion counter.
#[derive(Clone, Copy)]
pub(super) struct RouteFuse {
    router_quant: KernelHandle,
    select: KernelHandle,
    one: KernelHandle,
    /// `glm_moe_route_one`'s completion counter, then its router shares
    /// ([`ONE_PARTIALS`] bytes from `done + 16`).
    done: DevicePtr,
    /// `moe_silu_mul_touch` (null when the target lacks it): the shared
    /// expert's SiLU asks the router weight into L2 ([`MoeLayer::router_touch`]).
    pub(super) silu_touch: KernelHandle,
}

/// [`MoeLayer::router_touch`]: the touching SiLU and its
/// `ops::silu_mul_touch` region (router weight, bias, rows, row bytes).
pub(super) type RouterTouch = (KernelHandle, (DevicePtr, DevicePtr, u32, u32));

/// Widest batch, and most experts, `glm_moe_route_one` takes.
const ONE_MAX_ROWS: u32 = 8;
const ONE_MAX_EXPERTS: u32 = 320;
/// Router warp shares a `glm_moe_route_one` CTA runs (`ROUTE_ONE_VW`).
const ONE_VW: u32 = 3;
/// `glm_moe_route_one`'s share buffer: eight shares of 32 lanes x 4 f32 a
/// 16-expert tile.
const ONE_PARTIALS: usize = (ONE_MAX_EXPERTS as usize / 16) * 8 * 32 * 16;

/// The buffers and shape of one [`RouteFuse::launch`]. `total_tiles` holds
/// the tile count, the worklist 16 bytes past it.
pub(super) struct RouteArgs {
    pub input: DevicePtr,
    pub gate: DevicePtr,
    pub bias: DevicePtr,
    pub logits: DevicePtr,
    pub indices: DevicePtr,
    pub weights: DevicePtr,
    pub sorted_token_ids: DevicePtr,
    pub sorted_expert_ids: DevicePtr,
    pub expert_offsets: DevicePtr,
    pub token_to_perm: DevicePtr,
    pub local_ptrs: DevicePtr,
    pub total_tiles: DevicePtr,
    pub a_packed: DevicePtr,
    pub rows: u32,
    pub experts: u32,
    pub hidden: u32,
    pub top_k: u32,
    pub normalize: bool,
    pub scaling: f32,
    /// Device `[rows]` u32 liveness of dead verify rows
    /// (`moe_route_mask_rows`), or 0.
    pub row_live: DevicePtr,
    /// The NVFP4 input's `scale2`: 1, or the gate/up input scale
    /// (`ATLAS_GLM_FP4L_INPUT_SCALES`).
    pub a_scale2: f32,
}

/// Where steps 1-3 leave the routes of `forward_prefill`'s batch.
pub(super) struct Routes {
    pub indices: DevicePtr,
    pub weights: DevicePtr,
    pub sorted_token_ids: DevicePtr,
    pub expert_offsets: DevicePtr,
    pub token_to_perm: DevicePtr,
    /// The M16 worklist and the NVFP4 input are written too.
    pub prerouted: bool,
}

impl MoeLayer {
    /// Router GEMM, top-k and sort of `n` rows of `input` (`[n, h]`) whose
    /// routed experts are `[inter, h]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn route_and_sort(
        &self,
        input: DevicePtr,
        [n, h, inter]: [u32; 3],
        det: crate::det_trace::Taps<'_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Routes> {
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let (te, ne) = ((n * top_k) as usize, num_experts as usize);
        let rows = (0, n as usize);
        // The sort's outputs overlay the gate logits, which the top-k has
        // consumed by then.
        let gate_logits = ctx.buffers.gate_logits();
        let scratch = ctx.buffers.scratch();
        let routes = Routes {
            indices: scratch,
            weights: scratch.offset(te * 4),
            sorted_token_ids: gate_logits,
            expert_offsets: gate_logits.offset(te * 4 * 2),
            token_to_perm: gate_logits.offset(te * 4 * 2 + (ne + 1) * 4),
            prerouted: self.route_fused_applies(n, h, inter, ctx),
        };
        let sorted_expert_ids = gate_logits.offset(te * 4);
        if routes.prerouted {
            self.route_fused(input, &routes, sorted_expert_ids, [n, h], ctx, stream)?;
        } else {
            // Sequence-parallel Qwen prefill may already have routed this
            // rank's rows before the gather. Consume those routes before the
            // generic router; GLM's canonical/fused path remains unchanged.
            if !self.take_local_routes(
                input,
                n as usize,
                [routes.indices, routes.weights],
                ctx,
                stream,
            )? {
                // Gemma-4 router pre-norm (no-op for other models).
                let router_in = self.router_input(input, n, h, ctx, stream)?;
                super::dump::dump_gate_input(ctx.gpu, stream, router_in, n, h)?;
                // 1. Gate GEMM: [N, H] × [H, num_experts] → [N, num_experts]
                self.prefill_gate_gemm(router_in, gate_logits, n, num_experts, h, ctx, stream)?;
                super::dump::dump_gate_logits(
                    ctx.gpu,
                    stream,
                    gate_logits,
                    n,
                    num_experts,
                    self.router_f32,
                )?;
                det.tap(
                    "x_gate",
                    gate_logits,
                    rows,
                    ne * ctx.config.router_logit_bytes(),
                );
                // Feature-1: fold the router (`mlp.gate`) LoRA delta onto the
                // routing logits BEFORE top-k (reproduces PEFT `mlp.gate`). No-op
                // unless a router delta is installed (ATLAS_LORA_EXPERTS=1).
                self.apply_router_lora_prefill(router_in, gate_logits, n, ctx, stream)?;
                // 2. Batched top-k. DeepSeek-V3 / MiniMax-M2 use sigmoid +
                //    correction bias (detected via `correction_bias_dev`); every
                //    other model takes the softmax path.
                self.prefill_topk(
                    gate_logits,
                    routes.indices,
                    routes.weights,
                    n,
                    num_experts,
                    top_k,
                    ctx,
                    stream,
                )?;
            }
            super::dump::dump_expert_ids(
                ctx.gpu,
                stream,
                routes.indices,
                routes.weights,
                n,
                top_k,
            )?;
            // 3. Sort tokens by expert → L2-optimized ordering.
            ops::moe_sort_by_expert(
                ctx.gpu,
                self.moe_sort_by_expert,
                routes.indices,
                routes.sorted_token_ids,
                sorted_expert_ids,
                routes.expert_offsets,
                routes.token_to_perm,
                n * top_k,
                num_experts,
                top_k,
                stream,
            )?;
        }
        det.tap("rt_ids", routes.indices, rows, top_k as usize * 4);
        det.tap("rt_w", routes.weights, rows, top_k as usize * 4);
        Ok(routes)
    }

    /// Whether `ATLAS_GLM_MOE_ROUTE_FUSE` takes a batch of `n` rows: the
    /// canonical router on a BF16 weight with FP32 logits, the unadapted
    /// sigmoid top-k with correction bias, and a routed FFN on the M16 stream
    /// twins of the prequant path.
    fn route_fused_applies(&self, n: u32, h: u32, inter: u32, ctx: &ForwardContext) -> bool {
        self.route_fuse.loaded()
            && ctx.config.model_type == "glm5_next"
            && self.router_f32
            && crate::layers::canonical_verify::enabled()
            && (1..=MAX_ROWS).contains(&n)
            && self.gate_fp8.is_none()
            && self.gate_nvfp4.is_none()
            && self.weights.router_pre_norm.is_none()
            && self.pre_expert_norm.is_none()
            && self.tid2eid_dev.is_none()
            && self.correction_bias_dev.is_some()
            && ctx.config.scoring_func == "sigmoid"
            && self.router_logits_n == ctx.config.num_experts as u32
            && !self.btile_storage.is_published()
            && !super::dump::enabled()
            // The NVFP4 input lands in expert_down_out before the routed FFN
            // would zero it.
            && !self.ep_prefill_zeroes(ctx)
            && h.is_multiple_of(16)
            && self.stream_m16_prequant(n, h, inter)
    }

    /// The router weight and correction bias for the shared expert's SiLU to
    /// ask into L2 (`ops::silu_mul_touch`) when this batch routes through
    /// the fused launches: the SiLU waits through the attention all-reduce
    /// with the bus idle, while the router runs behind the shared down, cold.
    /// Same bytes either way.
    pub(super) fn router_touch(
        &self,
        n: u32,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
    ) -> Option<RouterTouch> {
        let kernel = self.route_fuse.silu_touch;
        (kernel.0 != 0 && self.route_fused_applies(n, h, inter, ctx)).then(|| {
            let bias = self
                .correction_bias_dev
                .expect("route_fused_applies checked");
            let experts = ctx.config.num_experts as u32;
            (kernel, (self.weights.gate.weight, bias, experts, h * 2))
        })
    }

    /// Steps 1-3 plus the M16 worklist and the NVFP4 input, in two launches.
    fn route_fused(
        &self,
        input: DevicePtr,
        routes: &Routes,
        sorted_expert_ids: DevicePtr,
        [n, h]: [u32; 2],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let top_k = ctx.config.num_experts_per_tok as u32;
        anyhow::ensure!(
            ctx.buffers.sizes().moe_router_in_f32 >= 16 + (n * top_k) as usize * 8,
            "fused MoE routing: M16 worklist exceeds router scratch"
        );
        let args = RouteArgs {
            input,
            gate: self.weights.gate.weight,
            bias: self
                .correction_bias_dev
                .expect("route_fused_applies checked"),
            logits: ctx.buffers.gate_logits(),
            indices: routes.indices,
            weights: routes.weights,
            sorted_token_ids: routes.sorted_token_ids,
            sorted_expert_ids,
            expert_offsets: routes.expert_offsets,
            token_to_perm: routes.token_to_perm,
            local_ptrs: self.gate_ptrs_t.as_ref().expect("checked").packed_ptrs,
            total_tiles: ctx.buffers.moe_router_in_f32(),
            a_packed: ctx.buffers.expert_down_out(),
            rows: n,
            experts: ctx.config.num_experts as u32,
            hidden: h,
            top_k,
            normalize: ctx.config.norm_topk_prob,
            scaling: ctx.config.routed_scaling_factor as f32,
            row_live: DevicePtr(0),
            a_scale2: self.a1_scale2(),
        };
        self.route_fuse.launch(ctx.gpu, &args, stream)
    }
}

#[cfg(test)]
#[path = "route_fuse_test_gpu.rs"]
mod gpu_tests;
