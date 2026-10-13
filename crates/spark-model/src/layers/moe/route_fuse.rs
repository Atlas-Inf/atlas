// SPDX-License-Identifier: AGPL-3.0-only

//! Steps 1-3 of `MoeLayer::forward_prefill` (router GEMM, top-k, sort by
//! expert), and `ATLAS_GLM_MOE_ROUTE_FUSE=1` (default off), which runs them
//! for a canonical verify block in two launches.
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
use spark_runtime::kernel_args::KernelLaunch;

/// Widest batch the switch takes. The select's top-k runs a warp per row
/// on eight warps, so past 16 rows it is slower than the per-row top-k CTAs
/// (glm_moe_route_bench: 12 us a layer saved at 1..=8 rows, 7 at 16, 5
/// lost at 32); wider batches run the five launches, same words.
const MAX_ROWS: u32 = 16;

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

/// The two routing kernels, both loaded or both null.
#[derive(Clone, Copy)]
pub(super) struct RouteFuse {
    router_quant: KernelHandle,
    select: KernelHandle,
}

impl RouteFuse {
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        let on = route_fuse_requested()? && config.model_type == "glm5_next";
        let this = if on { Self::load(gpu) } else { Self::NULL };
        if on && gpu.op_cache().once("moe:route_fuse") {
            if this.loaded() {
                tracing::info!("ATLAS_GLM_MOE_ROUTE_FUSE: routing chain in two launches");
            } else {
                tracing::warn!("ATLAS_GLM_MOE_ROUTE_FUSE=1 ignored: target lacks glm_moe_route");
            }
        }
        Ok(this)
    }

    const NULL: Self = Self {
        router_quant: KernelHandle(0),
        select: KernelHandle(0),
    };

    /// Both kernels of the target, or neither.
    pub(super) fn load(gpu: &dyn GpuBackend) -> Self {
        let kernel = |name| super::super::try_kernel(gpu, "glm_moe_route", name);
        let this = Self {
            router_quant: kernel("glm_moe_router_quant"),
            select: kernel("glm_moe_route_select"),
        };
        if this.router_quant.0 != 0 && this.select.0 != 0 {
            this
        } else {
            Self::NULL
        }
    }

    pub(super) fn loaded(&self) -> bool {
        self.select.0 != 0
    }

    /// The two launches over `rows` rows of `input` (`[rows, hidden]`): the
    /// router logits, the routes, the sort, the M16 worklist (one N tile of
    /// 64 rows per routed local expert, `local_ptrs` nonzero, as
    /// `decode_m16_grid`) and the NVFP4 input (`a_packed`, its group scales
    /// after `rows * hidden / 2` bytes, as `prequant_fp4_gate_up`).
    pub(super) fn launch(&self, gpu: &dyn GpuBackend, a: &RouteArgs, stream: u64) -> Result<()> {
        anyhow::ensure!(
            self.loaded() && a.hidden.is_multiple_of(16) && a.experts <= 512 && a.top_k <= 32,
            "fused MoE routing: unsupported shape (hidden {}, {} experts, top-{})",
            a.hidden,
            a.experts,
            a.top_k
        );
        let router_ctas = a.experts.div_ceil(16);
        let quant_ctas = (a.rows * (a.hidden / 16)).div_ceil(256);
        KernelLaunch::new(gpu, self.router_quant)
            .grid([router_ctas + quant_ctas, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.input)
            .arg_ptr(a.gate)
            .arg_ptr(a.bias)
            .arg_ptr(a.logits)
            .arg_ptr(a.a_packed)
            .arg_ptr(a.a_packed.offset(a.rows as usize * a.hidden as usize / 2))
            .arg_u32(a.rows)
            .arg_u32(a.experts)
            .arg_u32(a.hidden)
            .arg_u32(router_ctas)
            .arg_u32(a.experts)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.select)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.logits)
            .arg_ptr(a.bias)
            .arg_ptr(a.indices)
            .arg_ptr(a.weights)
            .arg_ptr(a.sorted_token_ids)
            .arg_ptr(a.sorted_expert_ids)
            .arg_ptr(a.expert_offsets)
            .arg_ptr(a.token_to_perm)
            .arg_ptr(a.local_ptrs)
            .arg_ptr(a.total_tiles.offset(16))
            .arg_ptr(a.total_tiles)
            .arg_u32(a.rows)
            .arg_u32(a.experts)
            .arg_u32(a.top_k)
            .arg_u32(a.normalize as u32)
            .arg_f32(a.scaling)
            .arg_u32(1)
            .arg_u32(64)
            .launch(stream)
    }
}

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
            // Gemma-4 router pre-norm (no-op for other models).
            let router_in = self.router_input(input, n, h, ctx, stream)?;
            super::dump::dump_gate_input(ctx.gpu, stream, router_in, n, h)?;
            // 1. Gate GEMM: [N, H] × [H, num_experts] → [N, num_experts]
            self.prefill_gate_gemm(router_in, gate_logits, n, num_experts, h, ctx, stream)?;
            super::dump::dump_gate_logits(ctx.gpu, stream, gate_logits, n, num_experts)?;
            det.tap("x_gate", gate_logits, rows, ne * 2);
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
    /// canonical BF16 router, the unadapted sigmoid top-k with correction
    /// bias, and a routed FFN on the M16 stream twins of the prequant path.
    fn route_fused_applies(&self, n: u32, h: u32, inter: u32, ctx: &ForwardContext) -> bool {
        self.route_fuse.loaded()
            && ctx.config.model_type == "glm5_next"
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
        };
        self.route_fuse.launch(ctx.gpu, &args, stream)
    }
}

#[cfg(test)]
#[path = "route_fuse_test_gpu.rs"]
mod gpu_tests;
