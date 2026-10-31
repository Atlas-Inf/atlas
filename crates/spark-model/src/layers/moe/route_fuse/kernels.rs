// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

impl RouteFuse {
    pub(in super::super) fn new(
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
        one: KernelHandle(0),
        done: DevicePtr(0),
        silu_touch: KernelHandle(0),
    };

    /// Both kernels of the target, or neither.
    pub(in super::super) fn load(gpu: &dyn GpuBackend) -> Self {
        let kernel = |name| crate::layers::try_kernel(gpu, "glm_moe_route", name);
        let mut this = Self {
            router_quant: kernel("glm_moe_router_quant"),
            select: kernel("glm_moe_route_select"),
            one: kernel("glm_moe_route_one"),
            done: DevicePtr(0),
            silu_touch: crate::layers::try_kernel(gpu, "moe_silu_mul", "moe_silu_mul_touch"),
        };
        if this.router_quant.0 == 0 || this.select.0 == 0 {
            return Self::NULL;
        }
        // The one-launch pair's counter, 0 between launches (the last CTA
        // returns it to 0), and its router shares.
        match gpu
            .alloc(16 + ONE_PARTIALS)
            .and_then(|p| gpu.memset(p, 0, 4).map(|()| p))
        {
            Ok(done) if this.one.0 != 0 => this.done = done,
            _ => this.one = KernelHandle(0),
        }
        this
    }

    pub(in super::super) fn loaded(&self) -> bool {
        self.select.0 != 0
    }

    /// The two launches over `rows` rows of `input` (`[rows, hidden]`): the
    /// router logits, the routes, the sort, the M16 worklist (one N tile of
    /// 64 rows per routed local expert, `local_ptrs` nonzero, as
    /// `decode_m16_grid`) and the NVFP4 input (`a_packed`, its group scales
    /// after `rows * hidden / 2` bytes, as `prequant_fp4_gate_up`).
    pub(in super::super) fn launch(
        &self,
        gpu: &dyn GpuBackend,
        a: &RouteArgs,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.loaded() && a.hidden.is_multiple_of(16) && a.experts <= 512 && a.top_k <= 32,
            "fused MoE routing: unsupported shape (hidden {}, {} experts, top-{})",
            a.hidden,
            a.experts,
            a.top_k
        );
        let router_ctas = a.experts.div_ceil(16);
        let quant_ctas = (a.rows * (a.hidden / 16)).div_ceil(256);
        if self.one.0 != 0 && a.rows <= ONE_MAX_ROWS && a.experts <= ONE_MAX_EXPERTS {
            let share_ctas = (router_ctas * 8).div_ceil(ONE_VW);
            return KernelLaunch::new(gpu, self.one)
                .grid([share_ctas + quant_ctas, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a.input)
                .arg_ptr(a.gate)
                .arg_ptr(a.bias)
                .arg_ptr(a.logits)
                .arg_ptr(self.done.offset(16))
                .arg_ptr(a.a_packed)
                .arg_ptr(a.a_packed.offset(a.rows as usize * a.hidden as usize / 2))
                .arg_u32(share_ctas)
                .arg_u32(a.experts)
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
                .arg_u32(a.hidden)
                .arg_u32(a.top_k)
                .arg_u32(a.normalize as u32)
                .arg_f32(a.scaling)
                .arg_u32(1)
                .arg_u32(64)
                .arg_ptr(self.done)
                .arg_ptr(a.row_live)
                .arg_f32(a.a_scale2)
                .launch(stream);
        }
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
            .arg_f32(a.a_scale2)
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
            .arg_ptr(a.row_live)
            .launch(stream)
    }
}
