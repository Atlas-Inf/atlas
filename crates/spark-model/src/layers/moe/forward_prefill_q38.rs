// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_MOE=1`: Qwen3.8-Flash-Next's MoE prefill GEMMs on
//! `moe_prefill_q38.cu`, every output byte unchanged:
//!
//! * the router logits ([`MoeLayer::try_q38_router`]): BF16 weight + the
//!   tile GEMM instead of `w4a16_gemm` (6.44 -> 0.82 ms a layer at 16K);
//! * the shared expert ([`MoeLayer::try_q38_shared`]): 4.88 -> 3.41 ms;
//! * the routed experts, below.
//!
//! The routed chain replaces the default `moe_w4a16_fused_gate_up_t_k64` ->
//! `moe_silu_mul` -> `moe_w4a16_grouped_gemm_ptrtable_t_k64` this target runs,
//! in three launches:
//!
//! 1. `moe_q38_a_to_e4m3`: the token-major input to E4M3 once (the default
//!    converts it inside the K loop of each of its ten column tiles);
//! 2. `moe_q38_gate_up_silu`: gate + up + SiLU*mul, writing the activation
//!    as E4M3 (the default's down converts its BF16 to exactly that);
//! 3. `moe_q38_down`: the down projection into `expert_down_out`.
//!
//! GB10, 256 local experts, 16000-token chunk (~80K routed rows): 24.5 ->
//! 16.5 ms a layer (`scripts/dev/qwen4exp_moe_prefill_bench.cu`, which checks
//! every byte of the activation's E4M3 image and of the down output).
//!
//! It takes only the shape that chain serves: NVFP4 transposed tables, SiLU,
//! no LoRA on the experts, none of the alternative gate/up or down arms
//! selected. Remote experts (EP) have null tables and return early, as in the
//! default kernels, so `prepare_ep_prefill_outputs`' zeroing still stands.
//!
//! `ATLAS_QWEN4EXP_PREFILL_MOE_CHECK=<n>`: for the first `n` layer calls, run
//! the q38 chain, keep its `expert_down_out` on the host, run the default
//! chain over the same buffers, and fail on any differing byte
//! (synchronizing, ~0.8 GB of host memory at a 16K chunk; diagnostic only).

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

/// `ATLAS_QWEN4EXP_PREFILL_MOE=1`.
pub(crate) fn q38_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::forward_prefill_routed::env_flag("ATLAS_QWEN4EXP_PREFILL_MOE"))
}

/// Layer calls left to cross-check (`1` or `true` means 16).
fn check_left() -> &'static std::sync::atomic::AtomicI64 {
    static LEFT: std::sync::OnceLock<std::sync::atomic::AtomicI64> = std::sync::OnceLock::new();
    LEFT.get_or_init(|| {
        let n = match std::env::var("ATLAS_QWEN4EXP_PREFILL_MOE_CHECK").as_deref() {
            Ok("1") | Ok("true") => 16,
            Ok(v) => v.parse().unwrap_or(0),
            Err(_) => 0,
        };
        std::sync::atomic::AtomicI64::new(n)
    })
}

/// The q38 chain's `expert_down_out`, held for [`finish_q38_check`].
static PENDING: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);

/// After the default chain ran a checked call: require its `expert_down_out`
/// to equal the q38 chain's, byte for byte.
pub(super) fn finish_q38_check(ctx: &ForwardContext, rows: u32, h: u32, stream: u64) -> Result<()> {
    let Some(fast) = PENDING
        .lock()
        .map_err(|_| anyhow::anyhow!("q38 check lock"))?
        .take()
    else {
        return Ok(());
    };
    ctx.gpu.synchronize(stream)?;
    let mut reference = vec![0u8; rows as usize * h as usize * 2];
    ctx.gpu
        .copy_d2h(ctx.buffers.expert_down_out(), &mut reference)?;
    let diff = reference.iter().zip(&fast).filter(|(a, b)| a != b).count();
    anyhow::ensure!(
        diff == 0,
        "ATLAS_QWEN4EXP_PREFILL_MOE_CHECK: the q38 MoE prefill differs from the default \
         in {diff}/{} expert_down_out bytes ({rows} routed rows)",
        reference.len()
    );
    tracing::info!("ATLAS_QWEN4EXP_PREFILL_MOE_CHECK: {rows} routed rows byte-identical");
    Ok(())
}

impl MoeLayer {
    /// The router logits of a prefill chunk, when the q38 arm serves them:
    /// `moe_q38_router_dequant` writes the NVFP4 router weight as the BF16
    /// `[N, K]` values `w4a16_gemm` forms in its loop, and
    /// `dense_gemm_bf16_pipelined` runs the same in-order m16n8k16 chain over
    /// them -- byte-identical logits, so identical routing. GB10, 16000
    /// tokens x 512 experts: 6.44 -> 0.82 ms a layer (2.7x at 300 tokens).
    /// The BF16 weight (2.6 MB) is staged in `expert_up_out`, which the
    /// routed GEMMs only fill after the router has run. `Ok(false)` launched
    /// nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_router(
        &self,
        router_in: DevicePtr,
        w: &QuantizedWeight,
        logits: DevicePtr,
        n: u32,
        n_out: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let bytes = n_out as usize * h as usize * 2;
        if !q38_requested()
            || ctx.config.model_type != "qwen4_exp"
            || n < 32
            || !h.is_multiple_of(16)
            || ctx.buffers.sizes().expert_up_out < bytes
        {
            return Ok(false);
        }
        let gpu = ctx.gpu;
        let k_dq = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_router_dequant");
        if k_dq.0 == 0 {
            return Ok(false);
        }
        let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
        let w_bf16 = ctx.buffers.expert_up_out();
        KernelLaunch::new(gpu, k_dq)
            .grid([(n_out * h / 2).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.weight)
            .arg_ptr(w.weight_scale)
            .arg_f32(w.weight_scale_2)
            .arg_ptr(w_bf16)
            .arg_u32(n_out)
            .arg_u32(h)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_gemm)
            .grid([n_out.div_ceil(128), n.div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(router_in)
            .arg_ptr(w_bf16)
            .arg_ptr(logits)
            .arg_u32(n)
            .arg_u32(n_out)
            .arg_u32(h)
            .launch(stream)?;
        Ok(true)
    }

    /// The shared expert of a prefill chunk, when the q38 arm serves it: the
    /// default's transposed arm (`w4a16_gemm_t` gate, up and down around
    /// `moe_silu_mul`) as `moe_q38_a_to_e4m3`, `moe_q38_dense_gate_up_silu`
    /// and `moe_q38_dense_down`, the same arithmetic as the routed pair --
    /// byte-identical `shared_down_out`. GB10, 16000 tokens: 4.88 -> 3.41
    /// ms a layer. `out` = [gate scratch, up scratch, down output] (the E4M3
    /// activation and input are staged in the first two). `Ok(false)`
    /// launched nothing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_shared(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        out: [DevicePtr; 3],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (Some(sg), Some(su), Some(sd)) =
            (&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t)
        else {
            return Ok(false);
        };
        let sizes = ctx.buffers.sizes();
        let serves = q38_requested()
            && ctx.config.model_type == "qwen4_exp"
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.shared_gate_fp8.is_none()
            && self.shared_up_fp8.is_none()
            && self.shared_down_fp8.is_none()
            && self.bf16_shared_expert.is_none()
            && !self.m5_projections.shared.enabled()
            && !self.gelu_activation
            && self.lora.is_none()
            && h.is_multiple_of(128)
            && inter.is_multiple_of(64)
            && sizes.ssm_deinterleaved >= n as usize * inter as usize
            && sizes.ssm_qkvz >= n as usize * h as usize;
        if !serves || out[0] != ctx.buffers.ssm_deinterleaved() || out[1] != ctx.buffers.ssm_qkvz()
        {
            return Ok(false);
        }
        let gpu = ctx.gpu;
        let k_a8 = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_a_to_e4m3");
        let k_gu = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_dense_gate_up_silu");
        let k_dn = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_dense_down");
        if k_a8.0 == 0 || k_gu.0 == 0 || k_dn.0 == 0 {
            return Ok(false);
        }
        let (act8, a8) = (out[0], out[1]);
        let cells = n * h;
        KernelLaunch::new(gpu, k_a8)
            .grid([(cells / 4).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(a8)
            .arg_u32(cells)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_gu)
            .grid([inter / 64, n.div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(a8)
            .arg_ptr(sg.weight)
            .arg_ptr(sg.weight_scale)
            .arg_f32(sg.weight_scale_2)
            .arg_ptr(su.weight)
            .arg_ptr(su.weight_scale)
            .arg_f32(su.weight_scale_2)
            .arg_ptr(act8)
            .arg_u32(n)
            .arg_u32(inter)
            .arg_u32(h)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_dn)
            .grid([h / 128, n.div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(act8)
            .arg_ptr(sd.weight)
            .arg_ptr(sd.weight_scale)
            .arg_f32(sd.weight_scale_2)
            .arg_ptr(out[2])
            .arg_u32(n)
            .arg_u32(h)
            .arg_u32(inter)
            .launch(stream)?;
        Ok(true)
    }

    /// Run the routed gate/up, SiLU and down of a prefill chunk on the q38
    /// kernels when they serve it; `Ok(false)` launched nothing.
    /// The unpermute + top-k reduce of a q38 chunk: `moe_q38_unpermute_local`
    /// sums only the routes of this rank's experts (all of them without EP),
    /// in slot order -- byte-identical to `moe_unpermute_reduce_indexed` over
    /// zeroed remote rows, without the 1.2 GB of clears a 16K layer needed
    /// for them (`prepare_ep_prefill_outputs`) and with 16-byte loads (GB10,
    /// 16000 tokens: 4.19 -> 2.17 ms). `dims` = [hidden, tokens, top_k].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_unpermute(
        &self,
        expert_output: DevicePtr,
        output: DevicePtr,
        token_to_perm: DevicePtr,
        topk_ids: DevicePtr,
        topk_weights: DevicePtr,
        dims: [u32; 3],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let [hidden, tokens, topk] = dims;
        let inter = ctx.config.routed_inter_local() as u32;
        if !hidden.is_multiple_of(8)
            || hidden / 8 > 1024
            || topk_ids.is_null()
            || !self.q38_routed_serves(hidden, inter, ctx)
        {
            return Ok(false);
        }
        let k = crate::layers::try_kernel(ctx.gpu, "moe_prefill_q38", "moe_q38_unpermute_local");
        if k.0 == 0 {
            return Ok(false);
        }
        let (start, end) = if ctx.config.ep_world_size > 1 {
            let (s, e) = ctx.config.local_expert_range();
            (s as u32, e as u32)
        } else {
            (0, ctx.config.num_experts as u32)
        };
        KernelLaunch::new(ctx.gpu, k)
            .grid([tokens, 1, 1])
            .block([hidden / 8, 1, 1])
            .arg_ptr(expert_output)
            .arg_ptr(output)
            .arg_ptr(token_to_perm)
            .arg_ptr(topk_ids)
            .arg_ptr(topk_weights)
            .arg_u32(hidden)
            .arg_u32(tokens)
            .arg_u32(topk)
            .arg_u32(start)
            .arg_u32(end)
            .launch(stream)?;
        Ok(true)
    }

    /// Whether the q38 routed chain serves this layer's prefill: the shape the
    /// default chain it replaces runs (NVFP4 transposed tables, SiLU, no expert
    /// LoRA, none of the alternative gate/up or down arms). Also read before
    /// the grid is sized: the q38 kernels stride their row tiles, so they do
    /// not need the exact per-expert tile count, nor the host round trip
    /// (`ATLAS_MOE_PREFILL_EXACT_TILES`) that buys it.
    pub(super) fn q38_routed_serves(&self, h: u32, inter: u32, ctx: &ForwardContext) -> bool {
        let fp8_down = std::env::var("ATLAS_MOE_PREFILL_FP8_DOWN").ok().as_deref() == Some("1");
        q38_requested()
            && ctx.config.model_type == "qwen4_exp"
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.btile_storage.is_published()
            && !self.nvfp4_mmq_layout
            && !self.nvfp4_prequant_moe
            && !self.gateup_fp4
            && !self.down_fp4
            && !self.nvfp4_gate_up_m128
            && !self.nvfp4_down_m32
            && !self.gelu_activation
            && self.lora.is_none()
            && !fp8_down
            && !super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
            && h.is_multiple_of(128)
            && inter.is_multiple_of(64)
            && crate::layers::try_kernel(ctx.gpu, "moe_prefill_q38", "moe_q38_down").0 != 0
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_q38_routed_prefill(
        &self,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        rows_per_expert_grid: u32,
        total_expanded: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let (Some(gp), Some(up), Some(dp)) =
            (&self.gate_ptrs_t, &self.up_ptrs_t, &self.down_ptrs_t)
        else {
            return Ok(false);
        };
        if !self.q38_routed_serves(h, inter, ctx) {
            return Ok(false);
        }
        let gpu = ctx.gpu;
        let k_a8 = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_a_to_e4m3");
        let k_gu = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_gate_up_silu");
        let k_dn = crate::layers::try_kernel(gpu, "moe_prefill_q38", "moe_q38_down");
        if k_a8.0 == 0 || k_gu.0 == 0 || k_dn.0 == 0 {
            return Ok(false);
        }
        // Scratch: the E4M3 input in `expert_up_out` ([n, h] bytes, inside its
        // [n * top_k, inter] BF16) and the E4M3 activation in
        // `expert_gate_out` ([rows, inter] bytes, inside the same).
        let a8 = ctx.buffers.expert_up_out();
        let act8 = ctx.buffers.expert_gate_out();
        let cells = n * h;
        KernelLaunch::new(gpu, k_a8)
            .grid([(cells / 4).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(expert_input)
            .arg_ptr(a8)
            .arg_u32(cells)
            .launch(stream)?;
        // The kernels stride over 128-row tiles, so any grid computes every
        // row; size it like the default's persistent grid (rows, not tiles).
        let grid_m = rows_per_expert_grid.div_ceil(128).max(1);
        KernelLaunch::new(gpu, k_gu)
            .grid([inter / 64, grid_m, num_experts])
            .block([256, 1, 1])
            .arg_ptr(a8)
            .arg_ptr(gp.packed_ptrs)
            .arg_ptr(gp.scale_ptrs)
            .arg_ptr(gp.scale2_vals)
            .arg_ptr(up.packed_ptrs)
            .arg_ptr(up.scale_ptrs)
            .arg_ptr(up.scale2_vals)
            .arg_ptr(act8)
            .arg_ptr(expert_offsets)
            .arg_ptr(sorted_token_ids)
            .arg_u32(num_experts)
            .arg_u32(inter)
            .arg_u32(h)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_dn)
            .grid([h / 128, grid_m, num_experts])
            .block([256, 1, 1])
            .arg_ptr(act8)
            .arg_ptr(dp.packed_ptrs)
            .arg_ptr(dp.scale_ptrs)
            .arg_ptr(dp.scale2_vals)
            .arg_ptr(ctx.buffers.expert_down_out())
            .arg_ptr(expert_offsets)
            .arg_u32(num_experts)
            .arg_u32(h)
            .arg_u32(inter)
            .launch(stream)?;
        if check_left().fetch_sub(1, std::sync::atomic::Ordering::Relaxed) > 0 {
            // Keep this result and let the default chain run over the same
            // buffers; `finish_q38_check` compares once it has.
            gpu.synchronize(stream)?;
            let mut fast = vec![0u8; total_expanded as usize * h as usize * 2];
            gpu.copy_d2h(ctx.buffers.expert_down_out(), &mut fast)?;
            *PENDING
                .lock()
                .map_err(|_| anyhow::anyhow!("q38 check lock"))? = Some(fast);
            return Ok(false);
        }
        Ok(true)
    }
}
