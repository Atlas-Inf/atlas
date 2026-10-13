// SPDX-License-Identifier: AGPL-3.0-only

//! Optional (`try_kernel`) kernel handles resolved for `MoeLayer::new_with_hash`.

use super::*;

/// Kernels a target may not ship; each is `KernelHandle(0)` when absent or
/// gated off, and the dispatch sites check the handle before launching.
pub(super) struct OptionalKernels {
    pub(super) moe_grouped_gemm_t_k64_m32: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_vecscale: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k128: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k128w: ops::K128wKernel,
    pub(super) moe_w4a4_prequant_gate_up_silu: ops::K128wKernel,
    /// Persistent CTAs of the K128W `_persist` twins above when resolved,
    /// else 0.
    pub(super) k128w_persist_ctas: u32,
    pub(super) moe_mtile_prefix_k: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_compact: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_vecscale_compact: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_compact_gate_up: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_vecscale_compact_gate_up: KernelHandle,
    pub(super) moe_nvfp4_mmq_gate_up_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_down_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_quantize_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_repack_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_silu_scale2_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_scale2_rows_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_mmq_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_mmq_k: KernelHandle,
    pub(super) silu_mul_quant_nvfp4_k: KernelHandle,
    pub(super) moe_grouped_gemm_e8m0: KernelHandle,
    pub(super) moe_grouped_gemm_t_e8m0: KernelHandle,
    pub(super) moe_grouped_gemm_t_k64_e8m0: KernelHandle,
    pub(super) moe_fused_gate_up_t_e8m0: KernelHandle,
    pub(super) moe_fused_gate_up_t_k64_e8m0: KernelHandle,
    pub(super) moe_fused_gate_up_t_k64_m128: KernelHandle,
    pub(super) moe_fused_gate_up_t_k64_fp4: KernelHandle,
    pub(super) moe_fp8_grouped_gemm_k: KernelHandle,
    pub(super) moe_build_tile_worklist_k: KernelHandle,
    pub(super) moe_w8a8_grouped_gemm_k: KernelHandle,
    pub(super) moe_w8a8_grouped_gemm_pm4_k: KernelHandle,
    pub(super) per_token_group_quant_fp8_k: KernelHandle,
    pub(super) silu_mul_quant_fp8_k: KernelHandle,
    pub(super) fp8_gemm_t_blockscaled_k: KernelHandle,
    pub(super) moe_bf16_grouped_gemm_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_bf16_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_bf16_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_bf16_batch2_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_bf16_batch2_k: KernelHandle,
    pub(super) w8a16_gemm_k: KernelHandle,
    pub(super) w8a16_gemm_pipelined_k: KernelHandle,
    pub(super) moe_gate_topk_fused_k: KernelHandle,
}

/// The batched router-logits GEMV (`dense_gemv_bf16_batchm` arguments and
/// grid), or 0. GLM's 288-expert router takes the bit-identical load-ahead
/// tier when the target ships it and its grid of `num_experts / 4` CTAs fits
/// that tier; no other target is asked for that symbol.
pub(super) fn router_gemv_batchm(
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> KernelHandle {
    let module = "dense_gemv_bf16_batchm";
    let ahead = super::super::try_kernel_gated(
        config.model_type == "glm5_next"
            && config.num_experts.div_ceil(4) <= ops::DENSE_GEMV_AHEAD_MAX_CTAS as usize,
        gpu,
        module,
        "dense_gemv_bf16_batchm_ahead",
    );
    if ahead.0 != 0 {
        return ahead;
    }
    super::super::try_kernel(gpu, module, "dense_gemv_bf16_batchm")
}

impl OptionalKernels {
    pub(super) fn resolve(gpu: &dyn GpuBackend, config: &atlas_core::config::ModelConfig) -> Self {
        let k128w = config.model_type == "glm5_next"
            && std::env::var("ATLAS_MOE_PREQUANT_K128").as_deref() == Ok("1")
            && std::env::var("ATLAS_MOE_PREQUANT_K128W").as_deref() != Ok("0");
        let (t_k128w, gate_up_silu, k128w_persist_ctas) = k128w_kernels(
            gpu,
            k128w,
            std::env::var("ATLAS_MOE_GATE_UP_SILU").as_deref() != Ok("0"),
            std::env::var("ATLAS_GLM_MOE_PREFILL_PERSIST").as_deref() == Ok("1"),
        );
        Self {
            moe_grouped_gemm_t_k64_m32: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t_k64_m32",
            ),
            moe_w4a4_prequant_t_k64: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a4_grouped_gemm_prequant_t_k64",
            ),
            moe_w4a4_prequant_t_k64_vecscale: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale",
            ),
            moe_w4a4_prequant_t_k128: if std::env::var("ATLAS_MOE_PREQUANT_K128").as_deref()
                == Ok("1")
            {
                super::super::try_kernel(gpu, "moe_w4a16", "moe_w4a4_grouped_gemm_prequant_t_k128")
            } else {
                KernelHandle(0)
            },
            moe_w4a4_prequant_t_k128w: t_k128w,
            moe_w4a4_prequant_gate_up_silu: gate_up_silu,
            k128w_persist_ctas,
            moe_mtile_prefix_k: if k128w {
                super::super::try_kernel(gpu, "moe_w4a16", "moe_mtile_prefix")
            } else {
                KernelHandle(0)
            },
            moe_w4a4_prequant_t_k64_compact: if config.model_type == "glm5_next" {
                super::super::try_kernel(
                    gpu,
                    "moe_w4a16",
                    "moe_w4a4_grouped_gemm_prequant_t_k64_compact",
                )
            } else {
                KernelHandle(0)
            },
            moe_w4a4_prequant_t_k64_vecscale_compact: if config.model_type == "glm5_next" {
                super::super::try_kernel(
                    gpu,
                    "moe_w4a16",
                    "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact",
                )
            } else {
                KernelHandle(0)
            },
            moe_w4a4_prequant_t_k64_compact_gate_up: if config.model_type == "glm5_next" {
                super::super::try_kernel(
                    gpu,
                    "moe_w4a16",
                    "moe_w4a4_grouped_gemm_prequant_t_k64_compact_gate_up",
                )
            } else {
                KernelHandle(0)
            },
            moe_w4a4_prequant_t_k64_vecscale_compact_gate_up: if config.model_type == "glm5_next" {
                super::super::try_kernel(
                    gpu,
                    "moe_w4a16",
                    "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up",
                )
            } else {
                KernelHandle(0)
            },
            moe_nvfp4_mmq_gate_up_k: super::super::try_kernel(
                gpu,
                "moe_nvfp4_mmq",
                "atlas_moe_nvfp4_mmq64_gate_up",
            ),
            moe_nvfp4_mmq_down_k: super::super::try_kernel(
                gpu,
                "moe_nvfp4_mmq",
                "atlas_moe_nvfp4_mmq64_down",
            ),
            moe_nvfp4_mmq_quantize_k: super::super::try_kernel(
                gpu,
                "moe_nvfp4_mmq",
                "atlas_moe_nvfp4_quantize_bf16",
            ),
            moe_nvfp4_mmq_repack_k: super::super::try_kernel(
                gpu,
                "moe_nvfp4_mmq",
                "atlas_moe_nvfp4_repack_batched",
            ),
            moe_nvfp4_mmq_silu_scale2_k: super::super::try_kernel(
                gpu,
                "moe_nvfp4_mmq",
                "atlas_moe_nvfp4_silu_scale2",
            ),
            moe_nvfp4_mmq_scale2_rows_k: super::super::try_kernel(
                gpu,
                "moe_nvfp4_mmq",
                "atlas_moe_nvfp4_scale2_rows",
            ),
            moe_expert_gate_up_shared_mmq_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_mmq",
                "moe_expert_gate_up_shared_mmq",
            ),
            moe_expert_silu_down_shared_mmq_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_mmq",
                "moe_expert_silu_down_shared_mmq",
            ),
            silu_mul_quant_nvfp4_k: super::super::try_kernel(
                gpu,
                "moe_silu_mul",
                "silu_mul_quant_nvfp4",
            ),
            // ARM-2 Phase-K native-MXFP4 (E8M0) prefill variants — try_kernel:
            // only the deepseek-v4-flash target's moe_w4a16 module ships them.
            moe_grouped_gemm_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_e8m0",
            ),
            moe_grouped_gemm_t_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t_e8m0",
            ),
            moe_grouped_gemm_t_k64_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t_k64_e8m0",
            ),
            moe_fused_gate_up_t_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_e8m0",
            ),
            moe_fused_gate_up_t_k64_e8m0: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_k64_e8m0",
            ),
            // M=128 variant only present in models where Block D #3 has
            // been ported (currently minimax-m2-229b). Other models keep
            // KernelHandle(0) and dispatch falls through to M=64.
            moe_fused_gate_up_t_k64_m128: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_k64_m128",
            ),
            // FUSED FP4 gate_up kernel (ATLAS_HOLO_MOE_GATEUP_FP4). try_kernel:
            // KernelHandle(0) on images that didn't compile it; the FP4 dispatch
            // checks this handle != 0 before firing.
            moe_fused_gate_up_t_k64_fp4: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_fused_gate_up_t_k64_fp4",
            ),
            // THE routed-expert FP8 prefill kernel: grid-compaction (persistent
            // 96-CTA grid over a compacted work-list). Handle may be 0 on older
            // images that don't ship it.
            moe_fp8_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_fp8_grouped_gemm",
                "moe_fp8_grouped_gemm",
            ),
            // Work-list builder (module "moe" = moe_permute.cu). Launched on the
            // SAME stream as the grouped GEMM (read-after-write of total_tiles).
            moe_build_tile_worklist_k: super::super::try_kernel(
                gpu,
                "moe",
                "moe_build_tile_worklist",
            ),
            moe_w8a8_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_w8a8_grouped_gemm",
                "moe_w8a8_grouped_gemm",
            ),
            // PM4-geometry W8A8 grouped GEMM (same module). Handle may be 0 on
            // targets/images without it; dispatch falls back to the dense grid.
            moe_w8a8_grouped_gemm_pm4_k: super::super::try_kernel(
                gpu,
                "moe_w8a8_grouped_gemm",
                "moe_w8a8_grouped_gemm_pm4",
            ),
            per_token_group_quant_fp8_k: super::super::try_kernel(
                gpu,
                "per_token_group_quant_fp8",
                "per_token_group_quant_fp8",
            ),
            // Fused silu_mul + per-token-group quant. Same module as
            // moe_silu_mul, so a model that shadows moe_silu_mul.cu without
            // this entry point gets handle 0 → unfused fallback.
            silu_mul_quant_fp8_k: super::super::try_kernel(
                gpu,
                "moe_silu_mul",
                "silu_mul_quant_fp8",
            ),
            fp8_gemm_t_blockscaled_k: super::super::try_kernel(
                gpu,
                "fp8_gemm_t_blockscaled",
                "fp8_gemm_t_blockscaled",
            ),
            moe_bf16_grouped_gemm_k: super::super::try_kernel(
                gpu,
                "moe_bf16_grouped_gemm",
                "moe_bf16_grouped_gemm",
            ),
            moe_expert_gate_up_shared_bf16_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16",
                "moe_expert_gate_up_shared_bf16",
            ),
            moe_expert_silu_down_shared_bf16_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16",
                "moe_expert_silu_down_shared_bf16",
            ),
            moe_expert_gate_up_shared_bf16_batch2_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16_batch2",
                "moe_expert_gate_up_shared_bf16_batch2",
            ),
            moe_expert_silu_down_shared_bf16_batch2_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_bf16_batch2",
                "moe_expert_silu_down_shared_bf16_batch2",
            ),
            w8a16_gemm_k: super::super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            moe_gate_topk_fused_k: super::super::try_kernel(
                gpu,
                "moe_gate_topk",
                "moe_gate_topk_fused",
            ),
        }
    }
}

/// The K128W down and fused gate/up kernels (gate/up only with `silu`),
/// with `persist` (`ATLAS_GLM_MOE_PREFILL_PERSIST=1`) also their `_persist`
/// twins, which write the same bytes, and the CTAs those launch: two per SM,
/// their `__launch_bounds__` residency. The grid kernels always load, since
/// chunks below `K128W_PERSIST_MIN_ROWS` rows keep them; the CTA count is 0
/// (grid only) unless every needed twin resolved.
fn k128w_kernels(
    gpu: &dyn GpuBackend,
    k128w: bool,
    silu: bool,
    persist: bool,
) -> (ops::K128wKernel, ops::K128wKernel, u32) {
    let pair = |name: &str, on: bool| ops::K128wKernel {
        grid: super::super::try_kernel_gated(k128w && on, gpu, "moe_w4a16", name),
        persist: super::super::try_kernel_gated(
            k128w && on && persist,
            gpu,
            "moe_w4a16",
            &format!("{name}_persist"),
        ),
    };
    let down = pair("moe_w4a4_grouped_gemm_prequant_t_k128w_compact", true);
    let gate_up = pair("moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w", silu);
    if !(k128w && persist) {
        return (down, gate_up, 0);
    }
    let ctas = 2 * gpu.sm_count().unwrap_or(0);
    let ok = down.persist.0 != 0 && (gate_up.persist.0 != 0 || !silu) && ctas > 0;
    if gpu.op_cache().once("moe:k128w_persist") {
        if ok {
            tracing::info!(
                "ATLAS_GLM_MOE_PREFILL_PERSIST: persistent K128W prefill, {ctas} CTAs from {} sorted rows",
                super::prequant_fp4::K128W_PERSIST_MIN_ROWS
            );
        } else {
            tracing::warn!(
                "ATLAS_GLM_MOE_PREFILL_PERSIST=1 ignored: persistent K128W kernels unavailable"
            );
        }
    }
    (down, gate_up, if ok { ctas } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::k128w_kernels;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn handles(k128w: bool, silu: bool, persist: bool) -> ([u64; 4], u32) {
        let (down, gate_up, ctas) = k128w_kernels(&MockGpuBackend::new(), k128w, silu, persist);
        let ids = [down.grid, down.persist, gate_up.grid, gate_up.persist].map(|h| h.0);
        (ids.map(|h| u64::from(h != 0)), ctas)
    }

    #[test]
    fn k128w_persist_adds_the_twins_and_keeps_the_grid_kernels() {
        // Flag off: the base grid kernels only.
        assert_eq!(handles(true, true, false), ([1, 0, 1, 0], 0));
        // Flag on: both sets, two CTAs per SM (the mock's 48 SMs).
        assert_eq!(handles(true, true, true), ([1, 1, 1, 1], 96));
        assert_eq!(handles(true, false, true), ([1, 1, 0, 0], 96));
        // K128W off: nothing, whatever the flag.
        assert_eq!(handles(false, true, true), ([0, 0, 0, 0], 0));
    }
}
