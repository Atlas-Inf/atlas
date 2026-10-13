// SPDX-License-Identifier: AGPL-3.0-only

use super::super::MoeLayer;
use crate::layers::moe::gate_up_repack::recording::{Event, Gpu};
use crate::weight_map::MoeWeights;
use atlas_core::config::ModelConfig;

// These symbols exist only in kernels/gb10/glm-5.3-flash/nvfp4. The
// production constructor must not ask other targets to resolve them.
const GLM_ONLY: &[&str] = &[
    "dense_gemm_bf16_router_m5",
    "dense_gemm_bf16_router_rows",
    "atlas_moe_nvfp4_mmq64_down",
    "atlas_moe_nvfp4_mmq64_gate_up",
    "atlas_moe_nvfp4_quantize_bf16",
    "atlas_moe_nvfp4_repack_batched",
    "atlas_moe_nvfp4_scale2_rows",
    "atlas_moe_nvfp4_silu_scale2",
    "moe_expert_gate_up_shared_mmq",
    "moe_expert_silu_down_shared_mmq",
    "silu_mul_quant_nvfp4",
    "moe_w4a16_grouped_gemm_ptrtable_t_k64_m32",
    "moe_w4a4_grouped_gemm_prequant_t_k64",
    "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale",
    "w4a16_gemv_batch5_dual",
    "moe_w4a4_grouped_gemm_prequant_t_k128",
];

fn lookups(gpu: &Gpu) -> Vec<String> {
    gpu.trace()
        .into_iter()
        .filter_map(|event| match event {
            Event::Lookup(_, name) => Some(name),
            _ => None,
        })
        .collect()
}

fn construct(gpu: &Gpu, model_type: &str, ep_world_size: usize) -> MoeLayer {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = model_type.into();
    config.ep_world_size = ep_world_size;
    MoeLayer::new(
        MoeWeights::empty(config.num_experts),
        config.num_experts,
        None,
        gpu,
        &config,
    )
    .expect("MoE construction with its target's kernels")
}

#[test]
fn constructor_scopes_glm_only_optional_lookups() {
    const CHILD: &str = "ATLAS_TEST_MOE_OPTIONAL_LOOKUPS";
    let Ok(mode) = std::env::var(CHILD) else {
        let name = concat!(
            module_path!(),
            "::constructor_scopes_glm_only_optional_lookups"
        );
        let name = name.split_once("::").unwrap().1;
        // Process-local flags: exercise the default and an inherited K128
        // setting without racing unrelated tests over the environment.
        for mode in ["0", "1"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(CHILD, mode)
                .env("ATLAS_MOE_PREQUANT_K128", mode)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "K128={mode}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    for ep_world_size in [1, 2] {
        let gpu = Gpu::new();
        *gpu.missing.lock().unwrap() = GLM_ONLY.iter().map(|s| (*s).into()).collect();
        let layer = construct(&gpu, "qwen3_5_moe", ep_world_size);
        let requests = lookups(&gpu);
        for symbol in GLM_ONLY {
            assert!(
                !requests.iter().any(|s| s == symbol),
                "Qwen EP{ep_world_size} queried GLM-only {symbol}"
            );
        }
        assert_eq!(layer.dense_gemm_router_m5.0, 0);
        assert_eq!(layer.dense_gemm_router_rows.0, 0);
        assert_eq!(layer.w4a16_batch5_dual_k.0, 0);
        // Keep optional kernels supported by other targets available.
        for symbol in [
            "moe_fp8_grouped_gemm",
            "moe_w8a8_grouped_gemm",
            "moe_w4a16_fused_gate_up_t_k64_fp4",
        ] {
            assert!(requests.iter().any(|s| s == symbol), "lost {symbol}");
        }
        let gpu = Gpu::new();
        let layer = construct(&gpu, "glm5_next", ep_world_size);
        let requests = lookups(&gpu);
        for symbol in GLM_ONLY {
            let expected = !symbol.ends_with("_k128") || mode == "1";
            assert_eq!(requests.iter().any(|s| s == symbol), expected, "{symbol}");
        }
        assert_ne!(layer.dense_gemm_router_m5.0, 0);
        assert_ne!(layer.dense_gemm_router_rows.0, 0);
        assert_ne!(layer.w4a16_batch5_dual_k.0, 0);
        // Missing GLM symbols still reach the backend audit; model scoping
        // must not turn a broken GLM build into an unobserved fallback.
        let gpu = Gpu::new();
        *gpu.missing.lock().unwrap() = GLM_ONLY.iter().map(|s| (*s).into()).collect();
        let _ = construct(&gpu, "glm5_next", ep_world_size);
        assert_eq!(lookups(&gpu), requests);
    }
}
