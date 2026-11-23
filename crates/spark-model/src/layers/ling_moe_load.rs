// SPDX-License-Identifier: AGPL-3.0-only

//! Loader half of the native Ling FFN. Kept separate from the forward path so
//! both files stay below Atlas's source-file size limit.

use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::ling_paging::LingPagingReader;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;
use crate::weight_map::DenseWeight;

impl LingMoeFfn {
    /// Load a native Ling MoE layer. `reader` is shared by all layers and
    /// bounds the number of distinct original checkpoint shards mapped.
    pub fn load(
        store: &WeightStore,
        layer_idx: usize,
        gpu: &dyn GpuBackend,
        config: &ModelConfig,
        reader: Arc<Mutex<LingPagingReader>>,
    ) -> Result<Self> {
        ensure!(
            config.model_type == "bailing_moe_v3_vl",
            "Ling FFN received model type `{}`",
            config.model_type
        );
        ensure!(
            config.hidden_size == HIDDEN
                && config.moe_intermediate_size == INTERMEDIATE
                && config.num_experts == EXPERTS
                && config.num_experts_per_tok == TOP_K,
            "Ling 3.2 FFN expects H={HIDDEN}, I={INTERMEDIATE}, E={EXPERTS}, top-k={TOP_K}; got H={}, I={}, E={}, top-k={}",
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            config.num_experts_per_tok
        );
        ensure!(
            !config.mlp_only_layers.contains(&layer_idx),
            "Ling layer {layer_idx} is a dense MLP, not a routed FFN"
        );

        let prefix = format!("model.layers.{layer_idx}.mlp");
        let gate = store.get(&format!("{prefix}.gate.weight"))?;
        ensure!(
            gate.dtype == WeightDtype::BF16 && gate.shape == [EXPERTS, HIDDEN],
            "Ling router weight has dtype/shape {:?}/{:?}; expected BF16/[{EXPERTS}, {HIDDEN}]",
            gate.dtype,
            gate.shape
        );
        let bias = store.get(&format!("{prefix}.gate.expert_bias"))?;
        ensure!(
            bias.dtype == WeightDtype::FP32 && bias.shape == [EXPERTS],
            "Ling expert bias has dtype/shape {:?}/{:?}; expected FP32/[{EXPERTS}]",
            bias.dtype,
            bias.shape
        );

        let shared_prefix = format!("{prefix}.shared_experts");
        let shared_gate = load_projection(store, &format!("{shared_prefix}.gate_proj"), gpu)?;
        let shared_up = load_projection(store, &format!("{shared_prefix}.up_proj"), gpu)?;
        let shared_down = load_projection(store, &format!("{shared_prefix}.down_proj"), gpu)?;
        let a16 = shared_gate.dense().is_some();
        for (name, projection, expected) in [
            ("shared gate", &shared_gate, (INTERMEDIATE, HIDDEN)),
            ("shared up", &shared_up, (INTERMEDIATE, HIDDEN)),
            ("shared down", &shared_down, (HIDDEN, INTERMEDIATE)),
        ] {
            ensure!(
                projection.dims() == expected
                    && if a16 {
                        projection.dense().is_some()
                    } else {
                        projection.resident().is_some()
                    },
                "Ling {name} has invalid storage for {} expected {:?}, got {:?}",
                if a16 {
                    "A16 dense"
                } else {
                    "strict FP8 resident"
                },
                expected,
                projection.dims(),
            );
        }

        let mut routed = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let ep = format!("{prefix}.experts.{expert}");
            let gate = load_projection(store, &format!("{ep}.gate_proj"), gpu)?;
            let up = load_projection(store, &format!("{ep}.up_proj"), gpu)?;
            let down = load_projection(store, &format!("{ep}.down_proj"), gpu)?;
            ensure!(
                gate.dims() == (INTERMEDIATE, HIDDEN),
                "Ling expert {expert} gate shape mismatch"
            );
            ensure!(
                up.dims() == (INTERMEDIATE, HIDDEN),
                "Ling expert {expert} up shape mismatch"
            );
            ensure!(
                down.dims() == (HIDDEN, INTERMEDIATE),
                "Ling expert {expert} down shape mismatch"
            );
            if a16 {
                ensure!(
                    gate.nvfp4().is_some() && up.nvfp4().is_some() && down.nvfp4().is_some(),
                    "Ling A16 expert {expert} must keep all packed NVFP4 projections resident"
                );
            } else {
                ensure!(
                    gate.nvfp4().is_none()
                        && up.nvfp4().is_none()
                        && down.nvfp4().is_none()
                        && gate.dense().is_none()
                        && up.dense().is_none()
                        && down.dense().is_none(),
                    "Ling strict FP8 expert {expert} has non-FP8 projection storage"
                );
            }
            routed.push([gate, up, down]);
        }
        let (a16_tables, a16_gate_up) = if a16 {
            let mut gate = Vec::with_capacity(EXPERTS);
            let mut up = Vec::with_capacity(EXPERTS);
            let mut down = Vec::with_capacity(EXPERTS);
            for expert in &routed {
                gate.push(expert[0].nvfp4().expect("validated A16 gate").0);
                up.push(expert[1].nvfp4().expect("validated A16 up").0);
                down.push(expert[2].nvfp4().expect("validated A16 down").0);
            }
            let tables = A16Tables {
                gate: crate::layers::moe::build_ptr_table_from_qw(&gate, gpu)?,
                up: crate::layers::moe::build_ptr_table_from_qw(&up, gpu)?,
                down: crate::layers::moe::build_ptr_table_from_qw(&down, gpu)?,
            };
            (Some(tables), gpu.alloc(TOP_K * INTERMEDIATE * 2 * 2)?)
        } else {
            (None, DevicePtr::NULL)
        };

        // The fast original-FP8 arm is optional: it is valid only when the
        // device can index the retained mmap ranges through host page tables
        // and both pointer-table kernels are present.  A failed capability
        // probe deliberately leaves the existing ordered staging path intact.
        let fp8_gate_up = if !a16 {
            crate::layers::try_kernel(gpu, "ling_ffn", "ling_fp8_gate_up_shared")
        } else {
            spark_runtime::gpu::KernelHandle(0)
        };
        let ling_down_fp8 = if !a16 {
            crate::layers::try_kernel(gpu, "ling_ffn", "ling_silu_down_shared_fp8")
        } else {
            spark_runtime::gpu::KernelHandle(0)
        };
        let strict_fast = if !a16 && fp8_gate_up.0 != 0 && ling_down_fp8.0 != 0 {
            fast::StrictFastTables::try_build(gpu, &reader, &routed).ok()
        } else {
            None
        };

        let max_weight_bytes = HIDDEN * INTERMEDIATE;
        let scale_elements = (HIDDEN / GROUP) * (INTERMEDIATE / GROUP);
        let stage = Stage {
            weight: gpu.alloc(max_weight_bytes)?,
            raw_scale: gpu.alloc(scale_elements * 4)?,
            scale_f32: gpu.alloc(scale_elements * 4)?,
            weight_bytes: max_weight_bytes,
            raw_scale_bytes: scale_elements * 4,
            scale_elements,
        };
        let input_fp8 = gpu.alloc(HIDDEN)?;
        let input_scale = gpu.alloc((HIDDEN / GROUP) * 4)?;
        let activation_fp8 = gpu.alloc(INTERMEDIATE)?;
        let activation_scale = gpu.alloc((INTERMEDIATE / GROUP) * 4)?;
        let activation_bf16 = gpu.alloc(INTERMEDIATE * 2)?;
        let kernels = Kernels {
            router_gemv: gpu.kernel("gemv", "dense_gemv_bf16_fp32out")?,
            router_gemm: if a16 {
                Some(gpu.kernel("gemm", "dense_gemm_bf16_f32out")?)
            } else {
                None
            },
            router: gpu.kernel("ling_router", "ling_grouped_top8")?,
            quant: gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
            gemm: gpu.kernel("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled")?,
            widen: gpu.kernel("widen_block_scale_f32", "widen_block_scale_f32")?,
            silu_quant: gpu.kernel("ling_ffn", "ling_silu_mul_quant_fp8")?,
            fp8_gate_up: (fp8_gate_up.0 != 0).then_some(fp8_gate_up),
            ling_down_fp8: (ling_down_fp8.0 != 0).then_some(ling_down_fp8),
            dense_gemv: if a16 {
                Some(gpu.kernel("gemv", "dense_gemv_bf16")?)
            } else {
                None
            },
            dense_gemm: if a16 {
                Some(gpu.kernel("gemm_tc", "dense_gemm_tc")?)
            } else {
                None
            },
            silu_bf16: if a16 {
                Some(gpu.kernel("ling_ffn", "ling_silu_mul_bf16")?)
            } else {
                None
            },
            moe_gate_up: if a16 {
                Some(gpu.kernel("moe_expert_gemv_fused", "moe_expert_gemv_gate_up")?)
            } else {
                None
            },
            moe_down: if a16 {
                Some(gpu.kernel("moe_expert_gemv", "moe_expert_gemv")?)
            } else {
                None
            },
            moe_wide: if a16 {
                let kernel =
                    crate::layers::try_kernel(gpu, "moe_expert_gemv_wide", "moe_expert_gemv_wide");
                (kernel.0 != 0).then_some(kernel)
            } else {
                None
            },
            blend_rows: if a16 {
                let kernel = crate::layers::try_kernel(
                    gpu,
                    "moe_expert_gemv",
                    "moe_weighted_sum_blend_rows",
                );
                (kernel.0 != 0).then_some(kernel)
            } else {
                None
            },
            weighted_sum: gpu.kernel("ling_ffn", "ling_weighted_sum")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
        };
        Ok(Self {
            layer_idx,
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            router: DenseWeight { weight: gate.ptr },
            expert_bias: bias.ptr,
            shared_gate,
            shared_up,
            shared_down,
            routed,
            reader,
            stage,
            input_fp8,
            input_scale,
            activation_fp8,
            activation_scale,
            activation_bf16,
            a16_gate_up,
            a16_tables,
            a16,
            strict_fast,
            kernels,
        })
    }
}

fn load_projection(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<LingProjection> {
    let weight_name = format!("{prefix}.weight");
    let scale_name = format!("{prefix}.weight_scale_inv");
    if store.contains(&format!("{prefix}.weight_packed")) {
        let packed = store.get(&format!("{prefix}.weight_packed"))?;
        ensure!(
            packed.dtype == WeightDtype::UInt8 && packed.shape.len() == 2,
            "Ling A16 projection {prefix} packed weight has dtype/shape {:?}/{:?}",
            packed.dtype,
            packed.shape
        );
        let n = packed.shape[0];
        let k = packed.shape[1] * 2;
        ensure!(
            k % 16 == 0,
            "Ling A16 projection {prefix} has odd packed K={k}"
        );
        let scale = store.get(&format!("{prefix}.weight_scale"))?;
        ensure!(
            scale.dtype == WeightDtype::FP8E4M3 && scale.shape == [n, k / 16],
            "Ling A16 projection {prefix} scale has dtype/shape {:?}/{:?}; expected FP8/[{}, {}]",
            scale.dtype,
            scale.shape,
            n,
            k / 16
        );
        let global = store.get(&format!("{prefix}.weight_global_scale"))?;
        ensure!(
            global.dtype == WeightDtype::FP32 && global.num_elements() == 1,
            "Ling A16 projection {prefix} global scale has dtype/shape {:?}/{:?}; expected FP32/[1]",
            global.dtype,
            global.shape
        );
        return Ok(LingProjection::Nvfp4 {
            weight: crate::weight_map::quantized_v2(store, prefix, gpu)?,
            n,
            k,
        });
    }
    if store.contains(&weight_name) {
        let weight = store.get(&weight_name)?;
        if weight.dtype == WeightDtype::BF16 {
            ensure!(
                weight.shape.len() == 2,
                "Ling dense projection {prefix} has invalid shape {:?}",
                weight.shape
            );
            ensure!(
                !store.contains(&scale_name) && !store.contains(&format!("{prefix}.weight_scale")),
                "Ling dense projection {prefix} unexpectedly has quantization scales"
            );
            return Ok(LingProjection::Dense {
                weight: DenseWeight { weight: weight.ptr },
                n: weight.shape[0],
                k: weight.shape[1],
            });
        }
        ensure!(
            store.contains(&scale_name),
            "Ling projection {prefix} is missing weight_scale_inv"
        );
        ensure!(weight.dtype == WeightDtype::FP8E4M3 && weight.shape.len() == 2);
        let scale = store.get(&scale_name)?;
        validate_scale(
            weight.shape[0],
            weight.shape[1],
            scale.dtype,
            &scale.shape,
            prefix,
        )?;
        return Ok(LingProjection::Resident(
            crate::weight_map::load_fp8_block_scaled_as_fp8weight(store, prefix, gpu)?,
        ));
    }
    let weight = store.deferred(&weight_name).cloned().ok_or_else(|| {
        anyhow::anyhow!("Ling projection {prefix} is neither resident nor deferred")
    })?;
    let scale = store.deferred(&scale_name).cloned().ok_or_else(|| {
        anyhow::anyhow!("Ling projection {prefix} scale is neither resident nor deferred")
    })?;
    ensure!(weight.dtype == WeightDtype::FP8E4M3 && weight.shape.len() == 2);
    validate_scale(
        weight.shape[0],
        weight.shape[1],
        scale.dtype,
        &scale.shape,
        prefix,
    )?;
    Ok(LingProjection::Paged {
        n: weight.shape[0],
        k: weight.shape[1],
        weight,
        scale,
    })
}

fn validate_scale(
    n: usize,
    k: usize,
    dtype: WeightDtype,
    shape: &[usize],
    prefix: &str,
) -> Result<()> {
    ensure!(
        k % GROUP == 0 && shape == [n.div_ceil(GROUP), k / GROUP],
        "Ling projection {prefix} has invalid block scale shape {:?}; expected [{}, {}]",
        shape,
        n.div_ceil(GROUP),
        k / GROUP
    );
    ensure!(
        matches!(dtype, WeightDtype::BF16 | WeightDtype::FP32),
        "Ling projection {prefix} has unsupported scale dtype {dtype:?}"
    );
    Ok(())
}
