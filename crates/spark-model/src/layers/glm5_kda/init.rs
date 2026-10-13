// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5KdaLayer::new`: config checks and kernel resolution.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::{Glm5KdaLayer, Glm5KdaWeights, indexed_core, recurrent};
use crate::layers::FfnComponent;
use crate::layers::qwen3_attention::HcWeights;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::DenseWeight;

impl Glm5KdaLayer {
    pub fn new(
        input_norm: DenseWeight,
        post_attn_norm: DenseWeight,
        weights: Glm5KdaWeights,
        ffn: FfnComponent,
        hc: HcWeights,
        layer_idx: usize,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            config.linear_key_head_dim == 128,
            "GLM-5 KDA requires head_dim=128"
        );
        ensure!(
            config.linear_num_key_heads == config.linear_num_value_heads,
            "GLM-5 KDA requires equal Q/K/V head counts"
        );
        ensure!(config.hc_mult == 4, "GLM-5 KDA requires hc_mult=4");
        let heads = config.linear_num_key_heads;
        let dim = config.linear_key_head_dim;
        let register_resident_prefill = recurrent::parse_register_resident_prefill(
            std::env::var("ATLAS_KDA_REGRESIDENT_PREFILL")
                .ok()
                .as_deref(),
        )?;
        let recurrent_regresident_k = if register_resident_prefill {
            gpu.kernel("kda", "kda_recurrent_bf16_regresident")?
        } else {
            KernelHandle(0)
        };
        let preprocess_regresident_k = if register_resident_prefill {
            ensure!(
                recurrent::scratch_is_sufficient(
                    heads,
                    dim,
                    config.num_experts_per_tok,
                    config.moe_intermediate_size,
                    config.hidden_size,
                ),
                "GLM-5 KDA register-resident prefill scratch does not fit expert buffers"
            );
            gpu.kernel("kda", "kda_preprocess_regresident")?
        } else {
            KernelHandle(0)
        };
        let hc_name = |base: &str| super::super::ops::hc_kernel_name(&config.model_type, base);
        // The fused verify triples are load-ahead kernels; a b / f_a / g_a
        // grid too wide for them keeps the bit-identical separate launches.
        let triple_fits = super::super::ops::dense_gemv_triple_fits(heads as u32, dim as u32);
        let triple =
            |func| super::super::try_kernel_gated(triple_fits, gpu, "dense_gemv_bf16_batchm", func);
        // ATLAS_GLM_DECODE_GEMV_BATCH: the GEMV touch twins live in this target only.
        super::super::ops::gemv_touch_resolve(gpu);
        let layer = Self {
            input_norm,
            post_attn_norm,
            weights,
            ffn,
            hc,
            layer_idx,
            ssm_ordinal: indexed_core::ordinal(config, layer_idx)?,
            indexed_trace_logged: std::sync::atomic::AtomicBool::new(false),
            hidden_size: config.hidden_size,
            heads,
            dim,
            conv_width: config.linear_conv_kernel_dim,
            lower_bound: config.kda_gate_lower_bound,
            h_state_bytes: heads * dim * dim * 4,
            conv_state_bytes: 3 * heads * dim * config.linear_conv_kernel_dim * 4,
            rms_norm_k: crate::layers::ops::glm_decode_fuse::twin(
                gpu,
                crate::layers::ops::glm_decode_fuse::RMS_NORM,
                gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
                ("glm_rms_norm_regs", "rms_norm_vanilla_regs"),
            )?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_batchm_k: gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
            dense_gemv_batch5_k: super::super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batch5",
            ),
            dense_gemv_batch5_dual_k: super::super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batch5_dual",
            ),
            dense_gemv_batch5_triple_n_k: triple("dense_gemv_bf16_batch5_triple_n"),
            dense_gemv_batchm_dual_k: super::super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm_dual",
            ),
            dense_gemv_batchm_triple_n_k: triple("dense_gemv_bf16_batchm_triple_n"),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemv_batch2_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemv_batch5_qkv_k: super::super::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch5_qkv",
            ),
            w4a16_gemv_batchm: W4a16BatchmTiers::resolve(gpu),
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_t_m128_k: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_pipelined_k: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            // Bit-identical to three pipelined launches, so on unless the
            // kill switch is exactly `0` (any other value leaves it on).
            dense_gemm_pipelined_triple_n_k: if std::env::var("ATLAS_GLM_KDA_FUSED_SMALL_PREFILL")
                .as_deref()
                == Ok("0")
            {
                KernelHandle(0)
            } else {
                super::super::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined_triple_n")
            },
            conv_prefill_k: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            conv_prefill_tp_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_tp",
            ),
            conv_prefill_tp_snap_k: super::super::try_kernel(
                gpu,
                "causal_conv1d",
                "causal_conv1d_update_prefill_tp_snap",
            ),
            pack_k: gpu.kernel("kda", "kda_pack_qkv")?,
            conv_indexed_k: super::super::try_kernel(gpu, "causal_conv1d", "glm_kda_conv_indexed"),
            recurrent_k: gpu.kernel("kda", "kda_recurrent_bf16")?,
            recurrent_indexed_k: super::super::try_kernel(gpu, "kda", "glm_kda_recurrent_indexed"),
            recurrent_verify_snap_k: super::super::try_kernel(
                gpu,
                "kda",
                "kda_recurrent_bf16_verify_snap",
            ),
            recurrent_verify_owners_k: if std::env::var("ATLAS_KDA_VERIFY_OWNERS").as_deref()
                == Ok("0")
            {
                KernelHandle(0)
            } else {
                super::super::try_kernel(gpu, "kda", "kda_recurrent_bf16_verify_snap_owners")
            },
            recurrent_verify_rec_k: super::super::try_kernel(
                gpu,
                "kda",
                "kda_recurrent_bf16_verify_rec_owners",
            ),
            preprocess_regresident_k,
            recurrent_regresident_k,
            register_resident_prefill,
            flash_prefill: super::flash_prefill::FlashPrefill::load(
                heads,
                dim,
                config.kda_gate_lower_bound,
            )?,
            gated_norm_k: gpu.kernel("kda", "kda_sigmoid_gated_rms_norm")?,
            // Highway-storage-specific mHC kernels (FP32, or BF16 twins).
            hc_expand_k: gpu.kernel("hyper_connection", &hc_name("hc_expand"))?,
            hc_pre_k: gpu.kernel("hyper_connection", &hc_name("hc_pre"))?,
            hc_pre_from_raw_mix_k: gpu
                .kernel("hyper_connection", &hc_name("hc_pre_from_raw_mix"))?,
            hc_pre_mix_k: super::super::try_kernel(gpu, "hyper_connection", &hc_name("hc_pre_mix")),
            hc_post_k: gpu.kernel("hyper_connection", &hc_name("hc_post"))?,
            hc_post_bf16_add_k: super::super::try_kernel(
                gpu,
                "hyper_connection",
                &hc_name("hc_post_bf16_add"),
            ),
            hc_post_moe_blend_k: super::super::try_kernel(
                gpu,
                "hyper_connection",
                &hc_name("hc_post_moe_blend"),
            ),
            hc_contract_k: gpu.kernel("hyper_connection", &hc_name("hc_contract"))?,
        };
        // Eager (boot-audited) lookup of the K = 128 dual tier, which
        // `ops::dense_gemv_batchm_dual` then reads from the backend's op cache.
        super::super::ops::dense_gemv_dual_k128_kernel(gpu);
        if crate::model::glm_independent::enabled(&config.model_type)? {
            let handles = std::array::from_fn(|i| match i + 2 {
                2 => layer.w4a16_gemv_batch2_k.0,
                3 => layer.w4a16_gemv_batch3_k.0,
                n => layer.w4a16_gemv_batchm.kernel(n as u32).0,
            });
            crate::model::glm_independent::validate_projection_handles(
                handles,
                layer.dense_gemv_batchm_k.0,
            )?;
            ensure!(
                layer.conv_indexed_k.0 != 0 && layer.recurrent_indexed_k.0 != 0,
                "independent KDA requires both indexed kernels"
            );
        }
        Ok(layer)
    }
}
