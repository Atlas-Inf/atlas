// SPDX-License-Identifier: AGPL-3.0-only
//! MiMo-V2 MTP module loader (`model.mtp.layers.{0,1,2}`).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::{MAX_DECODE_SEQS, QKV_TP_CHUNKS, load_dense_ffn, load_qkv_chunked};
use crate::layers::qwen3_attention::mimo::{
    MIMO_QK_DIM, MIMO_V_DIM, MimoAttn, MimoKernels, MimoQkv, mimo_workspace_bytes,
};
use crate::layers::{Qwen3AttentionLayer, mimo_mtp::MimoMtpModule};
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight, dense, dense_auto};

/// Load the checkpoint's independent MTP draft modules. Every module is an
/// MiMo SWA layer (8 KV heads, sink bias, dense FFN), not the Qwen-shaped
/// `MtpWeights` used by the generic proposer.
pub(crate) fn load_mimo_mtp_modules(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Vec<MimoMtpModule>> {
    if config.num_mtp_modules == 0 || !store.contains("model.mtp.layers.0.eh_proj.weight") {
        return Ok(Vec::new());
    }
    ensure!(
        config.head_dim == MIMO_QK_DIM as usize && config.v_head_dim == MIMO_V_DIM as usize,
        "mimo_v2 MTP: head dims {}/{} do not match the 192/128 kernels",
        config.head_dim,
        config.v_head_dim
    );
    let h = config.hidden_size;
    let nq = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let kernels = MimoKernels::resolve(gpu)?;
    let workspace_bytes = mimo_workspace_bytes(MAX_DECODE_SEQS, nq);
    let workspace = gpu.alloc(workspace_bytes)?;
    let mut modules = Vec::with_capacity(config.num_mtp_modules);

    for i in 0..config.num_mtp_modules {
        let lp = format!("model.mtp.layers.{i}");
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.pre_mlp_layernorm.weight"))?;
        let ffn = load_dense_ffn(store, config, gpu, &lp, absmax_k, quantize_k, stream)?;

        let p = format!("{lp}.self_attn");
        let (qkv, chunk_rows) = load_qkv_chunked(store, &format!("{p}.qkv_proj"), nq, nkv, h, gpu)?;
        let o_key = format!("{p}.o_proj.weight");
        {
            let o = store.get(&o_key)?;
            ensure!(
                o.shape == [h, nq * MIMO_V_DIM as usize],
                "{o_key}: shape {:?}, expected [{h}, {}]",
                o.shape,
                nq * MIMO_V_DIM as usize
            );
        }
        let o_proj = dense_auto(store, &o_key, gpu)?;
        let sinks = dense_auto(store, &format!("{p}.attention_sink_bias"), gpu)?.weight;
        ensure!(!sinks.is_null(), "{p}.attention_sink_bias is required");

        let attn = AttentionWeights {
            q_proj: null,
            k_proj: null,
            v_proj: null,
            o_proj: QuantizedWeight::null(),
            q_norm: null,
            k_norm: null,
            q_norm_full: None,
            k_norm_full: None,
            k_scale: 1.0,
            v_scale: 1.0,
        };
        let mut body = Qwen3AttentionLayer::new_ungated(
            input_norm,
            attn,
            post_attn_norm,
            ffn,
            0,
            None,
            None,
            None,
            gpu,
            spark_runtime::kv_cache::KvCacheDtype::Bf16,
            config.fp8_kv_calibration_tokens,
            config,
        )?;
        body.set_mimo(MimoAttn {
            qkv: MimoQkv::Bf16(qkv),
            o_proj,
            sinks,
            nq: nq as u32,
            nkv: nkv as u32,
            tp_chunks: QKV_TP_CHUNKS as u32,
            chunk_rows: chunk_rows as u32,
            window: config.sliding_window,
            rope_theta: config.swa_rope_theta as f32,
            rotary_dim: config.rotary_dim as u32,
            scale: 1.0 / (MIMO_QK_DIM as f32).sqrt(),
            out_scale: config.attention_value_scale as f32,
            workspace,
            workspace_bytes,
            kernels,
        });
        modules.push(MimoMtpModule {
            body,
            pre_fc_norm_embedding: dense(store, &format!("{lp}.enorm.weight"))?,
            pre_fc_norm_hidden: dense(store, &format!("{lp}.hnorm.weight"))?,
            fc: dense(store, &format!("{lp}.eh_proj.weight"))?,
            norm: dense(store, &format!("{lp}.final_layernorm.weight"))?,
        });
    }
    tracing::info!("mimo_v2: loaded {} MTP draft modules", modules.len());
    Ok(modules)
}
