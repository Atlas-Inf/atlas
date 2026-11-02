// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-zero GLM appended predictor using checkpoint weights without requantization.
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{DenseWeight, ExpertWeight, MoeWeights, dense_auto, quantized};
use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

pub struct Glm5MtpModule {
    pub body: Box<dyn TransformerLayer>,
    pub enorm: DenseWeight,
    pub hnorm: DenseWeight,
    pub eh_proj: DenseWeight,
    pub norm: DenseWeight,
}

/// Names that make the appended GLM predictor unambiguously present in a
/// checkpoint.  These are deliberately checked on every target rank before
/// the rank-0-only module is constructed: a missing marker on any worker must not
/// turn into a speculative path that only part of the target can execute.
pub(crate) fn glm5_mtp_marker_names(config: &ModelConfig) -> [String; 4] {
    let lp = config.layer_prefix(config.num_hidden_layers);
    [
        format!("{lp}.enorm.weight"),
        format!("{lp}.hnorm.weight"),
        format!("{lp}.eh_proj.weight"),
        format!("{lp}.shared_head.norm.weight"),
    ]
}

/// Derive the rank-0 MTP draft config from the target config.
///
/// The `WeightStore` advertises the full (unsharded) checkpoint tensors, so the
/// current loader reads rank-0-shaped weights directly. That is only correct
/// when the target's head counts have been un-split back to their
/// checkpoint-wide values and the draft runs as a single rank with all experts
/// replicated:
///
/// * `expert_tp = false` — the draft keeps every expert whole rather than a
///   Megatron slice.
/// * `tp_rank = 0`, `tp_world_size = 1` — the draft owns full projections and
///   issues no body collectives.
/// * `ep_rank = 0`, `ep_world_size = 1` — the draft owns every expert.
/// * `num_attention_heads` / `num_key_value_heads` are multiplied back by the
///   target's TP width, because the target config stores the *local* head
///   counts on each shard.
///
/// The target config is never mutated.
pub(crate) fn glm5_mtp_draft_config(target: &ModelConfig) -> Result<ModelConfig> {
    anyhow::ensure!(
        target.tp_world_size > 0 && target.ep_world_size > 0,
        "GLM-5 MTP: TP and EP world sizes must be nonzero"
    );
    anyhow::ensure!(
        target.tp_rank < target.tp_world_size && target.ep_rank < target.ep_world_size,
        "GLM-5 MTP: target rank is outside its parallel world"
    );
    anyhow::ensure!(
        target.num_attention_heads > 0 && target.num_key_value_heads > 0,
        "GLM-5 MTP: local attention head counts must be nonzero"
    );
    let mut draft = target.clone();
    draft.num_attention_heads = target
        .num_attention_heads
        .checked_mul(target.tp_world_size)
        .context("GLM-5 MTP: total query head count overflows")?;
    draft.num_key_value_heads = target
        .num_key_value_heads
        .checked_mul(target.tp_world_size)
        .context("GLM-5 MTP: total KV head count overflows")?;
    draft.expert_tp = false;
    draft.tp_rank = 0;
    draft.tp_world_size = 1;
    draft.ep_rank = 0;
    draft.ep_world_size = 1;
    Ok(draft)
}

/// Load the one full, unsharded predictor on rank zero before KV allocation.
/// The target retains its own TP/EP geometry and every acceptance decision.
pub fn load_glm5_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<Glm5MtpModule>> {
    if config.num_mtp_modules == 0 {
        return Ok(None);
    }
    ensure!(
        config.model_type == "glm5_next" && config.num_mtp_modules == 1,
        "native GLM MTP requires glm5_next with exactly one predictor"
    );
    ensure!(
        config.tp_rank == 0 && config.ep_rank == 0,
        "only rank zero may load native GLM MTP"
    );
    for name in glm5_mtp_marker_names(config) {
        ensure!(store.contains(&name), "native GLM MTP missing {name}");
    }
    // Neither private attention projections nor the shared head may acquire
    // optional lower-precision twins in strict mode.
    for name in ["ATLAS_GLM_MTP_NVFP4_WO", "ATLAS_GLM_MTP_NVFP4_EH"] {
        ensure!(
            std::env::var(name).as_deref() != Ok("1"),
            "native strict MTP refuses {name}=1"
        );
    }
    let draft = glm5_mtp_draft_config(config)?;
    let idx = config.num_hidden_layers;
    let lp = config.layer_prefix(idx);
    // FP4-Local's appended predictor is MXFP8 throughout its attention
    // projections (including kv_b/indexer), unlike the target's FP8-block MLA.
    // Exact expansion keeps all values and uses the established BF16 body.
    for name in [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "indexer.wq_b",
        "indexer.wk",
        "indexer.weights_proj",
    ] {
        validate_exact_dense(store, &format!("{lp}.self_attn.{name}"))?;
    }
    let ffn = load_native_moe(store, &lp, &draft, gpu)?;
    let body = super::layers::load_mla_layer(
        store,
        &lp,
        idx,
        0,
        super::fp4l::norm_bf16(store, &format!("{lp}.input_layernorm.weight"), gpu)?,
        super::fp4l::norm_bf16(store, &format!("{lp}.post_attention_layernorm.weight"), gpu)?,
        ffn,
        None,
        &draft,
        gpu,
        KvCacheDtype::Bf16,
        true,
        None,
    )?;
    let module = Glm5MtpModule {
        body,
        enorm: super::fp4l::norm_bf16(store, &format!("{lp}.enorm.weight"), gpu)?,
        hnorm: super::fp4l::norm_bf16(store, &format!("{lp}.hnorm.weight"), gpu)?,
        eh_proj: exact_dense(store, &format!("{lp}.eh_proj"), gpu)?,
        norm: super::fp4l::norm_bf16(store, &format!("{lp}.shared_head.norm.weight"), gpu)?,
    };
    let released = release_native_expanded(store, &lp, gpu)?;
    tracing::info!(
        layer = idx,
        released_source_bytes = released,
        "native GLM MTP loaded: checkpoint NVFP4 experts and exact BF16/MXFP8 shared weights"
    );
    Ok(Some(module))
}

/// Retire only source copies after the native predictor owns every BF16 expansion.
fn release_native_expanded(store: &WeightStore, lp: &str, gpu: &dyn GpuBackend) -> Result<usize> {
    // The replicated predictor forbids native MLA twins and loads its shared
    // experts through exact_dense, so none retain these source allocations.
    // Reuse the target loader's retirement list; routed experts are excluded.
    // BF16 sources (including the current router/EH tensors) remain resident.
    let stems = super::fp4l::decoder_expanded_stems(
        lp,
        atlas_core::config::LayerType::FullAttention,
        false,
    );
    stems
        .into_iter()
        .chain([format!("{lp}.eh_proj"), format!("{lp}.mlp.gate")])
        .try_fold(0, |released, stem| {
            Ok(released + super::fp4l::release_expanded(store, gpu, &stem, false)?)
        })
}

/// BF16 is resident as provided; MXFP8 expands exactly into BF16. Refuse
/// other source formats instead of silently quantizing the shared predictor.
fn exact_dense(store: &WeightStore, stem: &str, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    validate_exact_dense(store, stem)?;
    dense_auto(store, &format!("{stem}.weight"), gpu)
}

fn validate_exact_dense(store: &WeightStore, stem: &str) -> Result<()> {
    let w = store.get(&format!("{stem}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::BF16
            || (w.dtype == WeightDtype::FP8E4M3
                && crate::weight_map::mxfp8_geometry(store, stem)?.is_some()),
        "native MTP {stem} must be BF16 or MXFP8, got {:?}",
        w.dtype
    );
    Ok(())
}

fn load_native_moe(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FfnComponent> {
    let p = format!("{lp}.mlp");
    let mut experts = Vec::with_capacity(config.num_experts);
    for e in 0..config.num_experts {
        let stem = format!("{p}.experts.{e}");
        let load = |name: &str, n: usize, k: usize| -> Result<_> {
            let prefix = format!("{stem}.{name}");
            let w = store.get(&format!("{prefix}.weight"))?;
            ensure!(
                w.dtype == WeightDtype::UInt8 && w.shape == [n, k / 2],
                "native MTP expert {prefix} must retain checkpoint NVFP4 [{n}, {}]",
                k / 2
            );
            quantized(store, &prefix, gpu)
        };
        let (h, i) = (config.hidden_size, config.moe_intermediate_size);
        experts.push(ExpertWeight {
            gate_proj: load("gate_proj", i, h)?,
            up_proj: load("up_proj", i, h)?,
            down_proj: load("down_proj", h, i)?,
        });
    }
    let shared = format!("{p}.shared_experts");
    let gate = exact_dense(store, &format!("{shared}.gate_proj"), gpu)?;
    let up = exact_dense(store, &format!("{shared}.up_proj"), gpu)?;
    let down = exact_dense(store, &format!("{shared}.down_proj"), gpu)?;
    let bias = store.get(&format!("{p}.gate.e_score_correction_bias"))?;
    ensure!(
        bias.dtype == WeightDtype::FP32 && bias.num_elements() == config.num_experts,
        "native MTP routing bias must be FP32 with one element per expert"
    );
    let weights = MoeWeights {
        gate: exact_dense(store, &format!("{p}.gate"), gpu)?,
        // Fused routed kernels explicitly zero a null shared slot; the BF16
        // shared path overwrites it before blending. No temporary FP4 copy.
        shared_expert: ExpertWeight::null(),
        shared_expert_gate: DenseWeight {
            weight: DevicePtr::NULL,
        },
        experts,
        router_pre_norm: None,
        correction_bias: Some(DenseWeight { weight: bias.ptr }),
    };
    let mut layer = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    layer.set_bf16_shared_expert(gate, up, down)?;
    Ok(FfnComponent::Moe(layer))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-rank GLM-5 target config; tests tweak the parallel fields.
    fn glm5_base_config() -> ModelConfig {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.model_type = "glm5_next".into();
        c.num_hidden_layers = 45;
        c.num_mtp_modules = 1;
        c.tp_rank = 0;
        c.tp_world_size = 1;
        c.ep_rank = 0;
        c.ep_world_size = 1;
        c.expert_tp = false;
        c
    }

    #[test]
    fn draft_config_restores_target_heads_and_pins_parallel_geometry() {
        let mut target = glm5_base_config();
        target.tp_rank = 1;
        target.tp_world_size = 2;
        target.ep_rank = 1;
        target.ep_world_size = 2;
        target.expert_tp = true;
        // Head counts on a shard are local to the rank.
        target.num_attention_heads = 16;
        target.num_key_value_heads = 2;

        let draft = glm5_mtp_draft_config(&target).unwrap();

        // Restored to the checkpoint-wide values.
        assert_eq!(draft.num_attention_heads, 32);
        assert_eq!(draft.num_key_value_heads, 4);
        // Single rank, full projections, all experts.
        assert!(!draft.expert_tp);
        assert_eq!((draft.tp_rank, draft.tp_world_size), (0, 1));
        assert_eq!((draft.ep_rank, draft.ep_world_size), (0, 1));
    }

    #[test]
    fn tp4_predictor_retains_all_heads_experts_and_eh_workspace() {
        use spark_runtime::buffers::BufferSizes;
        for world in [2, 4] {
            for rank in 0..world {
                let mut target = glm5_base_config();
                target.hidden_size = 4096;
                target.head_dim = 256;
                target.kv_lora_rank = 512;
                target.qk_rope_head_dim = 0;
                target.tp_world_size = world;
                target.ep_world_size = world;
                target.tp_rank = rank;
                target.ep_rank = rank;
                target.num_attention_heads = 64 / world;
                target.num_key_value_heads = 64 / world;
                target.linear_num_key_heads = 64 / world;
                target.linear_num_value_heads = 64 / world;
                target.linear_key_head_dim = 128;
                target.linear_value_head_dim = 128;
                target.num_experts = 288;
                let draft = glm5_mtp_draft_config(&target).unwrap();
                assert_eq!(
                    (draft.num_attention_heads, draft.num_key_value_heads),
                    (64, 64)
                );
                assert_eq!((draft.tp_rank, draft.tp_world_size), (0, 1));
                assert_eq!((draft.ep_rank, draft.ep_world_size), (0, 1));
                assert_eq!(draft.local_expert_range(), (0, 288));
                let sizes = BufferSizes::from_config(&draft, 128, 1_048_576, 16, 1);
                assert!(sizes.ssm_qkvz >= 128 * 2 * 4096 * 2, "EH concatenation");
                assert!(sizes.ssm_deinterleaved >= 128 * 4096 * 2, "embedding rows");
                assert!(sizes.attn_output >= 128 * 64 * 512 * 2, "full-head MLA");
                assert!(
                    sizes.expert_down_out >= 1_048_576 / 4 * 4,
                    "full-context index"
                );
            }
        }
    }

    #[test]
    fn draft_config_never_mutates_the_target() {
        let mut target = glm5_base_config();
        target.tp_rank = 1;
        target.tp_world_size = 2;
        target.ep_rank = 1;
        target.ep_world_size = 2;
        target.expert_tp = true;
        target.num_attention_heads = 16;
        target.num_key_value_heads = 2;
        let before = target.clone();

        let _ = glm5_mtp_draft_config(&target).unwrap();

        assert_eq!(before.num_attention_heads, target.num_attention_heads);
        assert_eq!(before.num_key_value_heads, target.num_key_value_heads);
        assert_eq!(before.expert_tp, target.expert_tp);
        assert_eq!(before.tp_rank, target.tp_rank);
        assert_eq!(before.tp_world_size, target.tp_world_size);
        assert_eq!(before.ep_rank, target.ep_rank);
        assert_eq!(before.ep_world_size, target.ep_world_size);
    }

    #[test]
    fn draft_config_is_identity_on_a_single_rank_target() {
        let target = glm5_base_config();
        let draft = glm5_mtp_draft_config(&target).unwrap();
        assert_eq!(draft.num_attention_heads, target.num_attention_heads);
        assert_eq!(draft.num_key_value_heads, target.num_key_value_heads);
        assert_eq!((draft.tp_rank, draft.tp_world_size), (0, 1));
        assert_eq!((draft.ep_rank, draft.ep_world_size), (0, 1));
        assert!(!draft.expert_tp);
    }

    #[test]
    fn draft_config_replicates_experts_without_expert_tp() {
        // Full-copy expert plan: no expert is sliced across ranks, so both the
        // gate/up and the down projection must see whole experts.
        let mut target = glm5_base_config();
        target.tp_world_size = 2;
        target.ep_world_size = 2;
        target.expert_tp = true;
        let draft = glm5_mtp_draft_config(&target).unwrap();
        assert!(
            !draft.expert_tp,
            "full-copy expert plan requires expert_tp=false"
        );
        assert_eq!(draft.ep_world_size, 1);
    }

    #[test]
    fn draft_config_rejects_invalid_parallel_geometry() {
        for field in [
            "tp_world", "ep_world", "tp_rank", "ep_rank", "q_heads", "kv_heads",
        ] {
            let mut config = glm5_base_config();
            match field {
                "tp_world" => config.tp_world_size = 0,
                "ep_world" => config.ep_world_size = 0,
                "tp_rank" => config.tp_rank = 1,
                "ep_rank" => config.ep_rank = 1,
                "q_heads" => config.num_attention_heads = 0,
                "kv_heads" => config.num_key_value_heads = 0,
                _ => unreachable!(),
            }
            assert!(
                glm5_mtp_draft_config(&config).is_err(),
                "accepted invalid {field}"
            );
        }
    }

    #[test]
    fn draft_config_rejects_head_count_overflow() {
        for query_heads in [true, false] {
            let mut config = glm5_base_config();
            config.tp_world_size = 2;
            if query_heads {
                config.num_attention_heads = usize::MAX;
            } else {
                config.num_key_value_heads = usize::MAX;
            }
            assert!(glm5_mtp_draft_config(&config).is_err());
        }
    }
}

#[cfg(test)]
#[path = "mtp_native_tests.rs"]
mod native_tests;
