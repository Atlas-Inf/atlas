// SPDX-License-Identifier: AGPL-3.0-only

use super::super::weights::tensor;
use super::{kernels::Kernels, memory::Owned, projections};
use anyhow::{Context, Result, ensure};
use atlas_core::{
    config::{Hy4Attention as Kind, Hy4Rope, ModelConfig},
    hy4_reference::attention::HeadPlan,
};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    weights::{WeightDtype, WeightStore},
};

pub struct Hy4Attention<'a> {
    _store: &'a WeightStore,
    pub(super) memory: Owned<'a>,
    pub(super) kernels: Kernels,
    pub(super) scales: projections::Scales,
    pub(super) hidden: usize,
    pub(super) heads: usize,
    pub(super) qrank: usize,
    pub(super) rank: usize,
    pub(super) world: usize,
    pub(super) max_context: usize,
    pub(super) rope: Hy4Rope,
    pub(super) window: Option<usize>,
    pub(super) eps: f32,
    pub(super) qa: DevicePtr,
    pub(super) qan: DevicePtr,
    pub(super) qb: DevicePtr,
    pub(super) kva: DevicePtr,
    pub(super) kvn: DevicePtr,
    pub(super) wk: DevicePtr,
    pub(super) wv: DevicePtr,
    pub(super) output: DevicePtr,
    pub(super) gate: DevicePtr,
    pub(super) sink: DevicePtr,
    pub(super) norm: DevicePtr,
    pub(super) index: Option<IndexWeights>,
}
pub(super) struct IndexWeights {
    pub query: DevicePtr,
    pub key: DevicePtr,
    pub key_norm: DevicePtr,
    pub head: DevicePtr,
    pub value: DevicePtr,
    pub gates: DevicePtr,
    pub norm: DevicePtr,
    pub ape: DevicePtr,
    pub heads: usize,
}
impl<'a> Hy4Attention<'a> {
    /// Drain the bound compute stream before releasing derived matrices.
    pub fn release(&mut self) -> Result<()> {
        self.memory.release()
    }
    /// As in GLM-5, `config.num_attention_heads` is already TP-local.
    /// SWA's explicit Hy4 metadata remains global.
    pub fn load(
        store: &'a WeightStore,
        layer: usize,
        c: &ModelConfig,
        gpu: &'a dyn GpuBackend,
    ) -> Result<Self> {
        let hy = c.hy4.as_ref().context("Hy4 attention metadata required")?;
        let kind = hy
            .layer_types
            .get(layer)
            .context("Hy4 attention layer out of range")?;
        let swa = *kind == Kind::SlidingAttention;
        let global = if swa {
            hy.swa_num_attention_heads
        } else {
            c.num_attention_heads
                .checked_mul(c.tp_world_size)
                .context("Hy4 head overflow")?
        };
        let plan = HeadPlan::new(global, c.tp_rank, c.tp_world_size)?;
        ensure!(
            c.kv_lora_rank == 512
                && c.qk_nope_head_dim == 192
                && c.qk_rope_head_dim == 64
                && c.v_head_dim == 256
                && c.q_lora_rank > 0
                && c.q_lora_rank.is_multiple_of(c.tp_world_size)
                && c.hidden_size > 0
                && global <= 64
                && c.q_lora_rank <= u32::MAX as usize
                && c.hidden_size <= u32::MAX as usize
                && hy.swa_qk_nope_head_dim == 192
                && hy.swa_qk_rope_head_dim == 64
                && hy.swa_v_head_dim == 256
                && hy.gated_mla
                && hy.learnable_sink,
            "Hy4 requires gated latent512/rotary64 MLA with 192 NoPE and 256 values"
        );
        ensure!(
            c.max_position_embeddings > 0
                && c.max_position_embeddings < i32::MAX as usize
                && (c.rms_norm_eps as f32).is_finite()
                && c.rms_norm_eps as f32 > 0.,
            "Hy4 attention context/epsilon"
        );
        let rope = if swa {
            hy.rope_parameters.sliding_attention.clone()
        } else {
            hy.rope_parameters.deepseek_sparse_attention.clone()
        };
        ensure!(
            rope.rope_type == "default"
                && rope.mrope_section.iter().sum::<usize>() == 32
                && rope.rope_theta.is_finite()
                && rope.rope_theta > 0.
                && rope.rope_theta as f32 > 0.
                && (rope.rope_theta as f32).is_finite(),
            "Hy4 requires default 64-wide rotary"
        );
        let window = if swa {
            ensure!(c.sliding_window == 512, "Hy4 SWA requires window 512");
            Some(512)
        } else {
            None
        };
        let p = format!("{}.self_attn", c.layer_prefix(layer));
        let bf = |suffix: &str, shape: &[usize]| {
            tensor(store, &format!("{p}.{suffix}"), shape, WeightDtype::BF16)
        };
        let qa = bf("q_a_proj.weight", &[c.q_lora_rank, c.hidden_size])?;
        let qan = bf("q_a_layernorm.weight", &[c.q_lora_rank])?;
        let qb = bf("q_b_proj.weight", &[global * 256, c.q_lora_rank])?
            .offset(plan.start * 256 * c.q_lora_rank * 2);
        let kva = bf("kv_a_proj_with_mqa.weight", &[576, c.hidden_size])?;
        let kvn = bf("kv_a_layernorm.weight", &[512])?;
        let kvb = bf("kv_b_proj.weight", &[global * 448, 512])?;
        let gate = bf("linear_gate.weight", &[global * 256, c.hidden_size])?
            .offset(plan.start * 256 * c.hidden_size * 2);
        let sink = tensor(
            store,
            &format!("{p}.learnable_sink_param"),
            &[global],
            WeightDtype::FP32,
        )?
        .offset(plan.start * 4);
        let norm = if hy.sandwich_norm && (swa || !hy.dsa_skip_attn_post_norm) {
            bf("post_proj_norm.weight", &[c.hidden_size])?
        } else {
            DevicePtr::NULL
        };
        let o = bf("o_proj.weight", &[c.hidden_size, global * 256])?;
        let index = if swa {
            None
        } else {
            ensure!(
                hy.compressed_indexer_compress_ratio == 4
                    && hy.compressed_indexer_group_topk == 512
                    && hy.compressed_indexer_candidate_topk == 2048
                    && hy.compressed_indexer_key_pooling == "gated"
                    && hy.compressed_indexer_skip_fine_select
                    && hy.compressed_indexer_head_dim == 128
                    && (1..=64).contains(&hy.compressed_indexer_num_attention_heads),
                "Hy4 requires coarse-only four-token IndexPool, 511 historical groups"
            );
            let n = hy.compressed_indexer_num_attention_heads;
            let ix = |suffix: &str, shape: &[usize], dtype| {
                tensor(
                    store,
                    &format!("{p}.compressed_indexer.{suffix}"),
                    shape,
                    dtype,
                )
            };
            Some(IndexWeights {
                query: ix(
                    "query_projection.weight",
                    &[n * 128, c.q_lora_rank],
                    WeightDtype::BF16,
                )?,
                key: ix(
                    "token_key_projection.weight",
                    &[128, c.hidden_size],
                    WeightDtype::BF16,
                )?,
                key_norm: ix("token_key_norm.weight", &[128], WeightDtype::BF16)?,
                head: ix(
                    "head_weight_projection.weight",
                    &[n, c.hidden_size],
                    WeightDtype::BF16,
                )?,
                value: ix(
                    "key_value_projection.weight",
                    &[128, 128],
                    WeightDtype::FP32,
                )?,
                gates: ix("key_gate_projection.weight", &[128, 128], WeightDtype::FP32)?,
                norm: ix("key_norm.weight", &[128], WeightDtype::BF16)?,
                ape: ix("intra_group_ape", &[4, 128], WeightDtype::FP32)?,
                heads: n,
            })
        };
        let kernels = Kernels::load(gpu)?;
        let mut memory = Owned::new(gpu);
        let mut scales = projections::Scales::new();
        let mut kv = vec![0; plan.local * 448 * 512 * 2];
        gpu.copy_d2h(kvb.offset(plan.start * 448 * 512 * 2), &mut kv)?;
        let mut key = vec![0; plan.local * 512 * 192 * 2];
        let mut value = Vec::with_capacity(plan.local * 256 * 512 * 2);
        for h in 0..plan.local {
            for k in 0..192 {
                for l in 0..512 {
                    let src = (h * 448 * 512 + k * 512 + l) * 2;
                    let dst = (h * 512 * 192 + l * 192 + k) * 2;
                    key[dst..dst + 2].copy_from_slice(&kv[src..src + 2]);
                }
            }
            let start = (h * 448 + 192) * 512 * 2;
            value.extend_from_slice(&kv[start..start + 256 * 512 * 2]);
        }
        let mut full_o = vec![0; c.hidden_size * global * 256 * 2];
        gpu.copy_d2h(o, &mut full_o)?;
        let mut local_o = Vec::with_capacity(c.hidden_size * plan.local * 256 * 2);
        for row in full_o.chunks_exact(global * 256 * 2) {
            local_o
                .extend_from_slice(&row[plan.start * 256 * 2..(plan.start + plan.local) * 256 * 2]);
        }
        let fp8 = hy.attention_projection_fp8;
        let qa = projections::retain(
            &mut memory,
            qa.offset(c.tp_rank * c.q_lora_rank / c.tp_world_size * c.hidden_size * 2),
            c.q_lora_rank / c.tp_world_size,
            c.hidden_size,
            fp8,
            &mut scales,
        )?;
        let kva = projections::retain(
            &mut memory,
            kva.offset(c.tp_rank * 576 / c.tp_world_size * c.hidden_size * 2),
            576 / c.tp_world_size,
            c.hidden_size,
            fp8,
            &mut scales,
        )?;
        let qb = projections::retain(
            &mut memory,
            qb,
            plan.local * 256,
            c.q_lora_rank,
            fp8,
            &mut scales,
        )?;
        let gate = projections::retain(
            &mut memory,
            gate,
            plan.local * 256,
            c.hidden_size,
            fp8,
            &mut scales,
        )?;
        let wk = projections::upload(&mut memory, &key, plan.local * 512, 192, fp8, &mut scales)?;
        let wv = projections::upload(&mut memory, &value, plan.local * 256, 512, fp8, &mut scales)?;
        let output = projections::upload(
            &mut memory,
            &local_o,
            c.hidden_size,
            plan.local * 256,
            fp8,
            &mut scales,
        )?;
        for suffix in [
            "q_a_proj.weight",
            "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight",
            "kv_b_proj.weight",
            "linear_gate.weight",
            "o_proj.weight",
        ] {
            store.reclaim(gpu, &format!("{p}.{suffix}"))?;
        }
        tracing::info!(
            "HY4 attention layer={layer} rank={} projections={} TP-local; full sources reclaimed",
            c.tp_rank,
            if fp8 {
                "FP8 E4M3 per-row FP32 scales"
            } else {
                "BF16"
            }
        );
        Ok(Self {
            _store: store,
            memory,
            kernels,
            scales,
            hidden: c.hidden_size,
            heads: plan.local,
            qrank: c.q_lora_rank,
            rank: c.tp_rank,
            world: c.tp_world_size,
            max_context: c.max_position_embeddings,
            rope,
            window,
            eps: c.rms_norm_eps as f32,
            qa,
            qan,
            qb,
            kva,
            kvn,
            wk,
            wv,
            output,
            gate,
            sink,
            norm,
            index,
        })
    }
}
