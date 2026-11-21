// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2's DSpark drafter (`mtp.safetensors` of
//! `thinkingmachines/Inkling-2-Mini`): the tensors the DFlash head's Inkling
//! variant adds (`layers/dflash_head/inkling.rs`) and the `mtp_config` that
//! describes them. The drafter dir holds the checkpoint's `mtp.safetensors`
//! as `model.safetensors` beside a `config.json` that is the checkpoint's
//! `mtp_config` plus `vocab_size` (the target's tokenizer vocab). Reference:
//! vLLM PR #69028 `models/inkling2/nvidia/dspark.py` (Apache-2.0).

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::DflashConfig;
use crate::weight_map::{DenseWeight, dense_auto};

pub const INKLING2_DSPARK_ARCH: &str = "Inkling2DSparkModel";
/// Taps the fuse mixes: three target visits and the final-norm output.
pub const INKLING2_DSPARK_TAPS: usize = 4;

/// The drafter's own tensors beyond a DFlash layer stack.
pub struct Inkling2DsparkWeights {
    /// `softmax(fuse_weight)` rounded to BF16, as vLLM's `fuse_alpha` buffer:
    /// rows = draft layers then the Markov head, columns = taps.
    pub alpha: Vec<[f32; INKLING2_DSPARK_TAPS]>,
    pub mask_embed: DevicePtr,
    /// Per layer: the context mix's norm, `[k_proj_ctx; v_proj_ctx]` as one
    /// `[2 * kv_dim, H]` weight, and the context key norm.
    pub dflare_norms: Vec<DenseWeight>,
    pub kv_proj_ctx: Vec<DenseWeight>,
    pub k_norm_ctx: Vec<DenseWeight>,
    pub markov_w1: DenseWeight,
    pub rnn_proj: DenseWeight,
    pub rnn_bias: DevicePtr,
    pub hidden_out: DevicePtr,
    pub hidden_out_bias: DevicePtr,
    pub markov_norm: DenseWeight,
    pub markov_rank: usize,
    /// Target pieces the variant reads, filled by [`Self::attach_target`].
    pub target_embed_norm: DevicePtr,
    pub target_final_norm: DevicePtr,
    pub target_eps: f32,
}

impl DflashConfig {
    pub fn is_inkling2_dspark(&self) -> bool {
        self.architectures
            .as_ref()
            .is_some_and(|a| a.iter().any(|s| s == INKLING2_DSPARK_ARCH))
    }
}

/// An Inkling 2 `mtp_config` (keys `markov_head_type: "rnn_hidden"` and
/// `local_layer_ids`) in the DFlash config shape: bidirectional blocks, the
/// local layers as sliding layers of window `sliding_window_size + block`
/// (vLLM anchors that window at the block start; HF semantics measure it per
/// query, so draft rows see up to `block - 1` keys more at its far edge),
/// and the taps as `dflash_config.target_layer_ids`. Other configs pass
/// through unchanged.
pub(super) fn reshape_inkling2_mtp_config(raw: &mut Value) -> Result<()> {
    let is_inkling = raw.get("markov_head_type").and_then(Value::as_str) == Some("rnn_hidden")
        && raw.get("local_layer_ids").is_some();
    if !is_inkling {
        return Ok(());
    }
    let field = |k: &str| {
        raw.get(k)
            .with_context(|| format!("Inkling 2 mtp_config has no `{k}`"))
    };
    let layers = field("num_hidden_layers")?
        .as_u64()
        .context("num_hidden_layers")? as usize;
    let block = field("block_size")?.as_u64().context("block_size")? as usize;
    let window = field("sliding_window_size")?
        .as_u64()
        .context("sliding_window_size")? as usize;
    let local: Vec<u64> = serde_json::from_value(field("local_layer_ids")?.clone())
        .context("Inkling 2 mtp_config local_layer_ids")?;
    let taps = field("target_layer_ids")?.clone();
    ensure!(
        raw.get("vocab_size").is_some(),
        "an Inkling 2 DSpark drafter config.json needs `vocab_size` (the target's tokenizer vocab)"
    );
    ensure!(
        local.iter().all(|&l| (l as usize) < layers),
        "Inkling 2 mtp_config local_layer_ids {local:?} past {layers} layers"
    );
    let types: Vec<&str> = (0..layers as u64)
        .map(|i| {
            if local.contains(&i) {
                "sliding_attention"
            } else {
                "full_attention"
            }
        })
        .collect();
    let obj = raw
        .as_object_mut()
        .context("Inkling 2 mtp_config is not an object")?;
    obj.insert("architectures".into(), json!([INKLING2_DSPARK_ARCH]));
    obj.insert("layer_types".into(), json!(types));
    obj.insert("sliding_window".into(), json!(window + block));
    obj.insert("is_causal".into(), json!(false));
    obj.insert(
        "dflash_config".into(),
        json!({"mask_token_id": 0, "target_layer_ids": taps, "causal": false}),
    );
    Ok(())
}

fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// `softmax(fuse_weight, -1)` per row, each value rounded to BF16.
fn fuse_alpha(store: &WeightStore, gpu: &dyn GpuBackend, rows: usize) -> Result<Vec<[f32; 4]>> {
    let t = store.get("fuse_weight")?;
    ensure!(
        t.dtype == WeightDtype::BF16 && t.shape == [rows, INKLING2_DSPARK_TAPS],
        "Inkling 2 fuse_weight is {:?} {:?}, expected BF16 [{rows}, {INKLING2_DSPARK_TAPS}]",
        t.dtype,
        t.shape
    );
    let mut raw = vec![0u8; rows * INKLING2_DSPARK_TAPS * 2];
    gpu.copy_d2h(t.ptr, &mut raw)?;
    let vals: Vec<f32> = raw
        .chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect();
    Ok(vals
        .chunks_exact(INKLING2_DSPARK_TAPS)
        .map(|row| {
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = e.iter().sum();
            std::array::from_fn(|i| bf16_round(e[i] / s))
        })
        .collect())
}

fn ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    Ok(store.get(name)?.ptr)
}

/// The variant's tensors, when the store is an Inkling 2 drafter's.
pub fn load_inkling2_dspark(
    store: &WeightStore,
    config: &DflashConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<Inkling2DsparkWeights>> {
    if !config.is_inkling2_dspark() {
        return Ok(None);
    }
    let layers = config.num_hidden_layers;
    let kv = config.num_key_value_heads * config.head_dim;
    let h = config.hidden_size;
    let mut dflare_norms = Vec::with_capacity(layers);
    let mut kv_proj_ctx = Vec::with_capacity(layers);
    let mut k_norm_ctx = Vec::with_capacity(layers);
    for i in 0..layers {
        let lp = format!("layers.{i}");
        dflare_norms.push(dense_auto(store, &format!("{lp}.dflare_norm.weight"), gpu)?);
        k_norm_ctx.push(dense_auto(
            store,
            &format!("{lp}.self_attn.k_norm_ctx.weight"),
            gpu,
        )?);
        // [k_proj_ctx; v_proj_ctx]: one GEMM per layer writes [K | V] rows
        let fused = gpu
            .alloc(2 * kv * h * 2)
            .context("Inkling 2 DSpark ctx K/V weight")?;
        for (j, part) in ["k_proj_ctx", "v_proj_ctx"].iter().enumerate() {
            let src = ptr(store, &format!("{lp}.self_attn.{part}.weight"))?;
            gpu.copy_d2d(src, fused.offset(j * kv * h * 2), kv * h * 2)?;
        }
        kv_proj_ctx.push(DenseWeight { weight: fused });
    }
    let rank = config
        .markov_rank
        .context("Inkling 2 DSpark needs markov_rank")?;
    ensure!(
        rank == 256,
        "Inkling 2 DSpark kernels are built for markov_rank 256, got {rank}"
    );
    let rnn = store.get("markov_head.rnn_proj.weight")?;
    ensure!(
        rnn.shape == [3 * rank, 2 * rank + 2 * h],
        "Inkling 2 rnn_proj is {:?}, expected [{}, {}]",
        rnn.shape,
        3 * rank,
        2 * rank + 2 * h
    );
    tracing::info!(
        "Inkling 2 DSpark: {layers} layers, Markov rnn_hidden rank {rank}, taps {:?}",
        config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.as_slice())
    );
    Ok(Some(Inkling2DsparkWeights {
        alpha: fuse_alpha(store, gpu, layers + 1)?,
        mask_embed: ptr(store, "mask_embed")?,
        dflare_norms,
        kv_proj_ctx,
        k_norm_ctx,
        markov_w1: dense_auto(store, "markov_head.markov_w1.weight", gpu)?,
        rnn_proj: DenseWeight { weight: rnn.ptr },
        rnn_bias: ptr(store, "markov_head.rnn_proj.bias")?,
        hidden_out: ptr(store, "markov_head.hidden_out.weight")?,
        hidden_out_bias: ptr(store, "markov_head.hidden_out.bias")?,
        markov_norm: dense_auto(store, "markov_head.dflare_norm.weight", gpu)?,
        markov_rank: rank,
        target_embed_norm: DevicePtr::NULL,
        target_final_norm: DevicePtr::NULL,
        target_eps: 0.0,
    }))
}

impl Inkling2DsparkWeights {
    /// The target's embedding norm (the anchor row's input) and final norm
    /// (applied to the last tap, captured before it), from the target store.
    pub fn attach_target(&mut self, target: &WeightStore, eps: f32) -> Result<()> {
        self.target_embed_norm = ptr(target, "model.llm.embed_norm.weight")?;
        self.target_final_norm = ptr(target, "model.llm.norm.weight")?;
        self.target_eps = eps;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mtp() -> Value {
        json!({"hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24,
               "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9216,
               "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "local_layer_ids": [0, 2],
               "sliding_window_size": 1024, "block_size": 8, "markov_rank": 256,
               "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48],
               "vocab_size": 200058})
    }

    #[test]
    fn mtp_config_reads_as_a_bidirectional_dflash_drafter() {
        let c = super::super::parse_dflash_config(&mtp().to_string()).unwrap();
        assert!(c.is_inkling2_dspark());
        assert!(!c.query_causal());
        assert_eq!(c.block_size(), 8);
        assert_eq!(c.rope_theta, 1e6);
        assert_eq!(
            c.dflash_config.as_ref().unwrap().target_layer_ids,
            vec![5, 23, 41, 48]
        );
        assert_eq!(
            c.mixed_layer_windows().unwrap(),
            Some(vec![Some(1032), None, Some(1032)])
        );
    }

    #[test]
    fn mtp_config_without_vocab_fails() {
        let mut v = mtp();
        v.as_object_mut().unwrap().remove("vocab_size");
        assert!(super::super::parse_dflash_config(&v.to_string()).is_err());
    }

    #[test]
    fn other_configs_pass_through() {
        let mut v = json!({"hidden_size": 1, "markov_rank": 2});
        reshape_inkling2_mtp_config(&mut v).unwrap();
        assert_eq!(v, json!({"hidden_size": 1, "markov_rank": 2}));
    }

    #[test]
    fn bf16_rounding_is_nearest_even() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(f32::from_bits(0x3F80_8000)), 1.0); // tie -> even
        assert_eq!(
            bf16_round(f32::from_bits(0x3F81_8000)),
            f32::from_bits(0x3F82_0000)
        );
    }
}
