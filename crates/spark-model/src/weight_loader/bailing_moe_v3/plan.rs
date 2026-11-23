// SPDX-License-Identifier: AGPL-3.0-only

//! Strict tensor-name layout for Ling 3.2 air.

use std::collections::BTreeSet;

use anyhow::{Result, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};

pub(crate) type Header = BTreeSet<String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionLayout {
    Kda,
    Mla,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MlpLayout {
    Dense,
    Moe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerPlan {
    pub index: usize,
    pub attention: AttentionLayout,
    pub mlp: MlpLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Plan {
    pub layers: Vec<LayerPlan>,
    pub mtp: LayerPlan,
}

struct Expect<'a> {
    header: &'a Header,
    seen: BTreeSet<String>,
}

impl Expect<'_> {
    fn tensor(&mut self, name: &str) -> Result<()> {
        ensure!(
            self.header.contains(name),
            "bailing_moe_v3_vl checkpoint is missing {name}"
        );
        self.seen.insert(name.to_string());
        Ok(())
    }

    fn tensors(&mut self, prefix: &str, suffixes: &[&str]) -> Result<()> {
        for suffix in suffixes {
            self.tensor(&format!("{prefix}.{suffix}"))?;
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        let unknown: Vec<&String> = self
            .header
            .iter()
            .filter(|name| !self.seen.contains(*name) && !name.starts_with("model.visual."))
            .collect();
        if !unknown.is_empty() {
            bail!(
                "bailing_moe_v3_vl checkpoint has {} unread tensors, e.g. {:?}",
                unknown.len(),
                &unknown[..unknown.len().min(4)]
            );
        }
        Ok(())
    }
}

const ROOT: &[&str] = &[
    "model.word_embeddings.weight",
    "model.norm.weight",
    "lm_head.weight",
    "linear_proj.0.bias",
    "linear_proj.0.weight",
    "linear_proj.2.bias",
    "linear_proj.2.weight",
];

const COMMON: &[&str] = &["input_layernorm.weight", "post_attention_layernorm.weight"];

const KDA: &[&str] = &[
    "attention.A_log",
    "attention.b_proj.weight",
    "attention.dt_bias",
    "attention.f_proj.weight",
    "attention.f_proj.weight_scale_inv",
    "attention.g_proj.weight",
    "attention.g_proj.weight_scale_inv",
    "attention.k_conv1d.weight",
    "attention.k_proj.weight",
    "attention.k_proj.weight_scale_inv",
    "attention.o_norm.weight",
    "attention.o_proj.weight",
    "attention.o_proj.weight_scale_inv",
    "attention.q_conv1d.weight",
    "attention.q_proj.weight",
    "attention.q_proj.weight_scale_inv",
    "attention.v_conv1d.weight",
    "attention.v_proj.weight",
    "attention.v_proj.weight_scale_inv",
];

const MLA: &[&str] = &[
    "attention.dense.weight",
    "attention.dense.weight_scale_inv",
    "attention.g_proj.weight",
    "attention.kv_a_layernorm.weight",
    "attention.kv_a_proj_with_mqa.weight",
    "attention.kv_a_proj_with_mqa.weight_scale_inv",
    "attention.kv_b_proj.weight",
    "attention.kv_b_proj.weight_scale_inv",
    "attention.q_proj.weight",
    "attention.q_proj.weight_scale_inv",
];

const MLA_MTP_EXTRA: &[&str] = &[
    "enorm.weight",
    "eh_proj.weight",
    "hnorm.weight",
    "final_layernorm.weight",
];

const DENSE_MLP: &[&str] = &[
    "mlp.down_proj.weight",
    "mlp.down_proj.weight_scale_inv",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.weight_scale_inv",
    "mlp.up_proj.weight",
    "mlp.up_proj.weight_scale_inv",
];

fn expect_moe(e: &mut Expect<'_>, layer: usize, experts: usize, mtp: bool) -> Result<()> {
    let prefix = format!("model.layers.{layer}");
    let mut mla = MLA.to_vec();
    if mtp {
        mla.push("attention.g_proj.weight_scale_inv");
    }
    e.tensors(&prefix, &mla)?;
    e.tensors(&prefix, &["mlp.gate.expert_bias", "mlp.gate.weight"])?;
    for expert in 0..experts {
        let ep = format!("{prefix}.mlp.experts.{expert}");
        e.tensors(
            &ep,
            &[
                "down_proj.weight",
                "down_proj.weight_scale_inv",
                "gate_proj.weight",
                "gate_proj.weight_scale_inv",
                "up_proj.weight",
                "up_proj.weight_scale_inv",
            ],
        )?;
    }
    e.tensors(
        &format!("{prefix}.mlp.shared_experts"),
        &[
            "down_proj.weight",
            "down_proj.weight_scale_inv",
            "gate_proj.weight",
            "gate_proj.weight_scale_inv",
            "up_proj.weight",
            "up_proj.weight_scale_inv",
        ],
    )
}

pub(crate) fn plan(header: &Header, config: &ModelConfig) -> Result<Plan> {
    ensure!(
        config.model_type == "bailing_moe_v3_vl",
        "bailing_moe_v3_vl loader received model_type `{}`",
        config.model_type
    );
    ensure!(
        config.weight_prefix.is_empty(),
        "bailing_moe_v3_vl expects flat model.layers.* tensor names"
    );
    ensure!(
        config.num_hidden_layers > 0
            && config.mtp_num_hidden_layers == 1
            && config.num_experts > 0
            && config.kv_lora_rank > 0
            && config.linear_num_key_heads > 0,
        "bailing_moe_v3_vl config is missing the hybrid MLA/KDA dimensions"
    );
    ensure!(
        config.layer_types.len() == config.num_hidden_layers,
        "bailing_moe_v3_vl requires an explicit layer_types layout"
    );

    let mut e = Expect {
        header,
        seen: BTreeSet::new(),
    };
    for name in ROOT {
        e.tensor(name)?;
    }

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        e.tensors(&prefix, COMMON)?;
        let attention = match config.layer_type(layer) {
            LayerType::LinearAttention => {
                e.tensors(&prefix, KDA)?;
                AttentionLayout::Kda
            }
            LayerType::FullAttention => {
                // The reference excludes g_proj from FP8 conversion on every
                // main full-attention layer, hence no scale sibling here.
                e.tensors(&prefix, MLA)?;
                AttentionLayout::Mla
            }
            other => bail!("bailing_moe_v3_vl layer {layer} has unsupported type {other:?}"),
        };
        let mlp = if config.mlp_only_layers.contains(&layer) {
            e.tensors(&prefix, DENSE_MLP)?;
            MlpLayout::Dense
        } else {
            expect_moe(&mut e, layer, config.num_experts, false)?;
            MlpLayout::Moe
        };
        layers.push(LayerPlan {
            index: layer,
            attention,
            mlp,
        });
    }

    let mtp_index = config.num_hidden_layers;
    let mtp_prefix = format!("model.layers.{mtp_index}");
    e.tensors(&mtp_prefix, COMMON)?;
    e.tensors(&mtp_prefix, MLA_MTP_EXTRA)?;
    expect_moe(&mut e, mtp_index, config.num_experts, true)?;
    e.finish()?;
    Ok(Plan {
        layers,
        mtp: LayerPlan {
            index: mtp_index,
            attention: AttentionLayout::Mla,
            mlp: MlpLayout::Moe,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::{LayerType, ModelConfig};

    fn config() -> ModelConfig {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.model_type = "bailing_moe_v3_vl".into();
        c.num_hidden_layers = 2;
        c.mtp_num_hidden_layers = 1;
        c.num_experts = 2;
        c.kv_lora_rank = 1;
        c.mlp_only_layers = vec![0];
        c.layer_types = vec![LayerType::LinearAttention, LayerType::FullAttention];
        c
    }

    fn root(header: &mut Header) {
        for name in [
            "model.word_embeddings.weight",
            "model.norm.weight",
            "lm_head.weight",
            "linear_proj.0.bias",
            "linear_proj.0.weight",
            "linear_proj.2.bias",
            "linear_proj.2.weight",
        ] {
            header.insert(name.into());
        }
    }

    fn common(header: &mut Header, layer: usize) {
        let p = format!("model.layers.{layer}");
        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            header.insert(format!("{p}.{suffix}"));
        }
    }

    fn kda(header: &mut Header, layer: usize) {
        let p = format!("model.layers.{layer}.attention");
        for suffix in [
            "A_log",
            "b_proj.weight",
            "dt_bias",
            "f_proj.weight",
            "f_proj.weight_scale_inv",
            "g_proj.weight",
            "g_proj.weight_scale_inv",
            "k_conv1d.weight",
            "k_proj.weight",
            "k_proj.weight_scale_inv",
            "o_norm.weight",
            "o_proj.weight",
            "o_proj.weight_scale_inv",
            "q_conv1d.weight",
            "q_proj.weight",
            "q_proj.weight_scale_inv",
            "v_conv1d.weight",
            "v_proj.weight",
            "v_proj.weight_scale_inv",
        ] {
            header.insert(format!("{p}.{suffix}"));
        }
    }

    fn mla(header: &mut Header, layer: usize, mtp: bool) {
        let p = format!("model.layers.{layer}.attention");
        for suffix in [
            "dense.weight",
            "dense.weight_scale_inv",
            "g_proj.weight",
            "kv_a_layernorm.weight",
            "kv_a_proj_with_mqa.weight",
            "kv_a_proj_with_mqa.weight_scale_inv",
            "kv_b_proj.weight",
            "kv_b_proj.weight_scale_inv",
            "q_proj.weight",
            "q_proj.weight_scale_inv",
        ] {
            if !mtp && suffix == "g_proj.weight_scale_inv" {
                continue;
            }
            header.insert(format!("{p}.{suffix}"));
        }
    }

    fn mlp(header: &mut Header, layer: usize, config: &ModelConfig, dense: bool) {
        let p = format!("model.layers.{layer}.mlp");
        if dense {
            for suffix in [
                "down_proj.weight",
                "down_proj.weight_scale_inv",
                "gate_proj.weight",
                "gate_proj.weight_scale_inv",
                "up_proj.weight",
                "up_proj.weight_scale_inv",
            ] {
                header.insert(format!("{p}.{suffix}"));
            }
            return;
        }
        for suffix in ["gate.expert_bias", "gate.weight"] {
            header.insert(format!("{p}.{suffix}"));
        }
        for e in 0..config.num_experts {
            for suffix in [
                "down_proj.weight",
                "down_proj.weight_scale_inv",
                "gate_proj.weight",
                "gate_proj.weight_scale_inv",
                "up_proj.weight",
                "up_proj.weight_scale_inv",
            ] {
                header.insert(format!("{p}.experts.{e}.{suffix}"));
            }
        }
        for suffix in [
            "down_proj.weight",
            "down_proj.weight_scale_inv",
            "gate_proj.weight",
            "gate_proj.weight_scale_inv",
            "up_proj.weight",
            "up_proj.weight_scale_inv",
        ] {
            header.insert(format!("{p}.shared_experts.{suffix}"));
        }
    }

    fn valid_header(config: &ModelConfig) -> Header {
        let mut header = Header::new();
        root(&mut header);
        for layer in 0..config.num_hidden_layers {
            common(&mut header, layer);
            if config.layer_type(layer) == LayerType::LinearAttention {
                kda(&mut header, layer);
            } else {
                mla(&mut header, layer, false);
            }
            mlp(
                &mut header,
                layer,
                config,
                config.mlp_only_layers.contains(&layer),
            );
        }
        common(&mut header, config.num_hidden_layers);
        mla(&mut header, config.num_hidden_layers, true);
        for suffix in [
            "enorm.weight",
            "eh_proj.weight",
            "hnorm.weight",
            "final_layernorm.weight",
        ] {
            header.insert(format!(
                "model.layers.{}.{}",
                config.num_hidden_layers, suffix
            ));
        }
        mlp(&mut header, config.num_hidden_layers, config, false);
        header
    }

    #[test]
    fn accepts_the_hybrid_text_and_mtp_layout() {
        let config = config();
        let plan = plan(&valid_header(&config), &config).expect("valid layout");
        assert_eq!(
            plan.layers[0],
            LayerPlan {
                index: 0,
                attention: AttentionLayout::Kda,
                mlp: MlpLayout::Dense
            }
        );
        assert_eq!(plan.layers[1].attention, AttentionLayout::Mla);
        assert_eq!(plan.layers[1].mlp, MlpLayout::Moe);
        assert_eq!(plan.mtp.index, 2);
        assert_eq!(plan.mtp.attention, AttentionLayout::Mla);
    }

    #[test]
    fn rejects_missing_or_unknown_tensors_before_loading() {
        let config = config();
        let mut missing = valid_header(&config);
        missing.remove("model.layers.1.mlp.gate.weight");
        assert!(plan(&missing, &config).is_err());

        let mut unknown = valid_header(&config);
        unknown.insert("model.layers.1.attention.not_a_real_weight".into());
        assert!(plan(&unknown, &config).is_err());
    }
}
