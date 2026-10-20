// SPDX-License-Identifier: AGPL-3.0-only

//! The ModelOpt `quantized_layers` map as a per-module format plan.
//!
//! NVIDIA's FP4-Local checkpoints declare every module's storage format in
//! `config.json` → `quantization_config.quantized_layers` (or a sibling
//! `hf_quant_config.json` → `quantization.quantized_layers`), so nothing
//! about the load needs to be keyed on qwen4_exp's module names: the plan
//! says which algo and scale encoding each module uses, and a per-model
//! [`NativeConsumers`] registry says which of those the model's layers can
//! run natively. Everything else is host-expanded to BF16.
//!
//! `is_fp4_local` — a `weight_scale_encoding` on any module — is what makes a
//! checkpoint "FP4-Local": an encoding entry means the pack ships compressed
//! scale streams (`weight_scale.nvfp4_csf_*`) and the ModelOpt-local decode
//! path applies.

use std::collections::BTreeMap;

use atlas_core::config::ModelConfig;

/// The quantized-layer algorithm, normalized across spellings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantAlgo {
    /// `"NVFP4"` — packed E2M1 + E4M3 per-group scales (+ optional CSF).
    Nvfp4,
    /// `"W4A16_NVFP4"` and friends — same packed format, BF16 activations.
    W4A16Nvfp4,
    /// `"MXFP8"` — E4M3 + E8M0 per-32-input scales (`group_size` 32 or a
    /// `weight_block_size` of `[1, 32]`).
    Mxfp8,
    /// `"FP8_PB_WO"` / `"FP8_BLOCK_SCALES"` — E4M3 with a `[bn, bk]` tile.
    Fp8Block { bn: usize, bk: usize },
    /// Anything else, kept verbatim for error messages.
    Other(String),
}

/// One module's declared storage format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleFormat {
    pub algo: QuantAlgo,
    /// Scaling group along the input dim: `group_size`, or the `bk` of a
    /// `[1, k]` `weight_block_size`. `0` when neither is declared.
    pub group_size: usize,
    /// `weight_scale_encoding` verbatim (`"csf"`, `"csf_32x2_v1"`, …).
    pub scale_encoding: Option<String>,
}

/// The whole `quantized_layers` map, normalized.
#[derive(Debug, Default)]
pub struct ModeloptQuantPlan {
    formats: BTreeMap<String, ModuleFormat>,
}

impl ModeloptQuantPlan {
    /// From the parsed config (`config.json` or `hf_quant_config.json` —
    /// `parse_config` has already merged them). `None` without a map.
    pub fn from_config(config: &ModelConfig) -> Option<Self> {
        let qc = config.quantization_config.as_ref()?;
        if qc.quantized_layers.is_empty() {
            return None;
        }
        let mut formats = BTreeMap::new();
        for (module, spec) in &qc.quantized_layers {
            let algo = spec.quant_algo.trim();
            let algo = if algo.eq_ignore_ascii_case("MXFP8") {
                QuantAlgo::Mxfp8
            } else if algo.eq_ignore_ascii_case("NVFP4") {
                QuantAlgo::Nvfp4
            } else if atlas_core::config::is_nvfp4_quant_algo(algo) {
                QuantAlgo::W4A16Nvfp4
            } else if matches!(
                algo.to_ascii_uppercase().as_str(),
                "FP8_PB_WO" | "FP8_BLOCK_SCALES" | "FP8"
            ) && spec.weight_block_size.len() == 2
            {
                QuantAlgo::Fp8Block {
                    bn: spec.weight_block_size[0],
                    bk: spec.weight_block_size[1],
                }
            } else {
                QuantAlgo::Other(algo.to_string())
            };
            // `weight_block_size [1, k]` is the tiled spelling of
            // `group_size k`; a `[n>1, k]` tile groups n-rows instead and
            // keeps group_size 0 here (block scaling is a different codec).
            let group_size = if spec.group_size != 0 {
                spec.group_size
            } else if spec.weight_block_size.len() == 2 && spec.weight_block_size[0] == 1 {
                spec.weight_block_size[1]
            } else {
                0
            };
            formats.insert(
                module.clone(),
                ModuleFormat {
                    algo,
                    group_size,
                    scale_encoding: spec.weight_scale_encoding.clone(),
                },
            );
        }
        Some(Self { formats })
    }

    /// The declared format of `module`, if the map lists it.
    pub fn format_of(&self, module: &str) -> Option<&ModuleFormat> {
        self.formats.get(module)
    }

    /// Every module the map declares.
    pub fn modules(&self) -> impl Iterator<Item = &String> {
        self.formats.keys()
    }

    /// The FP4-Local family marker: any module ships a compressed
    /// `weight_scale` encoding (today `"csf"`).
    pub fn is_fp4_local(&self) -> bool {
        self.formats.values().any(|f| f.scale_encoding.is_some())
    }
}

/// Which module formats a model's layers run natively — declared by the
/// model loader, matched against module paths in the plan. Modules with no
/// registered consumer are host-expanded to BF16 by `materialize`.
#[derive(Debug, Clone, Copy)]
pub struct NativeConsumers {
    /// Modules whose MXFP8 tensors upload and dequant as shipped
    /// (`mxfp8_proj`).
    pub mxfp8: fn(&str) -> bool,
    /// Modules whose NVFP4 tensors are read natively (deferred to the layer's
    /// own loader — MoE experts, the vision tower, the PLE table).
    pub nvfp4: fn(&str) -> bool,
}

impl NativeConsumers {
    const NONE: Self = Self {
        mxfp8: |_| false,
        nvfp4: |_| false,
    };

    /// Whether `module`'s declared `algo` has a native consumer here.
    pub fn covers(&self, module: &str, algo: &QuantAlgo) -> bool {
        match algo {
            QuantAlgo::Mxfp8 => (self.mxfp8)(module),
            QuantAlgo::Nvfp4 | QuantAlgo::W4A16Nvfp4 => (self.nvfp4)(module),
            _ => false,
        }
    }
}

fn qwen4_exp_native_mxfp8(module: &str) -> bool {
    const NATIVE: [&str; 10] = [
        ".linear_attn.in_proj_qkv",
        ".linear_attn.in_proj_z",
        ".linear_attn.out_proj",
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
        ".mlp.shared_expert.gate_proj",
        ".mlp.shared_expert.up_proj",
        ".mlp.shared_expert.down_proj",
    ];
    (module.starts_with("model.language_model.layers.")
        && NATIVE.iter().any(|suffix| module.ends_with(suffix)))
    // The vision tower's MXFP8 attention projections upload as shipped and
    // dequant on-GPU into the encoder's BF16 scratch (`model.visual.` and
    // the `model.language_model.visual.` spelling both land here).
    || (module.contains(".visual.")
        && (module.ends_with(".attn.qkv") || module.ends_with(".attn.proj")))
}

fn qwen4_exp_native_nvfp4(module: &str) -> bool {
    // Routed + MTP experts (the per-expert NVFP4 MoE loaders), the vision
    // tower's linear_fc/merger projections (GPU dequant into BF16 scratch),
    // and the PLE n-gram table (row-gathered off disk).
    module.ends_with(".mlp.experts")
        || module.starts_with("mtp.")
        || module.contains(".visual.")
        || module.ends_with(".ngram_embedding")
}

/// The native-consumer registry, keyed by model type. Models not listed run
/// every declared module through the generic host expansion.
pub fn native_consumers(model_type: &str) -> NativeConsumers {
    match model_type {
        "qwen4_exp" => NativeConsumers {
            mxfp8: qwen4_exp_native_mxfp8,
            nvfp4: qwen4_exp_native_nvfp4,
        },
        _ => NativeConsumers::NONE,
    }
}

#[cfg(test)]
#[path = "modelopt_plan_tests.rs"]
mod tests;
