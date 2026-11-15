// SPDX-License-Identifier: AGPL-3.0-only

//! The tensors a DeepSeek-V4.1-Flash checkpoint must hold, from its config.
//!
//! Three packs of the 1017 weights are known: DeepSeek's release (FP8 dense
//! with 32x32 UE8M0 block scales, MXFP4 routed experts, FP8 Engram tables),
//! NVIDIA's NVFP4 pack (NVFP4 routed experts in the main model, MXFP4 Engram
//! tables) and the EXL3 packs (trellis-coded linears, no Engram tables).
//! [`manifest`] names every tensor of a pack with its safetensors dtype and
//! shape; [`check_headers`] holds a checkpoint's headers against it before
//! anything is loaded.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;

/// Which pack of the weights a checkpoint is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pack {
    Release,
    Nvfp4,
    Exl3,
}

/// The vision tower's `vision_config`, which `ModelConfig` does not carry.
#[derive(Debug, Clone, Copy)]
pub struct Vision {
    pub layers: usize,
    pub width: usize,
    pub intermediate: usize,
    pub patch: usize,
    pub downsample: usize,
}

/// One tensor: its name, safetensors dtype and shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
}

/// A dimension the pack's quantizer chose, which the config does not fix: the
/// last dimension of an EXL3 trellis is 16 times that tensor's bits per weight.
pub const ANY: usize = 0;

/// FP8 weights carry one UE8M0 scale per 32x32 block.
const BLOCK: usize = 32;

struct Manifest<'a> {
    pack: Pack,
    c: &'a ModelConfig,
    out: Vec<Tensor>,
}

impl Manifest<'_> {
    fn tensor(&mut self, name: String, dtype: &'static str, shape: &[usize]) {
        self.out.push(Tensor {
            name,
            dtype,
            shape: shape.to_vec(),
        });
    }

    /// A linear the EXL3 packs trellis-code: a sign vector per side, the
    /// codebook's multiplier, and the trellis in 16x16 tiles.
    fn exl3(&mut self, name: &str, rows: usize, cols: usize) {
        self.tensor(format!("{name}.mul1"), "I32", &[]);
        self.tensor(format!("{name}.suh"), "F16", &[cols]);
        self.tensor(format!("{name}.svh"), "F16", &[rows]);
        self.tensor(
            format!("{name}.trellis"),
            "I16",
            &[cols / 16, rows / 16, ANY],
        );
    }

    /// A linear the release stores as FP8 with block scales.
    fn fp8(&mut self, name: &str, rows: usize, cols: usize) {
        if self.pack == Pack::Exl3 {
            return self.exl3(name, rows, cols);
        }
        self.tensor(format!("{name}.weight"), "F8_E4M3", &[rows, cols]);
        self.tensor(
            format!("{name}.scale"),
            "F8_E8M0",
            &[rows.div_ceil(BLOCK), cols / BLOCK],
        );
    }

    /// A linear the release stores as BF16; `exl3` is how the EXL3 packs
    /// store it: trellis-coded, or as plain F16.
    fn bf16(&mut self, name: &str, rows: usize, cols: usize, exl3: Option<&'static str>) {
        match (self.pack, exl3) {
            (Pack::Exl3, None) => self.exl3(name, rows, cols),
            (Pack::Exl3, Some(dtype)) => {
                self.tensor(format!("{name}.weight"), dtype, &[rows, cols])
            }
            _ => self.tensor(format!("{name}.weight"), "BF16", &[rows, cols]),
        }
    }

    fn norm(&mut self, name: &str, width: usize) {
        self.tensor(format!("{name}.weight"), "BF16", &[width]);
    }

    /// One routed-expert projection: two FP4 values to a byte. The release
    /// scales each row in groups of 32 (MXFP4); NVIDIA re-scales the main
    /// model's in groups of 16 with two more scalars (NVFP4) and leaves the
    /// drafter's as released.
    fn routed(&mut self, name: &str, rows: usize, cols: usize, drafter: bool) {
        match self.pack {
            Pack::Exl3 => self.exl3(name, rows, cols),
            Pack::Nvfp4 if !drafter => {
                self.tensor(format!("{name}.weight"), "U8", &[rows, cols / 2]);
                self.tensor(
                    format!("{name}.weight_scale"),
                    "F8_E4M3",
                    &[rows, cols / 16],
                );
                self.tensor(format!("{name}.weight_scale_2"), "F32", &[]);
                self.tensor(format!("{name}.input_scale"), "F32", &[]);
            }
            _ => {
                self.tensor(format!("{name}.weight"), "I8", &[rows, cols / 2]);
                self.tensor(format!("{name}.scale"), "F8_E8M0", &[rows, cols / BLOCK]);
            }
        }
    }

    /// What a main layer and a drafter stage share: attention over the
    /// window, the MoE, and the two hyper-connection sites.
    fn block(&mut self, at: &str, experts: usize, drafter: bool) {
        let c = self.c;
        let (dim, inter) = (c.hidden_size, c.moe_intermediate_size);
        let heads = c.num_attention_heads * c.head_dim;
        self.tensor(
            format!("{at}.attn.attn_sink"),
            "F32",
            &[c.num_attention_heads],
        );
        self.norm(&format!("{at}.attn.kv_norm"), c.head_dim);
        self.norm(&format!("{at}.attn.q_norm"), c.q_lora_rank);
        self.fp8(&format!("{at}.attn.wq_a"), c.q_lora_rank, dim);
        self.fp8(&format!("{at}.attn.wq_b"), heads, c.q_lora_rank);
        self.fp8(&format!("{at}.attn.wkv"), c.head_dim, dim);
        if self.pack == Pack::Exl3 {
            for group in 0..c.o_groups {
                let name = format!("{at}.attn.wo_a.slice.{group}");
                self.exl3(&name, c.o_lora_rank, heads / c.o_groups);
            }
        } else {
            self.fp8(
                &format!("{at}.attn.wo_a"),
                c.o_groups * c.o_lora_rank,
                heads / c.o_groups,
            );
        }
        self.fp8(&format!("{at}.attn.wo_b"), dim, c.o_groups * c.o_lora_rank);
        self.norm(&format!("{at}.attn_norm"), dim);
        self.norm(&format!("{at}.ffn_norm"), dim);

        let exl3 = self.pack == Pack::Exl3;
        let gate = if exl3 { "F16" } else { "F32" };
        self.bf16(&format!("{at}.ffn.gate"), experts, dim, Some("F16"));
        self.tensor(format!("{at}.ffn.gate.bias"), gate, &[experts]);
        // The routing bias for image tokens; the EXL3 packs drop the drafter's.
        if !(exl3 && drafter) {
            self.tensor(format!("{at}.ffn.gate.bias_vl"), "F32", &[experts]);
        }
        for (projection, rows, cols) in [("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)]
        {
            for expert in 0..experts {
                let name = format!("{at}.ffn.experts.{expert}.{projection}");
                self.routed(&name, rows, cols, drafter);
            }
            self.fp8(&format!("{at}.ffn.shared_experts.{projection}"), rows, cols);
        }
        let mix = c.hc_mult * (c.hc_mult + 2);
        for site in ["hc_attn", "hc_ffn"] {
            self.tensor(format!("{at}.{site}_base"), "F32", &[mix]);
            self.tensor(format!("{at}.{site}_fn"), "F32", &[mix, c.hc_mult * dim]);
            self.tensor(format!("{at}.{site}_scale"), "F32", &[3]);
        }
    }

    /// What only some main layers own: the compressed KV and index keys a
    /// source layer produces, the indexer's scoring heads, an Engram table.
    fn layer_extras(&mut self, layer: usize) {
        let c = self.c;
        let (at, dim) = (format!("layers.{layer}.attn"), c.hidden_size);
        if c.kv_source_layer_ids.contains(&layer) {
            let ratio = c.compress_ratios[layer];
            self.norm(&format!("{at}.compressor.norm"), c.head_dim);
            self.bf16(&format!("{at}.compressor.wkv"), c.head_dim, dim, None);
            if ratio > 1 {
                self.bf16(&format!("{at}.compressor.wgate"), c.head_dim, dim, None);
            }
            if self.pack == Pack::Exl3 {
                self.tensor(format!("{at}.compressor.ape"), "F32", &[ratio, c.head_dim]);
            }
            self.norm(&format!("{at}.indexer.k_norm"), c.index_head_dim);
            self.bf16(
                &format!("{at}.indexer.wk"),
                c.index_head_dim,
                c.head_dim,
                None,
            );
        }
        if c.index_source_layer_ids.contains(&layer) {
            self.bf16(
                &format!("{at}.indexer.weights_proj"),
                c.index_n_heads,
                dim,
                Some("F16"),
            );
            let heads = c.index_n_heads * c.index_head_dim;
            self.fp8(&format!("{at}.indexer.wq_b"), heads, c.q_lora_rank);
        }
        if let Some(table) = c.engram_layer_ids.iter().position(|&id| id == layer) {
            let at = format!("layers.{layer}.engram");
            let (rows, width) = (c.engram_num_embeddings[table] as usize, c.engram_head_dim);
            match self.pack {
                Pack::Release => {
                    self.tensor(format!("{at}.embed.weight"), "F8_E4M3", &[rows, width])
                }
                // MXFP4: two values to a byte, under the release's scales.
                Pack::Nvfp4 => self.tensor(format!("{at}.embed.weight"), "U8", &[rows, width / 2]),
                Pack::Exl3 => {}
            }
            if self.pack != Pack::Exl3 {
                // One scale per row and 32 values: a row is read alone.
                self.tensor(
                    format!("{at}.embed.scale"),
                    "F8_E8M0",
                    &[rows, width / BLOCK],
                );
            }
            self.tensor(format!("{at}.k_weight"), "BF16", &[c.hc_mult, dim]);
            self.tensor(format!("{at}.q_weight"), "BF16", &[c.hc_mult, dim]);
            let lookups = (c.engram_max_ngram_size - 1) * c.engram_n_heads * width;
            self.fp8(&format!("{at}.wkv"), (c.hc_mult + 1) * dim, lookups);
        }
    }

    /// The DSpark drafter: stages over the target layers' outputs, then the
    /// Markov and confidence heads on the last stage.
    fn drafter(&mut self) {
        let c = self.c;
        let (dim, rank, vocab) = (c.hidden_size, c.dspark_markov_rank, c.vocab_size);
        for stage in 0..c.num_mtp_modules {
            self.block(&format!("mtp.{stage}"), c.dspark_n_routed_experts, true);
        }
        self.norm("mtp.0.main_norm", dim);
        self.fp8(
            "mtp.0.main_proj",
            dim,
            c.dspark_target_layer_ids.len() * dim,
        );
        let last = format!("mtp.{}", c.num_mtp_modules - 1);
        self.norm(&format!("{last}.norm"), dim);
        self.bf16(
            &format!("{last}.confidence_head.proj"),
            1,
            dim + rank,
            Some("F16"),
        );
        if self.pack == Pack::Exl3 {
            self.tensor(
                format!("{last}.markov_head.markov_w1.weight"),
                "BF16",
                &[vocab, rank],
            );
            self.tensor(
                format!("{last}.markov_head.markov_w2.weight"),
                "F16",
                &[vocab, rank],
            );
        } else {
            self.bf16(&format!("{last}.markov_head.embed"), vocab, rank, None);
            self.bf16(&format!("{last}.markov_head.head"), vocab, rank, None);
        }
    }

    fn vision(&mut self, v: &Vision) {
        let (dim, w) = (self.c.hidden_size, v.width);
        let mut linear = |name: &str, rows: usize, cols: usize, bias: bool| {
            self.tensor(format!("{name}.weight"), "BF16", &[rows, cols]);
            if bias {
                self.tensor(format!("{name}.bias"), "BF16", &[rows]);
            }
        };
        for block in 0..v.layers {
            let at = format!("vision.blocks.{block}");
            linear(&format!("{at}.attn.wqkv"), 3 * w, w, true);
            linear(&format!("{at}.attn.wo"), w, w, true);
            linear(&format!("{at}.mlp.w1"), 2 * v.intermediate, w, false);
            linear(&format!("{at}.mlp.w2"), w, v.intermediate, false);
        }
        linear("vision.patch_embed.proj", w, 3 * v.patch * v.patch, true);
        linear("aligner.w1", dim, w * v.downsample * v.downsample, true);
        linear("aligner.w2", dim, dim, true);
        for block in 0..v.layers {
            self.norm(&format!("vision.blocks.{block}.norm1"), w);
            self.norm(&format!("vision.blocks.{block}.norm2"), w);
        }
        self.norm("vision.norm", w);
        for name in ["image_start", "image_end", "image_newline"] {
            self.tensor(name.to_string(), "BF16", &[dim]);
        }
    }
}

/// Every tensor of a `pack` of the model `config` describes, in no order.
pub fn manifest(config: &ModelConfig, vision: Option<&Vision>, pack: Pack) -> Result<Vec<Tensor>> {
    let c = config;
    ensure!(
        c.model_type == "deepseek_v41",
        "not a DeepSeek-V4.1 config: model_type {}",
        c.model_type
    );
    ensure!(
        c.num_mtp_modules > 0 && c.o_groups > 0,
        "a DeepSeek-V4.1 checkpoint has DSpark stages and grouped output projections"
    );
    ensure!(
        c.compress_ratios.len() >= c.num_hidden_layers
            && c.engram_num_embeddings.len() == c.engram_layer_ids.len(),
        "compress_ratios must cover every layer, and engram_num_embeddings every Engram layer"
    );
    let mut m = Manifest {
        pack,
        c,
        out: Vec::new(),
    };
    m.tensor(
        "embed.weight".into(),
        "BF16",
        &[c.vocab_size, c.hidden_size],
    );
    if pack == Pack::Exl3 {
        m.exl3("head", c.vocab_size, c.hidden_size);
    } else {
        m.tensor("head.weight".into(), "BF16", &[c.vocab_size, c.hidden_size]);
    }
    m.norm("norm", c.hidden_size);
    for layer in 0..c.num_hidden_layers {
        m.block(&format!("layers.{layer}"), c.num_experts, false);
        m.layer_extras(layer);
    }
    m.drafter();
    if let Some(vision) = vision {
        m.vision(vision);
    }
    Ok(m.out)
}

/// Hold a checkpoint's headers (`name -> (dtype, shape)`) against `expected`.
/// Fails on a missing tensor, a tensor nobody expects, or a dtype or shape
/// that differs, and names the first few of each.
pub fn check_headers(
    expected: &[Tensor],
    headers: &HashMap<String, (String, Vec<usize>)>,
) -> Result<()> {
    let mut missing = Vec::new();
    let mut different = Vec::new();
    for tensor in expected {
        match headers.get(&tensor.name) {
            None => missing.push(tensor.name.clone()),
            Some((dtype, shape)) => {
                let same_shape = shape.len() == tensor.shape.len()
                    && shape
                        .iter()
                        .zip(&tensor.shape)
                        .all(|(&got, &want)| want == ANY || got == want);
                if dtype != tensor.dtype || !same_shape {
                    different.push(format!(
                        "{} is {dtype} {shape:?}, expected {} {:?}",
                        tensor.name, tensor.dtype, tensor.shape
                    ));
                }
            }
        }
    }
    let names: HashSet<&str> = expected.iter().map(|t| t.name.as_str()).collect();
    let mut extra: Vec<&str> = headers
        .keys()
        .map(String::as_str)
        .filter(|name| !names.contains(name))
        .collect();
    if missing.is_empty() && different.is_empty() && extra.is_empty() {
        return Ok(());
    }
    missing.sort();
    different.sort();
    extra.sort();
    let first = |list: &[String]| list.iter().take(4).cloned().collect::<Vec<_>>().join("; ");
    let extra: Vec<String> = extra.into_iter().map(str::to_string).collect();
    bail!(
        "checkpoint does not match the DeepSeek-V4.1 manifest: {} missing [{}], {} differ [{}], \
         {} unexpected [{}]",
        missing.len(),
        first(&missing),
        different.len(),
        first(&different),
        extra.len(),
        first(&extra),
    )
}

#[cfg(test)]
mod tests;
