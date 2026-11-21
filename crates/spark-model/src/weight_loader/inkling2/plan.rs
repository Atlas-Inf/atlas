// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2 Mini checkpoint layout, checked before anything is uploaded.
//!
//! The release keeps Thinking Machines' own tensor names under `model.llm.`
//! (transformers PR #51050 renames them on load; Atlas reads them as stored).
//! Strict loads exactly these formats: BF16 everywhere except the router
//! bias, the block-residual queries and the routed experts, which are NVFP4
//! (E2M1 codes, E4M3 per-16 scales, a per-expert `amax / denominator`) or
//! MXFP8 (E4M3 values, E8M0 per-32 scales), per layer. Any other dtype, shape,
//! missing tensor or unknown tensor under the prefix fails the load.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::{Inkling2Config, ModelConfig};
use spark_runtime::weights::WeightDtype;

/// Name -> (dtype, shape) of every tensor in the checkpoint.
pub(crate) type Header = HashMap<String, (WeightDtype, Vec<usize>)>;

/// Storage of one layer's routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertFormat {
    Nvfp4,
    Mxfp8,
}

/// What the loader needs to know about one physical layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PhysicalLayer {
    pub index: usize,
    /// Full attention (dynamic k/v convs, separate q/k/v input norms).
    pub full: bool,
    /// Inside the loop core: carries the cross-pass message gates.
    pub looped: bool,
    pub experts: ExpertFormat,
}

/// The checked layout: one entry per physical layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Plan {
    pub layers: Vec<PhysicalLayer>,
}

const PREFIX: &str = "model.llm";

/// Expected tensors, filled while checking, so unknown ones can be reported.
struct Expect<'a> {
    header: &'a Header,
    seen: BTreeSet<String>,
}

impl Expect<'_> {
    fn tensor(&mut self, name: &str, dtype: WeightDtype, shape: &[usize]) -> Result<()> {
        let (got_dtype, got_shape) = self
            .header
            .get(name)
            .with_context(|| format!("inkling2 checkpoint is missing {name}"))?;
        ensure!(
            *got_dtype == dtype && got_shape == shape,
            "inkling2 {name}: expected {dtype:?} {shape:?}, found {got_dtype:?} {got_shape:?}"
        );
        self.seen.insert(name.to_string());
        Ok(())
    }

    /// A BF16 `[rows, h]` vocabulary matrix with `rows >= vocab`.
    fn vocab_rows(&mut self, name: &str, vocab: usize, h: usize) -> Result<()> {
        let rows = self
            .header
            .get(name)
            .map(|t| t.1.first().copied().unwrap_or(0))
            .unwrap_or(0);
        self.tensor(name, WeightDtype::BF16, &[rows.max(vocab), h])
    }

    fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<()> {
        self.tensor(name, WeightDtype::BF16, shape)
    }

    fn f32(&mut self, name: &str, shape: &[usize]) -> Result<()> {
        self.tensor(name, WeightDtype::FP32, shape)
    }
}

/// Check `header` against `config` and describe every physical layer.
pub(crate) fn plan(header: &Header, config: &ModelConfig) -> Result<Plan> {
    let ink = config
        .inkling2
        .as_ref()
        .context("inkling2 loader needs the inkling2 config block")?;
    let mut e = Expect {
        header,
        seen: BTreeSet::new(),
    };
    let (h, v) = (config.hidden_size, config.vocab_size);
    // the release pads the vocabulary (201024 rows); the server may cap
    // `vocab_size` at the tokenizer's 200058, so accept at least that many
    for name in ["embed", "unembed"] {
        e.vocab_rows(&format!("{PREFIX}.{name}.weight"), v, h)?;
    }
    e.bf16(&format!("{PREFIX}.embed_norm.weight"), &[h])?;
    e.bf16(&format!("{PREFIX}.norm.weight"), &[h])?;
    e.f32(&format!("{PREFIX}.attn_res.output_query"), &[h])?;
    let core_len = ink.core_end - ink.core_start;
    // per loop layer: (q, mlp) and (k, v) message queries
    for which in ["queries", "kv_queries"] {
        e.f32(
            &format!("{PREFIX}.middle_cycle_cross_pass_{which}"),
            &[2 * core_len, h],
        )?;
    }

    let mut layers = Vec::with_capacity(ink.physical_layers);
    for l in 0..ink.physical_layers {
        let full = !ink.local_physical.contains(&l);
        // the residual query feeding layer l's attention input (none for layer 0)
        if l > 0 {
            let name = format!("{PREFIX}.attn_res.attn_queries.{}", l - 1);
            if full {
                e.f32(&name, &[3, h])?;
            } else {
                e.f32(&name, &[h])?;
            }
        }
        e.f32(&format!("{PREFIX}.attn_res.mlp_queries.{l}"), &[h])?;
        let looped = (ink.core_start..ink.core_end).contains(&l);
        let experts = layer(&mut e, config, ink, l, full, looped)?;
        layers.push(PhysicalLayer {
            index: l,
            full,
            looped,
            experts,
        });
    }

    let unknown: Vec<&String> = header
        .keys()
        .filter(|k| k.starts_with(&format!("{PREFIX}.")) && !e.seen.contains(*k))
        .collect();
    if !unknown.is_empty() {
        let mut unknown = unknown;
        unknown.sort();
        bail!(
            "inkling2 checkpoint has {} tensors this loader does not read, e.g. {:?}",
            unknown.len(),
            &unknown[..unknown.len().min(4)]
        );
    }
    Ok(Plan { layers })
}

fn layer(
    e: &mut Expect<'_>,
    config: &ModelConfig,
    ink: &Inkling2Config,
    l: usize,
    full: bool,
    looped: bool,
) -> Result<ExpertFormat> {
    let lp = format!("{PREFIX}.layers.{l}");
    let h = config.hidden_size;
    let (heads, kv_heads, hd) = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    );
    let (q_dim, kv_dim, k) = (heads * hd, kv_heads * hd, ink.sconv_kernel_size);

    if full {
        for n in ["q", "k", "v"] {
            e.bf16(&format!("{lp}.{n}_input_norm.weight"), &[h])?;
        }
    } else {
        e.bf16(&format!("{lp}.attn_norm.weight"), &[h])?;
    }
    let a = format!("{lp}.attn");
    e.bf16(&format!("{a}.wq_du.weight"), &[q_dim, h])?;
    e.bf16(&format!("{a}.wk_dv.weight"), &[kv_dim, h])?;
    e.bf16(&format!("{a}.wv_dv.weight"), &[kv_dim, h])?;
    e.bf16(&format!("{a}.wr_du.weight"), &[heads * ink.d_rel, h])?;
    e.bf16(
        &format!("{a}.rel_logits_proj.proj"),
        &[ink.d_rel, ink.rel_extent],
    )?;
    e.bf16(&format!("{a}.q_l2_clip.scale"), &[heads, hd])?;
    e.bf16(&format!("{a}.rel_l2_clip.scale"), &[heads, ink.d_rel])?;
    e.bf16(&format!("{a}.gate_attn.weight"), &[q_dim, h])?;
    e.bf16(&format!("{a}.wo_ud.weight"), &[h, q_dim])?;
    e.bf16(&format!("{a}.attn_sink_proj.weight"), &[heads, h])?;
    for kv in ["k", "v"] {
        if full {
            e.bf16(&format!("{a}.{kv}_sconv.static_w"), &[k, kv_dim])?;
            e.bf16(&format!("{a}.{kv}_sconv.weight_proj"), &[kv_heads * k, h])?;
        } else {
            e.bf16(&format!("{a}.{kv}_sconv.weight"), &[kv_dim, 1, k])?;
        }
    }
    e.bf16(&format!("{lp}.attn_sconv.weight"), &[h, 1, k])?;
    e.bf16(&format!("{lp}.mlp_sconv.weight"), &[h, 1, k])?;
    e.bf16(&format!("{lp}.mlp_norm.weight"), &[h])?;

    let (n_exp, inter) = (config.num_experts, config.moe_intermediate_size);
    e.bf16(&format!("{lp}.mlp.gate.weight"), &[n_exp, h])?;
    e.f32(&format!("{lp}.mlp.gate.bias"), &[n_exp])?;
    let ex = format!("{lp}.mlp.experts");
    let format = match e.header.get(&format!("{ex}.w13_weight")).map(|t| t.0) {
        Some(WeightDtype::UInt8) => ExpertFormat::Nvfp4,
        Some(WeightDtype::FP8E4M3) => ExpertFormat::Mxfp8,
        other => bail!(
            "inkling2 {ex}.w13_weight: expected U8 (NVFP4) or F8_E4M3 (MXFP8), found {other:?}"
        ),
    };
    // w13 rows interleave gate and up; w2 is the down projection
    for (w, n, kk) in [("w13", 2 * inter, h), ("w2", h, inter)] {
        let name = format!("{ex}.{w}_weight");
        match format {
            ExpertFormat::Nvfp4 => {
                e.tensor(&name, WeightDtype::UInt8, &[n_exp, n, kk / 2])?;
                e.tensor(
                    &format!("{name}_scale"),
                    WeightDtype::FP8E4M3,
                    &[n_exp, n, kk / 16],
                )?;
                e.f32(&format!("{name}_amax"), &[n_exp])?;
                e.f32(&format!("{name}_outer_scale_denominator"), &[1])?;
            }
            ExpertFormat::Mxfp8 => {
                e.tensor(&name, WeightDtype::FP8E4M3, &[n_exp, n, kk])?;
                e.tensor(
                    &format!("{name}_scale"),
                    WeightDtype::UInt8,
                    &[n_exp, n, kk / 32],
                )?;
            }
        }
    }

    if looped {
        for c in ["attn", "k", "v", "mlp"] {
            let cp = format!("{lp}.controllers.{c}");
            e.bf16(&format!("{cp}.down_weight"), &[ink.loop_gate_rank, 2 * h])?;
            e.bf16(&format!("{cp}.gain_weight"), &[h, ink.loop_gate_rank])?;
            e.bf16(&format!("{cp}.forget_weight"), &[h, ink.loop_gate_rank])?;
        }
    }
    Ok(format)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
