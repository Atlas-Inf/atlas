// SPDX-License-Identifier: AGPL-3.0-only

//! The checkpoint plan: one pass over the store's tensor names that
//! classifies all 253,044 of them and validates the layout the loader maps.
//!
//! Built ONCE per load (the performance guardrail for a 253k-tensor
//! checkpoint): a single `WeightStore::names()` iteration with fixed-size
//! per-layer counters — never per-tensor ad-hoc parsing at upload time.
//! After the pass, `O(num_layers * const)` presence probes verify the
//! tensors the counters cannot see (attention stems, norms, router, shared
//! expert, MTP conns).
//!
//! KDA vs gated-MLA disambiguation is by tensor signature, per the index
//! file: a layer carrying `attention.A_log` / `attention.dt_bias` is KDA; a
//! layer carrying `attention.kv_a_proj_with_mqa...` is gated MLA. (Indices
//! agree with the config rule — {5, 11, 17, 23, 29, 35, 41} plus MTP layer
//! 42 — but the store wins over derived config.)

use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

use super::config::{AttentionKind, MlpKind};
use super::mtp;
use super::names;

/// What the plan validates, taken from `ModelConfig` so the same build
/// works whatever config parser produced it.
pub(super) struct PlanSpec {
    /// Decoder layers (Ling: 42, `model.layers.0..42`).
    pub num_layers: usize,
    /// NEXTN MTP layers after them (Ling: 1, `model.layers.42.*`).
    pub num_mtp_layers: usize,
    /// Routed experts per MoE layer (Ling: 512, dense prefix excluded).
    pub num_experts: usize,
}

impl PlanSpec {
    fn from_config(config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            num_layers: config.num_hidden_layers,
            num_mtp_layers: config.mtp_num_hidden_layers,
            num_experts: config.num_experts,
        })
        .and_then(|spec| {
            ensure!(
                spec.num_layers > 0 && spec.num_experts > 0,
                "bailing: config must declare layers and experts"
            );
            Ok(spec)
        })
    }
}

/// One decoder or MTP layer's classified tensor inventory.
#[derive(Debug)]
pub(super) struct LayerPlan {
    pub attention: AttentionKind,
    pub mlp: MlpKind,
    /// Distinct routed experts seen (each must contribute all 12 members).
    pub experts_seen: usize,
}

/// The validated whole-checkpoint layout.
#[derive(Debug)]
pub(super) struct CheckpointPlan {
    pub layers: Vec<LayerPlan>,
    /// MTP layers in NEXTN order; empty when the checkpoint ships none.
    pub mtp_layers: Vec<LayerPlan>,
    /// `model.visual.*` tensors present (skipped, text-only phase).
    pub vision_tensors: usize,
    /// `linear_proj.*` projector tensors present (skipped, text-only phase).
    pub projector_tensors: usize,
    /// Layers below this index carry a dense MLP.
    pub first_k_dense_replace: usize,
}

impl CheckpointPlan {
    /// Structure summary for the load-time trace.
    pub(crate) fn summary(&self) -> String {
        let kda = self
            .layers
            .iter()
            .filter(|l| l.attention == AttentionKind::Kda)
            .count();
        let mla = self
            .layers
            .iter()
            .filter(|l| l.attention == AttentionKind::GatedMla)
            .count();
        let moe = self
            .layers
            .iter()
            .chain(self.mtp_layers.iter())
            .filter(|l| l.mlp == MlpKind::Moe)
            .count();
        format!(
            "{} decoder layers ({kda} KDA / {mla} gated-MLA), {} MTP layer(s), \
             {moe} MoE layers x {} experts each, dense prefix {}, \
             +{}/{} vision/projector tensors skipped",
            self.layers.len(),
            self.mtp_layers.len(),
            self.expert_count(),
            self.first_k_dense_replace,
            self.vision_tensors,
            self.projector_tensors,
        )
    }

    /// Experts per MoE layer (0 when the plan has none).
    fn expert_count(&self) -> usize {
        self.layers
            .iter()
            .chain(self.mtp_layers.iter())
            .find(|l| l.mlp == MlpKind::Moe)
            .map(|l| l.experts_seen)
            .unwrap_or(0)
    }
}

// ── single classification pass ───────────────────────────────────────────────

/// Per-layer accumulators. `expert_members[projection][part]` counts the
/// compression members of [`names::EXPERT_PROJ_STEMS`] x
/// [`names::EXPERT_PART_SUFFIXES`]; `experts` tracks distinct expert ids.
struct LayerBits {
    sees_kda_param: bool,
    sees_mla_projection: bool,
    sees_moe_router: bool,
    sees_dense_mlp: bool,
    expert_members: [[usize; 4]; 3],
    experts: Vec<bool>,
}

impl LayerBits {
    fn new(num_experts: usize) -> Self {
        Self {
            sees_kda_param: false,
            sees_mla_projection: false,
            sees_moe_router: false,
            sees_dense_mlp: false,
            expert_members: [[0; 4]; 3],
            experts: vec![false; num_experts],
        }
    }

    /// Signal the architecture from the `model.layers.<i>.`-suffix `rest`.
    fn mark(&mut self, rest: &str) {
        match rest {
            "attention.A_log" | "attention.dt_bias" => self.sees_kda_param = true,
            _ if rest.starts_with("attention.kv_a_proj_with_mqa") => {
                self.sees_mla_projection = true;
            }
            _ if rest.starts_with("mlp.gate.") => self.sees_moe_router = true,
            _ if rest.starts_with("mlp.gate_proj.weight")
                || rest.starts_with("mlp.up_proj.weight")
                || rest.starts_with("mlp.down_proj.weight") =>
            {
                self.sees_dense_mlp = true;
            }
            _ => {}
        }
    }
}

/// Build and validate the plan. Fails on any tensor the layout cannot
/// account for, any expected tensor missing, malformed expert indices, or
/// expert members that do not come in complete 12-tensor sets.
pub(super) fn build(store: &WeightStore, config: &ModelConfig) -> Result<CheckpointPlan> {
    let spec = PlanSpec::from_config(config)?;
    let total = spec.num_layers + spec.num_mtp_layers;
    let mut layers: Vec<LayerBits> = (0..total)
        .map(|_| LayerBits::new(spec.num_experts))
        .collect();
    let mut model = ModelLevel::default();
    let mut stray: Vec<&str> = Vec::new();

    for name in store.names() {
        if name.starts_with(names::SKIP_TOWER) {
            model.vision_tensors += 1;
            continue;
        }
        if name.starts_with(names::SKIP_PROJECTOR) {
            model.projector_tensors += 1;
            continue;
        }
        match name {
            names::WORD_EMBEDDINGS => model.has_embedding = true,
            names::FINAL_NORM => model.has_final_norm = true,
            names::LM_HEAD => model.has_lm_head = true,
            _ => {}
        }
        if let Some(rest) = name.strip_prefix("model.layers.") {
            match count_layer(rest, &mut layers)? {
                true => continue,
                false => bail!("bailing: malformed layer tensor {name:?}"),
            }
        } else if !matches!(
            name,
            names::WORD_EMBEDDINGS | names::FINAL_NORM | names::LM_HEAD
        ) && stray.len() < 8
        {
            stray.push(name);
        }
    }
    ensure!(
        stray.is_empty(),
        "bailing: unrecognized tensor names in store: {stray:?}"
    );
    ensure!(
        model.has_embedding && model.has_final_norm && model.has_lm_head,
        "bailing: model-level tensors missing \
         (word_embeddings / model.norm / lm_head)"
    );

    let first_k_dense_replace = dense_prefix(&layers);
    let mut plan = CheckpointPlan {
        layers: Vec::with_capacity(spec.num_layers),
        mtp_layers: Vec::with_capacity(spec.num_mtp_layers),
        vision_tensors: model.vision_tensors,
        projector_tensors: model.projector_tensors,
        first_k_dense_replace,
    };
    for (idx, bits) in layers.into_iter().enumerate() {
        let layer = finish_layer(idx, bits, &spec)?;
        if idx < spec.num_layers {
            plan.layers.push(layer);
        } else {
            plan.mtp_layers.push(layer);
        }
    }
    validate_expected(store, &spec, &plan)?;
    Ok(plan)
}

#[derive(Default)]
struct ModelLevel {
    vision_tensors: usize,
    projector_tensors: usize,
    has_embedding: bool,
    has_final_norm: bool,
    has_lm_head: bool,
}

/// Tally one `model.layers.<idx>.<rest>` name; `Ok(false)` marks a malformed
/// name the layout refuses.
fn count_layer(rest: &str, layers: &mut [LayerBits]) -> Result<bool> {
    let Some((idx, rest)) = split_index(rest) else {
        return Ok(false);
    };
    let Some(bits) = layers.get_mut(idx) else {
        bail!(
            "bailing: tensor belongs to layer {idx} but the config declares {}",
            layers.len()
        );
    };
    if let Some(after) = rest.strip_prefix("mlp.experts.") {
        let Some((expert, tail)) = after.split_once('.') else {
            return Ok(false);
        };
        let Some(expert) = expert.parse::<usize>().ok() else {
            return Ok(false);
        };
        let Some(expert) = bits.experts.get_mut(expert) else {
            bail!(
                "bailing: expert index {expert} out of range for \
                 {} experts",
                bits.experts.len()
            );
        };
        let [proj, part] = expert_member(tail)?;
        if !*expert {
            *expert = true;
        }
        bits.expert_members[proj][part] += 1;
        return Ok(true);
    }
    bits.mark(rest);
    Ok(true)
}

/// The `(projection, part)` indices of an expert compression member tail
/// (`<proj>.<part>`), or a malformed-name error.
fn expert_member(tail: &str) -> Result<[usize; 2]> {
    let Some((proj, part)) = tail.rsplit_once('.') else {
        bail!("bailing: malformed expert member {tail:?}");
    };
    let Some(proj_idx) = names::EXPERT_PROJ_STEMS.iter().position(|s| *s == proj) else {
        bail!("bailing: unknown expert projection {proj:?}");
    };
    let Some(part_idx) = names::EXPERT_PART_SUFFIXES.iter().position(|s| *s == part) else {
        bail!("bailing: unknown NVFP4 member {part:?}");
    };
    Ok([proj_idx, part_idx])
}

fn split_index(rest: &str) -> Option<(usize, &str)> {
    let (idx, rest) = rest.split_once('.')?;
    let idx = idx.parse::<usize>().ok()?;
    Some((idx, rest))
}

/// How many leading layers carry a dense MLP (must be a contiguous prefix;
/// the reference uses `first_k_dense_replace`).
fn dense_prefix(layers: &[LayerBits]) -> usize {
    layers
        .iter()
        .take_while(|b| b.sees_dense_mlp && !b.sees_moe_router)
        .count()
}

/// Convert one layer's accumulated signals into a validated [`LayerPlan`].
fn finish_layer(idx: usize, bits: LayerBits, spec: &PlanSpec) -> Result<LayerPlan> {
    let experts_seen = bits.experts.iter().filter(|seen| **seen).count();
    let is_moe = bits.sees_moe_router || experts_seen > 0;
    for (proj, counts) in bits.expert_members.iter().enumerate() {
        let expected = if is_moe { spec.num_experts } else { 0 };
        for (part, &count) in counts.iter().enumerate() {
            ensure!(
                count == expected,
                "bailing: layer {idx} {} {} has {count} members, expected {expected}",
                names::EXPERT_PROJ_STEMS[proj],
                names::EXPERT_PART_SUFFIXES[part],
            );
        }
    }
    ensure!(
        !is_moe || experts_seen == spec.num_experts,
        "bailing: layer {idx} has {experts_seen} distinct experts, expected {}",
        spec.num_experts
    );
    ensure!(
        is_moe || bits.sees_dense_mlp,
        "bailing: layer {idx} carries neither a dense MLP nor experts"
    );
    let attention = match (bits.sees_kda_param, bits.sees_mla_projection) {
        (true, false) => AttentionKind::Kda,
        (false, true) => AttentionKind::GatedMla,
        (true, true) => bail!("bailing: layer {idx} carries both KDA and MLA tensors"),
        (false, false) => bail!("bailing: layer {idx} carries neither KDA nor MLA tensors"),
    };
    let mlp = if is_moe { MlpKind::Moe } else { MlpKind::Dense };
    Ok(LayerPlan {
        attention,
        mlp,
        experts_seen,
    })
}

/// Presence probes for the tensors the pass counts cannot see: per-layer
/// attention stems, layernorms, router, shared expert, and the MTP-only
/// weights. `O(num_layers * const)` against the store, not `O(tensors)`.
fn validate_expected(store: &WeightStore, spec: &PlanSpec, plan: &CheckpointPlan) -> Result<()> {
    for (idx, layer) in plan.layers.iter().enumerate() {
        let lp = format!("model.layers.{idx}");
        probe(store, &names::layer_norm_tensors(&lp))?;
        probe(store, &attention_tensors(&lp, layer.attention))?;
        match layer.mlp {
            MlpKind::Dense => probe(store, &names::dense_mlp_tensors(&lp))?,
            MlpKind::Moe => {
                let mut moe_names = names::router_tensors(&lp);
                moe_names.extend(names::shared_expert_tensors(&lp));
                probe(store, &moe_names)?;
                // Completeness spot-probe of the first and last expert (the
                // single pass already counts all 12 members per expert).
                for expert in [0, layer_experts(layer).saturating_sub(1)] {
                    probe(store, &names::expert_tensor_names(&lp, expert))?;
                }
            }
        }
    }
    for (m, layer) in plan.mtp_layers.iter().enumerate() {
        let lp = format!("model.layers.{}", spec.num_layers + m);
        ensure!(
            layer.attention == mtp::mtp_attention_kind(),
            "bailing: MTP layer {m} must use gated MLA in this loader"
        );
        ensure!(
            layer.mlp == MlpKind::Moe,
            "bailing: MTP layer {m} has no MoE"
        );
        probe(store, &names::layer_norm_tensors(&lp))?;
        probe(store, &attention_tensors(&lp, layer.attention))?;
        let mut moe_names = names::router_tensors(&lp);
        moe_names.extend(names::shared_expert_tensors(&lp));
        probe(store, &moe_names)?;
        for expert in [0, layer_experts(layer).saturating_sub(1)] {
            probe(store, &names::expert_tensor_names(&lp, expert))?;
        }
        probe(store, &mtp::mtp_tensors(&lp))?;
    }
    Ok(())
}

fn attention_tensors(lp: &str, kind: AttentionKind) -> Vec<String> {
    match kind {
        AttentionKind::Kda => names::kda_attention_tensors(lp),
        AttentionKind::GatedMla => names::mla_attention_tensors(lp),
    }
}

fn layer_experts(layer: &LayerPlan) -> usize {
    layer.experts_seen
}

fn probe(store: &WeightStore, names: &[String]) -> Result<()> {
    for name in names {
        ensure!(store.contains(name), "bailing: missing tensor {name}");
    }
    Ok(())
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
