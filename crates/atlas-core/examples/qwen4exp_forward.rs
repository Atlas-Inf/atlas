// SPDX-License-Identifier: AGPL-3.0-only

//! A full `qwen4_exp` forward pass on CPU, from a checkpoint on disk.
//!
//! ```text
//! cargo run --release -p atlas-core --example qwen4exp_forward -- <ckpt> [fixture.json]
//! ```
//!
//! Not the serving path — this is slow and single-threaded. It exists so the
//! architecture can be run and checked end to end before any GPU layer is
//! written, and so a GPU layer has something to disagree with.
//!
//! Every block it calls is already checked against HuggingFace individually
//! (`qwen4exp_reference`); this adds the wiring, which is where the shapes stop
//! being obvious: the residual stream is `hc_count * hidden` wide throughout,
//! each block collapses it to `hidden`, and the result is broadcast back with
//! per-stream injection gains.

use anyhow::{Context, Result};
use atlas_core::config::{LayerType, ModelConfig, parse_config};
use atlas_core::ngram_table::NgramTable;
use atlas_core::qwen4exp_reference::*;
use atlas_core::weight_manifest::{TensorLocation, locate_checkpoint};
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;

struct Store {
    located: BTreeMap<String, TensorLocation>,
    cache: std::cell::RefCell<BTreeMap<String, std::rc::Rc<Vec<f32>>>>,
}

impl Store {
    fn get(&self, name: &str) -> Result<std::rc::Rc<Vec<f32>>> {
        if let Some(hit) = self.cache.borrow().get(name) {
            return Ok(hit.clone());
        }
        let loc = self
            .located
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        let file = std::fs::File::open(&loc.path)?;
        let mut raw = vec![0u8; loc.span.len as usize];
        file.read_exact_at(&mut raw, loc.span.abs_offset)?;
        let values: Vec<f32> = match loc.dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F8_E4M3" => raw
                .iter()
                .map(|b| atlas_core::numeric::fp8_e4m3_to_f32(*b))
                .collect(),
            other => anyhow::bail!("{name}: no f32 path for dtype {other}"),
        };
        let shared = std::rc::Rc::new(values);
        self.cache
            .borrow_mut()
            .insert(name.to_string(), shared.clone());
        Ok(shared)
    }
}

fn hc_weights<'a>(held: &'a [std::rc::Rc<Vec<f32>>], inject: bool) -> HyperConnectionWeights<'a> {
    HyperConnectionWeights {
        hc_norm: &held[0],
        mix_down: &held[1],
        mix_up: &held[2],
        block_inject: inject.then(|| held[3].as_slice()),
    }
}

fn load_hc(store: &Store, prefix: &str, inject: bool) -> Result<Vec<std::rc::Rc<Vec<f32>>>> {
    let mut held = vec![
        store.get(&format!("{prefix}.hc_norm.weight"))?,
        store.get(&format!("{prefix}.input_mix_weight_down.weight"))?,
        store.get(&format!("{prefix}.input_mix_weight_up.weight"))?,
    ];
    if inject {
        held.push(store.get(&format!("{prefix}.block_inject_weight.weight"))?);
    }
    Ok(held)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().context("usage: <ckpt> [fixture.json]")?);
    let fixture_path = args.next();

    let config: ModelConfig = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let store = Store {
        located: locate_checkpoint(&dir)?,
        cache: std::cell::RefCell::new(BTreeMap::new()),
    };
    let ids: Vec<u32> = match &fixture_path {
        Some(path) => {
            let f: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            f["input_ids"]
                .as_array()
                .context("fixture input_ids")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as u32)
                .collect()
        }
        None => vec![11, 42, 7, 300, 5],
    };
    let generate: usize = std::env::args()
        .position(|a| a == "--generate")
        .and_then(|_| {
            let all: Vec<String> = std::env::args().collect();
            all.iter()
                .position(|a| a == "--generate")
                .and_then(|i| all.get(i + 1).cloned())
        })
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut ids = ids;
    let started = std::time::Instant::now();
    for step in 0..=generate {
        eprintln!(
            "forward: {} tokens, {} layers  (step {step}/{generate}, {:.1?} elapsed)",
            ids.len(),
            config.num_hidden_layers,
            started.elapsed()
        );
        let logits = forward(&dir, &config, &store, &ids)?;
        let vocab = config.vocab_size;
        let argmax: Vec<usize> = (0..ids.len())
            .map(|t| {
                let row = &logits[t * vocab..(t + 1) * vocab];
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        if generate == 0 {
            println!("argmax per position: {argmax:?}");
            if let Some(path) = &fixture_path {
                let f: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
                let want: Vec<f32> = f["logits"]
                    .as_array()
                    .context("fixture logits")?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                let worst = logits
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                println!("HF argmax          : {:?}", f["argmax"]);
                println!(
                    "max|diff| {worst:.3e}  (logits up to {scale:.3e})  relative {:.3e}",
                    worst / scale.max(1e-9)
                );
            }
            return Ok(());
        }

        let next = *argmax.last().context("no logits")? as u32;
        eprintln!("  -> {next}");
        ids.push(next);
        // Emit after every step so a long run is watchable.
        println!("{}", serde_json::json!({ "ids": ids }));
    }
    Ok(())
}

/// One full forward. Returns logits `[seq, vocab]`.
fn forward(
    dir: &std::path::Path,
    config: &ModelConfig,
    store: &Store,
    ids: &[u32],
) -> Result<Vec<f32>> {
    let (h, hc) = (config.hidden_size, config.hc_count);
    let wide = h * hc;
    let lm = "model.language_model";
    let seq = ids.len();

    // Embedding, tiled across the hyper-connection streams.
    let embed = store.get(&format!("{lm}.embed_tokens.weight"))?;
    let mut hidden = vec![0f32; seq * wide];
    for (t, id) in ids.iter().enumerate() {
        let row = &embed[*id as usize * h..(*id as usize + 1) * h];
        for stream in 0..hc {
            hidden[t * wide + stream * h..t * wide + (stream + 1) * h].copy_from_slice(row);
        }
    }

    let rotary_dim = (config.head_dim as f64 * config.partial_rotary_factor) as usize;
    let (cos, sin) = rope_tables(seq, rotary_dim, config.rope_theta as f32);
    let eps = config.rms_norm_eps as f32;
    let ple_dims = PleDims {
        hidden: h,
        hc_count: hc,
        ple_embed_dim: config.ple_embed_dim,
        kernel: config.ple_conv_kernel_size,
        dilation: config.ngram_size,
        eps,
    };

    for layer in 0..config.num_hidden_layers {
        let base = format!("{lm}.layers.{layer}");

        // PLE, where the (one-indexed) ple_layer_ids puts it.
        if let Some(index) = config.ple_layer_ids.iter().position(|id| *id == layer + 1) {
            let ngram = config.qwen4exp_ngram(index)?.context("ngram geometry")?;
            let table = NgramTable::open(dir, config, index)?;
            let scale = store
                .get(&format!(
                    "{base}.ple.ple_embedding.ngram_embedding.weight_scale"
                ))
                .map(|s| s[0])
                .unwrap_or(1.0);
            // The reference seeds ngram_size-1 EOS tokens of carried context.
            let carry = config.ngram_size - 1;
            let mut stream: Vec<u32> = vec![config.eos_token_id; carry];
            stream.extend_from_slice(ids);
            let row_ids = ngram.ngram_ids(config.ngram_vocab_size_base, &stream);
            let heads = row_ids.len();
            let mut embeddings = vec![0f32; seq * heads * table.head_dim()];
            for t in 0..seq {
                let picks: Vec<u32> = row_ids.iter().map(|head| head[t + carry]).collect();
                let span = t * heads * table.head_dim()..(t + 1) * heads * table.head_dim();
                table.gather_dequant(&picks, scale, &mut embeddings[span])?;
            }
            let p = format!("{base}.ple");
            let held: Vec<_> = [
                "conv1d.weight",
                "key_proj.weight",
                "value_proj.weight",
                "norm_conv.weight",
                "norm_key.weight",
                "norm_query.weight",
            ]
            .iter()
            .map(|n| store.get(&format!("{p}.{n}")))
            .collect::<Result<_>>()?;
            let pw = PleWeights {
                conv1d: &held[0],
                key_proj: &held[1],
                value_proj: &held[2],
                norm_conv: &held[3],
                norm_key: &held[4],
                norm_query: &held[5],
            };
            let ple_out = ple_forward(&ple_dims, &pw, &embeddings, &hidden);
            for (slot, value) in hidden.iter_mut().zip(ple_out) {
                *slot += value;
            }
        }

        for (which, is_attn) in [
            ("attn_hyper_connection", true),
            ("mlp_hyper_connection", false),
        ] {
            let held = load_hc(store, &format!("{base}.{which}"), true)?;
            let hcw = hc_weights(&held, true);

            // Collapse every position, run the block, broadcast back.
            let mut mixed_all = vec![0f32; seq * h];
            let mut injects = vec![0f32; seq * hc];
            for t in 0..seq {
                let out = hyper_connection_forward(
                    &ple_dims,
                    &hcw,
                    config.hc_lowrank,
                    &hidden[t * wide..(t + 1) * wide],
                );
                mixed_all[t * h..(t + 1) * h].copy_from_slice(&out.mixed);
                injects[t * hc..(t + 1) * hc].copy_from_slice(&out.injection);
            }

            let block_out = if is_attn {
                match config.layer_types[layer] {
                    LayerType::FullAttention => {
                        let p = format!("{base}.self_attn");
                        let held: Vec<_> = [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "o_proj.weight",
                            "q_norm.weight",
                            "k_norm.weight",
                        ]
                        .iter()
                        .map(|n| store.get(&format!("{p}.{n}")))
                        .collect::<Result<_>>()?;
                        attention_forward(
                            &AttnDims {
                                hidden: h,
                                num_heads: config.num_attention_heads,
                                num_kv_heads: config.num_key_value_heads,
                                head_dim: config.head_dim,
                                rotary_dim,
                                eps,
                            },
                            &AttnWeights {
                                q_proj: &held[0],
                                k_proj: &held[1],
                                v_proj: &held[2],
                                o_proj: &held[3],
                                q_norm: &held[4],
                                k_norm: &held[5],
                            },
                            &mixed_all,
                            &cos,
                            &sin,
                        )
                    }
                    LayerType::LinearAttention => {
                        let p = format!("{base}.linear_attn");
                        let held: Vec<_> = [
                            "in_proj_qkv.weight",
                            "in_proj_z.weight",
                            "in_proj_a.weight",
                            "in_proj_b.weight",
                            "conv1d.weight",
                            "A_log",
                            "dt_bias",
                            "norm.weight",
                            "out_proj.weight",
                        ]
                        .iter()
                        .map(|n| store.get(&format!("{p}.{n}")))
                        .collect::<Result<_>>()?;
                        gdn_forward(
                            &GdnDims {
                                hidden: h,
                                num_k_heads: config.linear_num_key_heads,
                                key_head_dim: config.linear_key_head_dim,
                                num_v_heads: config.linear_num_value_heads,
                                value_head_dim: config.linear_value_head_dim,
                                conv_kernel: config.linear_conv_kernel_dim,
                                eps,
                                sigmoid_gate: config.output_gate_type == "sigmoid",
                            },
                            &GdnWeights {
                                in_proj_qkv: &held[0],
                                in_proj_z: &held[1],
                                in_proj_a: &held[2],
                                in_proj_b: &held[3],
                                conv1d: &held[4],
                                a_log: &held[5],
                                dt_bias: &held[6],
                                norm: &held[7],
                                out_proj: &held[8],
                            },
                            &mixed_all,
                        )
                    }
                    other => anyhow::bail!("unexpected layer type {other:?}"),
                }
            } else {
                let p = format!("{base}.mlp");
                let router = store.get(&format!("{p}.gate.weight"))?;
                let shared_gate = store.get(&format!("{p}.shared_expert_gate.weight"))?;
                let shared: Vec<_> = ["gate_proj", "up_proj", "down_proj"]
                    .iter()
                    .map(|n| store.get(&format!("{p}.shared_expert.{n}.weight")))
                    .collect::<Result<_>>()?;
                let dims = MoeDims {
                    hidden: h,
                    num_experts: config.num_experts,
                    top_k: config.num_experts_per_tok,
                    intermediate: config.moe_intermediate_size,
                    shared_intermediate: config.shared_expert_intermediate_size,
                    norm_topk_prob: config.norm_topk_prob,
                };
                let mw = MoeWeights {
                    router: &router,
                    shared_gate: &shared_gate,
                    shared_expert: [&shared[0], &shared[1], &shared[2]],
                };
                let mut out = vec![0f32; seq * h];
                for t in 0..seq {
                    // Fuse gate_up per routed expert on demand; a real loader
                    // would keep them fused, but on disk they are split.
                    let mut fused: BTreeMap<usize, (Vec<f32>, Vec<f32>)> = BTreeMap::new();
                    let x = &mixed_all[t * h..(t + 1) * h];
                    for (e, _) in moe_route(&dims, &router, x) {
                        let gate = store.get(&format!("{p}.experts.{e}.gate_proj.weight"))?;
                        let up = store.get(&format!("{p}.experts.{e}.up_proj.weight"))?;
                        let down = store.get(&format!("{p}.experts.{e}.down_proj.weight"))?;
                        let mut gate_up = gate.as_ref().clone();
                        gate_up.extend(up.iter());
                        fused.insert(e, (gate_up, down.as_ref().clone()));
                    }
                    let token = moe_forward(&dims, &mw, x, |e| {
                        fused.get(&e).map(|(gu, d)| (gu.as_slice(), d.as_slice()))
                    });
                    out[t * h..(t + 1) * h].copy_from_slice(&token);
                }
                out
            };

            for t in 0..seq {
                let scattered = broadcast_inject(
                    &block_out[t * h..(t + 1) * h],
                    &injects[t * hc..(t + 1) * hc],
                    h,
                );
                for (slot, value) in hidden[t * wide..(t + 1) * wide].iter_mut().zip(scattered) {
                    *slot += value;
                }
            }
        }
    }

    // No final norm: the trunk mixer is what normalises before the LM head.
    let held = load_hc(store, &format!("{lm}.hyper_connection_mixer"), false)?;
    let mixer = hc_weights(&held, false);
    let lm_head = store.get("lm_head.weight")?;
    let mut logits = vec![0f32; seq * config.vocab_size];
    for t in 0..seq {
        let out = hyper_connection_forward(
            &ple_dims,
            &mixer,
            config.hc_lowrank,
            &hidden[t * wide..(t + 1) * wide],
        );
        logits[t * config.vocab_size..(t + 1) * config.vocab_size].copy_from_slice(&linear(
            &out.mixed,
            &lm_head,
            config.vocab_size,
            h,
        ));
    }

    Ok(logits)
}
