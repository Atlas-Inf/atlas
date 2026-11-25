// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end driver parity vs the torch oracle (`scripts_debug/
//! ling_e2e_oracle.py`): the REAL loader path (WeightStore upload ->
//! `BailingWeightLoader::load_layers`, per-`alloc_state` sequence state)
//! driven through the eager prefill/decode composition, with the residual
//! stream snapshotted between every layer and diffed against the oracle's
//! per-layer hidden states.
//!
//! Run (f341):
//! ```text
//! ATLAS_LING_CKPT=<ckpt> ATLAS_LING_E2E_ORACLE=<oracle dir> \
//! ATLAS_LING_E2E_TOKENS=... ATLAS_LING_E2E_PROMPT=<P> \
//!   cargo test -p spark-model --lib bailing_c1::mla::e2e_gpu_parity_tests \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Env:
//! * `ATLAS_LING_E2E_ORACLE`: dir holding `oracle_pre.f32 [43, P, H]`,
//!   `oracle_dec.f32 [D, 43, H]` (+ oracle_meta.json for provenance).
//! * `ATLAS_LING_E2E_TOKENS`: comma-separated token ids, `P + D` of them
//!   (prompt first). Must match the oracle run's list.
//! * `ATLAS_LING_E2E_PROMPT`: `P` (default 16).
//!
//! The verdict line names the first layer whose relative error jumps
//! (>4x the previous layer's, or above the absolute floor) — with a healthy
//! bf16 stack the drift grows smoothly layer over layer, so a jump IS the
//! composition bug's address.

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use spark_runtime::buffers::BufferArena;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::weights::SafetensorsLoader;

use crate::layer::{ForwardContext, LayerState, MoeLoraRoute, TransformerLayer};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{DenseWeight, dense};

use super::LingKernels;
use super::LingScratch;

const H: usize = 2560;
const LAYERS: usize = 42;
const ATTN_LAYERS: usize = 7; // gated-MLA layers
const BS: usize = 8; // probe block size
/// The pool the rig builds: covers the run's positions plus headroom,
/// so a longer oracle prompt rides the same rig as the 16-token table.
fn n_blocks(max_pos: usize) -> usize {
    max_pos.div_ceil(BS) + 1
}

fn ckpt() -> Result<PathBuf> {
    std::env::var("ATLAS_LING_CKPT")
        .map(Into::into)
        .context("ATLAS_LING_CKPT=<Ling-3.2-air-NVFP4 dir>")
}

fn oracle_dir() -> Result<PathBuf> {
    std::env::var("ATLAS_LING_E2E_ORACLE")
        .map(Into::into)
        .context("ATLAS_LING_E2E_ORACLE=<dir with oracle_pre/oracle_dec>")
}

fn tokens_and_prompt() -> Result<(Vec<u32>, usize)> {
    let prompt: usize = std::env::var("ATLAS_LING_E2E_PROMPT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let tokens: Vec<u32> = std::env::var("ATLAS_LING_E2E_TOKENS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter(|s| !s.trim().is_empty())
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            vec![
                25951, 16278, 113926, 21053, 94766, 1061, 97529, 15953, 128054, 109169, 137868,
                60077, 69809, 26636, 133761, 41880, 40020, 45362, 38145, 33557,
            ]
        });
    ensure!(
        tokens.len() > prompt && tokens.len() - prompt <= 8,
        "tokens: {} with prompt {} (want P < total <= P+8)",
        tokens.len(),
        prompt
    );
    Ok((tokens, prompt))
}

fn backend() -> Result<&'static AtlasCudaBackend> {
    static GPU: std::sync::OnceLock<Result<AtlasCudaBackend, String>> = std::sync::OnceLock::new();
    let gpu = GPU.get_or_init(|| {
        let set = atlas_kernels::ptx_for_exact_target("ling-3.2-air", "nvfp4")
            .ok_or_else(|| "ling-3.2-air/nvfp4 absent from this build".to_string())?;
        AtlasCudaBackend::new(0, &set.modules).map_err(|e| format!("CUDA backend: {e}"))
    });
    let gpu = gpu.as_ref().map_err(|e| anyhow::anyhow!(e.clone()))?;
    gpu.bind_to_thread()?;
    Ok(gpu)
}

fn model_config() -> Result<atlas_core::config::ModelConfig> {
    let json = std::fs::read_to_string(ckpt()?.join("config.json")).context("config.json read")?;
    atlas_core::config::parse_config(&json).map_err(|e| anyhow::anyhow!("parse_config: {e:#}"))
}

/// Read `count` f32 values from a raw little-endian file.
fn read_f32(path: &std::path::Path, count: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() >= count * 4,
        "{}: {} bytes, want {} f32",
        path.display(),
        bytes.len(),
        count
    );
    Ok(bytes
        .chunks_exact(4)
        .take(count)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Per-layer accumulators: max abs diff, max rel diff (den floored at 1e-2),
/// 1 - cos similarity (summed over rows), over the row set.
#[derive(Default, Clone, Copy)]
struct Diff {
    max_abs: f32,
    max_rel: f32,
    one_minus_cos: f32,
}

impl Diff {
    fn add(&mut self, got: &[f32], want: &[f32]) {
        let mut dot = 0f64;
        let mut ng = 0f64;
        let mut nw = 0f64;
        for (g, w) in got.iter().zip(want) {
            let d = (g - w).abs();
            self.max_abs = self.max_abs.max(d);
            self.max_rel = self.max_rel.max(d / w.abs().max(1e-2));
            dot += *g as f64 * *w as f64;
            ng += *g as f64 * *g as f64;
            nw += *w as f64 * *w as f64;
        }
        let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-30);
        self.one_minus_cos = self.one_minus_cos.max((1.0 - cos) as f32);
    }
}

/// First layer whose rel error jumps over its predecessor's.
fn verdict(rows: &[(usize, Diff)]) -> Option<usize> {
    let mut prev_rel = rows.first().map(|(_, d)| d.max_rel).unwrap_or(0.0);
    for &(layer, d) in rows.iter().skip(1) {
        if d.max_rel > (prev_rel * 4.0).max(0.05) || d.max_abs > 1.0 {
            return Some(layer);
        }
        prev_rel = prev_rel.max(d.max_rel);
    }
    None
}

/// Per-row diff columns (prefill): which prompt row drives the error.
fn row_diffs(got: &[Vec<f32>], layer: usize, prompt: usize, oracle: &[f32]) -> Vec<Diff> {
    (0..prompt)
        .map(|t| {
            let mut d = Diff::default();
            let base = (layer * prompt + t) * H;
            d.add(&got[t], &oracle[base..base + H]);
            d
        })
        .collect()
}

fn print_table(title: &str, rows: &[(usize, Diff)]) {
    eprintln!("== {title}: per-layer residual diff vs oracle ==");
    for &(layer, d) in rows {
        eprintln!(
            "  layer {layer:>2}: max_abs {:+.5}  max_rel {:.5}  1-cos {:.5}",
            d.max_abs, d.max_rel, d.one_minus_cos
        );
    }
    match verdict(rows) {
        Some(first) => eprintln!("  VERDICT {title}: FIRST DIVERGENCE at layer {first}"),
        None => eprintln!("  VERDICT {title}: no divergence jump detected"),
    }
}

#[test]
#[ignore = "GB10 + ling-3.2-air kernel build + ATLAS_LING_CKPT + ATLAS_LING_E2E_ORACLE"]
fn bailing_e2e_oracle_parity() -> Result<()> {
    let gpu = backend()?;
    let config = model_config()?;
    ensure!(
        config.num_hidden_layers == LAYERS
            && config.hidden_size == H
            && config.num_attention_layers() == ATTN_LAYERS,
        "unexpected geometry: {} layers x hidden {} ({} attention)",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_layers()
    );

    // ---- weights: the real loader path ------------------------------------
    let mut loader_disk = SafetensorsLoader::new();
    loader_disk.skip_prefixes = vec!["model.visual.".into(), "linear_proj.".into()];
    let store = {
        use spark_runtime::weights::WeightLoader;
        loader_disk.load(ckpt()?.as_path(), gpu, 8 << 30)?
    };
    let loader = crate::factory::loader_for_config(&config)?;
    let kv_dtypes = vec![KvCacheDtype::Bf16; ATTN_LAYERS];
    let layers = loader.load_layers(&store, &config, gpu, &kv_dtypes)?;
    ensure!(layers.len() == LAYERS, "loaded {} layers", layers.len());
    let embed = loader.load_embedding(&store, &config, gpu)?;
    let _final_norm = loader.load_final_norm(&store, &config, gpu)?;

    // ---- per-sequence state: alloc through the layers themselves ----------
    let mut states: Vec<Box<dyn LayerState>> = layers
        .iter()
        .map(|l| l.alloc_state(gpu))
        .collect::<Result<Vec<_>>>()?;

    // ---- pools / arena / context ------------------------------------------
    let (tokens, prompt) = tokens_and_prompt()?;
    let decode_n = tokens.len() - prompt;
    let max_pos = prompt + decode_n;
    let kv_blocks = n_blocks(max_pos);
    let mut kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: BS,
            num_kv_heads: 1,
            head_dim: 576, // MLA K row: [kv_lora 512 | rope 64]
            num_layers: ATTN_LAYERS,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        kv_blocks,
        gpu,
    )?;
    let block_table: Vec<u32> = (0..kv_blocks as u32).collect();
    let disk_block_ids: Vec<u32> = Vec::new();
    let disk_last_offloaded: Vec<u32> = Vec::new();
    ensure!(
        kv_blocks * BS >= max_pos,
        "block table covers only {} positions, want {max_pos}",
        kv_blocks * BS
    );

    let arena = BufferArena::new(&config, 64, 64, BS, 1, gpu)?;
    let dispatch = GemmDispatch::default();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &arena,
        gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        ssm_batch: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Skip,
    };
    let stream = gpu.default_stream();

    let hidden = gpu.alloc(prompt * H * 2)?;
    let residual = gpu.alloc(prompt * H * 2)?; // Ling layers carry the residual inline
    let mut scratch_row = vec![0u8; H * 2];

    // ---- oracle -------------------------------------------------------------
    let odir = oracle_dir()?;
    let pre_oracle = read_f32(&odir.join("oracle_pre.f32"), (LAYERS + 1) * prompt * H)?;
    let dec_oracle = read_f32(&odir.join("oracle_dec.f32"), decode_n * (LAYERS + 1) * H)?;
    ensure!(pre_oracle.len() == (LAYERS + 1) * prompt * H);

    let embed_row = |t: u32| -> Result<Vec<u16>> {
        let mut bytes = vec![0u8; H * 2];
        gpu.copy_d2h(embed.weight.offset(t as usize * H * 2), &mut bytes)?;
        bytes
            .chunks_exact(2)
            .map(|c| Ok(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };
    let bf_to_f32 = |v: &[u16]| -> Vec<f32> {
        v.iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect()
    };

    // ---------------- PHASE P: prompt prefill through all layers -------------
    // `ATLAS_LING_E2E_SERIAL=1`: one-row prefill calls instead of the batched
    // `num_tokens = P` call. Semantics must be identical (KDA state carries
    // across calls; MLA writes KV at `seq_len_start` and attends over the
    // paged pool) — a divergence that appears only batched isolates the
    // bug to a multi-row kernel arm, not the composition.
    let serial = std::env::var("ATLAS_LING_E2E_SERIAL").as_deref() == Ok("1");
    let mut pre_rows: Vec<Vec<f32>> = vec![vec![]; prompt];
    for (t, id) in tokens[..prompt].iter().enumerate() {
        let row = embed_row(*id)?;
        let f = bf_to_f32(&row);
        pre_rows[t] = f;
        gpu.copy_h2d(
            unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, H * 2) },
            hidden.offset(t * H * 2),
        )?;
    }

    let mut diffs: Vec<(usize, Diff)> = Vec::with_capacity(LAYERS + 1);
    let mut snapshot =
        |gpu: &dyn GpuBackend, hidden: DevicePtr, prompt: usize| -> Result<Vec<Vec<f32>>> {
            let mut out = vec![vec![]; prompt];
            for t in 0..prompt {
                gpu.copy_d2h(hidden.offset(t * H * 2), unsafe {
                    std::slice::from_raw_parts_mut(scratch_row.as_mut_ptr(), H * 2)
                })?;
                let row: Vec<u16> = scratch_row
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                out[t] = bf_to_f32(&row);
            }
            Ok(out)
        };

    // index 0: embed rows
    {
        let mut d = Diff::default();
        for t in 0..prompt {
            let want = &pre_oracle[t * H..(t + 1) * H];
            d.add(&pre_rows[t], want);
        }
        diffs.push((0usize, d));
    }
    for (i, layer) in layers.iter().enumerate() {
        if serial {
            for t in 0..prompt {
                layer.prefill(
                    hidden.offset(t * H * 2),
                    residual.offset(t * H * 2),
                    1,
                    states[i].as_mut(),
                    &mut kv,
                    t,
                    &mut block_table.clone(),
                    &mut disk_block_ids.clone(),
                    &mut disk_last_offloaded.clone(),
                    0,
                    &ctx,
                    stream,
                )?;
            }
        } else {
            layer.prefill(
                hidden,
                residual,
                prompt,
                states[i].as_mut(),
                &mut kv,
                0,
                &mut block_table.clone(),
                &mut disk_block_ids.clone(),
                &mut disk_last_offloaded.clone(),
                0,
                &ctx,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        let got = snapshot(gpu, hidden, prompt)?;
        let mut d = Diff::default();
        for t in 0..prompt {
            let base = ((i + 1) * prompt + t) * H;
            let want = &pre_oracle[base..base + H];
            d.add(&got[t], want);
        }
        // Per-row breakdown: which prompt row carries the error.
        let rd = row_diffs(&got, i + 1, prompt, &pre_oracle);
        eprintln!(
            "  layer {i:>2} per-row: {}",
            rd.iter()
                .enumerate()
                .map(|(t, r)| format!("r{t} abs {:.4} cos {:.4}", r.max_abs, r.one_minus_cos))
                .collect::<Vec<_>>()
                .join(" | ")
        );
        diffs.push((i + 1, d));
    }
    print_table("prefill", &diffs);

    // ---------------- PHASE D: decode tokens, one at a time ------------------
    for step in 0..decode_n {
        let id = tokens[prompt + step];
        let row = embed_row(id)?;
        gpu.copy_h2d(
            unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, H * 2) },
            hidden,
        )?;
        let seq_len = prompt + step; // positions 0..seq_len already cached
        let mut diffs_d: Vec<(usize, Diff)> = Vec::with_capacity(LAYERS + 1);
        {
            let mut d = Diff::default();
            let f = bf_to_f32(&row);
            let base = (step * (LAYERS + 1)) * H;
            d.add(&f, &dec_oracle[base..base + H]);
            diffs_d.push((0usize, d));
        }
        for (i, layer) in layers.iter().enumerate() {
            layer.decode(
                hidden,
                residual,
                states[i].as_mut(),
                &mut kv,
                seq_len,
                &mut block_table.clone(),
                &mut disk_block_ids.clone(),
                &mut disk_last_offloaded.clone(),
                &ctx,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let got = snapshot(gpu, hidden, 1)?;
            let mut d = Diff::default();
            let base = (step * (LAYERS + 1) + (i + 1)) * H;
            d.add(&got[0], &dec_oracle[base..base + H]);
            diffs_d.push((i + 1, d));
        }
        print_table(&format!("decode step {step} (pos {seq_len})"), &diffs_d);
    }

    eprintln!("e2e oracle parity: harness complete; see tables above");
    Ok(())
}

/// Lane-level trace of ONE KDA layer's prefill against the oracle's stage
/// dump (`trace_<L>_<stage>.f32` from the oracle's `ATLAS_LING_E2E_TRACE_
/// LAYER`). Rebuilds layer `L` exactly as `BailingWeightLoader::load_layers`
/// does (same weight names, same kernels) but over a locally-owned
/// `LingScratch`, so every intermediate lane is readable after the prefill.
/// The first mismatching stage names the op — and the file to inspect.
#[test]
#[ignore = "GB10 + ling-3.2-air kernel build + ATLAS_LING_CKPT + ATLAS_LING_E2E_ORACLE"]
fn bailing_e2e_layer_trace() -> Result<()> {
    use crate::model::bailing_c1::{LingDenseWeights, LingFfn, LingKdaLayer, LingKdaWeights};

    const T: usize = 32 * 128;
    const INTER: usize = 6144;

    let gpu = backend()?;
    let config = model_config()?;
    let mut loader_disk = SafetensorsLoader::new();
    loader_disk.skip_prefixes = vec!["model.visual.".into(), "linear_proj.".into()];
    let store = {
        use spark_runtime::weights::WeightLoader;
        loader_disk.load(ckpt()?.as_path(), gpu, 8 << 30)?
    };
    let loader = crate::factory::loader_for_config(&config)?;
    let embed = loader.load_embedding(&store, &config, gpu)?;

    let layer: usize = std::env::var("ATLAS_LING_E2E_TRACE_LAYER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    ensure!(layer <= 4, "trace: layers 0-4 are the KDA+dense-MLP prefix");
    let lp = format!("model.layers.{layer}");
    let attn = |stem: &str| format!("{lp}.attention.{stem}.weight");

    // ---- layer L weights, exactly as `load_kda_weights` maps them ---------
    // NOTE: the buggy prefill-conv weight offset in kda.rs cannot be
    // compensated from here — the mis-read tap regions OVERLAP the correct
    // ones (plane 1 reads start inside plane 0's taps), so no blob layout
    // satisfies both launches. The wrong-offset emulation against the
    // oracle trace (see scripts_debug/ling_e2e_trace_analysis.py) is the
    // proof; the fix belongs in kda.rs:166.
    let conv_blob = {
        let bytes = 3 * T * 4 * 2;
        let p = gpu.alloc(bytes)?;
        for (plane, stem) in ["q_conv1d", "k_conv1d", "v_conv1d"].iter().enumerate() {
            let t = store.get(&attn(stem))?;
            ensure!(
                matches!(t.dtype, spark_runtime::weights::WeightDtype::BF16)
                    && t.num_elements() == T * 4,
                "trace: {stem} must be BF16 [T x 4]"
            );
            gpu.copy_d2d(t.ptr, p.offset(plane * T * 4 * 2), T * 4 * 2)?;
        }
        p
    };
    let f32_raw = |name: &str, elems: usize| -> Result<DevicePtr> {
        let t = store.get(name)?;
        ensure!(
            matches!(t.dtype, spark_runtime::weights::WeightDtype::FP32)
                && t.num_elements() == elems,
            "trace: {name} must be FP32 x {elems}"
        );
        Ok(t.ptr)
    };
    let kernels = std::sync::Arc::new(LingKernels::resolve(gpu)?);
    let scratch = LingScratch::arc(gpu, 64, H, 512, 8, INTER.max(768), 8)?;
    let weights = LingKdaWeights {
        q_proj: dense(&store, &attn("q_proj"))?,
        k_proj: dense(&store, &attn("k_proj"))?,
        v_proj: dense(&store, &attn("v_proj"))?,
        b_proj: dense(&store, &attn("b_proj"))?,
        f_proj: dense(&store, &attn("f_proj"))?,
        g_proj: dense(&store, &attn("g_proj"))?,
        o_proj: dense(&store, &attn("o_proj"))?,
        conv: DenseWeight { weight: conv_blob },
        a_log: f32_raw(&format!("{lp}.attention.A_log"), 32)?,
        dt_bias: f32_raw(&format!("{lp}.attention.dt_bias"), T)?,
        o_norm: dense(&store, &attn("o_norm"))?,
    };
    let ffn = LingFfn::dense(
        LingDenseWeights {
            gate: dense(&store, &format!("{lp}.mlp.gate_proj.weight"))?,
            up: dense(&store, &format!("{lp}.mlp.up_proj.weight"))?,
            down: dense(&store, &format!("{lp}.mlp.down_proj.weight"))?,
        },
        INTER,
        H,
        std::sync::Arc::clone(&scratch),
        std::sync::Arc::clone(&kernels),
    )?;
    let ssm_ordinal = (0..layer)
        .filter(|i| config.layer_type(*i) == atlas_core::config::LayerType::LinearAttention)
        .count();
    let kda = LingKdaLayer::new(
        dense(&store, &format!("{lp}.input_layernorm.weight"))?,
        dense(&store, &format!("{lp}.post_attention_layernorm.weight"))?,
        weights,
        ffn,
        std::sync::Arc::clone(&kernels),
        std::sync::Arc::clone(&scratch),
        ssm_ordinal,
        config.kda_gate_lower_bound,
        config.rms_norm_eps as f32,
    )?;

    // ---- context ------------------------------------------------------------
    let (tokens, prompt) = tokens_and_prompt()?;
    let kv_blocks = n_blocks(tokens.len());
    let mut kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: BS,
            num_kv_heads: 1,
            head_dim: 576,
            num_layers: ATTN_LAYERS,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        kv_blocks,
        gpu,
    )?;
    let block_table: Vec<u32> = (0..kv_blocks as u32).collect();
    let arena = BufferArena::new(&config, 64, 64, BS, 1, gpu)?;
    let dispatch = GemmDispatch::default();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &arena,
        gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        ssm_batch: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Skip,
    };
    let stream = gpu.default_stream();

    let mut state = kda.alloc_state(gpu)?; // fresh conv + h states
    let hidden = gpu.alloc(prompt.max(2) * H * 2)?;
    let residual = gpu.alloc(prompt.max(2) * H * 2)?;

    // embed rows into hidden
    for (t, id) in tokens[..prompt].iter().enumerate() {
        let mut src = vec![0u8; H * 2];
        gpu.copy_d2h(embed.weight.offset(*id as usize * H * 2), &mut src)?;
        gpu.copy_h2d(&src, hidden.offset(t * H * 2))?;
    }

    // Serial knob: one-row prefill calls (the decode-arm kernels; state
    // carries across calls). Lanes are dumped after EACH row so the
    // per-row trace pins which op of row t first diverges.
    let odir = oracle_dir()?;
    let lane_bf = |ptr: DevicePtr, elems: usize| -> Result<Vec<f32>> {
        let mut b = vec![0u8; elems * 2];
        gpu.copy_d2h(ptr, &mut b)?;
        Ok(b.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect())
    };
    let serial = std::env::var("ATLAS_LING_E2E_SERIAL").as_deref() == Ok("1");
    if serial {
        for t in 0..prompt {
            kda.prefill(
                hidden.offset(t * H * 2),
                residual.offset(t * H * 2),
                1,
                state.as_mut(),
                &mut kv,
                t,
                &mut block_table.clone(),
                &mut Vec::new(),
                &mut Vec::new(),
                0,
                &ctx,
                stream,
            )?;
            gpu.synchronize(stream)?;
            // h_state dump: the recurrent carried state, for offline per-head
            // comparison against the fp32 oracle state.
            {
                let ssm = state
                    .as_mut()
                    .as_any_mut()
                    .downcast_mut::<crate::layer::SsmLayerState>()
                    .expect("ling KDA state");
                let elems = 32 * 128 * 128;
                let mut b = vec![0u8; elems * 4];
                gpu.copy_d2h(ssm.h_state, &mut b)?;
                std::fs::write(odir.join(format!("atlas_lane_{layer}_hstate_t{t}.f32")), &b)?;
            }
            let s: &LingScratch = &scratch;
            let check_row = |stage: &str, ptr: DevicePtr, width_row: usize| -> Result<()> {
                let got = lane_bf(ptr, width_row)?;
                // raw dump for offline state forensics
                let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(
                    odir.join(format!("atlas_lane_{layer}_{stage}_t{t}.f32")),
                    &bytes,
                )?;
                let want = read_f32(
                    &odir.join(format!("trace_{layer}_{stage}.f32")),
                    width_row * prompt,
                )?;
                let mut d = Diff::default();
                let base = t * width_row;
                d.add(&got, &want[base..base + width_row]);
                eprintln!(
                    "  serial row {t} stage {stage:>10}: max_abs {:+.6}  1-cos {:.6}",
                    d.max_abs, d.one_minus_cos
                );
                Ok(())
            };
            check_row("normed", s.normed, H)?;
            check_row("qkv", s.kda_qkv, 3 * T)?;
            check_row("f", s.kda_gate, T)?;
            check_row("g", s.kda_o_g, T)?;
            check_row("b_raw", s.kda_beta, 32)?;
            check_row("kda_out", s.kda_out, T)?;
            check_row("kda_o", s.kda_o, T)?;
            check_row("attn_out", s.attn_out, H)?;
            check_row("ffn_in", s.ffn_in, H)?;
            check_row("shared_h", s.shared_h, INTER)?;
            check_row("ffn_out", s.ffn_out, H)?;
        }
        eprintln!("layer {layer} serial trace complete");
        return Ok(());
    }

    kda.prefill(
        hidden,
        residual,
        prompt,
        state.as_mut(),
        &mut kv,
        0,
        &mut block_table.clone(),
        &mut Vec::new(),
        &mut Vec::new(),
        0,
        &ctx,
        stream,
    )?;
    gpu.synchronize(stream)?;

    // ---- the lane snapshot ---------------------------------------------------
    // Raw lane dumps for offline structural analysis (layout forensics).
    let dump_lane = |got: &[f32], stage: &str| -> Result<()> {
        let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(odir.join(format!("atlas_lane_{layer}_{stage}.f32")), &bytes)?;
        Ok(())
    };
    let check_stage = |got: Vec<f32>, stage: &str| -> Result<()> {
        dump_lane(&got, stage)?;
        let want = read_f32(&odir.join(format!("trace_{layer}_{stage}.f32")), got.len())?;
        let mut d = Diff::default();
        d.add(&got, &want);
        eprintln!(
            "  stage {stage:>10}: max_abs {:+.6}  max_rel {:.6}  1-cos {:.6}",
            d.max_abs, d.max_rel, d.one_minus_cos
        );
        Ok(())
    };

    eprintln!("== layer {layer} trace: got vs oracle (rows = prompt) ==");
    let s: &LingScratch = &scratch;
    // `normed` last written by the attention half's input rms_norm.
    check_stage(lane_bf(s.normed, prompt * H)?, "normed")?;
    // `preconv`: plane-major [3 planes][prompt][T] per the strided gemm.
    check_stage(lane_bf(s.kda_preconv, 3 * prompt * T)?, "preconv")?;
    check_stage(lane_bf(s.kda_qkv, prompt * 3 * T)?, "qkv")?;
    check_stage(lane_bf(s.kda_gate, prompt * T)?, "f")?;
    check_stage(lane_bf(s.kda_o_g, prompt * T)?, "g")?;
    check_stage(lane_bf(s.kda_beta, prompt * 32)?, "b_raw")?;
    check_stage(lane_bf(s.kda_out, prompt * T)?, "kda_out")?;
    check_stage(lane_bf(s.kda_o, prompt * T)?, "kda_o")?;
    check_stage(lane_bf(s.attn_out, prompt * H)?, "attn_out")?;
    check_stage(lane_bf(s.ffn_in, prompt * H)?, "ffn_in")?;
    check_stage(lane_bf(s.shared_h, prompt * INTER)?, "shared_h")?;
    check_stage(lane_bf(s.ffn_out, prompt * H)?, "ffn_out")?;
    eprintln!("layer {layer} trace complete");
    Ok(())
}
