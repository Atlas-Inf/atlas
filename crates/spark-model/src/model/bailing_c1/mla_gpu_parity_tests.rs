// SPDX-License-Identifier: AGPL-3.0-only

//! GPU parity probes (`#[ignore]`: a GB10, a `ling-3.2-air` kernel build and
//! the real NVFP4 checkpoint) of the gated-MLA half of layer 5 (the first
//! gated-MLA layer) and the head tail, against a naive f64 host reference
//! that mirrors `modeling_bailing_moe_v3_vl.py`
//! (`BailingMoeV3MultiLatentAttention` lines ~1101-1233, the interleave
//! rope helper at ~1047 and the norm/lm_head sections):
//!
//! - `mla_layer5_parity` (default; `ATLAS_LING_MLA_PARITY_LAYER` selects
//!   any gated-MLA layer of {5, 11, 17, 23, 29, 35, 41}, with its paged
//!   pool index): the paged launch path end to end — a 12-row prefill from
//!   position 0 (the paged kernel in an 8+4-row group pair, NOT the
//!   contiguous fast path), a 4-row continuation chunk at
//!   `seq_len_start = 12`, and one decode row at position 16 — every stage
//!   buffer (absorbed q, paged K/V rows, attention latent, gated V,
//!   o_proj, residual) checked against the reference at each phase.
//!   Validated on layer 5 (pool index 0) and layer 41 (pool index 6,
//!   95 stage checks each, all <= 2% rel, worst single stage 0.03 abs).
//! - `mla_layer5_prefill_fast_parity`: the same rows through the
//!   `ling_mla_prefill_attn` contiguous single-pass path (25 stage
//!   checks clean; worst single stage 0.031 abs on the residual row).
//! - `ling_head_lm_head_parity`: `model.norm` + `lm_head` gemv over the
//!   full 157,184-row BF16 head vs the reference, plus the embed/vocab
//!   shape sanity of `model.word_embeddings` (final norm exact; worst
//!   logit abs err 0.029 on an 8.6-scale argmax winner; the GPU and
//!   reference argmax agree at 100036).
//!
//! ```sh
//! ATLAS_LING_CKPT=<checkpoint-dir> cargo test --release -p spark-model --lib \
//!     bailing_c1::mla::mla_gpu_parity -- --ignored --nocapture --test-threads=1
//!
//! # Whole-sweep over all seven gated-MLA layers:
//! for L in 5 11 17 23 29 35 41; do ATLAS_LING_MLA_PARITY_LAYER=$L \
//!     cargo test --release -p spark-model --lib mla_layer5_parity -- --ignored \
//!     --exact --nocapture; done
//! ```

use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use atlas_core::numeric::f32_to_bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::*;
use crate::layers::ops;
use crate::weight_map::DenseWeight;
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;

// The Ling MLA geometry (see gated_mla.cu's header block).
const HID: usize = 2560;
const HEADS: usize = 32;
const NOPE: usize = 128;
const ROPE: usize = 64;
const LORA: usize = 512;
const VD: usize = 128;
const KVD: usize = 576;
const BS: usize = 8; // probe block size (the driver takes the pool's)

const ROPE_THETA: f64 = 6_000_000.0;
const EPS: f64 = 1e-6;

fn ckpt() -> Result<std::path::PathBuf> {
    std::env::var("ATLAS_LING_CKPT")
        .map(Into::into)
        .context("ATLAS_LING_CKPT=<Ling-3.2-air-NVFP4 dir>")
}

fn model_config() -> Result<ModelConfig> {
    static CFG: OnceLock<Result<ModelConfig, String>> = OnceLock::new();
    let cfg = CFG.get_or_init(|| {
        let dir = match ckpt() {
            Ok(d) => d,
            Err(e) => return Err(format!("{e:#}")),
        };
        let json = match std::fs::read_to_string(dir.join("config.json")) {
            Ok(j) => j,
            Err(e) => return Err(format!("config.json read: {e}")),
        };
        atlas_core::config::parse_config(&json).map_err(|e| format!("parse_config: {e:#}"))
    });
    let cfg = cfg.as_ref().map_err(|e| anyhow::anyhow!(e.clone()))?;
    Ok(cfg.clone())
}

fn backend() -> Result<&'static AtlasCudaBackend> {
    static GPU: OnceLock<Result<AtlasCudaBackend, String>> = OnceLock::new();
    let gpu = GPU.get_or_init(|| {
        let set = atlas_kernels::ptx_for_exact_target("ling-3.2-air", "nvfp4")
            .ok_or_else(|| "ling-3.2-air/nvfp4 absent from this build".to_string())?;
        AtlasCudaBackend::new(0, &set.modules).map_err(|e| format!("CUDA backend: {e:#}"))
    });
    let gpu = gpu.as_ref().map_err(|e| anyhow::anyhow!(e.clone()))?;
    gpu.bind_to_thread()?;
    Ok(gpu)
}

// ---------------------------------------------------------------- host ref

fn bf(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// f32 -> BF16 bits (round-to-nearest, the house `f32_to_bf16`).
fn rbf(x: f64) -> u16 {
    f32_to_bf16(x as f32)
}

/// Pseudo-random BF16 rows (xorshift, seed-stable; values in [-0.5, 0.5)).
fn rand_rows(seed: u64, rows: usize, cols: usize) -> Vec<u16> {
    let mut s = 0x9e37_79b9_7f4a_7c15_u64 ^ seed;
    (0..rows * cols)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            rbf(((s >> 40) as f64 / (1u64 << 24) as f64) - 0.5)
        })
        .collect()
}

/// `BailingMoeV3RMSNorm`: fp32 stats, one BF16 round of `x * rsqrt`, then
/// the BF16 weight product.
fn ref_rms(x: &[u16], w: &[u16]) -> Vec<u16> {
    let ss: f64 = x.iter().map(|&v| bf(v) as f64 * bf(v) as f64).sum();
    let inv = ((ss / x.len() as f64 + EPS).sqrt()).recip();
    x.iter()
        .zip(w)
        .map(|(&v, &wv)| rbf(bf(rbf(bf(v) as f64 * inv)) as f64 * bf(wv) as f64))
        .collect()
}

/// The house `rms_norm_vanilla` kernel semantics: out = bf(x * rsqrt * w),
/// ONE BF16 round (input bf16, weight bf16, fp32 math).
fn ref_rms_vanilla(x: &[u16], w: &[u16]) -> Vec<u16> {
    let ss: f64 = x.iter().map(|&v| bf(v) as f64 * bf(v) as f64).sum();
    let inv = ((ss / x.len() as f64 + EPS).sqrt()).recip();
    x.iter()
        .zip(w)
        .map(|(&v, &wv)| rbf(bf(v) as f64 * inv * bf(wv) as f64))
        .collect()
}

/// GEMV `out[i] = x . W[i, :]` (W row-major [n, k] bf16), f64 dot, one round.
fn ref_gemv(w: &[u16], x: &[u16], n: usize, k: usize) -> Vec<u16> {
    let mut out = vec![0u16; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for j in 0..k {
            acc += bf(w[i * k + j]) as f64 * bf(x[j]) as f64;
        }
        out[i] = rbf(acc);
    }
    out
}

/// Pair-interleaved partial rope over a 64-wide slice at `pos`
/// (inv_freq[p] = theta^(-2p/64); channels (2p, 2p+1)).
fn ref_rope(slice: &[u16], pos: usize) -> Vec<u16> {
    let mut out = slice.to_vec();
    for p in 0..ROPE / 2 {
        let freq = 1.0 / ROPE_THETA.powf((2 * p) as f64 / ROPE as f64);
        let (cos, sin) = ((freq * pos as f64).cos(), (freq * pos as f64).sin());
        let (even, odd) = (bf(slice[2 * p]) as f64, bf(slice[2 * p + 1]) as f64);
        out[2 * p] = rbf(even * cos - odd * sin);
        out[2 * p + 1] = rbf(odd * cos + even * sin);
    }
    out
}

/// Host tensors of layer 5.
struct Host {
    q_proj: Vec<u16>,    // [HEADS*192, HID]
    kv_a: Vec<u16>,      // [576, HID]
    kv_a_norm: Vec<u16>, // [512]
    kv_b: Vec<u16>,      // [HEADS*256, 512]
    g_proj: Vec<u16>,    // [32, HID]
    dense: Vec<u16>,     // [HID, HEADS*128]
    input_norm: Vec<u16>,
    post_norm: Vec<u16>,
    w_uk: Vec<u16>, // [HEADS, 512, 128] (the loader's transpose)
}

fn host_bf16(name: &str) -> Result<Vec<u16>> {
    let dir = ckpt()?;
    let (dtype, shape, bytes) = read_checkpoint_tensor(&dir, name)?;
    ensure!(
        dtype == "BF16" && bytes.len() % 2 == 0,
        "{name}: expected BF16, got {dtype}"
    );
    let want: usize = shape.iter().product();
    ensure!(
        want * 2 == bytes.len(),
        "{name}: shape {shape:?} vs {} bytes",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Probed gated-MLA layer (`ATLAS_LING_MLA_PARITY_LAYER`, default 5; one of
/// {5, 11, 17, 23, 29, 35, 41}) and its paged-pool attention index.
fn probe_layer() -> Result<(usize, usize)> {
    let idx: usize = match std::env::var("ATLAS_LING_MLA_PARITY_LAYER") {
        Ok(v) => v
            .parse()
            .map_err(|_| anyhow::anyhow!("ATLAS_LING_MLA_PARITY_LAYER must be an integer"))?,
        Err(_) => 5,
    };
    ensure!(
        idx >= 5 && (idx - 5).is_multiple_of(6) && idx <= 41,
        "ATLAS_LING_MLA_PARITY_LAYER must be one of the gated-MLA layers {{5, 11, 17, 23, 29, 35, 41}}"
    );
    let attn_idx = (idx - 5) / 6;
    Ok((idx, attn_idx))
}

fn load_host() -> Result<Host> {
    let (l5, _) = (format!("model.layers.{}", probe_layer()?.0), 0);
    let l5 = l5.as_str();
    let mut h = Host {
        q_proj: host_bf16(&format!("{l5}.attention.q_proj.weight"))?,
        kv_a: host_bf16(&format!("{l5}.attention.kv_a_proj_with_mqa.weight"))?,
        kv_a_norm: host_bf16(&format!("{l5}.attention.kv_a_layernorm.weight"))?,
        kv_b: host_bf16(&format!("{l5}.attention.kv_b_proj.weight"))?,
        g_proj: host_bf16(&format!("{l5}.attention.g_proj.weight"))?,
        dense: host_bf16(&format!("{l5}.attention.dense.weight"))?,
        input_norm: host_bf16(&format!("{l5}.input_layernorm.weight"))?,
        post_norm: host_bf16(&format!("{l5}.post_attention_layernorm.weight"))?,
        w_uk: vec![0u16; HEADS * LORA * NOPE],
    };
    // The loader's `build_w_uk` transpose (weight_loader/bailing/layers.rs):
    // w_uk[h][n][j] == kv_b[h * 256 + j][n].
    for hh in 0..HEADS {
        for n in 0..LORA {
            for j in 0..NOPE {
                h.w_uk[(hh * LORA + n) * NOPE + j] = h.kv_b[(hh * 256 + j) * LORA + n];
            }
        }
    }
    Ok(h)
}

/// Per-row (token) reference state of the MLA half.
struct RefRow {
    /// rms_norm_vanilla(input_layernorm) stage (the GEMVs' input).
    normed: Vec<u16>,
    /// q_proj output row [HEADS, 192] per head [nope | rope raw].
    q: Vec<u16>,
    /// kv_a output row [512 | 64 raw rope].
    kv_a: Vec<u16>,
    /// kv_a_layernorm'd latent (double-round RMS) [512] bf16.
    latent: Vec<u16>,
    g: Vec<u16>, // g_proj logits [32]
}

fn ref_row(h: &Host, input: &[u16]) -> RefRow {
    let normed = ref_rms_vanilla(input, &h.input_norm);
    let q = ref_gemv(&h.q_proj, &normed, HEADS * (NOPE + ROPE), HID);
    let kv_a = ref_gemv(&h.kv_a, &normed, LORA + ROPE, HID);
    let g = ref_gemv(&h.g_proj, &normed, HEADS, HID);
    let latent = ref_rms(&kv_a[..LORA], &h.kv_a_norm);
    RefRow {
        normed,
        q,
        kv_a,
        latent,
        g,
    }
}

/// Eager per-head attention of query `r` (at position `ctx - 1`) over the
/// reference rows [0, ctx), plus the head gate: [HEADS * VD] bf16.
fn ref_attended(h: &Host, rows: &[RefRow], ctx: usize) -> Vec<u16> {
    let q_pos = ctx - 1;
    let q_row = &rows[q_pos];
    let mut v_gated = vec![0u16; HEADS * VD];
    for hh in 0..HEADS {
        let q_nope = &q_row.q[hh * (NOPE + ROPE)..hh * (NOPE + ROPE) + NOPE];
        let q_rot = ref_rope(
            &q_row.q[hh * (NOPE + ROPE) + NOPE..hh * (NOPE + ROPE) + NOPE + ROPE],
            q_pos,
        );
        // Per-context k/v per head.
        let mut scores = vec![0f64; ctx];
        let mut vs = vec![Vec::new(); ctx];
        for (j, rj) in rows.iter().enumerate().take(ctx) {
            let k_rot = ref_rope(&rj.kv_a[LORA..], j);
            let mut dot = 0.0f64;
            let mut vj = vec![0f64; VD];
            for jj in 0..NOPE {
                let mut kn = 0.0f64;
                for n in 0..LORA {
                    kn += bf(h.kv_b[(hh * 256 + jj) * LORA + n]) as f64 * bf(rj.latent[n]) as f64;
                }
                dot += bf(q_nope[jj]) as f64 * bf(rbf(kn)) as f64;
            }
            for p in 0..ROPE {
                dot += bf(q_rot[p]) as f64 * bf(k_rot[p]) as f64;
            }
            scores[j] = dot / (192f64).sqrt();
            for nn in 0..VD {
                let mut vv = 0.0f64;
                for n in 0..LORA {
                    vv += bf(h.kv_b[(hh * 256 + NOPE + nn) * LORA + n]) as f64
                        * bf(rj.latent[n]) as f64;
                }
                vj[nn] = vv;
            }
            vs[j] = vj;
        }
        let m = scores.iter().cloned().fold(f64::MIN, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        let mut out = vec![0f64; VD];
        for j in 0..ctx {
            for nn in 0..VD {
                out[nn] += exps[j] / z * vs[j][nn];
            }
        }
        let gate = 1.0f64 / (1.0 + (-(bf(q_row.g[hh]) as f64)).exp());
        for nn in 0..VD {
            v_gated[hh * VD + nn] = rbf(out[nn] * gate);
        }
    }
    v_gated
}

/// Absorbed-q reference [HEADS, LORA + ROPE] bf16 (the
/// `ling_mla_q_absorb_rope` contract).
fn ref_q_final(h: &Host, r: &RefRow, pos: usize) -> Vec<u16> {
    let mut out = vec![0u16; HEADS * (LORA + ROPE)];
    for hh in 0..HEADS {
        let q_nope = &r.q[hh * (NOPE + ROPE)..hh * (NOPE + ROPE) + NOPE];
        for n in 0..LORA {
            let mut acc = 0.0f64;
            for j in 0..NOPE {
                acc += bf(q_nope[j]) as f64 * bf(h.w_uk[(hh * LORA + n) * NOPE + j]) as f64;
            }
            out[hh * (LORA + ROPE) + n] = rbf(acc);
        }
        let q_rot = ref_rope(
            &r.q[hh * (NOPE + ROPE) + NOPE..hh * (NOPE + ROPE) + NOPE + ROPE],
            pos,
        );
        for p in 0..ROPE {
            out[hh * (LORA + ROPE) + LORA + p] = q_rot[p];
        }
    }
    out
}

/// Softmax-weighted latent reference [HEADS, LORA] bf16 (the paged kernel's
/// output) — eager scores over absorbed q_k dot products.
fn ref_latent(h: &Host, rows: &[RefRow], ctx: usize) -> Vec<u16> {
    let q_pos = ctx - 1;
    let q_row = &rows[q_pos];
    let q_final = ref_q_final(h, q_row, q_pos);
    let mut out = vec![0u16; HEADS * LORA];
    for hh in 0..HEADS {
        let mut scores = vec![0f64; ctx];
        for (j, rj) in rows.iter().enumerate().take(ctx) {
            let k_rot = ref_rope(&rj.kv_a[LORA..], j);
            let mut dot = 0.0f64;
            for n in 0..LORA {
                dot += bf(q_final[hh * (LORA + ROPE) + n]) as f64 * bf(rj.latent[n]) as f64;
            }
            for p in 0..ROPE {
                dot += bf(q_final[hh * (LORA + ROPE) + LORA + p]) as f64 * bf(k_rot[p]) as f64;
            }
            scores[j] = dot / (192f64).sqrt();
        }
        let m = scores.iter().cloned().fold(f64::MIN, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        for n in 0..LORA {
            let mut acc = 0.0f64;
            for j in 0..ctx {
                acc += exps[j] / z * bf(rows[j].latent[n]) as f64;
            }
            out[hh * LORA + n] = rbf(acc);
        }
    }
    out
}

/// The attention-half output row (o_proj + residual) [HID] bf16.
fn ref_attn_half(h: &Host, rows: &[RefRow], ctx: usize, input: &[u16]) -> Vec<u16> {
    let v_gated = ref_attended(h, rows, ctx);
    let o = ref_gemv(&h.dense, &v_gated, HID, HEADS * VD);
    o.iter()
        .zip(input)
        .map(|(&a, &b)| rbf(bf(a) as f64 + bf(b) as f64))
        .collect()
}

// --------------------------------------------------------------- comparing

struct Stage {
    label: &'static str,
    max_abs: f64,
    scale: f64,
}

fn check_stage(label: &'static str, got: &[u16], want: &[u16]) -> Result<Stage> {
    ensure!(
        got.len() == want.len(),
        "{label}: length {} vs {}",
        got.len(),
        want.len()
    );
    let mut max_abs = 0.0f64;
    let mut scale = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (bf(*g) as f64, bf(*w) as f64);
        max_abs = max_abs.max((g - w).abs());
        scale = scale.max(w.abs());
    }
    ensure!(
        got.iter().all(|&v| bf(v).is_finite()),
        "{label}: non-finite BF16"
    );
    let rel = max_abs / scale.max(1e-3);
    ensure!(
        rel <= 2.0e-2,
        "{label}: max_abs {max_abs:.5} on scale {scale:.3} (rel {rel:.4}) exceeds 2%"
    );
    Ok(Stage {
        label,
        max_abs,
        scale,
    })
}

fn bf16s(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn upload_bf16(gpu: &dyn GpuBackend, host: &[u16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

// ------------------------------------------------------------ test plumbing

struct Rig {
    layer: LingMlaLayer,
    scratch: Arc<LingScratch>,
    kv: PagedKvCache,
    block_table: Vec<u32>,
    attn_idx: usize,
}

fn build_rig(gpu: &dyn GpuBackend, host: &Host) -> Result<Rig> {
    let config = model_config()?;
    ensure!(
        config.kv_lora_rank == LORA
            && config.qk_nope_head_dim == NOPE
            && config.qk_rope_head_dim == ROPE
            && config.v_head_dim == VD
            && config.num_attention_heads == HEADS
            && config.hidden_size == HID,
        "probe: unexpected geometry in the parsed config"
    );
    let kernels = Arc::new(LingKernels::resolve(gpu)?);
    let scratch = LingScratch::arc(
        gpu, 64, // rows cap (probe-scale; the env knob default is 512)
        HID, 512, 8, 6144, 64,
    )?;
    let weights = LingMlaWeights {
        q_proj: DenseWeight {
            weight: upload_bf16(gpu, &host.q_proj)?,
        },
        kv_a: DenseWeight {
            weight: upload_bf16(gpu, &host.kv_a)?,
        },
        kv_a_norm: DenseWeight {
            weight: upload_bf16(gpu, &host.kv_a_norm)?,
        },
        g_proj: DenseWeight {
            weight: upload_bf16(gpu, &host.g_proj)?,
        },
        o_proj: DenseWeight {
            weight: upload_bf16(gpu, &host.dense)?,
        },
        w_uk: DenseWeight {
            weight: upload_bf16(gpu, &host.w_uk)?,
        },
        kv_b: DenseWeight {
            weight: upload_bf16(gpu, &host.kv_b)?,
        },
    };
    let dummy = DenseWeight {
        weight: DevicePtr(0),
    };
    let ffn = LingFfn::dense(
        super::super::LingDenseWeights {
            gate: dummy,
            up: dummy,
            down: dummy,
        },
        6144,
        HID,
        Arc::clone(&scratch),
        Arc::clone(&kernels),
    )?;
    let (_, attn_idx) = probe_layer()?;
    let layer = LingMlaLayer::new(
        DenseWeight {
            weight: upload_bf16(gpu, &host.input_norm)?,
        },
        DenseWeight {
            weight: upload_bf16(gpu, &host.post_norm)?,
        },
        weights,
        ffn,
        kernels,
        scratch.clone(),
        attn_idx, // paged-pool region of this gated-MLA layer
        &config,
    )?;
    let kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: BS,
            num_kv_heads: 1,
            head_dim: MLA_K_ROW_BF16,
            num_layers: 7,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        4,
        gpu,
    )?;
    Ok(Rig {
        layer,
        scratch,
        kv,
        block_table: vec![0, 1, 2, 3],
        attn_idx,
    })
}

/// Read the paged K row (`[latent | roped rope]`) of pool slot `s`.
fn pool_k_row(gpu: &dyn GpuBackend, attn_idx: usize, rig: &Rig, s: usize) -> Result<Vec<u16>> {
    bf16s(
        gpu,
        DevicePtr(rig.kv.k_pool_ptr(attn_idx).0 + (s * MLA_K_ROW_BF16 * 2) as u64),
        MLA_K_ROW_BF16,
    )
}

fn pool_v_row(gpu: &dyn GpuBackend, rig: &Rig, s: usize) -> Result<Vec<u16>> {
    let p = *rig.layer.v_pool.lock();
    ensure!(p.is_some(), "v pool not allocated");
    bf16s(gpu, DevicePtr(p.unwrap().0 + (s * LORA * 2) as u64), LORA)
}

/// Reference cache row of reference row `j` (position j): [latent | rope].
fn ref_k_row(_h: &Host, r: &RefRow, pos: usize) -> Vec<u16> {
    let mut out = r.latent.clone();
    out.extend_from_slice(&ref_rope(&r.kv_a[LORA..], pos));
    out
}

// ------------------------------------------------------------------- tests

/// End-to-end layer-5 parity over the PAGED launch path: an 8-row prefill,
/// a 4-row continuation chunk, and a decode row.
#[test]
#[ignore = "GB10 + ling-3.2-air kernel build + ATLAS_LING_CKPT"]
fn mla_layer5_parity() -> Result<()> {
    let gpu = backend()?;
    let host = load_host()?;
    let mut rig = build_rig(gpu, &host)?;
    let stream = gpu.default_stream();

    // 18 probe rows: prefill 0..12 (two paged groups: 8 + 4), continuation
    // 12..16, decode 16 — covering every row-count regime of the paged path.
    const N: usize = 12 + 4 + 1;
    let inputs = rand_rows(0xDA1A, N, HID);
    let rows: Vec<RefRow> = (0..N)
        .map(|r| ref_row(&host, &inputs[r * HID..(r + 1) * HID]))
        .collect();
    let hidden = upload_bf16(gpu, &inputs)?;

    // ---- phase 1: 12-row prefill from position 0 (paged path, 8+4 groups).
    rig.layer.attention_half_prefill(
        gpu,
        hidden,
        &mut rig.kv,
        12,
        0,
        false,
        &rig.block_table,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let s = &rig.scratch;
    let mut stages = Vec::new();
    for r in 0..12 {
        let got = bf16s(
            gpu,
            DevicePtr(s.mla_q_final.0 + (r * HEADS * KVD * 2) as u64),
            HEADS * KVD,
        )?;
        stages.push(check_stage(
            "prefill q_final",
            &got,
            &ref_q_final(&host, &rows[r], r),
        )?);
    }
    for pos in 0..12 {
        stages.push(check_stage(
            "prefill pool K row",
            &pool_k_row(gpu, rig.attn_idx, &rig, pos)?,
            &ref_k_row(&host, &rows[pos], pos),
        )?);
        stages.push(check_stage(
            "prefill pool V row",
            &pool_v_row(gpu, &rig, pos)?,
            &rows[pos].latent,
        )?);
    }
    for r in 0..12 {
        let got = bf16s(
            gpu,
            DevicePtr(s.mla_latent.0 + (r * HEADS * LORA * 2) as u64),
            HEADS * LORA,
        )?;
        stages.push(check_stage(
            "prefill attn latent",
            &got,
            &ref_latent(&host, &rows, r + 1),
        )?);
        let got = bf16s(
            gpu,
            DevicePtr(s.attn_out.0 + (r * HEADS * VD * 2) as u64),
            HEADS * VD,
        )?;
        stages.push(check_stage(
            "prefill gated V",
            &got,
            &ref_attended(&host, &rows, r + 1),
        )?);
        let got = bf16s(gpu, DevicePtr(s.ffn_in.0 + (r * HID * 2) as u64), HID)?;
        let want = ref_gemv(
            &host.dense,
            &ref_attended(&host, &rows, r + 1),
            HID,
            HEADS * VD,
        );
        stages.push(check_stage("prefill o_proj", &got, &want)?);
    }
    {
        let got = bf16s(gpu, hidden, 12 * HID)?;
        let want: Vec<u16> = (0..12)
            .flat_map(|r| ref_attn_half(&host, &rows, r + 1, &inputs[r * HID..(r + 1) * HID]))
            .collect();
        stages.push(check_stage("prefill hidden out", &got, &want)?);
    }

    // ---- phase 2: 4-row continuation chunk at seq_len_start = 12.
    let chunk = DevicePtr(hidden.0 + (12 * HID * 2) as u64);
    rig.layer.attention_half_prefill(
        gpu,
        chunk,
        &mut rig.kv,
        4,
        12,
        false,
        &rig.block_table,
        stream,
    )?;
    gpu.synchronize(stream)?;
    for r in 12..16 {
        let got = bf16s(
            gpu,
            DevicePtr(s.mla_q_final.0 + ((r - 12) * HEADS * KVD * 2) as u64),
            HEADS * KVD,
        )?;
        stages.push(check_stage(
            "cont q_final",
            &got,
            &ref_q_final(&host, &rows[r], r),
        )?);
    }
    for pos in 12..16 {
        stages.push(check_stage(
            "cont pool K row",
            &pool_k_row(gpu, rig.attn_idx, &rig, pos)?,
            &ref_k_row(&host, &rows[pos], pos),
        )?);
    }
    for r in 12..16 {
        let got = bf16s(
            gpu,
            DevicePtr(s.mla_latent.0 + ((r - 12) * HEADS * LORA * 2) as u64),
            HEADS * LORA,
        )?;
        stages.push(check_stage(
            "cont attn latent",
            &got,
            &ref_latent(&host, &rows, r + 1),
        )?);
        let got = bf16s(
            gpu,
            DevicePtr(s.attn_out.0 + ((r - 12) * HEADS * VD * 2) as u64),
            HEADS * VD,
        )?;
        stages.push(check_stage(
            "cont gated V",
            &got,
            &ref_attended(&host, &rows, r + 1),
        )?);
    }
    {
        let got = bf16s(gpu, chunk, 4 * HID)?;
        let want: Vec<u16> = (12..16)
            .flat_map(|r| ref_attn_half(&host, &rows, r + 1, &inputs[r * HID..(r + 1) * HID]))
            .collect();
        stages.push(check_stage("cont hidden out", &got, &want)?);
    }

    // ---- phase 3: one decode row at position 16 (context 16 + self).
    let dec = DevicePtr(hidden.0 + (16 * HID * 2) as u64);
    rig.layer
        .attention_half_decode(gpu, dec, &mut rig.kv, 16, &rig.block_table, stream)?;
    gpu.synchronize(stream)?;
    stages.push(check_stage(
        "decode pool K row",
        &pool_k_row(gpu, rig.attn_idx, &rig, 16)?,
        &ref_k_row(&host, &rows[16], 16),
    )?);
    {
        let got = bf16s(gpu, DevicePtr(s.mla_q_final.0), HEADS * KVD)?;
        stages.push(check_stage(
            "decode q_final",
            &got,
            &ref_q_final(&host, &rows[16], 16),
        )?);
        let got = bf16s(gpu, DevicePtr(s.mla_latent.0), HEADS * LORA)?;
        stages.push(check_stage(
            "decode attn latent",
            &got,
            &ref_latent(&host, &rows, 17),
        )?);
        let got = bf16s(gpu, DevicePtr(s.attn_out.0), HEADS * VD)?;
        stages.push(check_stage(
            "decode gated V",
            &got,
            &ref_attended(&host, &rows, 17),
        )?);
        let got = bf16s(gpu, dec, HID)?;
        let want = ref_attn_half(&host, &rows, 17, &inputs[16 * HID..17 * HID]);
        stages.push(check_stage("decode hidden out", &got, &want)?);
    }

    for st in &stages {
        eprintln!(
            "{}: max_abs {:.5} on scale {:.3}",
            st.label, st.max_abs, st.scale
        );
    }
    eprintln!(
        "mla_layer5_parity: {} stage checks CLEAN (<= 2% rel)",
        stages.len()
    );
    Ok(())
}

/// The single-pass contiguous prefill fast path (`ling_mla_prefill_attn`).
#[test]
#[ignore = "GB10 + ling-3.2-air kernel build + ATLAS_LING_CKPT"]
fn mla_layer5_prefill_fast_parity() -> Result<()> {
    let gpu = backend()?;
    let host = load_host()?;
    let mut rig = build_rig(gpu, &host)?;
    let stream = gpu.default_stream();
    const N: usize = 8;
    let inputs = rand_rows(0xF457, N, HID);
    let rows: Vec<RefRow> = (0..N)
        .map(|r| ref_row(&host, &inputs[r * HID..(r + 1) * HID]))
        .collect();
    let hidden = upload_bf16(gpu, &inputs)?;
    rig.layer.attention_half_prefill(
        gpu,
        hidden,
        &mut rig.kv,
        N,
        0,
        true, // whole prompt: contiguous K/V + ling_mla_prefill_attn
        &rig.block_table,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let s = &rig.scratch;
    let mut stages = Vec::new();
    for r in 0..N {
        let got = bf16s(
            gpu,
            DevicePtr(s.mla_latent.0 + (r * HEADS * LORA * 2) as u64),
            HEADS * LORA,
        )?;
        stages.push(check_stage(
            "fast attn latent",
            &got,
            &ref_latent(&host, &rows, r + 1),
        )?);
        let got = bf16s(
            gpu,
            DevicePtr(s.attn_out.0 + (r * HEADS * VD * 2) as u64),
            HEADS * VD,
        )?;
        stages.push(check_stage(
            "fast gated V",
            &got,
            &ref_attended(&host, &rows, r + 1),
        )?);
    }
    // The paged rows must ALSO be populated (both destinations written).
    for pos in 0..N {
        stages.push(check_stage(
            "fast pool K row",
            &pool_k_row(gpu, rig.attn_idx, &rig, pos)?,
            &ref_k_row(&host, &rows[pos], pos),
        )?);
    }
    let got = bf16s(gpu, hidden, N * HID)?;
    let want: Vec<u16> = (0..N)
        .flat_map(|r| ref_attn_half(&host, &rows, r + 1, &inputs[r * HID..(r + 1) * HID]))
        .collect();
    stages.push(check_stage("fast hidden out", &got, &want)?);
    for st in &stages {
        eprintln!(
            "{}: max_abs {:.5} on scale {:.3}",
            st.label, st.max_abs, st.scale
        );
    }
    eprintln!(
        "mla_layer5_prefill_fast_parity: {} stage checks CLEAN (<= 2% rel)",
        stages.len()
    );
    Ok(())
}

/// Head tail: `model.norm` + `lm_head` gemv over the full BF16 157,184-row
/// head vs an f64 reference; the argmax pair must agree. Also checks the
/// embed/vocab shapes the driver asserts at load.
#[test]
#[ignore = "GB10 + ling-3.2-air kernel build + ATLAS_LING_CKPT"]
fn ling_head_lm_head_parity() -> Result<()> {
    let gpu = backend()?;
    let config = model_config()?;
    let stream = gpu.default_stream();
    const VOCAB: usize = 157_184;
    let model_norm = host_bf16("model.norm.weight")?;
    let lm_head = host_bf16("lm_head.weight")?;
    ensure!(lm_head.len() == VOCAB * HID, "lm_head rows != vocab");
    // Embed/vocab sanity: the embedding table is [VOCAB, HID] and the row
    // the engine's plain d2d copy would fetch is the checkpoint's own row.
    let (_, emb_shape, emb_bytes) =
        read_checkpoint_tensor(&ckpt()?, "model.word_embeddings.weight")?;
    ensure!(
        emb_shape == vec![VOCAB, HID] && emb_bytes.len() == VOCAB * HID * 2,
        "word_embeddings shape {emb_shape:?}"
    );
    let emb_row: Vec<u16> = emb_bytes[1234 * HID * 2..1235 * HID * 2]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    ensure!(emb_row.len() == HID, "embed row width {}", emb_row.len());
    ensure!(
        config.vocab_size == VOCAB,
        "config vocab {}",
        config.vocab_size
    );

    let hidden = rand_rows(0x5EE0, 1, HID);
    let dev = upload_bf16(gpu, &hidden)?;
    let kernels = LingKernels::resolve(gpu)?;
    let norm_w = upload_bf16(gpu, &model_norm)?;
    let normed = gpu.alloc(HID * 2)?;
    ops::rms_norm(
        gpu,
        kernels.rms_norm_k,
        dev,
        &DenseWeight { weight: norm_w },
        normed,
        1,
        HID as u32,
        EPS as f32,
        stream,
    )?;
    let logits = gpu.alloc(VOCAB * 2)?;
    ops::dense_gemv(
        gpu,
        kernels.gemv_k,
        normed,
        &DenseWeight {
            weight: upload_bf16(gpu, &lm_head)?,
        },
        logits,
        VOCAB as u32,
        HID as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;

    let normed_ref = ref_rms_vanilla(&hidden, &model_norm);
    let normed_stage = check_stage("head final norm", &bf16s(gpu, normed, HID)?, &normed_ref)?;

    let mut ref_max = (-f64::INFINITY, 0usize);
    let mut worst = 0.0f64;
    let got = bf16s(gpu, logits, VOCAB)?;
    let mut top: Vec<(f64, usize)> = Vec::new();
    for i in 0..VOCAB {
        let mut acc = 0.0f64;
        for j in 0..HID {
            acc += bf(lm_head[i * HID + j]) as f64 * bf(normed_ref[j]) as f64;
        }
        let acc = acc;
        if acc > ref_max.0 {
            ref_max = (acc, i);
        }
        worst = worst.max((bf(got[i]) as f64 - acc).abs());
        top.push((bf(got[i]) as f64, i));
    }
    top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let gpu_argmax = top[0].1;
    ensure!(
        gpu_argmax == ref_max.1,
        "head argmax mismatch: gpu {gpu_argmax} vs ref {} (gpu {:.4} vs ref {:.4})",
        ref_max.1,
        top[0].0,
        ref_max.0
    );
    eprintln!(
        "head final norm: max_abs {:.5} on scale {:.3}",
        normed_stage.max_abs, normed_stage.scale
    );
    eprintln!(
        "head lm_head: worst abs err {worst:.5} over {VOCAB} logits; argmax {} == ref ({:.4})",
        ref_max.1, ref_max.0
    );
    eprintln!(
        "head top-5 gpu: {:?}",
        top.iter()
            .take(5)
            .map(|(v, i)| (i, f64::round(*v * 100.0) / 100.0))
            .collect::<Vec<_>>()
    );
    eprintln!("head probe CLEAN: norm <=2% rel, full-vocab gemv argmax agrees, embed row sane");
    Ok(())
}
