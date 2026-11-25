// SPDX-License-Identifier: AGPL-3.0-only

//! On-GPU parity of the whole Ling-3.2-air MoE chain — route + activation
//! quantization, the routed experts' fused gate/up/swiglu/re-quant/down,
//! the BF16 shared expert and the unpermute — against a slow host
//! reference of `BailingMoeV3SparseMoeBlock`
//! (context/modeling_bailing_moe_v3_vl.py:498-586) computed from the
//! checkpoint's own bytes. Modeled on `glm5/fp4l_test_gpu.rs`: `#[ignore]`d,
//! it needs a GB10, a `ling-3.2-air` kernel build and the real checkpoint:
//!
//! ```sh
//! ATLAS_LING_CKPT=/home/spark/ling/Ling-3.2-air-NVFP4 \
//! cargo test --release -p spark-model --lib \
//!     bailing::layers::layers_moe_parity_gpu -- --ignored --nocapture
//! ```
//!
//! Layers probed: 3 (plain routed layer, swiglu limits 0/0) and 35 (the
//! last 7 MoE layers' routed limit 4, shared limit 7). Batches M=1 (the
//! decode contract) and M=8 (an MTP-verify-shaped batch: rows>1, sorting,
//! multi-row expert tiles). The kernel quantization helpers
//! (`ling_e2m1_quantize`, `ling_float_to_e4m3`, `ling_quantize_group`,
//! `ling_group_scale`) are mirrored bit for bit on the host, so the
//! quantized-activation stage is asserted BIT-EXACT and the re-quantized
//! intermediate at the nibble level.
//!
//! Ground-truth scale conventions (the `weight_map::quantized_v2`
//! compressed-tensors finding, checked per checkpoint scalar here): both
//! `weight_global_scale` and `input_global_scale` are stored as
//! RECIPROCALS of the dequant multipliers —
//! `w = E2M1 * E4M3_group / weight_global_scale`, and the quantized
//! activation dequantizes as `q * sf / input_global_scale` (the quantizer
//! multiplies `x * ig / sf`, saturating past the calibrated amax
//! `6 * 448 / ig`). The test also asserts the convention's plausibility
//! in the data itself (dequantized routed-gate std vs the shared expert's
//! BF16 gate std), so a wrong convention in the reference cannot pass.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};
use atlas_core::config::parse_config;
use atlas_core::numeric::{FP4_E2M1, f32_to_bf16, fp8_e4m3_to_f32};
use spark_runtime::cuda_backend::{AtlasCudaBackend, configure_pdl};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use crate::model::bailing_c1::{LingFfn, LingKernels, LingScratch, LingSpec};

const HIDDEN: usize = 2560;
const EXPERTS: usize = 512;
const TOP_K: usize = 8;
const INTER: usize = 768;
const PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// BF16 bits widened to f32 (exact).
pub(super) fn widen(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// `f32 -> BF16 -> f32` (one round-trip: the reference chain's
/// per-tensor rounding).
pub(super) fn bf16(v: f32) -> f32 {
    f32::from_bits((f32_to_bf16(v) as u32) << 16)
}

pub(super) fn f32_le(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

// ── GB10 backend with the ling-3.2-air target modules ──────────────────────

pub(super) fn gpu_backend() -> Result<&'static AtlasCudaBackend> {
    static GPU: OnceLock<Result<AtlasCudaBackend, String>> = OnceLock::new();
    let gpu = GPU.get_or_init(|| {
        configure_pdl("ling-3.2-air");
        let set = atlas_kernels::ptx_for_exact_target("ling-3.2-air", "nvfp4")
            .ok_or_else(|| "ling-3.2-air/nvfp4 absent from this build".to_owned())?;
        let gpu = AtlasCudaBackend::new(0, &set.modules).map_err(|e| format!("{e:#}"))?;
        gpu.bind_to_thread().map_err(|e| format!("{e:#}"))?;
        Ok(gpu)
    });
    gpu.as_ref().map_err(|e| anyhow::anyhow!(e.clone()))
}

fn ckpt_dir() -> Result<PathBuf> {
    Ok(std::env::var("ATLAS_LING_CKPT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/spark/ling/Ling-3.2-air-NVFP4")))
}

// ── Checkpoint reader: index parsed once, shard handles kept open ──────────

pub(super) struct Ckpt {
    dir: PathBuf,
    map: HashMap<String, String>,
    /// Open shard handles with their parsed headers (one header read+parse
    /// per shard, then raw offset reads).
    files: HashMap<String, (std::fs::File, u64, serde_json::Value)>,
}

impl Ckpt {
    pub(super) fn open() -> Result<Self> {
        let dir = ckpt_dir()?;
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("model.safetensors.index.json"))
                .with_context(|| format!("read index in {}", dir.display()))?,
        )
        .with_context(|| format!("parse index in {}", dir.display()))?;
        let map = index["weight_map"]
            .as_object()
            .context("index without weight_map")?
            .iter()
            .map(|(k, v)| Ok((k.clone(), v.as_str().context("non-str shard")?.to_owned())))
            .collect::<Result<_>>()?;
        Ok(Self {
            dir,
            map,
            files: HashMap::new(),
        })
    }

    /// `(dtype, shape, bytes)` of one tensor.
    pub(super) fn tensor(&mut self, name: &str) -> Result<(WeightDtype, Vec<usize>, Vec<u8>)> {
        let shard = self
            .map
            .get(name)
            .with_context(|| format!("{name}: not in the index"))?
            .clone();
        if !self.files.contains_key(&shard) {
            let mut f = std::fs::File::open(self.dir.join(&shard))?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len)?;
            let header_len = u64::from_le_bytes(len);
            let mut header = vec![0u8; header_len as usize];
            f.read_exact(&mut header)?;
            let header: serde_json::Value = serde_json::from_slice(&header)
                .with_context(|| format!("parse shard {shard} header"))?;
            self.files.insert(shard.clone(), (f, header_len, header));
        }
        let (file, header_len, header) = self.files.get_mut(&shard).context("just inserted")?;
        let t = &header[name];
        let shape: Vec<usize> = serde_json::from_value(t["shape"].clone())?;
        let [lo, hi]: [u64; 2] = serde_json::from_value(t["data_offsets"].clone())?;
        file.seek(SeekFrom::Start(8 + *header_len + lo))?;
        let mut data = vec![0u8; (hi - lo) as usize];
        file.read_exact(&mut data)?;
        let dtype = WeightDtype::from_safetensors_str(t["dtype"].as_str().unwrap_or(""))?;
        Ok((dtype, shape, data))
    }

    pub(super) fn scalar(&mut self, name: &str) -> Result<f32> {
        let (dt, shape, data) = self.tensor(name)?;
        ensure!(
            dt == WeightDtype::FP32 && shape == [1] && data.len() == 4,
            "{name}: expected an F32 scalar, got {dt:?} {shape:?}"
        );
        Ok(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
    }
}

// ── Host mirrors of ling_moe_quant.cuh (bit-identical by construction) ───────

/// `ling_e2m1_quantize`.
pub(super) fn e2m1_quantize(v: f32) -> u32 {
    let absv = v.abs();
    let sign = u32::from(v < 0.0) << 3;
    let idx = if absv <= 0.25 {
        0
    } else if absv <= 0.75 {
        1
    } else if absv <= 1.25 {
        2
    } else if absv <= 1.75 {
        3
    } else if absv <= 2.5 {
        4
    } else if absv <= 3.5 {
        5
    } else if absv <= 5.0 {
        6
    } else {
        7
    };
    sign | idx
}

/// `ling_float_to_e4m3`.
pub(super) fn f32_to_e4m3(v: f32) -> u8 {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    if bits & 0x7FFF_FFFF == 0 {
        return (sign << 7) as u8;
    }
    let mut a = v.abs();
    if a > 448.0 {
        a = 448.0;
    }
    let abits = a.to_bits();
    let f32_exp = (((abits >> 23) & 0xFF) as i32) - 127;
    let f32_man = abits & 0x7F_FFFF;
    if f32_exp < -9 {
        return (sign << 7) as u8;
    }
    if f32_exp < -6 {
        let man = ((a * 512.0 + 0.5) as i32).clamp(0, 7);
        return ((sign << 7) | man as u32) as u8;
    }
    let mut e = f32_exp + 7;
    if e < 1 {
        e = 1;
    }
    let man;
    if e > 15 {
        e = 15;
        man = 6;
    } else {
        let mut m = ((f32_man + (1u32 << 19)) >> 20) as i32;
        if m > 7 {
            m = 0;
            e += 1;
            if e > 15 {
                e = 15;
                m = 6;
            }
        }
        man = m;
    }
    ((sign << 7) | ((e as u32) << 3) | man as u32) as u8
}

/// The e4m3 decode of `ling_quantize_group` (sign honored).
pub(super) fn e4m3_signed(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let man = b & 7;
    let mag = if exp == 0 {
        man as f32 * 0.001_953_125
    } else if exp == 15 && man == 7 {
        0.0
    } else {
        f32::from_bits((((exp as u32) + 120) << 23) | ((man as u32) << 20))
    };
    if sign == 1 { -mag } else { mag }
}

/// `ling_group_scale`'s decode (group scales are unsigned).
pub(super) fn e4m3_unsigned(b: u8) -> f32 {
    fp8_e4m3_to_f32(b & 0x7F)
}

/// `ling_quantize_group`: 16 f32 values -> (scale byte, 8 packed bytes),
/// with the launcher's dequant-global `scale2` (`ffn.rs` passes
/// `inv = 1.0 / scale2`, mirrored exactly).
pub(super) fn quantize_group_mirror(x16: &[f32], scale2: f32, inv_scale2: f32) -> (u8, [u8; 8]) {
    let mut group_max = 0.0f32;
    for v in x16 {
        group_max = group_max.max(v.abs());
    }
    let fp8_float = if group_max > 0.0 {
        group_max * inv_scale2 / 6.0
    } else {
        0.0
    };
    let fp8_byte = f32_to_e4m3(fp8_float);
    let decoded = e4m3_signed(fp8_byte);
    let effective = decoded * scale2;
    let inv_eff = if effective > 0.0 {
        1.0 / effective
    } else {
        0.0
    };
    let mut packed = [0u8; 8];
    for i in 0..8 {
        let n0 = e2m1_quantize(x16[2 * i] * inv_eff);
        let n1 = e2m1_quantize(x16[2 * i + 1] * inv_eff);
        packed[i] = ((n1 << 4) | n0) as u8;
    }
    (fp8_byte, packed)
}

/// `ling_group_scale`: the re-quantizer's scale byte and the e2m1
/// multiplier (`inv = g / sf`).
pub(super) fn group_scale_mirror(group_max: f32, g: f32) -> (u8, f32) {
    let byte = f32_to_e4m3(group_max * g / 6.0);
    let decoded = e4m3_unsigned(byte);
    let inv = if decoded > 0.0 { g / decoded } else { 0.0 };
    (byte, inv)
}

/// `ling_silu_bf16`.
pub(super) fn silu_clamped(g_bf16: f32, limit: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-g_bf16).exp());
    let s = bf16(g_bf16 * sig);
    if limit > 0.0 && s > limit { limit } else { s }
}

/// `ling_gate_up_mul`.
pub(super) fn gate_up_mul(gate_silu: f32, up: f32, limit: f32) -> f32 {
    let mut u = up;
    if limit > 0.0 {
        u = u.clamp(-limit, limit);
    }
    bf16(gate_silu * u)
}

pub(super) fn e2m1(code: u32) -> f32 {
    FP4_E2M1[(code & 0xF) as usize]
}

pub(super) fn nibbles(b: u8) -> (u32, u32) {
    ((b & 0xF) as u32, (b >> 4) as u32)
}

/// One expert's dequantized projections (the reference's weight side).
struct ExpertWeights {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

// ── Host-side checkpoint meta + the reference of the MoE block ─────────────

struct LayerHost {
    gate_w: Vec<f32>,
    bias: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_down: Vec<f32>,
    /// Per expert [gate, up, down] checkpoint scalars.
    wgs: Vec<[f32; 3]>,
    ig: Vec<[f32; 3]>,
    routed_limit: f32,
    shared_limit: f32,
}

impl LayerHost {
    /// One projection dequantized to f32 `[n, k]` (reciprocal global).
    fn dequant_projection(
        &self,
        ckpt: &mut Ckpt,
        lp: &str,
        e: usize,
        proj: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let stem = format!("{lp}.mlp.experts.{e}.{}", PROJS[proj]);
        let (_, _, packed) = ckpt.tensor(&format!("{stem}.weight_packed"))?;
        let (_, _, scale) = ckpt.tensor(&format!("{stem}.weight_scale"))?;
        let wgs = self.wgs[e][proj];
        ensure!(packed.len() == n * k / 2, "{stem}: packed shape");
        ensure!(scale.len() == n * k / 16, "{stem}: scale shape");
        let mut w = vec![0f32; n * k];
        for r in 0..n {
            for g in 0..k / 16 {
                let sf = fp8_e4m3_to_f32(scale[r * k / 16 + g]);
                for i in 0..8 {
                    let (n0, n1) = nibbles(packed[r * k / 2 + g * 8 + i]);
                    w[r * k + g * 16 + 2 * i] = e2m1(n0) * sf / wgs;
                    w[r * k + g * 16 + 2 * i + 1] = e2m1(n1) * sf / wgs;
                }
            }
        }
        Ok(w)
    }

    fn expert_weights(&self, ckpt: &mut Ckpt, lp: &str, e: usize) -> Result<ExpertWeights> {
        Ok(ExpertWeights {
            gate: self.dequant_projection(ckpt, lp, e, 0, INTER, HIDDEN)?,
            up: self.dequant_projection(ckpt, lp, e, 1, INTER, HIDDEN)?,
            down: self.dequant_projection(ckpt, lp, e, 2, HIDDEN, INTER)?,
        })
    }
}

/// The reference FFN of one routed expert on one row: the down output
/// `y`, the dequantized down input `h_deq`, the pre-requant `h`, the
/// TRUE-convention packed/scale bytes and the PRODUCTION-mirror nibbles
/// (the launcher's `a2_gscale[e]` — for the bit-level comparison).
#[allow(clippy::too_many_arguments)]
struct ExpertRow {
    y: Vec<f32>,
    h: Vec<f32>,
    packed: Vec<u8>,
    scale: Vec<u8>,
    mirror_packed: Vec<u8>,
}

fn expert_row(
    host: &LayerHost,
    w: &ExpertWeights,
    a_deq: &[f32],
    e: usize,
    a2g_mirror: f32,
) -> ExpertRow {
    let limit = host.routed_limit;
    let ig_down = host.ig[e][2];
    let mut h = vec![0f32; INTER];
    for c in 0..INTER {
        let mut g_acc = 0f64;
        let mut u_acc = 0f64;
        for t in 0..HIDDEN {
            g_acc += (a_deq[t] * w.gate[c * HIDDEN + t]) as f64;
            u_acc += (a_deq[t] * w.up[c * HIDDEN + t]) as f64;
        }
        let g = bf16(g_acc as f32);
        let u = bf16(u_acc as f32);
        h[c] = gate_up_mul(silu_clamped(g, limit), u, limit);
    }
    let mut packed = Vec::with_capacity(INTER / 2);
    let mut scale = Vec::with_capacity(INTER / 16);
    let mut mirror = Vec::with_capacity(INTER / 2);
    let mut h_deq = vec![0f32; INTER];
    let scale2_v = 1.0f32 / ig_down;
    for g in 0..INTER / 16 {
        let ch = &h[g * 16..(g + 1) * 16];
        let amax = ch.iter().fold(0.0f32, |mx, v| mx.max(v.abs()));
        // TRUE re-quantization (g = ig_down) and the production mirror.
        let (byte, inv) = group_scale_mirror(amax, ig_down);
        let (_, inv_m) = group_scale_mirror(amax, a2g_mirror);
        scale.push(byte);
        let sf = e4m3_unsigned(byte);
        for i in 0..8 {
            let c0 = e2m1_quantize(ch[2 * i] * inv);
            let c1 = e2m1_quantize(ch[2 * i + 1] * inv);
            let m0 = e2m1_quantize(ch[2 * i] * inv_m);
            let m1 = e2m1_quantize(ch[2 * i + 1] * inv_m);
            packed.push(((c1 << 4) | c0) as u8);
            mirror.push(((m1 << 4) | m0) as u8);
            h_deq[g * 16 + 2 * i] = e2m1(c0) * sf * scale2_v;
            h_deq[g * 16 + 2 * i + 1] = e2m1(c1) * sf * scale2_v;
        }
    }
    let mut y = vec![0f32; HIDDEN];
    for c in 0..HIDDEN {
        let mut acc = 0f64;
        for t in 0..INTER {
            acc += (h_deq[t] * w.down[c * INTER + t]) as f64;
        }
        y[c] = bf16(acc as f32);
    }
    ExpertRow {
        y,
        h,
        packed,
        scale,
        mirror_packed: mirror,
    }
}

struct RouteRef {
    ids: Vec<Vec<u32>>,
    weights: Vec<Vec<f32>>,
}

/// Router reference (modeling :486-494): fp32 sigmoid + bias group-limited
/// top-8 over f64-accumulated logits (ties -> lowest index, torch.topk
/// convention the kernels encode).
fn route_reference(host: &LayerHost, x: &[f32], m: usize) -> RouteRef {
    const N_GROUP: usize = 8;
    const TOPK_GROUP: usize = 4;
    let gsize = EXPERTS / N_GROUP;
    let mut ids = Vec::with_capacity(m);
    let mut weights = Vec::with_capacity(m);
    for r in 0..m {
        let xs = &x[r * HIDDEN..(r + 1) * HIDDEN];
        let logits: Vec<f32> = (0..EXPERTS)
            .map(|e| {
                let w = &host.gate_w[e * HIDDEN..(e + 1) * HIDDEN];
                let mut acc = 0f64;
                for t in 0..HIDDEN {
                    acc += (xs[t] * w[t]) as f64;
                }
                acc as f32
            })
            .collect();
        let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let sel: Vec<f32> = (0..EXPERTS).map(|i| scores[i] + host.bias[i]).collect();
        let gsel: Vec<f32> = sel
            .chunks(gsize)
            .map(|chunk| {
                let mut top = chunk.to_vec();
                top.sort_by(|a, b| b.partial_cmp(a).unwrap());
                top[0] + top[1]
            })
            .collect();
        let mut picked = [false; N_GROUP];
        for _ in 0..TOPK_GROUP {
            let mut bg = 0;
            let mut best = f32::NEG_INFINITY;
            for (g, v) in gsel.iter().enumerate() {
                if !picked[g] && *v > best {
                    best = *v;
                    bg = g;
                }
            }
            picked[bg] = true;
        }
        let mut work: Vec<f32> = sel
            .chunks(gsize)
            .enumerate()
            .flat_map(|(g, chunk)| {
                chunk
                    .iter()
                    .map(|v| if picked[g] { *v } else { f32::NEG_INFINITY })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut chosen = Vec::with_capacity(TOP_K);
        for _ in 0..TOP_K {
            let mut at = 0;
            let mut best = f32::NEG_INFINITY;
            for (i, v) in work.iter().enumerate() {
                if *v > best {
                    best = *v;
                    at = i;
                }
            }
            work[at] = f32::NEG_INFINITY;
            chosen.push(at as u32);
        }
        // Raw sigmoid scores of the selection, normalized by their sum, x2.5.
        let vals: Vec<f32> = chosen.iter().map(|&i| scores[i as usize]).collect();
        let mut sum = 0.0f32;
        for v in &vals {
            sum += *v;
        }
        let ws: Vec<f32> = vals.iter().map(|v| *v / (sum + 1e-20) * 2.5).collect();
        ids.push(chosen);
        weights.push(ws);
    }
    RouteRef { ids, weights }
}

/// The reference shared expert (BF16 weights, fp32-dot opmath, the
/// share_expert clamp list).
fn shared_reference(host: &LayerHost, x: &[f32], m: usize) -> Vec<Vec<f32>> {
    (0..m)
        .map(|r| {
            let xs = &x[r * HIDDEN..(r + 1) * HIDDEN];
            let mut h = vec![0f32; INTER];
            for c in 0..INTER {
                let mut g_acc = 0f64;
                let mut u_acc = 0f64;
                for t in 0..HIDDEN {
                    g_acc += (xs[t] * host.shared_gate[c * HIDDEN + t]) as f64;
                    u_acc += (xs[t] * host.shared_up[c * HIDDEN + t]) as f64;
                }
                h[c] = gate_up_mul(
                    silu_clamped(bf16(g_acc as f32), host.shared_limit),
                    bf16(u_acc as f32),
                    host.shared_limit,
                );
            }
            let mut y = vec![0f32; HIDDEN];
            for c in 0..HIDDEN {
                let mut acc = 0f64;
                for t in 0..INTER {
                    acc += (h[t] * host.shared_down[c * INTER + t]) as f64;
                }
                y[c] = bf16(acc as f32);
            }
            y
        })
        .collect()
}

/// BF16 tensor read to f32 (widened exactly).
pub(super) fn read_bf16(ckpt: &mut Ckpt, name: &str, len: usize) -> Result<Vec<f32>> {
    let (dt, sh, b) = ckpt.tensor(name)?;
    ensure!(
        dt == WeightDtype::BF16 && sh.len() == 2 && b.len() == len * 2,
        "{name}: {dt:?} {sh:?}"
    );
    Ok(b.chunks_exact(2)
        .map(|c| widen(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// Router tensor read to f32: BF16 (widened exactly) or FP32 on disk —
/// the load path's `f32_or_upcast` takes both.
pub(super) fn read_f32s(ckpt: &mut Ckpt, name: &str, len: usize) -> Result<Vec<f32>> {
    let (dt, sh, b) = ckpt.tensor(name)?;
    let elems = sh.iter().product::<usize>();
    ensure!(elems == len, "{name}: {sh:?} for {len} elements");
    let expect = if dt == WeightDtype::FP32 { 4 } else { 2 };
    ensure!(b.len() == len * expect, "{name}: {dt:?} {sh:?}");
    Ok(match dt {
        WeightDtype::BF16 => b
            .chunks_exact(2)
            .map(|c| widen(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        WeightDtype::FP32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => anyhow::bail!("{name}: {other:?}"),
    })
}

// ── Input rows: deterministic, with hotter rows so the clamps and the
//    saturation band of the static activation scale bind ────────────────────

struct Xorshift64(u64);

impl Xorshift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub(super) fn x_rows(m: usize, layer: usize) -> Vec<f32> {
    let mut rng = Xorshift64(0x9E37_79B9_7F4A_7C15 ^ ((layer as u64) << 32) ^ 0x1234);
    (0..m)
        .flat_map(|r| {
            let std = match r {
                0..=4 => 1.0f64,
                5 => 2.5,
                _ => 6.0,
            };
            (0..HIDDEN)
                .map(|_| {
                    // Box-Muller.
                    let u1 = rng.unit().max(1e-12);
                    let u2 = rng.unit();
                    let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    bf16((g * std) as f32)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

// ── The test ────────────────────────────────────────────────────────────────

#[test]
#[ignore = "needs a GB10, a ling-3.2-air kernel build and ATLAS_LING_CKPT"]
fn ling_moe_layer_parity_gpu() -> Result<()> {
    let gpu = gpu_backend()?;
    let dir = ckpt_dir()?;
    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let spec = LingSpec::from_config(&config)?;
    ensure!(
        spec.hidden == HIDDEN && spec.experts == EXPERTS && spec.top_k == TOP_K,
        "unexpected Ling shapes: {} x {} x {}",
        spec.hidden,
        spec.experts,
        spec.top_k
    );
    let kernels = std::sync::Arc::new(LingKernels::resolve(gpu)?);
    let scratch = LingScratch::arc(gpu, 8, HIDDEN, EXPERTS, TOP_K, INTER, 16)?;
    let mut ckpt = Ckpt::open()?;
    for layer in [3usize, 35usize] {
        parity_one_layer(
            gpu,
            &mut ckpt,
            &spec,
            &kernels,
            &scratch,
            &format!("model.layers.{layer}"),
            layer,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn parity_one_layer(
    gpu: &dyn GpuBackend,
    ckpt: &mut Ckpt,
    spec: &LingSpec,
    kernels: &std::sync::Arc<LingKernels>,
    scratch: &std::sync::Arc<LingScratch>,
    lp: &str,
    layer: usize,
) -> Result<()> {
    // ── Upload the layer's tensors under the production names, keeping
    //    host bytes for the reference.
    let mut names = vec![
        format!("{lp}.mlp.gate.weight"),
        format!("{lp}.mlp.gate.expert_bias"),
        format!("{lp}.mlp.shared_experts.gate_proj.weight"),
        format!("{lp}.mlp.shared_experts.up_proj.weight"),
        format!("{lp}.mlp.shared_experts.down_proj.weight"),
    ];
    for e in 0..EXPERTS {
        for p in PROJS {
            for part in [
                "weight_packed",
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            ] {
                names.push(format!("{lp}.mlp.experts.{e}.{p}.{part}"));
            }
        }
    }
    let mut map = HashMap::new();
    for name in &names {
        let (dtype, shape, data) = ckpt.tensor(name)?;
        let ptr = gpu.alloc(data.len().max(1))?;
        gpu.copy_h2d(&data, ptr)?;
        map.insert(name.clone(), WeightTensor { ptr, dtype, shape });
    }
    let store = WeightStore::from_map(map);

    let gate_w = read_f32s(ckpt, &format!("{lp}.mlp.gate.weight"), EXPERTS * HIDDEN)?;
    let bias = read_f32s(ckpt, &format!("{lp}.mlp.gate.expert_bias"), EXPERTS)?;
    // Shared expert weights are BF16.
    let shared_gate = read_bf16(
        ckpt,
        &format!("{lp}.mlp.shared_experts.gate_proj.weight"),
        INTER * HIDDEN,
    )?;
    let shared_up = read_bf16(
        ckpt,
        &format!("{lp}.mlp.shared_experts.up_proj.weight"),
        INTER * HIDDEN,
    )?;
    let shared_down = read_bf16(
        ckpt,
        &format!("{lp}.mlp.shared_experts.down_proj.weight"),
        HIDDEN * INTER,
    )?;
    let mut wgs = vec![[0f32; 3]; EXPERTS];
    let mut ig = vec![[0f32; 3]; EXPERTS];
    for e in 0..EXPERTS {
        for (pi, p) in PROJS.iter().enumerate() {
            wgs[e][pi] = ckpt.scalar(&format!("{lp}.mlp.experts.{e}.{p}.weight_global_scale"))?;
            ig[e][pi] = ckpt.scalar(&format!("{lp}.mlp.experts.{e}.{p}.input_global_scale"))?;
        }
    }
    // The route kernel quantizes with ONE activation global; production
    // asserts the gate/up agreement — assert what the reference assumes.
    for e in 0..EXPERTS {
        ensure!(
            ig[e][0] == ig[0][0] && ig[e][1] == ig[0][0],
            "expert {e} gate/up input_global_scale differs from expert 0"
        );
    }

    let layer_spec = spec.layers[layer];
    let host = LayerHost {
        gate_w,
        bias,
        shared_gate,
        shared_up,
        shared_down,
        wgs,
        ig,
        routed_limit: layer_spec.routed_swiglu_limit.max(0) as f32,
        shared_limit: layer_spec.shared_swiglu_limit.max(0) as f32,
    };

    // ── Reference-convention plausibility: the dequantized routed-gate std
    //    must sit in a sane band of the shared expert's BF16 gate std.
    let std_of = |v: &[f32]| -> f64 {
        let s: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        (s / v.len() as f64).sqrt()
    };
    let probe = host.dequant_projection(ckpt, lp, 0, 0, INTER, HIDDEN)?;
    let probe_std = std_of(&probe);
    let shared_std = std_of(&host.shared_gate);
    eprintln!(
        "layer {layer}: routed gate dequant std {probe_std:.5} vs shared gate bf16 std {shared_std:.5}"
    );
    ensure!(
        probe_std > 0.2 * shared_std && probe_std < 5.0 * shared_std,
        "scale-convention plausibility failed (routed {probe_std} vs shared {shared_std})"
    );

    // ── Production load (the real folds + repack) and the FFN under test.
    let weights = super::load_moe_weights(&store, lp, layer_spec, spec, gpu, kernels)?;
    let a_scale2 = weights.a_scale2;
    let a2_gscale = weights.a2_gscale;
    let mut a2g = vec![0f32; EXPERTS];
    {
        let mut b = vec![0u8; EXPERTS * 4];
        gpu.copy_d2h(a2_gscale, &mut b)?;
        for e in 0..EXPERTS {
            a2g[e] = f32_le(&b, e * 4);
        }
    }
    let ffn = LingFfn::moe(
        weights,
        EXPERTS,
        TOP_K,
        spec.n_group,
        spec.topk_group,
        spec.routed_scaling,
        INTER,
        HIDDEN,
        std::sync::Arc::clone(scratch),
        std::sync::Arc::clone(kernels),
    )?;

    let stream = gpu.default_stream();
    for m in [1usize, 8usize] {
        let x = x_rows(m, layer);
        let x_bits: Vec<u16> = x.iter().map(|v| f32_to_bf16(*v)).collect();
        let x_bytes: Vec<u8> = x_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let x_dev = gpu.alloc(x_bytes.len())?;
        gpu.copy_h2d(&x_bytes, x_dev)?;
        let out = gpu.alloc(m * HIDDEN * 2)?;
        gpu.memset(out, 0, m * HIDDEN * 2)?;
        ffn.forward(gpu, x_dev, out, m, stream)?;
        gpu.synchronize(stream)?;
        compare_stages(
            gpu, ckpt, &host, scratch, &a2g, a_scale2, lp, layer, &x, out, m,
        )?;
        gpu.free(x_dev)?;
        gpu.free(out)?;
    }
    Ok(())
}

/// Pull the stage buffers off the device and compare, stage by stage:
/// input quant -> route -> gate/up re-quant -> down -> final.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compare_stages(
    gpu: &dyn GpuBackend,
    ckpt: &mut Ckpt,
    host: &LayerHost,
    scratch: &LingScratch,
    a2g: &[f32],
    a_scale2: f32,
    lp: &str,
    layer: usize,
    x: &[f32],
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    let d2h = |gpu: &dyn GpuBackend, p: DevicePtr, n: usize| -> Result<Vec<u8>> {
        let mut b = vec![0u8; n];
        gpu.copy_d2h(p, &mut b)?;
        Ok(b)
    };
    let a_packed = d2h(gpu, scratch.a_packed, m * HIDDEN / 2)?;
    let a_scale = d2h(gpu, scratch.a_scale, m * HIDDEN / 16)?;
    let topk_ids_b = d2h(gpu, scratch.topk_ids, m * TOP_K * 4)?;
    let topk_w_b = d2h(gpu, scratch.topk_weights, m * TOP_K * 4)?;
    let token_to_perm_b = d2h(gpu, scratch.token_to_perm, m * TOP_K * 4)?;
    let gu_packed = d2h(gpu, scratch.gu_packed, m * TOP_K * INTER / 2)?;
    let gu_scale = d2h(gpu, scratch.gu_scale, m * TOP_K * INTER / 16)?;
    let expert_out_b = d2h(gpu, scratch.expert_out, m * TOP_K * HIDDEN * 2)?;
    let out_b = d2h(gpu, out, m * HIDDEN * 2)?;

    // ── Stage A: the quantized MoE input must be BIT-IDENTICAL to the
    //    TRUE-Convention quantization of the checkpoint (the reciprocal
    //    `input_global_scale`; see the file comment). A launcher scalar run
    //    in the wrong direction shows up here immediately — and the
    //    downstream reference consumes these bytes, never the GPU's, so a
    //    zeroed quantization cannot pass against itself.
    let ig_a = host.ig[0][0];
    let true_scale2 = 1.0f32 / ig_a;
    if (a_scale2 - true_scale2).abs() > true_scale2 * 1e-6 {
        eprintln!(
            "layer {layer}: route quantizer dequant-global a_scale2={a_scale2:e} but the checkpoint convention needs 1/input_global_scale={true_scale2:e}"
        );
    }
    let inv_scale2 = 1.0f32 / true_scale2;
    let mut ref_a_packed = vec![0u8; m * HIDDEN / 2];
    let mut ref_a_scale = vec![0u8; m * HIDDEN / 16];
    for r in 0..m {
        for g in 0..HIDDEN / 16 {
            let vals = &x[r * HIDDEN + g * 16..r * HIDDEN + g * 16 + 16];
            let (sb, pk) = quantize_group_mirror(vals, true_scale2, inv_scale2);
            ref_a_scale[r * HIDDEN / 16 + g] = sb;
            ref_a_packed[r * HIDDEN / 2 + g * 8..r * HIDDEN / 2 + g * 8 + 8].copy_from_slice(&pk);
        }
    }
    let a_pack_diff = a_packed
        .iter()
        .zip(&ref_a_packed)
        .filter(|(a, b)| a != b)
        .count();
    let a_scale_diff = a_scale
        .iter()
        .zip(&ref_a_scale)
        .filter(|(a, b)| a != b)
        .count();
    let a_scale_zero = a_scale.iter().filter(|b| **b == 0).count();
    eprintln!(
        "layer {layer} M={m} stage A (input quant): {a_pack_diff}/{} packed bytes, {a_scale_diff}/{} scale bytes differ ({a_scale_zero} GPU scale bytes are ZERO)",
        ref_a_packed.len(),
        ref_a_scale.len()
    );
    ensure!(
        a_pack_diff == 0 && a_scale_diff == 0,
        "stage A diverges: {a_pack_diff} packed + {a_scale_diff} scale bytes differ from the TRUE quantization"
    );

    // ── Route: selection, weights and the sort permutation.
    let route_ref = route_reference(host, x, m);
    let flat: Vec<u32> = route_ref.ids.iter().flatten().copied().collect();
    let mut order: Vec<usize> = (0..flat.len()).collect();
    order.sort_by_key(|&s| flat[s]);
    let mut perm_of = vec![0usize; flat.len()];
    for (pos, &slot) in order.iter().enumerate() {
        perm_of[slot] = pos;
    }
    let mut id_mismatch = 0usize;
    let mut perm_mismatch = 0usize;
    let mut w_max_rel = 0f64;
    for r in 0..m {
        for t in 0..TOP_K {
            let at4 = (r * TOP_K + t) * 4;
            let gpu_id = u32::from_le_bytes([
                topk_ids_b[at4],
                topk_ids_b[at4 + 1],
                topk_ids_b[at4 + 2],
                topk_ids_b[at4 + 3],
            ]);
            let gpu_perm = i32::from_le_bytes([
                token_to_perm_b[at4],
                token_to_perm_b[at4 + 1],
                token_to_perm_b[at4 + 2],
                token_to_perm_b[at4 + 3],
            ]);
            if gpu_id != route_ref.ids[r][t] {
                eprintln!(
                    "layer {layer} row {r} slot {t}: GPU id {gpu_id} != ref {}",
                    route_ref.ids[r][t]
                );
                id_mismatch += 1;
            }
            if gpu_perm as usize != perm_of[r * TOP_K + t] {
                eprintln!(
                    "layer {layer} row {r} slot {t}: GPU perm {gpu_perm} != ref {}",
                    perm_of[r * TOP_K + t]
                );
                perm_mismatch += 1;
            }
            let w = f32_le(&topk_w_b, at4);
            let r_w = route_ref.weights[r][t] as f64;
            w_max_rel = w_max_rel.max(((w as f64) - r_w).abs() / r_w.abs().max(1e-9));
        }
    }
    ensure!(
        id_mismatch == 0,
        "route: {id_mismatch} selected experts differ"
    );
    ensure!(
        perm_mismatch == 0,
        "route: {perm_mismatch} permed slots differ"
    );
    ensure!(w_max_rel < 1e-4, "route: weights off (max rel {w_max_rel})");
    eprintln!(
        "layer {layer} M={m} route: ids/perms exact, topk weight max rel diff {w_max_rel:.3e}"
    );

    // The activation the reference experts consume: the TRUE quantized
    // bytes, dequantized by `q * sf / ig`.
    let a_deq_rows: Vec<Vec<f32>> = (0..m)
        .map(|r| {
            let mut row = vec![0f32; HIDDEN];
            for g in 0..HIDDEN / 16 {
                let sf = e4m3_unsigned(ref_a_scale[r * HIDDEN / 16 + g]);
                for i in 0..8 {
                    let (n0, n1) = nibbles(ref_a_packed[r * HIDDEN / 2 + g * 8 + i]);
                    row[g * 16 + 2 * i] = e2m1(n0) * sf / ig_a;
                    row[g * 16 + 2 * i + 1] = e2m1(n1) * sf / ig_a;
                }
            }
            row
        })
        .collect();

    // ── Gate/up + down: per slot, the reference expert FFN vs the GPU's
    //    re-quantized intermediate and its bf16 down row.
    let mut expert_cache: HashMap<usize, ExpertWeights> = HashMap::new();
    let mut slot_y: Vec<Vec<f32>> = vec![Vec::new(); flat.len()];
    let mut nib_diff = 0usize;
    let mut mirror_diff = 0usize;
    let mut nib_total = 0usize;
    let mut scale_diff = 0usize;
    let mut bound_hits = 0usize;
    let mut down_err2 = 0f64;
    let mut down_ref2 = 0f64;
    for r in 0..m {
        for t in 0..TOP_K {
            let e = route_ref.ids[r][t] as usize;
            if !expert_cache.contains_key(&e) {
                expert_cache.insert(e, host.expert_weights(ckpt, lp, e)?);
            }
            let w = expert_cache.get(&e).expect("just inserted");
            let row = expert_row(host, w, &a_deq_rows[r], e, a2g[e]);
            let perm = perm_of[r * TOP_K + t];
            let gpu_pk = &gu_packed[perm * INTER / 2..(perm + 1) * INTER / 2];
            let gpu_sc = &gu_scale[perm * INTER / 16..(perm + 1) * INTER / 16];
            let ig_dn = host.ig[e][2];
            nib_total += INTER;
            for g in 0..INTER / 16 {
                let sf = e4m3_unsigned(gpu_sc[g]);
                for i in 0..8 {
                    let (g0, g1) = nibbles(gpu_pk[g * 8 + i]);
                    let (p0, p1) = nibbles(row.packed[g * 8 + i]);
                    let (m0, m1) = nibbles(row.mirror_packed[g * 8 + i]);
                    nib_diff += usize::from(g0 != p0) + usize::from(g1 != p1);
                    mirror_diff += usize::from(g0 != m0) + usize::from(g1 != m1);
                    // TRUE dequant of the GPU bytes vs the pre-requant h,
                    // one e2m1 step of the group (diagnostic only — the
                    // static activation scale saturates whole groups past
                    // 6 * 448 / ig by design, and the bit-exact nibble and
                    // scale-byte comparisons above carry the parity proof).
                    let d0 = e2m1(g0) * sf / ig_dn;
                    let d1 = e2m1(g1) * sf / ig_dn;
                    let ch = &row.h[g * 16..(g + 1) * 16];
                    let step = sf / ig_dn;
                    bound_hits += usize::from((d0 - ch[2 * i]).abs() > step + 1e-12)
                        + usize::from((d1 - ch[2 * i + 1]).abs() > step + 1e-12);
                }
                scale_diff += usize::from(gpu_sc[g] != row.scale[g]);
            }
            for c in 0..HIDDEN {
                let i = (perm * HIDDEN + c) * 2;
                let y = widen(u16::from_le_bytes([expert_out_b[i], expert_out_b[i + 1]]));
                ensure!(
                    y.is_finite(),
                    "expert_out non-finite at perm {perm} col {c}"
                );
                down_err2 += ((y - row.y[c]) as f64).powi(2);
                down_ref2 += (row.y[c] as f64).powi(2);
            }
            slot_y[r * TOP_K + t] = row.y;
        }
    }
    let nib_pct = 100.0 * nib_diff as f64 / nib_total as f64;
    let down_rms = (down_err2 / down_ref2.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!(
        "layer {layer} M={m} gate/up: {nib_diff}/{nib_total} nibbles ({nib_pct:.3}%) differ from the true pack ({mirror_diff} mirror), {scale_diff} scale bytes, {bound_hits} outside the e2m1 bound"
    );
    eprintln!("layer {layer} M={m} down rows: rel RMS {down_rms:.5}");
    ensure!(
        nib_pct < 1.0,
        "gate/up intermediate nibbles diverge ({nib_pct:.3}%)"
    );
    ensure!(
        scale_diff <= nib_total / 400,
        "gate/up intermediate scale bytes diverge ({scale_diff})"
    );
    ensure!(down_rms < 0.02, "down rows diverge (rel RMS {down_rms})");

    // ── Final: weighted reduce + shared (modeling :560-586, :549-556).
    let shared = shared_reference(host, x, m);
    let mut final_err2 = 0f64;
    let mut final_ref2 = 0f64;
    for r in 0..m {
        for c in 0..HIDDEN {
            let mut acc = 0f32;
            for t in 0..TOP_K {
                acc += route_ref.weights[r][t] * slot_y[r * TOP_K + t][c];
            }
            let routed = bf16(acc);
            let final_ref = bf16(routed + shared[r][c]);
            let i = (r * HIDDEN + c) * 2;
            let y = widen(u16::from_le_bytes([out_b[i], out_b[i + 1]]));
            ensure!(y.is_finite(), "MoE out non-finite at row {r} col {c}");
            final_err2 += ((y - final_ref) as f64).powi(2);
            final_ref2 += (final_ref as f64).powi(2);
        }
    }
    let final_rms = (final_err2 / final_ref2.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("layer {layer} M={m} FINAL (routed reduce + shared): rel RMS {final_rms:.5}");
    ensure!(
        final_rms < 0.02,
        "layer {layer} M={m} MoE output diverges (rel RMS {final_rms})"
    );
    Ok(())
}
