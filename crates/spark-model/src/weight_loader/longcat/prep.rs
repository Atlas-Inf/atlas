// SPDX-License-Identifier: AGPL-3.0-only

//! LongCat MLA weight preparation: the two load-time transforms that let the
//! shared (DeepSeek/Mistral-lineage) MLA runtime serve LongCat unchanged.
//!
//! 1. ROPE CONVENTION. Atlas's rope kernels are `rotate_half`: they pair
//!    element `i` with `i + rope/2`. LongCat (like DeepSeek HF) stores the
//!    rope slice INTERLEAVED — its `apply_rotary_pos_emb_interleave` first
//!    de-interleaves `[x0,x1,x2,x3,…] → [x0,x2,…,x1,x3,…]` and then applies
//!    the same rotate_half math. That de-interleave is a fixed PERMUTATION of
//!    the projection's OUTPUT rows, so folding it into the weights at load
//!    (rows `j → 2j`, `j + rope/2 → 2j+1`) makes the runtime kernel produce
//!    exactly the reference's rotated values with no kernel change.
//!
//!    Applies to the rope rows of `q_b_proj` (per head, rows
//!    `nope..nope+rope` of each `qk_head_dim` block) and the trailing `rope`
//!    rows of `kv_a_proj_with_mqa`.
//!
//! 2. MLA LoRA SCALING. LongCat multiplies `q_pass`/`q_rot` by
//!    `sqrt(hidden/q_lora_rank)` and `k_pass` by `sqrt(hidden/kv_lora_rank)`
//!    (`mla_scale_q_lora` / `mla_scale_kv_lora`). Both fold into weights:
//!      - q: scale ALL of `q_b_proj` (both nope and rope halves are scaled).
//!      - kv: scale `kv_a_layernorm.weight` — `k_pass` is the norm's OUTPUT,
//!        and RMSNorm is scale-invariant in its input, so scaling the norm
//!        gain reproduces `scale * norm(x)` exactly. `k_rot` correctly stays
//!        UNSCALED: it bypasses the norm (it is split off before it).

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense};

const BF16: usize = 2;

fn to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn to_bf16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| {
            let bits = x.to_bits();
            // round-to-nearest-even, matching the repo's other host converters
            let r = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
            r.to_le_bytes()
        })
        .collect()
}

/// De-interleave `rows` rope rows in place: source row `2j` becomes dest row
/// `j`, source `2j+1` becomes dest `j + rows/2` (the reference's
/// `view(-1, d/2, 2).transpose(4,3)` on the OUTPUT axis).
fn deinterleave_rope_rows(host: &mut [u8], base_row: usize, rows: usize, cols: usize) {
    let half = rows / 2;
    let row_bytes = cols * BF16;
    let start = base_row * row_bytes;
    let src: Vec<u8> = host[start..start + rows * row_bytes].to_vec();
    for j in 0..half {
        let (a, b) = (2 * j, 2 * j + 1);
        host[start + j * row_bytes..start + (j + 1) * row_bytes]
            .copy_from_slice(&src[a * row_bytes..(a + 1) * row_bytes]);
        host[start + (half + j) * row_bytes..start + (half + j + 1) * row_bytes]
            .copy_from_slice(&src[b * row_bytes..(b + 1) * row_bytes]);
    }
}

fn upload(host: &[u8], gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let p = gpu.alloc(host.len())?;
    gpu.copy_h2d(host, p)?;
    Ok(p)
}

/// `q_b_proj` `[n_heads*qk_head_dim, q_lora]`: de-interleave each head's rope
/// rows and fold `mla_scale_q_lora` over the whole tensor.
pub(super) fn prep_q_b(
    store: &WeightStore,
    name: &str,
    n_heads: usize,
    nope: usize,
    rope: usize,
    q_lora: usize,
    scale_q: f32,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let hd = nope + rope;
    let bytes = n_heads * hd * q_lora * BF16;
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    for head in 0..n_heads {
        deinterleave_rope_rows(&mut host, head * hd + nope, rope, q_lora);
    }
    if (scale_q - 1.0).abs() > f32::EPSILON {
        let mut f = to_f32(&host);
        for v in &mut f {
            *v *= scale_q;
        }
        host = to_bf16(&f);
    }
    Ok(DenseWeight {
        weight: upload(&host, gpu)?,
    })
}

/// `kv_a_proj_with_mqa` `[kv_lora + rope, hidden]`: de-interleave the trailing
/// rope rows (k_rot). NOT scaled — `k_rot` bypasses `kv_a_layernorm`.
pub(super) fn prep_kv_a(
    store: &WeightStore,
    name: &str,
    kv_lora: usize,
    rope: usize,
    hidden: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let bytes = (kv_lora + rope) * hidden * BF16;
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    deinterleave_rope_rows(&mut host, kv_lora, rope, hidden);
    Ok(DenseWeight {
        weight: upload(&host, gpu)?,
    })
}

/// `kv_a_layernorm.weight` `[kv_lora]` scaled by `mla_scale_kv_lora` — the
/// fold that reproduces the reference's `k_pass * sqrt(hidden/kv_lora)`.
pub(super) fn prep_kv_a_norm(
    store: &WeightStore,
    name: &str,
    kv_lora: usize,
    scale_kv: f32,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    if (scale_kv - 1.0).abs() <= f32::EPSILON {
        return Ok(w);
    }
    let mut host = vec![0u8; kv_lora * BF16];
    gpu.copy_d2h(w.weight, &mut host)?;
    let mut f = to_f32(&host);
    for v in &mut f {
        *v *= scale_kv;
    }
    Ok(DenseWeight {
        weight: upload(&to_bf16(&f), gpu)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deinterleave_moves_even_rows_first() {
        // 4 rope rows of width 1: [a,b,c,d] (interleaved pairs (a,b),(c,d))
        // → [a,c,b,d]: evens to the front half, odds to the back half, which
        // is exactly what rotate_half then pairs as (a,b) and (c,d).
        let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut host = to_bf16(&vals);
        deinterleave_rope_rows(&mut host, 0, 4, 1);
        assert_eq!(to_f32(&host), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn deinterleave_respects_base_row_and_width() {
        // 2 leading rows untouched, then 4 rope rows of width 2.
        let vals: [f32; 12] = [
            -1.0, -1.0, -2.0, -2.0, // leading (nope) rows
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
        ];
        let mut host = to_bf16(&vals);
        deinterleave_rope_rows(&mut host, 2, 4, 2);
        let got = to_f32(&host);
        assert_eq!(&got[..4], &[-1.0, -1.0, -2.0, -2.0]);
        assert_eq!(&got[4..], &[1.0, 1.5, 3.0, 3.5, 2.0, 2.5, 4.0, 4.5]);
    }
}
