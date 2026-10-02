// SPDX-License-Identifier: AGPL-3.0-only
// Portions of this file are derived from ExLlamaV3 (MIT, see NOTICE.md).

//! The decoder itself: state extraction, the mul1 codebook, tile placement and
//! the two 128-wide Hadamard passes. Ported from exllamav3's CPU
//! `decode_state_scalar` / `make_tc_perm` and the GPU `decode_3inst<2>`.

use half::f16;

use super::Exl3Shape;

/// The mul1 codebook's integer multiplier (`codebook.cuh`).
const MUL1_MULT: u32 = 0x83DCD12D;
/// fp16 0x1eee ~ 0.00677 = 1/147.7, the codebook's inverse scale.
const K_INV: u16 = 0x1eee;
/// fp16 0xc931 ~ -10.39 = (-1024.0 - 510.0) * k_inv, its bias.
const K_BIAS: u16 = 0xc931;
/// The Hadamard block width, and so the feature count every EXL3 tensor must be
/// a multiple of.
const HAD: usize = 128;
/// 1/sqrt(128), the normalized Hadamard's entry magnitude. The literal is the
/// f32 nearest the exact value; a test pins it against the computed one.
pub(crate) const HAD_SCALE: f32 = 0.088_388_346;

/// Decode one mul1 state to its fp16 weight value.
///
/// `x = state * MUL1_MULT` (wrapping), `s` = the sum of x's four bytes, and the
/// value is `f16((1024 + s) * k_inv + k_bias)`. The arithmetic is done in f64 -
/// exact for every operand - and rounded ONCE to fp16, which is what reproduces
/// the GPU's single `__hfma` over the dp4a accumulator in `decode_3inst<2>`.
/// Rounding the sum to fp16 first and then fma-ing would not.
pub fn mul1_decode(state: u16) -> f16 {
    let x = (state as u32).wrapping_mul(MUL1_MULT);
    let s: u64 = x.to_le_bytes().iter().map(|&b| u64::from(b)).sum();
    let k_inv = f16::from_bits(K_INV).to_f64();
    let k_bias = f16::from_bits(K_BIAS).to_f64();
    f16::from_f64((1024.0 + s as f64) * k_inv + k_bias)
}

/// The 256 trellis states of one tile, in tensor-core thread order.
///
/// The tile's `16*bits` u16 words are read as `8*bits` u32 words
/// (`u32[i] = w16[2i] | w16[2i+1] << 16`); state `t` is the 16 bits starting at
/// bit `t*bits + 256*bits + bits - 16` of that bitstream, gathered across a
/// word boundary with a 64-bit merge and a funnel shift — exllamav3's
/// `decode_state_scalar`, applied per state (two word loads, so O(1) each).
///
/// Panics if `tile` is not exactly one tile (`16*bits` words) for a `bits` in
/// `1..=8`; the public callers validate that first.
pub fn tile_states(tile: &[u16], bits: u32) -> [u16; 256] {
    assert!(
        (1..=8).contains(&bits) && tile.len() == 16 * bits as usize,
        "tile of {} words is not one {bits}-bit tile (expected {})",
        tile.len(),
        16 * bits as usize
    );
    let words32 = bits as usize * 256 / 32;
    // One tile's worth of u32 words: at most 8*8 = 64, so no heap per tile -
    // this runs over every tile of a multi-megabyte tensor at load time.
    let mut word = [0u32; 64];
    for i in 0..words32 {
        word[i] = u32::from(tile[2 * i]) | (u32::from(tile[2 * i + 1]) << 16);
    }
    let mut states = [0u16; 256];
    for t in 0..256usize {
        // The additions come first so the -16 cannot underflow the usize.
        let b0 = t * bits as usize + 256 * bits as usize + bits as usize - 16;
        let b1 = b0 + 16;
        let shift = ((b1 - 1) / 32 + 1) * 32 - b1;
        let merged = (u64::from(word[(b0 / 32) % words32]) << 32)
            | u64::from(word[((b1 - 1) / 32) % words32]);
        states[t] = (merged >> shift) as u16;
    }
    states
}

/// `W_inner` as fp16, row-major `[in_features, out_features]`, from the trellis.
///
/// Tile `(i, j)` covers rows `i*16..i*16+16` and columns `j*16..j*16+16`; its
/// t-th decoded state lands at tile position `perm[t]`, the tensor-core
/// permutation baked into the storage format by exllamav3.
pub fn decode_inner(trellis: &[u16], shape: &Exl3Shape) -> anyhow::Result<Vec<f16>> {
    shape.validate()?;
    let tiles_i = shape.in_features / 16;
    let tiles_j = shape.out_features / 16;
    let want = tiles_i * tiles_j * shape.words_per_tile();
    anyhow::ensure!(
        trellis.len() == want,
        "trellis has {} u16 words, expected {want} for shape {:?}",
        trellis.len(),
        shape
    );
    let perm = tc_perm();
    let mut inner = vec![f16::from_bits(0); shape.in_features * shape.out_features];
    for ti in 0..tiles_i {
        for tj in 0..tiles_j {
            let base = (ti * tiles_j + tj) * shape.words_per_tile();
            let states = tile_states(&trellis[base..base + shape.words_per_tile()], shape.bits);
            for (t, &state) in states.iter().enumerate() {
                let p = perm[t];
                let idx = (ti * 16 + p / 16) * shape.out_features + tj * 16 + p % 16;
                inner[idx] = mul1_decode(state);
            }
        }
    }
    Ok(inner)
}

/// The full weight as f32, row-major `[in_features, out_features]`:
/// `W = diag(suh) . H . W_inner . H . diag(svh)`.
///
/// `H` is block-diagonal in 128, so it is applied as an in-place fast
/// Walsh-Hadamard butterfly (Sylvester order, no bit reversal) down each column
/// within every 128-row block and then along each row within every 128-column
/// block, each pass scaled by 1/sqrt(128). The features must be multiples of
/// 128 for the blocks to tile; every checkpoint tensor satisfies that, and a
/// tensor that does not is refused rather than padded.
///
/// The passes are ordered for load-time locality — this runs over 2560x10240
/// GDN projections times 36 layers and a 2560x248320 lm_head, so every pass
/// touches only contiguous rows:
///
/// 1. the left Hadamard as row-pair butterflies over whole rows (never a
///    strided column gather), unscaled, with its 1/sqrt(128) folded into the
///    suh multiply of pass 2,
/// 2. `diag(suh) * (1/sqrt(128))` and the right Hadamard row by row — the
///    right pass scales itself inside `hadamard`, so svh is applied plain and
///    no separate column walk is needed.
pub fn reconstruct(
    trellis: &[u16],
    suh: &[f16],
    svh: &[f16],
    shape: &Exl3Shape,
) -> anyhow::Result<Vec<f32>> {
    for (what, len) in [
        ("in_features", shape.in_features),
        ("out_features", shape.out_features),
    ] {
        anyhow::ensure!(
            len.is_multiple_of(HAD),
            "EXL3 reconstruct needs {what} a multiple of the {HAD}-wide Hadamard block, got {len}"
        );
    }
    shape.check_scales(suh.len(), svh.len())?;

    let n = shape.out_features;
    let mut w = decode_inner(trellis, shape)?
        .into_iter()
        .map(f16::to_f32)
        .collect::<Vec<f32>>();

    // H down the columns, as butterflies between row pairs of each 128-row
    // block: rows r and r+len of the block combine element-wise across the
    // whole row, so every access is contiguous. The 1/sqrt(128) is folded into
    // the suh multiply below.
    let mut rows: Vec<&mut [f32]> = w.chunks_exact_mut(n).collect();
    for block in rows.chunks_mut(HAD) {
        let mut len = 1usize;
        while len < HAD {
            for base in (0..HAD).step_by(len * 2) {
                let pair = block[base..base + 2 * len].split_at_mut(len);
                for (x, y) in pair.0.iter_mut().zip(pair.1) {
                    for (a, b) in x.iter_mut().zip(y.iter_mut()) {
                        let (u, v) = (*a, *b);
                        *a = u + v;
                        *b = u - v;
                    }
                }
            }
            len *= 2;
        }
    }
    // diag(suh) * HAD_SCALE, then H along each row within every 128-column
    // block, then * diag(svh) — one contiguous sweep per row. Each pass's
    // 1/sqrt(128) rides elsewhere: the left butterflies above are unscaled and
    // their scale is folded into the suh multiply here, while `hadamard` scales
    // the right pass itself, so svh is applied plain.
    let svh_f32: Vec<f32> = svh.iter().map(|&v| v.to_f32()).collect();
    for (k, row) in w.chunks_mut(n).enumerate() {
        let s = suh[k].to_f32() * HAD_SCALE;
        for v in row.iter_mut() {
            *v *= s;
        }
        for block in row.chunks_mut(HAD) {
            hadamard(block);
        }
        for (v, s) in row.iter_mut().zip(&svh_f32) {
            *v *= *s;
        }
    }
    Ok(w)
}

/// In-place normalized fast Walsh-Hadamard transform of 128 floats, Sylvester
/// ordering: butterflies at lengths 1, 2, 4 .. 64, then one scale.
fn hadamard(x: &mut [f32]) {
    debug_assert_eq!(x.len(), HAD);
    let mut len = 1usize;
    while len < HAD {
        for base in (0..HAD).step_by(len * 2) {
            for i in base..base + len {
                let a = x[i];
                let b = x[i + len];
                x[i] = a + b;
                x[i + len] = a - b;
            }
        }
        len *= 2;
    }
    for v in x.iter_mut() {
        *v *= HAD_SCALE;
    }
}

/// The tensor-core permutation: state `t` lands at tile position `p`.
///
/// `make_tc_perm` in `moe_mul1.cpp`: thread `thr` owns rows
/// `(thr%4)*2, +1, +8, +9` at columns `thr/4` and `thr/4 + 8`.
fn tc_perm() -> [usize; 256] {
    let mut p = [0usize; 256];
    for thr in 0..32 {
        let r0 = (thr % 4) * 2;
        let c0 = thr / 4;
        for (i, r) in [r0, r0 + 1, r0 + 8, r0 + 9].iter().enumerate() {
            p[thr * 8 + i] = r * 16 + c0;
            p[thr * 8 + 4 + i] = r * 16 + c0 + 8;
        }
    }
    p
}
