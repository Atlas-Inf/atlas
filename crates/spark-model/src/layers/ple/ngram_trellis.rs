// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from ExLlamaV3 `ngram_codec.py` (`unpack_rows` / `dequant_rows`).

//! Host dequant of EXL3 n-gram trellis rows (slow path).
//!
//! exllamav3 `modules/quant/exl3_lib/ngram_codec.py`: each 160-dim row is
//! packed as `1 + 10*K` little-endian int16 words (`words_per_row`). Word 0
//! holds the fp16 row-scale bit pattern. The remaining words are a `160*K`-bit
//! tail-biting ring. Reconstruction, from that file's docstring and
//! `dequant_rows`:
//!
//! ```text
//! row[i] = decode_mul1(state_i) * scale + head_bias[head]
//! ```
//!
//! `unpack_rows` places bit `m` of `state_i` at stream bit
//! `((i - m // K) mod 160) * K + (m % K)`, LSB-first inside each uint16.
//! The codebook is the tree's existing mul1 decoder
//! (`weight_map::exl3::mul1_decode`, bit-exact with `decode_3inst<2>`).

use anyhow::Result;
use half::{bf16, f16};

use crate::weight_map::exl3::mul1_decode;

/// `ROW_DIM` in `ngram_codec.py`.
pub const ROW_DIM: usize = 160;

/// Packed EXL3 n-gram table metadata needed to turn a gathered I16 row into
/// the 160-dim embedding the PLE GEMM already expects.
pub struct NgramTrellis {
    pub k: u32,
    pub packed_words: usize,
    pub heads: usize,
    /// fp16 bits, row-major `[heads, ROW_DIM]`.
    pub bias_fp16: Vec<u16>,
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
}

/// `words_per_row(K) = 1 + 10*K`. Inverse: K from a packed width such as 51.
pub fn k_from_packed_words(words: usize) -> Option<u32> {
    if words <= 1 {
        return None;
    }
    let rest = words - 1;
    if !rest.is_multiple_of(10) {
        return None;
    }
    let k = (rest / 10) as u32;
    if (1..=8).contains(&k) { Some(k) } else { None }
}

/// The cache still stores packed I16 bytes. Refuse a scale-array or a stride
/// that is not `packed_words * 2`.
pub fn check_packed_stride(stride: usize, spec: &NgramTrellis, scaled: bool) -> Result<()> {
    anyhow::ensure!(
        !scaled,
        "PLE: EXL3 n-gram trellis rows carry a per-row fp16 scale in word 0, \
         not a cache scale array"
    );
    let expect = spec.packed_words * 2;
    anyhow::ensure!(
        stride == expect,
        "PLE: trellis row stride {stride} != packed_words {} x 2",
        spec.packed_words
    );
    Ok(())
}

fn stream_bit(words: &[u16], bit: usize) -> u16 {
    (words[bit / 16] >> (bit % 16)) & 1
}

/// One-row `unpack_rows`: 160 trellis states plus the fp16 row scale.
pub fn unpack_row(packed: &[u16], k: u32) -> Result<([u16; ROW_DIM], f16)> {
    let expect = 1 + ROW_DIM * k as usize / 16;
    anyhow::ensure!(
        packed.len() == expect,
        "PLE trellis: row has {} words, expected {expect} for K={k}",
        packed.len()
    );
    let scale = f16::from_bits(packed[0]);
    let words = &packed[1..];
    let mut states = [0u16; ROW_DIM];
    let k_us = k as usize;
    for i in 0..ROW_DIM {
        let mut state = 0u16;
        for m in 0..16 {
            // Python `unpack_rows`: `((i - m // K) % ROW_DIM) * K + m % K`.
            let src = (i as isize - (m as isize / k as isize)).rem_euclid(ROW_DIM as isize)
                as usize
                * k_us
                + (m % k_us);
            state |= stream_bit(words, src) << m;
        }
        states[i] = state;
    }
    Ok((states, scale))
}

pub fn head_for_row(spec: &NgramTrellis, row: u64) -> Result<usize> {
    for (h, (&off, &vocab)) in spec
        .head_offsets
        .iter()
        .zip(spec.head_vocab_sizes.iter())
        .enumerate()
    {
        if row >= off && row < off.saturating_add(vocab) {
            return Ok(h);
        }
    }
    anyhow::bail!("PLE trellis: row {row} is outside every head offset range")
}

/// Concatenated BF16 embeddings, one `ROW_DIM` slice per id, in `ids` order.
/// That is the `[T, heads * 160]` matrix the PLE GEMM reads from `emb`.
pub fn decode_rows_bf16(spec: &NgramTrellis, ids: &[u64], raw_rows: &[Vec<u8>]) -> Result<Vec<u8>> {
    anyhow::ensure!(
        ids.len() == raw_rows.len(),
        "PLE trellis: {} ids but {} cached rows",
        ids.len(),
        raw_rows.len()
    );
    let nbytes = spec.packed_words * 2;
    let mut out = vec![0u8; ids.len() * ROW_DIM * 2];
    for (n, (id, raw)) in ids.iter().zip(raw_rows.iter()).enumerate() {
        anyhow::ensure!(
            raw.len() == nbytes,
            "PLE trellis: cached row is {} bytes, expected {nbytes}",
            raw.len()
        );
        let mut words = vec![0u16; spec.packed_words];
        for (i, chunk) in raw.chunks_exact(2).enumerate() {
            words[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }
        let (states, scale) = unpack_row(&words, spec.k)?;
        let scale_f = scale.to_f32();
        let head = head_for_row(spec, *id)?;
        let bias_off = head * ROW_DIM;
        anyhow::ensure!(
            spec.bias_fp16.len() >= bias_off + ROW_DIM,
            "PLE trellis: head_bias is {} fp16 values, need {} for head {head}",
            spec.bias_fp16.len(),
            bias_off + ROW_DIM
        );
        for i in 0..ROW_DIM {
            let w = mul1_decode(states[i]).to_f32() * scale_f
                + f16::from_bits(spec.bias_fp16[bias_off + i]).to_f32();
            let bits = bf16::from_f32(w).to_bits().to_le_bytes();
            let dst = (n * ROW_DIM + i) * 2;
            out[dst] = bits[0];
            out[dst + 1] = bits[1];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_width_51_is_k5() {
        assert_eq!(k_from_packed_words(51), Some(5));
        assert_eq!(1 + ROW_DIM * 5 / 16, 51);
    }
}
