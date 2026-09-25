// SPDX-License-Identifier: AGPL-3.0-only
// Portions of this file are derived from ExLlamaV3 (MIT, see NOTICE.md).

//! Bit-exact CPU reference decoder for ExLlamaV3's EXL3 (mul1) weight format.
//!
//! An EXL3 linear layer with `in` input and `out` output features is stored as
//!
//! - `trellis`: int16 `[in/16, out/16, 16*bits]` — one 16x16 weight tile per
//!   `[i, j]`, its 256 trellis states packed into `16*bits` words (`bits` = the
//!   codebook's instruction count K, 1..=8; the Qwen3.8-Flash-Next checkpoints
//!   use 3..=6),
//! - `suh` / `svh`: fp16 `[in]` / `[out]` sign-and-scale vectors,
//! - `mul1`: an int32 scalar whose presence selects the "mul1" codebook. Only
//!   that codebook is implemented here, and it is what every tensor in these
//!   checkpoints carries.
//!
//! The weight it encodes is
//!
//! ```text
//!   W = diag(suh) · H · W_inner · H · diag(svh)
//! ```
//!
//! with `H` the block-diagonal normalized 128x128 Sylvester Hadamard matrix
//! (entries ±1/√128) and `W_inner` fp16 `[in, out]`, built tile by tile from the
//! decoded trellis states through the mul1 codebook and the tensor-core
//! permutation. Hence the 128-multiple requirement on both feature counts.
//!
//! This module is the reference every later GPU kernel is tested against, so it
//! is pure Rust, CPU-only and allocation-light: it runs at load time over
//! multi-megabyte tensors. Ported from exllamav3 (MIT, see NOTICE.md):
//! `exllamav3/exllamav3_ext/cpu/moe_mul1.cpp` (state extraction,
//! `make_tc_perm`), `exllamav3/exllamav3_ext/quant/codebook.cuh`
//! (`decode_3inst<2>`) and `exllamav3/exllamav3/modules/quant/exl3.py` (weight
//! assembly). `bench/exl3/reference.py` is the NumPy oracle the fixtures in
//! `bench/exl3/fixtures.json` were generated from.

mod cpu_ref;
#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
#[path = "tests.rs"]
mod tests;
mod weight;
#[cfg(test)]
#[path = "weight_tests.rs"]
mod weight_tests;

#[cfg(test)]
pub(crate) use cpu_ref::{HAD_SCALE, reconstruct as reconstruct_ref};
pub use cpu_ref::{decode_inner, mul1_decode, reconstruct, tile_states};
pub use weight::{Exl3Weight, MUL1_TAG, exl3_from_store};

/// The shape of one EXL3 trellis tensor, as derived from its `trellis` dims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exl3Shape {
    pub in_features: usize,
    pub out_features: usize,
    /// Codebook instructions per state (K), 1..=8.
    pub bits: u32,
}

impl Exl3Shape {
    /// From the dims of a `trellis` tensor: `[in/16, out/16, 16*bits]`.
    ///
    /// The third dim is the only place `bits` is recorded, so it is what the
    /// whole decode keys off — a tensor whose last dim is not `16 * K` for an
    /// in-range K is not an EXL3 mul1 tensor and must not be guessed at.
    pub fn from_trellis_dims(dims: &[usize]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            dims.len() == 3,
            "EXL3 trellis must be rank 3, got {:?}",
            dims
        );
        let last = dims[2];
        anyhow::ensure!(
            last != 0 && last.is_multiple_of(16),
            "EXL3 trellis last dim {last} must be a non-zero multiple of 16 (16 * bits)"
        );
        let bits = (last / 16) as u32;
        anyhow::ensure!(
            (1..=8).contains(&bits),
            "EXL3 trellis last dim {last} implies bits = {bits}, outside the codebook range 1..=8"
        );
        anyhow::ensure!(
            dims[0] != 0 && dims[1] != 0,
            "EXL3 trellis has a zero tile dim: {:?}",
            dims
        );
        Ok(Self {
            in_features: dims[0] * 16,
            out_features: dims[1] * 16,
            bits,
        })
    }

    /// Number of u16 words per tile.
    pub fn words_per_tile(&self) -> usize {
        16 * self.bits as usize
    }

    /// Self-consistency of a hand-built shape. The fields are public, so a
    /// shape can exist that no `from_trellis_dims` would produce — a feature
    /// count that is not a whole number of 16-wide tiles (the decode would
    /// silently drop the ragged remainder) or an out-of-range `bits` (the
    /// state bitstream would not fit the tile). Every decode entry point
    /// checks this first.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (1..=8).contains(&self.bits),
            "EXL3 bits = {}, outside the codebook range 1..=8",
            self.bits
        );
        for (what, len) in [
            ("in_features", self.in_features),
            ("out_features", self.out_features),
        ] {
            anyhow::ensure!(
                len != 0 && len.is_multiple_of(16),
                "EXL3 {what} {len} must be a non-zero multiple of the 16-wide trellis tile"
            );
        }
        Ok(())
    }

    /// The scale vectors must cover the features they scale, or the decode would
    /// silently read past them (or stop short of the tensor).
    pub fn check_scales(&self, suh_len: usize, svh_len: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            suh_len == self.in_features,
            "suh has {suh_len} entries, expected {} (in_features)",
            self.in_features
        );
        anyhow::ensure!(
            svh_len == self.out_features,
            "svh has {svh_len} entries, expected {} (out_features)",
            self.out_features
        );
        Ok(())
    }
}
