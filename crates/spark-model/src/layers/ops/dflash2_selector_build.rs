// SPDX-License-Identifier: AGPL-3.0-only

//! Which compiled DFlash2 candidate selector serves a checkpoint's
//! `selector_top_k`: the 16-candidate common module every Qwen3.8 / GLM-5.3
//! DFlash2 drafter launches, or its 32-candidate twin
//! (`kernels/gb10/common/dflash2_candidate_selector_k32.cu`, Ripple-65B's
//! DFlash2 drafter). The kernels return without writing when the launch block
//! is not their `DF2_SEL_THREADS`, so the module and the block size are chosen
//! together here, from the one `top_k`.

use anyhow::{Result, ensure};

use super::{DFLASH2_SELECTOR_MAX_TOP_K, DFLASH2_SELECTOR_SYMBOL, DFLASH2_SELECTOR_THREADS};

/// Mirrors `DF2_SEL_MAX_TOP_K` / `DF2_SEL_THREADS` of the 32-candidate twin.
pub const DFLASH2_SELECTOR_WIDE_MAX_TOP_K: usize = 32;
pub const DFLASH2_SELECTOR_WIDE_THREADS: u32 = 1024;

/// One compiled selector: its module and fused entry, and the block it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dflash2SelectorBuild {
    pub module: &'static str,
    pub symbol: &'static str,
    pub threads: u32,
    pub max_top_k: usize,
    /// Whether the module carries the confidence, batched and rank-split
    /// entries (only the 16-candidate module does).
    pub full_family: bool,
}

/// The 16-candidate selector.
pub const DFLASH2_SELECTOR_BASE: Dflash2SelectorBuild = Dflash2SelectorBuild {
    module: "dflash2_candidate_selector",
    symbol: DFLASH2_SELECTOR_SYMBOL,
    threads: DFLASH2_SELECTOR_THREADS,
    max_top_k: DFLASH2_SELECTOR_MAX_TOP_K,
    full_family: true,
};

/// The 32-candidate twin.
pub const DFLASH2_SELECTOR_WIDE: Dflash2SelectorBuild = Dflash2SelectorBuild {
    module: "dflash2_candidate_selector_k32",
    symbol: "dflash2_candidate_selector_k32",
    threads: DFLASH2_SELECTOR_WIDE_THREADS,
    max_top_k: DFLASH2_SELECTOR_WIDE_MAX_TOP_K,
    full_family: false,
};

/// The selector for `top_k` candidates: the 16-candidate module up to its
/// cap (so existing drafters launch exactly what they did), the twin above it.
pub fn dflash2_selector_build(top_k: usize) -> Result<Dflash2SelectorBuild> {
    match top_k {
        1..=DFLASH2_SELECTOR_MAX_TOP_K => Ok(DFLASH2_SELECTOR_BASE),
        k if (1..=DFLASH2_SELECTOR_WIDE_MAX_TOP_K).contains(&k) => Ok(DFLASH2_SELECTOR_WIDE),
        _ => anyhow::bail!(
            "DFlash2 selector top_k {top_k} is outside the on-device selectors' range \
             1..={DFLASH2_SELECTOR_WIDE_MAX_TOP_K}"
        ),
    }
}

/// Entries only the 16-candidate module ships (rank-split walk, batched
/// selector) refuse a wider `top_k` instead of returning without writing.
pub(super) fn ensure_base_selector(top_k: u32, entry: &str) -> Result<()> {
    ensure!(
        (1..=DFLASH2_SELECTOR_MAX_TOP_K).contains(&(top_k as usize)),
        "{entry} supports selector top_k 1..={DFLASH2_SELECTOR_MAX_TOP_K}, found {top_k}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_caps_keep_the_base_module() {
        for k in [1, 4, 16] {
            assert_eq!(dflash2_selector_build(k).unwrap(), DFLASH2_SELECTOR_BASE);
        }
        assert_eq!(dflash2_selector_build(17).unwrap(), DFLASH2_SELECTOR_WIDE);
        assert_eq!(dflash2_selector_build(32).unwrap(), DFLASH2_SELECTOR_WIDE);
        assert!(dflash2_selector_build(0).is_err());
        assert!(dflash2_selector_build(33).is_err());
        assert!(ensure_base_selector(32, "walk").is_err());
        assert!(ensure_base_selector(16, "walk").is_ok());
    }
}
