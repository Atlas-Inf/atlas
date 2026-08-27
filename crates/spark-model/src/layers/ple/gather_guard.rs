// SPDX-License-Identifier: AGPL-3.0-only

//! The guard that ties the n-gram table's element type to its gather kernel.
//!
//! Split from `layer.rs` for the 500-LoC cap. Pure, so the bug it guards —
//! which needed a 126 GB checkpoint and a GB10 to reveal — is now caught by a
//! test that needs neither.

use anyhow::Result;

/// Refuse a row stride and scale that do not describe the same element type.
///
/// `stride == head_dim` is one byte per element, which is FP8 and MUST carry a
/// dequant scale. `stride == head_dim * 2` is BF16 and must NOT — a scale
/// there would mean the gather multiplies by something the reference does not.
/// Anything else is a geometry the gather kernels cannot express.
///
/// Pure so it is testable without a GPU: the bug it guards needed a 126 GB
/// checkpoint and a GB10 to reveal, and this needs neither.
// Reached from `PleLayer::new` only under cuda; its own tests exercise it on
// every backend, which is the point — the guard is pure.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn gather_matches_element_size(
    stride: usize,
    head_dim: usize,
    scaled: bool,
) -> Result<()> {
    // EXACT multiples only. `stride / head_dim` would call 240/160 "1 byte"
    // and wave a bad geometry through — which is what this file's own test
    // caught on the first run.
    let elem = if head_dim == 0 {
        0
    } else if stride == head_dim {
        1
    } else if stride == head_dim * 2 {
        2
    } else {
        0
    };
    match (elem, scaled) {
        (1, true) | (2, false) => Ok(()),
        (1, false) => anyhow::bail!(
            "PLE: the n-gram table is FP8 (row stride {stride} = head_dim \
             {head_dim} x 1 B) but carries NO dequant scale, so the gather \
             would return raw E4M3 magnitudes"
        ),
        (2, true) => anyhow::bail!(
            "PLE: the n-gram table is BF16 (row stride {stride} = head_dim \
             {head_dim} x 2 B) but carries a dequant scale, which the BF16 \
             gather would ignore and the FP8 gather would misapply"
        ),
        _ => anyhow::bail!(
            "PLE: row stride {stride} is not head_dim {head_dim} x 1 or x 2; \
             the gather kernels read FP8 or BF16 rows and nothing else"
        ),
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::gather_matches_element_size as check;

    /// The two shapes that exist, and the two that are the bug.
    #[test]
    fn element_size_and_scale_must_agree() {
        // FP8 with its scale, and BF16 without: the only legal pairs.
        assert!(check(160, 160, true).is_ok(), "FP8 + scale");
        assert!(check(320, 160, false).is_ok(), "BF16, no scale");

        // THE SHIPPED BUG, in the form it would now be caught: RadixArk's
        // F8_E4M3 table reached the BF16 gather, which is this row once the
        // scale is the thing selecting the kernel.
        let e = check(160, 160, false).unwrap_err().to_string();
        assert!(e.contains("FP8"), "{e}");
        assert!(e.contains("NO dequant scale"), "{e}");

        // And its mirror, which would silently apply a scale twice.
        let e = check(320, 160, true).unwrap_err().to_string();
        assert!(e.contains("BF16"), "{e}");

        // A stride that is neither is refused rather than rounded.
        assert!(check(240, 160, true).is_err());
        assert!(check(0, 160, false).is_err());
    }
}
