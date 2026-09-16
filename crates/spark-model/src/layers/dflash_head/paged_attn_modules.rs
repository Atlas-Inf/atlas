// SPDX-License-Identifier: AGPL-3.0-only
//! (module, symbol) pairs for the DFlash Option-B sink paged-attention
//! kernels, selected by the drafter's `head_dim`. The wrappers include
//! `prefill_paged_compute.cuh` whose tile geometry is baked at compile time
//! via `HDIM`; dispatch must pick the build matching `kv_config.head_dim`.

use anyhow::bail;

pub(super) struct PagedSinkModules {
    pub sink: (&'static str, &'static str),
    pub indirect: (&'static str, &'static str),
    pub batched: (&'static str, &'static str),
}

pub(super) fn paged_sink_modules_for_head_dim(head_dim: usize) -> anyhow::Result<PagedSinkModules> {
    match head_dim {
        128 => Ok(PagedSinkModules {
            sink: (
                "prefill_paged_sink_h128",
                "inferspark_prefill_paged_sink_h128",
            ),
            indirect: (
                "prefill_paged_indirect_sink_h128",
                "inferspark_prefill_paged_indirect_sink_h128",
            ),
            batched: (
                "inferspark_prefill_paged_batched_sink_h128",
                "inferspark_prefill_paged_batched_sink_h128",
            ),
        }),
        256 => Ok(PagedSinkModules {
            sink: ("prefill_paged_sink", "inferspark_prefill_paged_sink"),
            indirect: (
                "prefill_paged_indirect_sink",
                "inferspark_prefill_paged_indirect_sink",
            ),
            batched: (
                "inferspark_prefill_paged_batched_sink",
                "inferspark_prefill_paged_batched_sink",
            ),
        }),
        other => bail!(
            "DFlash paged sink attention has no HDIM={other} kernel build \
             (supported head_dim: 128, 256)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_128_selects_h128_builds() {
        let m = paged_sink_modules_for_head_dim(128).unwrap();
        assert_eq!(m.sink.0, "prefill_paged_sink_h128");
        assert_eq!(m.indirect.0, "prefill_paged_indirect_sink_h128");
        assert_eq!(m.batched.0, "inferspark_prefill_paged_batched_sink_h128");
    }

    #[test]
    fn head_dim_256_keeps_the_original_names() {
        let m = paged_sink_modules_for_head_dim(256).unwrap();
        assert_eq!(m.sink.0, "prefill_paged_sink");
        assert_eq!(m.indirect.0, "prefill_paged_indirect_sink");
        assert_eq!(m.batched.0, "inferspark_prefill_paged_batched_sink");
    }

    #[test]
    fn unsupported_head_dim_errors() {
        assert!(paged_sink_modules_for_head_dim(64).is_err());
    }
}
