// SPDX-License-Identifier: AGPL-3.0-only

//! Whether a dense qwen3_5 full-attention layer keeps its checkpoint FP8
//! q/k/v/o (`w8a16_gemv` decode, FP8 prefill) instead of the FP8 -> BF16 ->
//! NVFP4 requant. Always on HIP; on CUDA only with
//! `ATLAS_DENSE_NATIVE_FP8_ATTN=1` (default off; nvidia/Qwen3.8-27B-NVFP4
//! ships these sixteen layers FP8 per-tensor, which the requant double
//! quantizes). `ATLAS_NO_ATTN_FP8` forces the requant on both, for A/B.
//! Single GPU only (no FP8 TP sharding here), and every projection must carry
//! a scale the block-scaled kernels read (`proj_is_fp8_any_scale`).

use spark_runtime::weights::WeightStore;

use crate::layers::ops::ModelLevers;

/// The CUDA opt-in.
pub(super) const LEVER: &str = "ATLAS_DENSE_NATIVE_FP8_ATTN";

const PROJS: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// The decision for `{p}.{q,k,v,o}_proj` (`p` = `...self_attn`).
pub(super) fn native(store: &WeightStore, p: &str, tp_size: usize) -> bool {
    let lever = ModelLevers::opt_in(LEVER);
    let on = native_from(
        cfg!(atlas_hip),
        lever,
        std::env::var_os("ATLAS_NO_ATTN_FP8").is_some(),
        tp_size,
        || all_fp8(store, p),
    );
    if lever && !cfg!(atlas_hip) {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| match on {
            true => tracing::info!(
                "{LEVER}=1: full-attention q/k/v/o stay native FP8 (w8a16 decode \
                 and verify, FP8 prefill; no NVFP4 requant)"
            ),
            false => tracing::warn!(
                "{LEVER}=1 has no effect on {p}: needs TP=1 (tp={tp_size}), no \
                 ATLAS_NO_ATTN_FP8, and FP8 q/k/v/o with a per-tensor or 128x128 scale"
            ),
        });
    }
    on
}

/// Every projection FP8 with a scale the block-scaled kernels read.
fn all_fp8(store: &WeightStore, p: &str) -> bool {
    PROJS
        .iter()
        .all(|n| super::proj_is_fp8_any_scale(store, &format!("{p}.{n}")))
}

/// [`native`] on its inputs; `all_fp8` is read only when the rest allows.
fn native_from(
    hip: bool,
    lever: bool,
    killed: bool,
    tp_size: usize,
    all_fp8: impl FnOnce() -> bool,
) -> bool {
    (hip || lever) && !killed && tp_size == 1 && all_fp8()
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::weights::{WeightDtype, WeightTensor};
    use std::collections::HashMap;

    fn tensor(shape: Vec<usize>, dtype: WeightDtype) -> WeightTensor {
        WeightTensor {
            ptr: DevicePtr::NULL,
            shape,
            dtype,
        }
    }

    /// q/k/v/o as nvidia/Qwen3.8-27B-NVFP4 ships them: FP8 + scalar F32 scale;
    /// `o` overrides the o_proj weight dtype.
    fn attention(o: WeightDtype) -> WeightStore {
        let mut m = HashMap::new();
        for (name, n, k) in [
            ("q_proj", 12288, 5120),
            ("k_proj", 1024, 5120),
            ("v_proj", 1024, 5120),
            ("o_proj", 5120, 6144),
        ] {
            let dtype = if name == "o_proj" {
                o
            } else {
                WeightDtype::FP8E4M3
            };
            m.insert(format!("a.{name}.weight"), tensor(vec![n, k], dtype));
            m.insert(
                format!("a.{name}.weight_scale"),
                tensor(vec![], WeightDtype::FP32),
            );
        }
        WeightStore::from_map(m)
    }

    #[test]
    fn cuda_needs_the_lever() {
        assert!(!native_from(false, false, false, 1, || true));
        assert!(native_from(false, true, false, 1, || true));
        // HIP: on by default, as before the lever existed.
        assert!(native_from(true, false, false, 1, || true));
    }

    #[test]
    fn kill_switch_tp_and_format_refuse() {
        assert!(!native_from(false, true, true, 1, || true));
        assert!(!native_from(true, false, true, 1, || true));
        assert!(!native_from(false, true, false, 2, || true));
        assert!(!native_from(false, true, false, 1, || false));
    }

    #[test]
    fn the_checkpoint_is_read_only_when_the_lever_allows() {
        let read = std::cell::Cell::new(false);
        native_from(false, false, false, 1, || {
            read.set(true);
            true
        });
        assert!(!read.get());
    }

    #[test]
    fn every_projection_must_be_native_fp8() {
        assert!(all_fp8(&attention(WeightDtype::FP8E4M3), "a"));
        assert!(!all_fp8(&attention(WeightDtype::BF16), "a"));
        assert!(!all_fp8(&attention(WeightDtype::FP8E4M3), "missing"));
    }
}
