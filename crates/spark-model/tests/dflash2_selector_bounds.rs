// SPDX-License-Identifier: AGPL-3.0-only
//! Bounds + parity-of-source checks for the DFlash2 on-device candidate
//! selector.
//!
//! The kernel keeps per-thread/warp/block top-k lists in fixed-size arrays
//! (`DF2_SEL_MAX_TOP_K`) and the context vector in `s_context`
//! (`DF2_SEL_MAX_RANK`). `Dflash2CandidateSelector::new` fails fast when a
//! checkpoint's `selector_top_k`/`selector_rank` exceeds the Rust mirrors —
//! so the mirrors and the kernel `#define`s must never drift apart. CUDA
//! and native HIP both use the grid-wide single-sequence selector.
use std::fs;
use std::path::{Path, PathBuf};

use spark_model::layers::ops::{
    DFLASH2_SELECTOR_MAX_RANK, DFLASH2_SELECTOR_MAX_SPLITS, DFLASH2_SELECTOR_MAX_TOP_K,
    DFLASH2_SELECTOR_THREADS,
};

const KERNEL_GB10: &str = "gb10/common/dflash2_candidate_selector.cu";
const KERNEL_STRIX: &str = "strix-hip/common/dflash2_candidate_selector.cu";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

fn kernel_path(rel: &str) -> PathBuf {
    repo_root().join("kernels").join(rel)
}

fn kernel_src(rel: &str) -> String {
    fs::read_to_string(kernel_path(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

fn kernel_define(src: &str, name: &str) -> usize {
    src.lines()
        .find_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix("#define")?.trim_start();
            let rest = rest.strip_prefix(name)?;
            let value = rest.trim();
            (!value.is_empty() && rest.starts_with(char::is_whitespace))
                .then(|| value.parse::<usize>().expect("numeric #define"))
        })
        .unwrap_or_else(|| panic!("missing #define {name}"))
}

#[test]
fn caps_match_kernel_defines() {
    for rel in [KERNEL_GB10, KERNEL_STRIX] {
        let src = kernel_src(rel);
        assert_eq!(
            kernel_define(&src, "DF2_SEL_MAX_TOP_K"),
            DFLASH2_SELECTOR_MAX_TOP_K,
            "DF2_SEL_MAX_TOP_K drifted from DFLASH2_SELECTOR_MAX_TOP_K in {rel}"
        );
        assert_eq!(
            kernel_define(&src, "DF2_SEL_MAX_RANK"),
            DFLASH2_SELECTOR_MAX_RANK,
            "DF2_SEL_MAX_RANK drifted from DFLASH2_SELECTOR_MAX_RANK in {rel}"
        );
    }
    for rel in [KERNEL_GB10, KERNEL_STRIX] {
        let src = kernel_src(rel);
        assert_eq!(
            kernel_define(&src, "DF2_SEL_MAX_SPLITS"),
            DFLASH2_SELECTOR_MAX_SPLITS
        );
        assert_eq!(
            kernel_define(&src, "DF2_SEL_THREADS"),
            DFLASH2_SELECTOR_THREADS as usize
        );
    }
}

fn kernel_params(src: &str, name: &str) -> Vec<String> {
    let sig_start = src.find(&format!("{name}(")).expect("kernel signature");
    let open = src[sig_start..].find('(').unwrap() + sig_start;
    let mut depth = 0i32;
    let mut close = None;
    for (i, c) in src[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    src[open + 1..close.expect("closing paren")]
        .split(',')
        .map(|param| param.trim().to_owned())
        .collect()
}

/// Source-level arity pin: the extern "C" signature must take exactly
/// sixteen parameters — the device anchor and ban-depth pointers after
/// `out_tokens`, `unsigned int top_k` eleventh, the four end ids, then the
/// scratch pointer —
/// matching the launcher's `.arg_*` calls in `layers/ops/sampling.rs`. (The
/// PTX-side pin in `atlas-kernels/tests/kernel_arity.rs` covers compiled
/// builds; this one also holds under `ATLAS_SKIP_BUILD`.)
#[test]
fn kernel_signature_has_sixteen_params_with_device_anchor() {
    for rel in [KERNEL_GB10, KERNEL_STRIX] {
        let src = kernel_src(rel);
        let params = kernel_params(&src, "dflash2_candidate_selector");
        assert_eq!(
            params.len(),
            16,
            "dflash2_candidate_selector must take 16 parameters: {params:?}"
        );
        assert_eq!(
            params[5], "const unsigned int* __restrict__ anchor",
            "sixth parameter must be the device anchor: {params:?}"
        );
        assert_eq!(
            params[10], "unsigned int top_k",
            "eleventh parameter must be `unsigned int top_k`: {params:?}"
        );
        assert_eq!(
            params[14], "unsigned int end3",
            "fifteenth parameter must be `unsigned int end3`: {params:?}"
        );
        assert_eq!(
            params[15], "unsigned int* __restrict__ scratch",
            "last parameter must be the scratch pointer: {params:?}"
        );
    }
}

/// Batching has a distinct one-CTA-per-sequence ABI. It must not call the
/// grid-wide single selector, whose scratch and device-ban arguments differ.
#[test]
fn batched_selector_keeps_ten_param_per_sequence_body() {
    for rel in [KERNEL_GB10, KERNEL_STRIX] {
        let src = kernel_src(rel);
        let params = kernel_params(&src, "dflash2_candidate_selector_batched");
        assert_eq!(params.len(), 10);
        assert_eq!(params[5], "const unsigned int* __restrict__ last_tokens");
        assert_eq!(params[9], "unsigned int top_k");
        let name = if rel == KERNEL_STRIX {
            "df2_selector_batched_body"
        } else {
            "df2_batch_selector_body"
        };
        let body = kernel_params(&src, name);
        assert_eq!(body.len(), 10);
        assert_eq!(body[5], "unsigned int last_token");
    }
}

#[test]
fn confidence_selector_uses_the_same_sixteen_param_entry() {
    let src = kernel_src("gb10/glm-5.3-flash/nvfp4/glm_dflash2_selector_conf.cu");
    assert!(src.contains("#define DF2_SEL_CONF 1"));
    assert!(src.contains("#define dflash2_candidate_selector dflash2_candidate_selector_conf"));
    assert!(src.contains("#include \"../../common/dflash2_candidate_selector.cu\""));
}
