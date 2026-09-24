// SPDX-License-Identifier: AGPL-3.0-only
//! Bounds + parity-of-source checks for the DFlash2 on-device candidate
//! selector.
//!
//! The kernel keeps per-thread/warp/block top-k lists in fixed-size arrays
//! (`DF2_SEL_MAX_TOP_K`) and the context vector in `s_context`
//! (`DF2_SEL_MAX_RANK`). `Dflash2CandidateSelector::new` fails fast when a
//! checkpoint's `selector_top_k`/`selector_rank` exceeds the Rust mirrors —
//! so the mirrors and the kernel `#define`s must never drift apart, and the
//! gb10/strix-hip copies of the kernel must stay byte-identical.
use std::fs;
use std::path::{Path, PathBuf};

use spark_model::layers::ops::{DFLASH2_SELECTOR_MAX_RANK, DFLASH2_SELECTOR_MAX_TOP_K};

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
}

#[test]
fn gb10_and_strix_hip_copies_are_byte_identical() {
    let gb10 = fs::read(kernel_path(KERNEL_GB10)).expect("read gb10 selector");
    let strix = fs::read(kernel_path(KERNEL_STRIX)).expect("read strix-hip selector");
    assert_eq!(
        gb10, strix,
        "{KERNEL_GB10} and {KERNEL_STRIX} must stay byte-identical copies"
    );
}

/// Source-level arity pin: the extern "C" signature must take exactly ten
/// parameters, the last being `unsigned int top_k` — matching the launcher's
/// ten `.arg_*` calls in `layers/ops/sampling.rs`. (The PTX-side pin in
/// `atlas-kernels/tests/kernel_arity.rs` covers compiled builds; this one
/// also holds under `ATLAS_SKIP_BUILD`.)
#[test]
fn kernel_signature_has_ten_params_ending_in_top_k() {
    let src = kernel_src(KERNEL_GB10);
    let sig_start = src
        .find("dflash2_candidate_selector(")
        .expect("kernel signature");
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
    let params: Vec<&str> = src[open + 1..close.expect("closing paren")]
        .split(',')
        .map(str::trim)
        .collect();
    assert_eq!(
        params.len(),
        10,
        "dflash2_candidate_selector must take 10 parameters: {params:?}"
    );
    assert_eq!(
        params[9], "unsigned int top_k",
        "tenth parameter must be `unsigned int top_k`: {params:?}"
    );
}
