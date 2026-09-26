// SPDX-License-Identifier: AGPL-3.0-only

//! Dedicated Nemotron 3.5 Lightning target contract pins.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn model_toml() -> toml::Value {
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("kernels/gb10/nemotron-3.5-lightning-30b-a3b/MODEL.toml");
    toml::from_str(&std::fs::read_to_string(root).expect("Lightning MODEL.toml")).unwrap()
}

#[test]
fn lightning_declares_exact_proven_try_kernel_fallbacks() {
    let value = model_toml();
    let expected = value
        .get("expected_absent")
        .and_then(toml::Value::as_table)
        .expect("Lightning must declare its expected-absent try_kernel probes");
    let found: BTreeSet<(String, String)> = expected
        .iter()
        .flat_map(|(module, entries)| {
            entries
                .as_table()
                .expect("module expected_absent is a table")
                .iter()
                .map(move |(kernel, reason)| {
                    assert!(
                        reason.as_str().is_some_and(|s| !s.trim().is_empty()),
                        "{module}::{kernel} requires a non-empty fallback reason"
                    );
                    (module.clone(), kernel.clone())
                })
        })
        .collect();
    let required: BTreeSet<(String, String)> = [
        ("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_relu2"),
        ("moe_w4a4", "moe_w4a4_grouped_gemm_relu2"),
        ("w4a16", "fp8_fp8_gemm_t_m128_mfast"),
        ("w4a16", "fp8_gemm_t_m128_mfast"),
        ("w4a16", "fp8_gemm_t_row_scaled"),
        ("w4a16", "fp8_gemm_t_row_scaled_m16"),
        ("w4a16", "w4a16_gemm_t_k64_n64_p3"),
        ("w4a16", "w4a16_gemm_t_k64_p3"),
        ("w4a16", "w4a16_gemm_t_m128_bf16"),
        ("w4a16", "w4a16_gemm_t_p3"),
        ("w4a4", "w4a4_gemm_mfast"),
    ]
    .into_iter()
    .map(|(m, k)| (m.to_owned(), k.to_owned()))
    .collect();
    assert_eq!(found, required, "Lightning expected-absent set drifted");
}
