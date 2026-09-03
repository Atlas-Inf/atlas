// SPDX-License-Identifier: AGPL-3.0-only

//! Shared harness for the qwen4_exp oracle-parity tests (split out of
//! `qwen4exp_oracle_tests.rs` for the 500-LoC cap).

pub(crate) fn backend() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    // By identity, NOT `ptx_modules()`: in a wildcard build that is an alias
    // for target 0, and `hyper_connection` in another target's set is
    // DeepSeek-V4's Sinkhorn kernel — the same name over a different argument
    // list, which is a segfault or, worse, plausible numbers.
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with ATLAS_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend")
}
