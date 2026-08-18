// SPDX-License-Identifier: AGPL-3.0-only

//! Pins per-stream causal offsets for batched varlen paged attention.

use std::path::PathBuf;

#[test]
fn varlen_batched_attention_derives_q_offset_per_stream() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let src = std::fs::read_to_string(root.join("kernels/gb10/common/prefill_paged_compute.cuh"))
        .unwrap();
    assert!(src.contains("q_offset = kv_len - q_len_eff;"));
    assert!(src.contains("if (cu_seqlens != nullptr && kv_len >= q_len_eff)"));
}

#[test]
fn dspark_batched_attention_requires_sink_aware_kernel() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let src = std::fs::read_to_string(
        root.join("kernels/gb10/common/inferspark_prefill_paged_batched_sink.cu"),
    )
    .unwrap();
    assert!(src.contains("#define PREFILL_BATCHED"));
    assert!(src.contains("#define ATLAS_ATTN_SINKS"));
    assert!(src.contains("#define KERNEL_NAME inferspark_prefill_paged_batched_sink"));
    assert!(src.contains("const __nv_bfloat16* __restrict__ sinks"));
}

#[test]
fn dspark_markov_kernels_are_batch_wide_and_depth_indexed() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let src =
        std::fs::read_to_string(root.join("kernels/gb10/common/dflash_batch_markov.cu")).unwrap();
    assert!(src.contains("dflash_batch_add_depth_bias"));
    assert!(src.contains("sequence * gamma + depth"));
    assert!(src.contains("dflash_batch_store_depth_tokens"));
    assert!(src.contains("tokens[sequence * gamma + depth]"));
}
