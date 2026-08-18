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
