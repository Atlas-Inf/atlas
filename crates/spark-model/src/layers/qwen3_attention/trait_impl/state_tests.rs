// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn test_alloc_state_returns_empty() {
    let gpu = MockGpuBackend::new();
    assert!(gpu.kernel("norm", "rms_norm").is_ok());
    assert!(gpu.kernel("rope", "rope_forward").is_ok());
    assert!(
        gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")
            .is_ok()
    );
}
