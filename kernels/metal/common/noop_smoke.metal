// SPDX-License-Identifier: AGPL-3.0-only
//
// Minimal smoke kernel exercised by `metal_alloc_copy_launch_roundtrip`
// in `crates/spark-runtime/src/metal_backend.rs`. Zeros the first `n`
// floats of `out`. Kept around as a self-contained correctness probe
// for the build + launch pipeline; production kernels live alongside.

#include <metal_stdlib>
using namespace metal;

kernel void noop_smoke(
    device float *out [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        out[gid] = 0.0f;
    }
}
