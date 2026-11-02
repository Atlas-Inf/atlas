// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Programmatic dependent launch (PDL). A kernel launched with the runtime's
// programmatic-serialization attribute (ATLAS_PDL=1, kernels on the runtime's
// PDL list) may be scheduled before its stream predecessor finishes. Entry
// order: let our own successor be scheduled once all our blocks have started,
// then wait until the predecessor's writes are visible. Every block runs this
// before any other work (early exits included), so "kernel N complete"
// still implies "kernel N-1 complete". Without the attribute both are no-ops.
__device__ __forceinline__ void atlas_pdl_enter() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
