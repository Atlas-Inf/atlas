#pragma once
// SPDX-License-Identifier: AGPL-3.0-only
//
// CUDA cache-hint load intrinsics that HIP does not declare.
//
// `__ldcg` ("cache global": serve the load from L2, not L1) is how the shared
// kernels read rows another thread of the same kernel just wrote: the L1 may
// hold a stale copy, and L2 is the coherence point. HIP ships `__ldg` but no
// `__ldcg`, so every kernel using it stopped compiling on gfx1151 once #174
// added it to the shared DFlash2 selector and the QSA indexer.
//
// On AMDGPU a volatile global load is emitted coherent (glc/dlc on gfx11:
// it bypasses the non-coherent per-CU L0/L1 and reads from L2), which is the
// same cache behaviour. Like `__ldcg` itself it is not a fence: the callers
// order their cross-block reads with __threadfence() plus an atomic ticket
// (the DFlash2 selector's last-block walk) or a block barrier (the QSA
// indexer), and on gfx11 the agent-scope fence already invalidates L0/L1.
// Scalar types only (one load instruction): every current call site loads a
// float, an unsigned int or an unsigned short.
//
// Force-included on every HIP compile (build_target.rs), after hip_runtime.h.
#if defined(__HIP__)
template <typename T>
__device__ __forceinline__ T __ldcg(const T* p) {
    static_assert(sizeof(T) <= 8, "__ldcg shim: scalar loads only (a wider volatile load may tear)");
    return *reinterpret_cast<const volatile T*>(p);
}
#endif
