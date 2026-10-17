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
// same guarantee. Scalar types only: every current call site loads a float,
// an unsigned int or an unsigned short.
//
// Force-included on every HIP compile (build_target.rs), after hip_runtime.h.
#if defined(__HIP__)
template <typename T>
__device__ __forceinline__ T __ldcg(const T* p) {
    return *reinterpret_cast<const volatile T*>(p);
}
#endif
