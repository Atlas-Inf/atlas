// SPDX-License-Identifier: AGPL-3.0-only

// cudaGetLastError() returns, and clears, the calling thread's sticky runtime
// error whoever set it. CUTLASS's Gemm::run() reads it right after its launch,
// and so do the pack/transpose launches here. An error an unrelated earlier
// runtime call left unchecked therefore made a correct launch report failure;
// callers that then recomputed in another backend got differently-rounded
// output on some calls only.
//
// Every wrapper entry point drains the sticky error first, so an error read
// after a launch belongs to that launch.

#pragma once

#include <atomic>

#include <cuda_runtime_api.h>

// Drained errors so far and the most recent code, for the host to report once.
inline std::atomic<unsigned long long> atlas_stale_cuda_error_count{0};
inline std::atomic<int> atlas_stale_cuda_error_last{0};

inline void atlas_drain_stale_cuda_error() {
  const cudaError_t stale = cudaGetLastError();
  if (stale != cudaSuccess) {
    atlas_stale_cuda_error_last.store(static_cast<int>(stale), std::memory_order_relaxed);
    atlas_stale_cuda_error_count.fetch_add(1, std::memory_order_relaxed);
  }
}

// Status a wrapper returns when the GEMM launch itself failed: the output may
// be partly written, which no caller may paper over with another backend.
// Operand rejections before any launch keep their small CUTLASS codes.
constexpr int ATLAS_CUTLASS_LAUNCH_FAILED = 1000;

inline int atlas_launch_status(int cutlass_status) {
  return cutlass_status == 0 ? 0 : ATLAS_CUTLASS_LAUNCH_FAILED + cutlass_status;
}
