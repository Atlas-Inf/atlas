// SPDX-License-Identifier: AGPL-3.0-only

//! Raw-GEMM plumbing split out of `hyper_connection_lowrank.rs` to keep
//! that file under the 500-LoC cap; same module surface via re-import.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use spark_runtime::kernel_args::KernelLaunch;

/// `dense_gemm_bf16_pipelined` launch over raw BF16 pointers (the hc weights
/// are plain `DevicePtr`s, not `DenseWeight`s). Mirrors
/// `ops::dense_gemm_bf16_pipelined` exactly: out[m,n] = a[m,k] x w[n,k]^T.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_raw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), m.div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Block width for stage 3 (`ATLAS_HC_FIN_BLOCK`, default 128). Kept as a knob
/// because it is pure launch geometry — it cannot change the arithmetic, only
/// how much of the machine runs it.
/// `ATLAS_HC_FIN_X4`: stream-per-warp `hc_pre_finish_x4` (default on) vs the
/// thread-per-`d` `hc_pre_finish` (`=0`). Read once per process.
pub(crate) fn hc_finish_x4() -> bool {
    static X4: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *X4.get_or_init(|| std::env::var("ATLAS_HC_FIN_X4").as_deref() != Ok("0"))
}

pub(crate) fn hc_finish_block() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_HC_FIN_BLOCK")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|n| (32..=1024).contains(n) && n % 32 == 0)
            .unwrap_or(128)
    })
}

/// One projection of the collapse, on whichever GEMM actually fills the part.
///
/// `dense_gemm_bf16_pipelined` emits a 128x128 output tile, so its grid is
/// `[ceil(N/128), ceil(M/128)]`. Two of this collapse's three projections are
/// skinny in N, and that grid leaves the machine idle. nsys over the 96-token
/// prefill chunk of a 118-token prompt (2026-08-30, 483 ms window):
///
///   projection      M      N      K      grid    us/call   total
///   up             96  10240    320    80x1x1       43.7    4.3 ms
///   down           96    320  10240     3x1x1      374.5   36.3 ms
///   inject         96      4  10240     1x1x1      409.4   39.3 ms
///
/// `up` and `down` are the SAME 96x320x10240 MAC count and differ 8.6x, and
/// `inject` does 80x LESS arithmetic than `down` in MORE time. The cost is the
/// grid, not the math: three CTAs and one CTA on a 48-SM part. Together the
/// two skinny projections were 75.6 ms, **15.7% of the whole prefill window**.
///
/// N <= 320 against K = 10240 is a split-K shape, which the tile kernel does
/// not have and cuBLASLt picks automatically. So route by whether the tile
/// grid can cover the part -- the same machine-fill rule
/// `qwen3_ssm::kernel_select` already uses (`n.div_ceil(128) >= sm_count`) --
/// and leave `up`, whose 80 CTAs already do, where it is.
///
/// The fallback is not a policy knob. cuBLASLt is linked unconditionally and
/// pre-warmed at load (`serve_load.rs`), but if the handle cannot be created
/// the collapse must still compute rather than fail the request; it warns
/// once and then runs the tile kernel forever.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    if n.div_ceil(128) * m.div_ceil(128) < sm_count {
        match crate::layers::ops::cublas_bf16_proj_dense(a, w, out, m, n, k, stream) {
            Ok(()) => return Ok(()),
            Err(e) => {
                static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                if WARNED.set(()).is_ok() {
                    tracing::warn!(
                        "mHC collapse: cuBLASLt unavailable ({e}); the skinny \
                         projections fall back to the tile kernel, which \
                         launches {} CTAs on {sm_count} SMs",
                        n.div_ceil(128) * m.div_ceil(128),
                    );
                }
            }
        }
    }
    gemm_raw(gpu, kernel, a, w, out, m, n, k, stream)
}
