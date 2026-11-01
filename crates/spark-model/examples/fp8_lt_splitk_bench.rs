// SPDX-License-Identifier: AGPL-3.0-only
//! KDA Lt FP8 projection (`cublaslt::fp8_gemm_act_weight_t_tensorwise`,
//! N = K = 4096) per chunk row count: cuBLASLt's default heuristic vs the
//! split-K 1 pin (`no_split_k`, `ATLAS_GLM_KDA_PREFILL_LT_FP8_SPLITK1=1`).
//! Cold-L2 ms per call (median of CUDA-event-bracketed reps, each after its
//! own 64 MiB memset) and `=N` differing BF16 outputs (0 where the default
//! already runs without split-K).
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example fp8_lt_splitk_bench

use anyhow::{Result, bail};
use spark_runtime::cublaslt::fp8_gemm_act_weight_t_tensorwise as lt;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[allow(dead_code)]
#[path = "moe_verify_bench/timing.rs"]
mod timing;

const FLUSH_BYTES: usize = 64 << 20;

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (n, k, max_m) = (4096u32, 4096u32, 8196u32);
    let mut rng = 0x5eed_u64;
    // Finite E4M3 below 2.0: sign, exponent <= 7, any mantissa.
    let bytes: Vec<u8> = (0..(max_m + n) as usize * k as usize)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 40) as u8) & 0xBF
        })
        .collect();
    let act = g.alloc(max_m as usize * k as usize)?;
    let weight = g.alloc(n as usize * k as usize)?;
    let (split, single) = (
        g.alloc(max_m as usize * n as usize * 2)?,
        g.alloc(max_m as usize * n as usize * 2)?,
    );
    let timer = timing::Timer {
        stream: 0,
        scratch: g.alloc(FLUSH_BYTES)?,
        reps: 15,
    };
    let (a_bytes, w_bytes) = bytes.split_at(max_m as usize * k as usize);
    g.copy_h2d(a_bytes, act)?;
    g.copy_h2d(w_bytes, weight)?;
    let time =
        |f: &dyn Fn() -> Result<()>| -> Result<f64> { Ok(timer.time(g, true, &|_| f())? * 1e-6) };
    let read = |p: DevicePtr, count: usize| -> Result<Vec<u8>> {
        let mut b = vec![0u8; count * 2];
        g.copy_d2h(p, &mut b)?;
        Ok(b)
    };
    for m in [2048u32, 2560, 3072, 3692, 3723, 4096, 6144, 8196] {
        let t_default = time(&|| lt(act.0, weight.0, split.0, m, n, k, false, 0))?;
        let t_pinned = time(&|| lt(act.0, weight.0, single.0, m, n, k, true, 0))?;
        let count = (m * n) as usize;
        let (x, y) = (read(split, count)?, read(single, count)?);
        let differing = x
            .chunks_exact(2)
            .zip(y.chunks_exact(2))
            .filter(|(p, q)| p != q)
            .count();
        println!(
            "M={m:5}: default {:6.3} ms | split-K 1 {:6.3} ms ={differing}",
            t_default * 1e3,
            t_pinned * 1e3
        );
    }
    Ok(())
}
