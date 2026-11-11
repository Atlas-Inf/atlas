// SPDX-License-Identifier: AGPL-3.0-only

//! The join of row-split outputs after a chain gather: `[m, n]` BF16 rows
//! from every rank's `[m, rows_r]` share (`rank_split_join_bf16`), one launch
//! in place of one pitched copy-engine copy a rank (4-12 us each between the
//! drafter's collectives in the dec32 nsys, 2026-10-18).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// `rank_split_join_bf16`, or 0 where the target lacks it (callers then
/// copy one share at a time).
pub fn rank_join_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    gpu.kernel("rank_split_join", "rank_split_join_bf16")
        .unwrap_or(KernelHandle(0))
}

/// Write `[m, n]` rows at `dst` from `ranks` shares `stride` bytes apart
/// from `shares`: rank r's `[m, rows_r]`, its columns from
/// `n * r / ranks / 16 * 16` (the last rank takes the rest).
#[allow(clippy::too_many_arguments)]
pub fn rank_join_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    dst: DevicePtr,
    shares: DevicePtr,
    stride: usize,
    [m, n]: [usize; 2],
    ranks: usize,
    stream: u64,
) -> Result<()> {
    let blocks = (m * n).div_ceil(256).clamp(1, 512) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(dst)
        .arg_ptr(shares)
        .arg_u64(stride as u64)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(ranks as u32)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU (`#[ignore]`): every joined word is the share's word the pitched
    /// copies would have put there, for CTA-cut and equal shares, on the
    /// pair and the chain.
    ///
    ///   cargo test --release -p spark-model --lib rank_join -- --ignored
    #[test]
    #[ignore = "needs a GB10 and a kernel build"]
    fn rank_join_matches_the_per_rank_copies() -> Result<()> {
        let gpu =
            spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
        let g: &dyn GpuBackend = &gpu;
        let kernel = rank_join_kernel(g);
        anyhow::ensure!(kernel.0 != 0, "rank_split_join_bf16 is missing");
        for (m, n, ranks) in [
            (8, 4096, 4),
            (8, 12288, 4),
            (3, 1040, 4),
            (8, 1040, 2),
            (1, 32, 2),
        ] {
            let cut = |r: usize| {
                if r == ranks {
                    n
                } else {
                    n * r / ranks / 16 * 16
                }
            };
            let widest = (0..ranks).map(|r| cut(r + 1) - cut(r)).max().unwrap();
            let stride = m * widest * 2;
            let mut shares = vec![0u16; ranks * stride / 2];
            let mut want = vec![0u16; m * n];
            for r in 0..ranks {
                let w = cut(r + 1) - cut(r);
                for i in 0..m {
                    for j in 0..w {
                        let v = (r * 7919 + i * 131 + j) as u16;
                        shares[r * stride / 2 + i * w + j] = v;
                        want[i * n + cut(r) + j] = v;
                    }
                }
            }
            let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let src = g.alloc(shares.len() * 2)?;
            let dst = g.alloc(m * n * 2)?;
            g.copy_h2d(&bytes(&shares), src)?;
            let stream = g.default_stream();
            rank_join_bf16(g, kernel, dst, src, stride, [m, n], ranks, stream)?;
            g.synchronize(stream)?;
            let mut got = vec![0u8; m * n * 2];
            g.copy_d2h(dst, &mut got)?;
            assert_eq!(got, bytes(&want), "m {m} n {n} ranks {ranks}");
            g.free(src)?;
            g.free(dst)?;
        }
        Ok(())
    }
}
