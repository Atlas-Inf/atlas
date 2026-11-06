// SPDX-License-Identifier: AGPL-3.0-only

//! GDN `in_proj_ba` + gates for prefill on BF16 tensor cores
//! (`gdn_ba_gates_prefill.cu`), part of the dense BF16-MMA path.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{DenseBf16mma, KS};

/// `gdn_ba_gates_prefill` CTA tiles, tokens x outputs: the large one, and
/// the small one below `BA_SMALL_ROWS` rows (more CTAs where the K loop is
/// latency-bound: 29 vs 45 us at 96 rows, 99 vs 64 us at 1056).
const BA_TILES: [(u32, u32); 2] = [(32, 48), (16, 32)];
const BA_SMALL_ROWS: u32 = 640;
impl DenseBf16mma {
    /// Whether `ba_gates` serves an `[n, k]` BF16 in_proj_ba. At every row
    /// count: it sums in another order than the token-parallel kernel, so a
    /// choice by the pass's rows would make a request's gates depend on what
    /// it is batched with. Its two tile shapes give identical results (each
    /// output's MMA sequence is the same), so they may be picked by rows.
    pub fn serves_ba(&self, [n, k]: [u32; 2]) -> bool {
        self.ba.0 != 0
            && self.ba_small.0 != 0
            && BA_TILES.iter().all(|&(_, out)| n.is_multiple_of(out))
            && k.is_multiple_of(KS)
    }

    /// The GDN BA projection + gates on tensor cores; the arguments of
    /// `ops::dense_gemm_ba_gates_prefill`, `[m, n, k, k_stride, gate_stride,
    /// nv, vheads_per_group]`.
    pub fn ba_gates(
        &self,
        gpu: &dyn GpuBackend,
        [input, weight, a_log, dt_bias, gate_out]: [DevicePtr; 5],
        dims: [u32; 7],
        stream: u64,
    ) -> Result<()> {
        let [m, n, ..] = dims;
        let (kernel, (tok, out)) = match m < BA_SMALL_ROWS {
            true => (self.ba_small, BA_TILES[1]),
            false => (self.ba, BA_TILES[0]),
        };
        dims.iter()
            .fold(
                KernelLaunch::new(gpu, kernel)
                    .grid([n / out, m.div_ceil(tok), 1])
                    .block([128, 1, 1])
                    .arg_ptr(input)
                    .arg_ptr(weight)
                    .arg_ptr(a_log)
                    .arg_ptr(dt_bias)
                    .arg_ptr(gate_out),
                |l, &d| l.arg_u32(d),
            )
            .launch(stream)
    }
}
