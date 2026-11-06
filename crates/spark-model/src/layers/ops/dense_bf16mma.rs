// SPDX-License-Identifier: AGPL-3.0-only

//! Dense prefill GEMMs on BF16 tensor cores with register dequant
//! (`dense_bf16mma_prefill.cu`, lever `ATLAS_DENSE_PREFILL_BF16MMA=1`): the
//! NVFP4 MLP (gate+up with SiLU·mul in the epilogue, down) on the transposed
//! `[K/2, N]` prefill tables, and per-tensor FP8 projections on the
//! checkpoint `[N, K]` rows. Weights dequantize exactly to BF16 and
//! activations stay BF16, where the default kernels round activations to
//! E4M3 (W4A16 tile GEMMs) or per-128 FP8 (block-scaled FP8 GEMM).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};

const MODULE: &str = "dense_bf16mma_prefill";
/// Fewer rows are decode / verify batches; they keep their small-M kernels.
const MIN_ROWS: u32 = 32;
/// Rows per CTA chunk (`DB_ROWS`).
const ROWS: u32 = 64;
/// Threads per CTA: two 128-thread feature halves.
const THREADS: u32 = 256;
/// Output features per CTA: gate/up (64 of each per half), the others.
const GATE_UP_FEATURES: u32 = 128;
const FEATURES: u32 = 256;
/// K per pipeline stage (`DB_KS`); the FP8 scale grid's block.
const KS: u32 = 64;
const FP8_BLOCK: u32 = 128;
/// Dynamic shared memory: two 8 KiB token stages, then three NVFP4 weight
/// stages (per half 4 KiB weights + 512 B scales) or two FP8 ones (per half
/// 8 KiB) — `DbTok` / `DbW` in the kernel. Both fit two CTAs per SM.
const SMEM_NVFP4: u32 = 2 * 8192 + 3 * 2 * (4096 + 512);
const SMEM_FP8: u32 = 2 * 8192 + 2 * 2 * 8192;
const CTAS_PER_SM: u32 = 2;
/// Per-CTA cost of streaming its weight tile, in token tiles' worth of work
/// (`chunks`).
const CTA_FIXED: u32 = 4;

/// The three kernels; handle 0 where the target does not ship them.
#[derive(Clone, Copy, Debug)]
pub struct DenseBf16mma {
    gate_up: KernelHandle,
    nvfp4: KernelHandle,
    fp8: KernelHandle,
    /// GDN in_proj_ba + gates (`gdn_ba_gates_prefill.cu`).
    ba: KernelHandle,
    /// Resident CTAs (SMs x 2).
    slots: u32,
}

impl Default for DenseBf16mma {
    fn default() -> Self {
        Self {
            gate_up: KernelHandle(0),
            nvfp4: KernelHandle(0),
            fp8: KernelHandle(0),
            ba: KernelHandle(0),
            slots: 0,
        }
    }
}

/// Row chunks for `m` rows over `tiles` feature tiles: of the counts from
/// 64-row to 16-row chunks, the one with the fewest waves x per-CTA work
/// (its weight tile plus its 8-row token tiles). A short M x narrow N grid
/// (down at 288 rows: 5 chunks x 20 tiles = 100 CTAs on 96 slots) takes
/// smaller chunks instead of a nearly empty second wave. Results do not
/// depend on it: every output is one CTA's fixed-order K loop.
fn chunks(m: u32, tiles: u32, slots: u32) -> u32 {
    (m.div_ceil(ROWS)..=m.div_ceil(16))
        .min_by_key(|&c| (tiles * c).div_ceil(slots) * (CTA_FIXED + m.div_ceil(c).div_ceil(8)))
        .unwrap_or(1)
}

/// A transposed NVFP4 table this kernel reads: scalar tensor scale, shapes
/// that tile exactly.
fn nvfp4_ok(w: &QuantizedWeight, n: u32, k: u32, features: u32) -> bool {
    !w.weight.is_null()
        && !w.weight_scale.is_null()
        && !w.has_per_row_scale2()
        && n.is_multiple_of(features)
        && k.is_multiple_of(KS)
}

impl DenseBf16mma {
    /// `ATLAS_DENSE_PREFILL_BF16MMA=1`, read where the layers are built: the
    /// kernels ship only in the qwen3.6/3.8-27B target, and a lookup that
    /// misses fails the boot audit, so other targets must not issue it.
    /// Dispatch reads the same lever from `ModelLevers`.
    pub fn lever() -> bool {
        std::env::var("ATLAS_DENSE_PREFILL_BF16MMA").as_deref() == Ok("1")
    }

    /// The kernels when the lever is on, else all handles 0.
    pub fn load(gpu: &dyn GpuBackend) -> Self {
        if !Self::lever() {
            return Self::default();
        }
        Self::resolve(gpu)
    }

    fn resolve(gpu: &dyn GpuBackend) -> Self {
        Self {
            gate_up: crate::layers::try_kernel(gpu, MODULE, "dense_bf16mma_gate_up_silu"),
            nvfp4: crate::layers::try_kernel(gpu, MODULE, "dense_bf16mma_nvfp4"),
            fp8: crate::layers::try_kernel(gpu, MODULE, "dense_bf16mma_fp8"),
            ba: crate::layers::try_kernel(gpu, "gdn_ba_gates_prefill", "gdn_ba_gates_prefill"),
            slots: CTAS_PER_SM * gpu.sm_count().unwrap_or(48),
        }
    }

    /// `[grid, nchunks]` for `m` rows of `n` features, `features` per CTA.
    fn geometry(&self, m: u32, n: u32, features: u32) -> ([u32; 3], u32) {
        let c = chunks(m, n / features, self.slots.max(1));
        ([(n / features) * c, 1, 1], c)
    }

    /// Whether `gate_up_silu` + `nvfp4` serve `m` rows of an MLP with these
    /// transposed tables (`[inter, h]` gate/up, `[h, inter]` down).
    pub fn serves_mlp(
        &self,
        [gate, up, down]: [&QuantizedWeight; 3],
        [m, h, inter]: [u32; 3],
    ) -> bool {
        m >= MIN_ROWS
            && self.gate_up.0 != 0
            && self.nvfp4.0 != 0
            && nvfp4_ok(gate, inter, h, GATE_UP_FEATURES)
            && nvfp4_ok(up, inter, h, GATE_UP_FEATURES)
            && nvfp4_ok(down, h, inter, FEATURES)
    }

    /// `out[m, inter] = silu(input · gateᵀ) * (input · upᵀ)`, both rounded to
    /// BF16 before the SiLU as the unfused chain does.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up_silu(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        [gate, up]: [&QuantizedWeight; 2],
        out: DevicePtr,
        [m, inter, h]: [u32; 3],
        stream: u64,
    ) -> Result<()> {
        let (grid, nchunks) = self.geometry(m, inter, GATE_UP_FEATURES);
        KernelLaunch::new(gpu, self.gate_up)
            .grid(grid)
            .block([THREADS, 1, 1])
            .shared_mem(SMEM_NVFP4)
            .arg_ptr(input)
            .arg_ptr(gate.weight)
            .arg_ptr(gate.weight_scale)
            .arg_f32(gate.weight_scale_2)
            .arg_ptr(up.weight)
            .arg_ptr(up.weight_scale)
            .arg_f32(up.weight_scale_2)
            .arg_ptr(out)
            .arg_u32(m)
            .arg_u32(inter)
            .arg_u32(h)
            .arg_u32(nchunks)
            .launch(stream)
    }

    /// Whether `nvfp4` serves `m` rows of an `[n, k]` transposed NVFP4 table.
    pub fn serves_nvfp4(&self, w: &QuantizedWeight, [m, n, k]: [u32; 3]) -> bool {
        m >= MIN_ROWS && self.nvfp4.0 != 0 && nvfp4_ok(w, n, k, FEATURES)
    }

    /// `out[m, n] = input[m, k] · wᵀ` for a transposed NVFP4 table.
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &QuantizedWeight,
        out: DevicePtr,
        [m, n, k]: [u32; 3],
        stream: u64,
    ) -> Result<()> {
        let (grid, nchunks) = self.geometry(m, n, FEATURES);
        KernelLaunch::new(gpu, self.nvfp4)
            .grid(grid)
            .block([THREADS, 1, 1])
            .shared_mem(SMEM_NVFP4)
            .arg_ptr(input)
            .arg_ptr(w.weight)
            .arg_ptr(w.weight_scale)
            .arg_f32(w.weight_scale_2)
            .arg_ptr(out)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(n)
            .arg_u32(nchunks)
            .launch(stream)
    }

    /// Whether `fp8` serves `m` rows of this weight: kernel shipped, shapes
    /// tile, and the block-scale grid is constant along K (`k_uniform`, see
    /// [`fp8_scale_k_uniform`]).
    pub fn serves_fp8(&self, w: &Fp8Weight, k_uniform: bool, m: u32) -> bool {
        m >= MIN_ROWS
            && self.fp8.0 != 0
            && k_uniform
            && w.scale_format == WeightQuantFormat::Fp8BlockScaled
            && w.n.is_multiple_of(FEATURES)
            && w.k.is_multiple_of(FP8_BLOCK)
    }

    /// `out[m, w.n] = input[m, w.k] · wᵀ` for an `[N, K]` e4m3 weight.
    pub fn fp8(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &Fp8Weight,
        out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        let (grid, nchunks) = self.geometry(m, w.n, FEATURES);
        KernelLaunch::new(gpu, self.fp8)
            .grid(grid)
            .block([THREADS, 1, 1])
            .shared_mem(SMEM_FP8)
            .arg_ptr(input)
            .arg_ptr(w.weight)
            .arg_ptr(w.row_scale)
            .arg_ptr(out)
            .arg_u32(m)
            .arg_u32(w.n)
            .arg_u32(w.k)
            .arg_u32(w.n)
            .arg_u32(nchunks)
            .launch(stream)
    }
}

/// `gdn_ba_gates_prefill` CTA tile: tokens x outputs.
const BA_TOK: u32 = 32;
const BA_OUT: u32 = 48;
/// Below this its few CTAs lose to the token-parallel kernel (one score pass,
/// per layer: 96 rows 79 vs 62 us; 1056 rows 155 vs 674 us).
const BA_MIN_ROWS: u32 = 256;

impl DenseBf16mma {
    /// Whether `ba_gates` serves `m` rows of an `[n, k]` BF16 in_proj_ba.
    pub fn serves_ba(&self, [m, n, k]: [u32; 3]) -> bool {
        m >= BA_MIN_ROWS && self.ba.0 != 0 && n.is_multiple_of(BA_OUT) && k.is_multiple_of(KS)
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
        dims.iter()
            .fold(
                KernelLaunch::new(gpu, self.ba)
                    .grid([n / BA_OUT, m.div_ceil(BA_TOK), 1])
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

/// Whether an FP8 weight's FP32 block-scale grid `[N/128, K/128]` is constant
/// along K (true for per-tensor and per-channel checkpoints), so the kernel
/// can apply it per output feature in the epilogue. Load time: one D2H copy.
pub fn fp8_scale_k_uniform(gpu: &dyn GpuBackend, w: &Fp8Weight) -> Result<bool> {
    if w.scale_format != WeightQuantFormat::Fp8BlockScaled
        || w.row_scale.is_null()
        || !w.n.is_multiple_of(FP8_BLOCK)
        || !w.k.is_multiple_of(FP8_BLOCK)
    {
        return Ok(false);
    }
    let (nb, kb) = ((w.n / FP8_BLOCK) as usize, (w.k / FP8_BLOCK) as usize);
    let mut bytes = vec![0u8; nb * kb * 4];
    gpu.copy_d2h(w.row_scale, &mut bytes)?;
    let grid: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(grid
        .chunks_exact(kb)
        .all(|row| row.iter().all(|&s| s == row[0])))
}

#[cfg(test)]
#[path = "dense_bf16mma_test_gpu.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::chunks;

    #[test]
    fn chunks_stay_within_the_kernel_tile_and_fill_short_grids() {
        for m in 1..5000 {
            for tiles in [20, 64, 136] {
                let c = chunks(m, tiles, 96);
                assert!(
                    c >= m.div_ceil(64) && m.div_ceil(c) <= 64,
                    "m={m} tiles={tiles} c={c}"
                );
                // The kernel's balanced split never leaves an empty chunk.
                assert!((c - 1) * m.div_ceil(c) < m, "m={m} tiles={tiles} c={c}");
            }
        }
        // down / out_proj at 288 rows: not 5 x 20 = 100 CTAs on 96 slots.
        assert!(chunks(288, 20, 96) * 20 <= 96 || chunks(288, 20, 96) * 20 > 150);
        // Full waves already: keep 64-row chunks.
        assert_eq!(chunks(4128, 136, 96), 65);
    }
}
