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

use crate::weight_map::{Fp8Weight, Fp8WeightTransposed, QuantizedWeight, WeightQuantFormat};

const MODULE: &str = "dense_bf16mma_prefill";
/// Fewer rows are decode / verify batches; they keep their small-M kernels.
const MIN_ROWS: u32 = 32;
/// Rows per CTA chunk (`DB_ROWS`; the low-precision kernels `DL_ROWS`).
const ROWS: u32 = 64;
const LOWP_ROWS: u32 = 128;
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
/// Low-precision kernels (`db_smem_bytes`, one CTA per SM, four stages):
/// the 128-row BF16 epilogue tile bounds W4A4 (4.5 KiB NVFP4 tokens and
/// 9 KiB weights per stage); W8A8 8 KiB E4M3 tokens and 16 KiB weights,
/// W4A8 8 KiB tokens and 9 KiB weights.
const SMEM_W4A4: u32 = LOWP_ROWS * (256 + 8) * 2;
/// The quantizers hold a row in registers: 16 x 256 x `DL_QB` values.
const QUANT_MAX_K: u32 = 16 * 256 * 5;
const SMEM_W8A8: u32 = 4 * (8192 + 2 * 8192);
const SMEM_W4A8: u32 = 4 * (8192 + 2 * (4096 + 512));
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
    /// Low-precision twins and the activation quantizers (loaded only for a
    /// non-empty `lowp`).
    w4a4_gate_up: KernelHandle,
    w4a4: KernelHandle,
    w4a8_gate_up: KernelHandle,
    w4a8: KernelHandle,
    w8a8: KernelHandle,
    /// W8A8 on the transposed `[K, N]` prefill copy.
    w8a8_t: KernelHandle,
    quant_nvfp4: KernelHandle,
    quant_fp8: KernelHandle,
    lowp: LowpMix,
    /// GDN in_proj_ba + gates (`gdn_ba_gates_prefill.cu`).
    ba: KernelHandle,
    /// Its 16 x 32 tile twin (`gdn_ba_gates_prefill_small`).
    ba_small: KernelHandle,
    /// Resident CTAs (SMs x 2).
    slots: u32,
}

impl Default for DenseBf16mma {
    fn default() -> Self {
        Self {
            gate_up: KernelHandle(0),
            nvfp4: KernelHandle(0),
            fp8: KernelHandle(0),
            w4a4_gate_up: KernelHandle(0),
            w4a4: KernelHandle(0),
            w4a8_gate_up: KernelHandle(0),
            w4a8: KernelHandle(0),
            w8a8: KernelHandle(0),
            w8a8_t: KernelHandle(0),
            quant_nvfp4: KernelHandle(0),
            quant_fp8: KernelHandle(0),
            lowp: LowpMix(0),
            ba: KernelHandle(0),
            ba_small: KernelHandle(0),
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
fn chunks(m: u32, tiles: u32, slots: u32, rows: u32) -> u32 {
    (m.div_ceil(rows)..=m.div_ceil(16))
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
        Self::resolve_lowp(gpu, LowpMix::from_env())
    }

    fn resolve_lowp(gpu: &dyn GpuBackend, lowp: LowpMix) -> Self {
        let k = |name| {
            if lowp == LowpMix(0) {
                KernelHandle(0)
            } else {
                crate::layers::try_kernel(gpu, MODULE, name)
            }
        };
        Self {
            w4a4_gate_up: k("dense_w4a4_gate_up_silu"),
            w4a4: k("dense_w4a4_nvfp4"),
            w4a8_gate_up: k("dense_w4a8_gate_up_silu"),
            w4a8: k("dense_w4a8_nvfp4"),
            w8a8: k("dense_w8a8_fp8"),
            w8a8_t: k("dense_w8a8_fp8t"),
            quant_nvfp4: k("dl_quant_nvfp4"),
            quant_fp8: k("dl_quant_fp8"),
            lowp,
            gate_up: crate::layers::try_kernel(gpu, MODULE, "dense_bf16mma_gate_up_silu"),
            nvfp4: crate::layers::try_kernel(gpu, MODULE, "dense_bf16mma_nvfp4"),
            fp8: crate::layers::try_kernel(gpu, MODULE, "dense_bf16mma_fp8"),
            ba: crate::layers::try_kernel(gpu, "gdn_ba_gates_prefill", "gdn_ba_gates_prefill"),
            ba_small: crate::layers::try_kernel(
                gpu,
                "gdn_ba_gates_prefill",
                "gdn_ba_gates_prefill_small",
            ),
            slots: CTAS_PER_SM * gpu.sm_count().unwrap_or(48),
        }
    }

    /// The launch up to the weights: the BF16 `kernel` on `input`, or the
    /// activations quantized and the low-precision kernel `lp`; its grid for
    /// `m` rows of `n` features, `features` per CTA, and the chunk count.
    #[allow(clippy::too_many_arguments)]
    fn head<'a>(
        &self,
        gpu: &'a dyn GpuBackend,
        (kernel, smem): (KernelHandle, u32),
        lp: Option<(bool, KernelHandle, u32)>,
        input: DevicePtr,
        lowp: &LowpArgs,
        [m, n, k, features]: [u32; 4],
        stream: u64,
    ) -> Result<(KernelLaunch<'a>, u32)> {
        let (launch, rows, slots) = match lp {
            None => (
                KernelLaunch::new(gpu, kernel)
                    .shared_mem(smem)
                    .arg_ptr(input),
                ROWS,
                self.slots,
            ),
            Some((nvfp4, kernel, smem)) => {
                let token_scales = self.quantize(gpu, nvfp4, input, lowp, [m, k], stream)?;
                let launch = KernelLaunch::new(gpu, kernel)
                    .shared_mem(smem)
                    .arg_ptr(lowp.codes);
                let launch = match nvfp4 {
                    true => launch.arg_ptr(lowp.scales),
                    false => launch,
                };
                (
                    launch.arg_ptr(token_scales),
                    LOWP_ROWS,
                    self.slots / CTAS_PER_SM,
                )
            }
        };
        let c = chunks(m, n / features, slots.max(1), rows);
        Ok((
            launch
                .grid([(n / features) * c, 1, 1])
                .block([THREADS, 1, 1]),
            c,
        ))
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
        lowp: LowpArgs,
        stream: u64,
    ) -> Result<()> {
        let lp = self.lowp_kernel(
            &lowp,
            h,
            (self.w4a4_gate_up, SMEM_W4A4),
            (lowp.group << LowpMix::A8, self.w4a8_gate_up, SMEM_W4A8),
        );
        let (launch, nchunks) = self.head(
            gpu,
            (self.gate_up, SMEM_NVFP4),
            lp,
            input,
            &lowp,
            [m, inter, h, GATE_UP_FEATURES],
            stream,
        )?;
        launch
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
        lowp: LowpArgs,
        stream: u64,
    ) -> Result<()> {
        let lp = self.lowp_kernel(
            &lowp,
            k,
            (self.w4a4, SMEM_W4A4),
            (lowp.group << LowpMix::A8, self.w4a8, SMEM_W4A8),
        );
        let (launch, nchunks) = self.head(
            gpu,
            (self.nvfp4, SMEM_NVFP4),
            lp,
            input,
            &lowp,
            [m, n, k, FEATURES],
            stream,
        )?;
        launch
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

    /// `out[m, w.n] = input[m, w.k] · wᵀ` for an `[N, K]` e4m3 weight. W8A8
    /// reads the transposed `[K, N]` prefill copy `wt` where there is one
    /// (64 rows of 256 contiguous features a stage instead of 256 rows of 64
    /// bytes; the same MMA operands, so the same results).
    #[allow(clippy::too_many_arguments)]
    pub fn fp8(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &Fp8Weight,
        wt: Option<&Fp8WeightTransposed>,
        out: DevicePtr,
        m: u32,
        lowp: LowpArgs,
        stream: u64,
    ) -> Result<()> {
        let wt = wt.filter(|t| t.n == w.n && t.k == w.k && self.w8a8_t.0 != 0);
        // E4M3 tokens only: the FP8 group's own bit selects W8A8.
        let w8a8 = if wt.is_some() { self.w8a8_t } else { self.w8a8 };
        let lp = self.lowp_kernel(
            &lowp,
            w.k,
            (KernelHandle(0), 0),
            (lowp.group, w8a8, SMEM_W8A8),
        );
        let (weight, scale) = match (lp, wt) {
            (Some(_), Some(t)) => (t.weight_t, t.scale_t),
            _ => (w.weight, w.row_scale),
        };
        let (launch, nchunks) = self.head(
            gpu,
            (self.fp8, SMEM_FP8),
            lp,
            input,
            &lowp,
            [m, w.n, w.k, FEATURES],
            stream,
        )?;
        launch
            .arg_ptr(weight)
            .arg_ptr(scale)
            .arg_ptr(out)
            .arg_u32(m)
            .arg_u32(w.n)
            .arg_u32(w.k)
            .arg_u32(w.n)
            .arg_u32(nchunks)
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

#[path = "dense_bf16mma_lowp.rs"]
mod lowp;
pub use lowp::{LowpArgs, LowpMix};

#[path = "dense_bf16mma_ba.rs"]
mod ba;

#[cfg(test)]
#[path = "dense_bf16mma_test_gpu.rs"]
mod gpu_tests;

#[cfg(test)]
#[path = "dense_bf16mma_lowp_test_gpu.rs"]
mod lowp_gpu_tests;

#[cfg(test)]
mod tests {
    use super::{LOWP_ROWS, ROWS, chunks};

    #[test]
    fn chunks_stay_within_the_kernel_tile_and_fill_short_grids() {
        for (rows, slots) in [(ROWS, 96), (LOWP_ROWS, 48)] {
            for m in 1..5000 {
                for tiles in [20, 64, 136] {
                    let c = chunks(m, tiles, slots, rows);
                    assert!(
                        c >= m.div_ceil(rows) && m.div_ceil(c) <= rows,
                        "m={m} tiles={tiles} c={c}"
                    );
                    // The kernel's balanced split never leaves an empty chunk.
                    assert!((c - 1) * m.div_ceil(c) < m, "m={m} tiles={tiles} c={c}");
                }
            }
        }
        // down / out_proj at 288 rows: not 5 x 20 = 100 CTAs on 96 slots.
        assert!(chunks(288, 20, 96, ROWS) * 20 <= 96 || chunks(288, 20, 96, ROWS) * 20 > 150);
        // Full waves already: keep 64-row chunks.
        assert_eq!(chunks(4128, 136, 96, ROWS), 65);
    }
}
