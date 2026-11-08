// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_WIDE_GEMV=1` (default off): 5..=16-row NVFP4 / block-FP8 GEMVs in
//! one `wide_gemv.cu` launch (`kernels/gb10/qwen3.6-27b/nvfp4/`, so the
//! qwen3.6-27b and qwen3.8-27b targets) instead of the narrow scalar tiers.
//!
//! | family | single-row kernel | replaces (5..=16 rows) |
//! |---|---|---|
//! | [`Family::W4a16`] | `w4a16_gemv` | `w4a16_gemv_batch5..8/16/32` ([`super::w4a16_gemv_batchm`]) |
//! | [`Family::W8a16`] | `w8a16_gemv` | `w8a16_gemv_batch16` ([`super::w8a16_gemv_batch4`], `_div` at 4) |
//! | [`Family::Qg`] | `w4a16_gemv_qg` | `w4a16_gemv_qg_batch4` chunks (the exact verify's Q+gate) |
//!
//! Row `t` of every launch is the single-row kernel's bytes on row `t`, as
//! the kernels it replaces already are (`examples/wide_gemv_bitparity_microtest.rs`
//! gates every tier at M = 1..=16 on the 27B shapes), so the switch changes
//! launch shape only: no output byte, no collective, and the ranks need not
//! agree on it. That is also why the hook sits inside the shared launch ops
//! rather than at each caller: any scalar-template launch of 5..=16 rows may
//! take it, the exact verify's (`model/exact_verify.rs`) and the batched
//! multi-row decode's alike. Tensor-core, DP4A and `_vl2`/`_n2` launches are
//! other arithmetic and never do.
//!
//! The narrowest resolved tier covering the rows serves a launch;
//! `ATLAS_WIDE_GEMV_NB=4` picks the 4-outputs-a-thread variants (256-thread
//! CTAs) over the default 2 (512 threads). Outside its shapes (K not a
//! multiple of 16, K below [`Family::min_k`]) or on a target without the
//! module, a launch keeps its narrow kernel.

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

/// The kernels' module (file stem).
pub const MODULE: &str = "wide_gemv";
/// Row capacities of the tiers (the kernels' MAX_M), narrowest first.
pub const TIERS: [u32; 4] = [8, 9, 12, 16];
/// Outputs a thread of the two variants.
pub const NBS: [u32; 2] = [2, 4];
/// Fewer rows than this keep the narrow tiers (batch4 runs them near the
/// DRAM floor).
pub const MIN_ROWS: u32 = 5;
/// Outputs a CTA tile (the kernels' `WG_TILE`).
const TILE_OUTPUTS: u32 = 16;
/// Persistent grid: CTAs an SM (a launch is exact at any grid).
const CTAS_PER_SM: u32 = 2;

/// Which single-row kernel a tier reproduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    W4a16,
    Qg,
    W8a16,
}

impl Family {
    pub const ALL: [Family; 3] = [Family::W4a16, Family::Qg, Family::W8a16];

    fn prefix(self) -> &'static str {
        match self {
            Family::W4a16 => "w4a16_gemv",
            Family::Qg => "w4a16_gemv_qg",
            Family::W8a16 => "w8a16_gemv",
        }
    }

    /// Smallest K the kernel's walk serves: every thread must own a chunk of
    /// every pass (W4: K/16 >= 128 over two phases; Q+gate: K/8 >= 64; W8:
    /// K/16 >= 64), or the in-flight next chunk would skip one.
    pub fn min_k(self) -> u32 {
        match self {
            Family::W4a16 => 2048,
            Family::Qg => 512,
            Family::W8a16 => 1024,
        }
    }
}

/// `w4a16_gemv_wide9_nb2`, `w4a16_gemv_qg_wide16_nb4`, ...
pub fn kernel_name(family: Family, rows: u32, nb: u32) -> String {
    format!("{}_wide{rows}_nb{nb}", family.prefix())
}

/// One resolved tier.
#[derive(Clone, Copy, Debug)]
pub struct WideKernel {
    pub handle: KernelHandle,
    pub family: Family,
    pub rows: u32,
    pub nb: u32,
}

impl WideKernel {
    /// Look the tier up; a zero handle when the target does not ship it.
    pub fn resolve(gpu: &dyn GpuBackend, family: Family, rows: u32, nb: u32) -> Self {
        Self {
            handle: try_kernel(gpu, MODULE, &kernel_name(family, rows, nb)),
            family,
            rows,
            nb,
        }
    }

    /// Whether this tier can run `m` rows of a `k`-wide weight.
    pub fn serves(&self, m: u32, k: u32) -> bool {
        self.handle.0 != 0
            && (1..=self.rows).contains(&m)
            && k.is_multiple_of(16)
            && k >= self.family.min_k()
    }

    /// The launch geometry for `n` outputs: 64 threads an output group,
    /// 16 / nb groups, persistent over the 16-output tiles.
    fn launch<'a>(&self, gpu: &'a dyn GpuBackend, n: u32, sms: u32) -> KernelLaunch<'a> {
        let grid = div_ceil(n, TILE_OUTPUTS)
            .min(CTAS_PER_SM * sms.max(1))
            .max(1);
        KernelLaunch::new(gpu, self.handle)
            .grid([grid, 1, 1])
            .block([64 * TILE_OUTPUTS / self.nb, 1, 1])
    }

    fn check(&self, family: Family, m: u32, k: u32) -> Result<()> {
        ensure!(
            self.family == family && self.serves(m, k),
            "{}: m={m} k={k} not served",
            kernel_name(self.family, self.rows, self.nb)
        );
        Ok(())
    }
}

/// `m` rows of `input` (`[m, k]` BF16) through the NVFP4 `w` (`[n, k]`)
/// into `output` rows `c_stride` elements apart; row `t` is `w4a16_gemv`'s
/// bytes. `sms`: the device's SM count (sizes the persistent grid).
#[allow(clippy::too_many_arguments)]
pub fn launch_w4a16(
    gpu: &dyn GpuBackend,
    kernel: &WideKernel,
    sms: u32,
    input: DevicePtr,
    w: &QuantizedWeight,
    output: DevicePtr,
    (m, n, k): (u32, u32, u32),
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    kernel.check(Family::W4a16, m, k)?;
    kernel
        .launch(gpu, n, sms)
        .arg_ptr(input)
        .arg_ptr(w.weight)
        .arg_ptr(w.weight_scale)
        .arg_f32(w.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(c_stride)
        .launch(stream)
}

/// `m` rows through the interleaved NVFP4 Q+gate weight (`n` = heads x
/// head_dim x 2), output row `t` deinterleaved `[Q | G]` at `output + t*n`;
/// row `t` is `w4a16_gemv_qg`'s bytes.
#[allow(clippy::too_many_arguments)]
pub fn launch_qg(
    gpu: &dyn GpuBackend,
    kernel: &WideKernel,
    sms: u32,
    input: DevicePtr,
    w: &QuantizedWeight,
    output: DevicePtr,
    (m, n, k): (u32, u32, u32),
    (heads, head_dim): (u32, u32),
    stream: u64,
) -> Result<()> {
    kernel.check(Family::Qg, m, k)?;
    ensure!(
        n == heads * head_dim * 2,
        "q+gate: n={n} != 2 x {heads} x {head_dim}"
    );
    kernel
        .launch(gpu, n, sms)
        .arg_ptr(input)
        .arg_ptr(w.weight)
        .arg_ptr(w.weight_scale)
        .arg_f32(w.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// `m` rows through the block-scaled FP8 `weight` (`[n, k]`, FP32
/// `block_scale` `[ceil(n/128), ceil(k/128)]`) into `output` rows `c_stride`
/// apart; row `t` is `w8a16_gemv`'s bytes.
#[allow(clippy::too_many_arguments)]
pub fn launch_w8a16(
    gpu: &dyn GpuBackend,
    kernel: &WideKernel,
    sms: u32,
    input: DevicePtr,
    (weight, block_scale): (DevicePtr, DevicePtr),
    output: DevicePtr,
    (m, n, k): (u32, u32, u32),
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    kernel.check(Family::W8a16, m, k)?;
    kernel
        .launch(gpu, n, sms)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(c_stride)
        .launch(stream)
}

/// `ATLAS_WIDE_GEMV=1`, read once.
pub fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_WIDE_GEMV").as_deref() == Ok("1"))
}

/// Outputs a thread for an `ATLAS_WIDE_GEMV_NB` value: 4 when it says so,
/// else 2.
pub fn nb_from(value: Option<&str>) -> u32 {
    if value == Some("4") { 4 } else { 2 }
}

static WIDE: OnceLock<WideGemv> = OnceLock::new();

/// Resolve the lever's tiers once, when it is on (layer init calls this
/// through `W4a16BatchmTiers::resolve`); a no-op otherwise, so no default
/// boot looks the module up.
pub fn init(gpu: &dyn GpuBackend) {
    if requested() {
        WIDE.get_or_init(|| WideGemv::resolve(gpu));
    }
}

/// The lever's tiers, `None` when it is off.
pub fn active() -> Option<&'static WideGemv> {
    WIDE.get()
}

/// The resolved tiers of the chosen variant.
#[derive(Debug)]
pub struct WideGemv {
    kernels: Vec<WideKernel>,
    sms: u32,
}

impl WideGemv {
    fn resolve(gpu: &dyn GpuBackend) -> Self {
        let nb = nb_from(std::env::var("ATLAS_WIDE_GEMV_NB").ok().as_deref());
        let kernels: Vec<WideKernel> = Family::ALL
            .iter()
            .flat_map(|&f| TIERS.map(|rows| WideKernel::resolve(gpu, f, rows, nb)))
            .collect();
        let found = kernels.iter().filter(|k| k.handle.0 != 0).count();
        tracing::info!(
            "ATLAS_WIDE_GEMV=1: {found}/{} wide row-exact GEMV tiers (nb{nb}) resolved",
            kernels.len()
        );
        Self {
            kernels,
            sms: gpu.sm_count().unwrap_or(0),
        }
    }

    /// The narrowest resolved tier of `family` serving `m` rows at `k`, for
    /// `m` in [`MIN_ROWS`]..=16.
    pub fn pick(&self, family: Family, m: u32, k: u32) -> Option<&WideKernel> {
        if m < MIN_ROWS {
            return None;
        }
        self.kernels
            .iter()
            .filter(|w| w.family == family && w.serves(m, k))
            .min_by_key(|w| w.rows)
    }

    /// [`launch_w4a16`] when a tier serves the shape (`Ok(true)`), else
    /// nothing launched (`Ok(false)`).
    #[allow(clippy::too_many_arguments)]
    pub fn try_w4a16(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        (m, n, k): (u32, u32, u32),
        c_stride: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(kernel) = self.pick(Family::W4a16, m, k) else {
            return Ok(false);
        };
        launch_w4a16(
            gpu,
            kernel,
            self.sms,
            input,
            w,
            output,
            (m, n, k),
            c_stride,
            stream,
        )?;
        Ok(true)
    }

    /// [`launch_qg`] when a tier serves the shape.
    #[allow(clippy::too_many_arguments)]
    pub fn try_qg(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &QuantizedWeight,
        output: DevicePtr,
        (m, n, k): (u32, u32, u32),
        heads_dim: (u32, u32),
        stream: u64,
    ) -> Result<bool> {
        let Some(kernel) = self.pick(Family::Qg, m, k) else {
            return Ok(false);
        };
        launch_qg(
            gpu,
            kernel,
            self.sms,
            input,
            w,
            output,
            (m, n, k),
            heads_dim,
            stream,
        )?;
        Ok(true)
    }

    /// [`launch_w8a16`] when a tier serves the shape.
    #[allow(clippy::too_many_arguments)]
    pub fn try_w8a16(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight_scale: (DevicePtr, DevicePtr),
        output: DevicePtr,
        (m, n, k): (u32, u32, u32),
        c_stride: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(kernel) = self.pick(Family::W8a16, m, k) else {
            return Ok(false);
        };
        launch_w8a16(
            gpu,
            kernel,
            self.sms,
            input,
            weight_scale,
            output,
            (m, n, k),
            c_stride,
            stream,
        )?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(present: &[(Family, u32)]) -> WideGemv {
        let kernels = Family::ALL
            .iter()
            .flat_map(|&family| {
                TIERS.map(|rows| WideKernel {
                    handle: KernelHandle(
                        u64::from(present.contains(&(family, rows))) * rows as u64,
                    ),
                    family,
                    rows,
                    nb: 2,
                })
            })
            .collect();
        WideGemv { kernels, sms: 48 }
    }

    #[test]
    fn names_match_the_kernels() {
        assert_eq!(kernel_name(Family::W4a16, 9, 2), "w4a16_gemv_wide9_nb2");
        assert_eq!(kernel_name(Family::Qg, 16, 4), "w4a16_gemv_qg_wide16_nb4");
        assert_eq!(kernel_name(Family::W8a16, 12, 2), "w8a16_gemv_wide12_nb2");
    }

    #[test]
    fn narrowest_covering_tier_from_five_rows() {
        let all: Vec<_> = Family::ALL
            .iter()
            .flat_map(|&f| TIERS.map(|r| (f, r)))
            .collect();
        let t = table(&all);
        for m in 0..MIN_ROWS {
            assert!(t.pick(Family::W4a16, m, 5120).is_none(), "m={m}");
        }
        let want = |m: u32| match m {
            5..=8 => 8,
            9 => 9,
            10..=12 => 12,
            _ => 16,
        };
        for f in Family::ALL {
            for m in MIN_ROWS..=16 {
                assert_eq!(
                    t.pick(f, m, 5120).map(|k| k.rows),
                    Some(want(m)),
                    "{f:?} m={m}"
                );
            }
            assert!(t.pick(f, 17, 5120).is_none());
        }
    }

    #[test]
    fn missing_tiers_widen_and_shapes_are_refused() {
        let t = table(&[(Family::W4a16, 16), (Family::W8a16, 12)]);
        assert_eq!(t.pick(Family::W4a16, 9, 5120).map(|k| k.rows), Some(16));
        assert_eq!(t.pick(Family::W8a16, 9, 6144).map(|k| k.rows), Some(12));
        assert!(t.pick(Family::W8a16, 13, 6144).is_none());
        assert!(t.pick(Family::Qg, 9, 5120).is_none());
        // K below the walk's floor, or not a multiple of 16.
        assert!(t.pick(Family::W4a16, 9, 1024).is_none());
        assert!(t.pick(Family::W4a16, 9, 5128).is_none());
        assert!(t.pick(Family::W4a16, 9, 2048).is_some());
    }

    #[test]
    fn nb_switch_reads_exactly_four() {
        assert_eq!(nb_from(None), 2);
        assert_eq!(nb_from(Some("2")), 2);
        assert_eq!(nb_from(Some("4")), 4);
        assert_eq!(nb_from(Some("8")), 2);
    }
}
