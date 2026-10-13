// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_DECODE_GEMV_BATCH=1`: weight touch before the PDL wait for the
//! verify-step GEMVs.
//!
//! A verify step launches every kernel with programmatic dependent launch, so
//! a projection's CTAs are resident while the small kernels in front of it
//! (HC pre, norm, conv, recurrent) still run, waiting with the memory bus idle
//! (2026-09-30 nsys, prose / code: 49 / 30 us before KDA q, 28 / 37 before o,
//! 31 before the shared gate, 50 / 30 before MLA q_a). The `*_touch` twins of
//! the GEMV kernels spend that wait pulling their weight rows into L2
//! (`atlas_pdl_enter_touch`, kernels/gb10/glm-5.3-flash/nvfp4/atlas_pdl_touch.cuh);
//! the body then reads L2 at about twice the DRAM rate. A twin runs the body of
//! the kernel it replaces, so every output bit is unchanged
//! (scripts/dev/glm_decode_gemv_touch_bench.cu).
//!
//! The twins of the scalar `w4a16_gemv_batch2/3` and `_batch5_qkv` tiers are
//! also what puts those kernels on the PDL list: their untouched originals
//! are launched without it and start only after their predecessor completes.
//! `ATLAS_GLM_DECODE_GEMV_TOUCH_MB=0` keeps that PDL entry and touches nothing,
//! separating the two effects.
//!
//! The shared expert's gate and up (64 CTAs each at N = 1024: neither fills
//! the GPU) run as one two-plane launch, `w4a16_gemv_tc*_pair_touch`, the up
//! plane beside the gate plane instead of after it (4-8 us a KDA layer in the
//! bench). KDA q/k/v stay three launches: one three-plane launch saved 11 us
//! on that segment alone but cost 9-65 us per emulated layer at 8 and 16 rows.
//!
//! The twins exist only in the GLM-5.3-Flash kernel target and only pay off
//! with PDL, so the GLM KDA layer resolves them ([`gemv_touch_resolve`]) and
//! only under `ATLAS_PDL=1` on a PDL target; everywhere else [`gemv_touch`]
//! returns None and every site launches its original kernel.
//!
//! Prior art (docs/glm-prior-art.md): TensorFold's L2 weight touch
//! (<https://github.com/jayleaton/glm53-tensorfold-spark> patches 0040 and
//! 0440, the latter prefetching before `griddepcontrol.wait`; Apache-2.0) and
//! knapcio's `GLM_L2_PREFETCH`; compare mmastrac's arx L2 prefetch during
//! all-reduce waits. Our twins are discarded byte loads from the kernel's own
//! CTAs, not a side kernel or `cp.async.bulk.prefetch`. No code copied.

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// A touch twin: an index into `TWINS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Twin {
    W4Tc8,
    W4Tc16,
    W4Tc32,
    W4Batch2,
    W4Batch3,
    W4Batch5Qkv,
    MxTc8,
    MxTc16,
    MxTc32,
    W4Tc8Pair,
    W4Tc16Pair,
    W4Tc32Pair,
}

/// `(module, function)` of every touch twin, in [`Twin`] order.
const TWINS: [(&str, &str); 12] = [
    ("w4a16_gemv", "w4a16_gemv_tc8_touch"),
    ("w4a16_gemv", "w4a16_gemv_tc16_touch"),
    ("w4a16_gemv", "w4a16_gemv_tc32_touch"),
    ("w4a16_gemv", "w4a16_gemv_batch2_touch"),
    ("w4a16_gemv", "w4a16_gemv_batch3_touch"),
    ("w4a16_gemv", "w4a16_gemv_batch5_qkv_touch"),
    ("mxfp8_gemv", "mxfp8_gemv_tc8_touch"),
    ("mxfp8_gemv", "mxfp8_gemv_tc16_touch"),
    ("mxfp8_gemv", "mxfp8_gemv_tc32_touch"),
    ("w4a16_gemv", "w4a16_gemv_tc8_pair_touch"),
    ("w4a16_gemv", "w4a16_gemv_tc16_pair_touch"),
    ("w4a16_gemv", "w4a16_gemv_tc32_pair_touch"),
];

/// The touch twin of the NVFP4 tensor-core tier of `rows` (8, 16 or 32) rows.
pub fn w4a16_tc_twin(rows: u32) -> Option<Twin> {
    match rows {
        8 => Some(Twin::W4Tc8),
        16 => Some(Twin::W4Tc16),
        32 => Some(Twin::W4Tc32),
        _ => None,
    }
}

/// The pair twin (two projections of one input in one launch) of the NVFP4
/// tensor-core tier of `rows` (8, 16 or 32) rows.
pub fn w4a16_pair_twin(rows: u32) -> Option<Twin> {
    match rows {
        8 => Some(Twin::W4Tc8Pair),
        16 => Some(Twin::W4Tc16Pair),
        32 => Some(Twin::W4Tc32Pair),
        _ => None,
    }
}

/// The touch twin of the MXFP8 tier serving `m` (1..=32) rows.
pub fn mxfp8_tc_twin(m: u32) -> Twin {
    match m {
        0..=8 => Twin::MxTc8,
        9..=16 => Twin::MxTc16,
        _ => Twin::MxTc32,
    }
}

/// Read once: the flag, and the touch geometry `(bytes, ctas)`.
///
/// `bytes` bounds what one kernel pulls in (half of GB10's 24 MiB L2, so a
/// weight wider than that does not evict its own first rows; 0 touches
/// nothing); `ctas` is how many CTAs issue the loads (they must all be
/// resident during the wait, and more of them only delays a kernel that has
/// nothing to wait for). `ATLAS_GLM_DECODE_GEMV_TOUCH_MB` (0..=1024) /
/// `_TOUCH_CTAS` (1..=1024) override them for tuning.
fn settings() -> Option<(u64, u32)> {
    static SETTINGS: OnceLock<Option<(u64, u32)>> = OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let var = |name: &str, min: u64, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| (min..=1024).contains(v))
                .unwrap_or(default)
        };
        (std::env::var("ATLAS_GLM_DECODE_GEMV_BATCH").as_deref() == Ok("1")).then(|| {
            (
                var("ATLAS_GLM_DECODE_GEMV_TOUCH_MB", 0, 12) << 20,
                var("ATLAS_GLM_DECODE_GEMV_TOUCH_CTAS", 1, 32) as u32,
            )
        })
    })
}

/// The twin handles, in [`Twin`] order, when `on`: looked up with the kernels
/// around them so the boot audit sees them before it seals. No lookup when
/// off, so a target without the twins leaves no failed row in the audit.
fn twin_table(gpu: &dyn GpuBackend, on: bool) -> Option<[KernelHandle; TWINS.len()]> {
    on.then(|| TWINS.map(|(module, func)| crate::layers::try_kernel(gpu, module, func)))
}

static RESOLVED: OnceLock<Option<[KernelHandle; TWINS.len()]>> = OnceLock::new();

/// Resolve the twins once, from the GLM KDA layer constructor (the twins live
/// only in the GLM-5.3-Flash target): with the flag on and PDL active. Without
/// PDL a twin's touch would run in series right before its body (about +2 us
/// a launch, nothing to hide it), so the flag is then ignored.
pub fn gemv_touch_resolve(gpu: &dyn GpuBackend) {
    RESOLVED.get_or_init(|| {
        let on = settings().is_some();
        let pdl = spark_runtime::cuda_backend::pdl_enabled();
        if on && !pdl {
            tracing::warn!("ATLAS_GLM_DECODE_GEMV_BATCH=1 ignored: it needs ATLAS_PDL=1");
        }
        twin_table(gpu, on && pdl)
    });
}

/// A resolved touch twin and the geometry it is launched with.
#[derive(Clone, Copy, Debug)]
pub struct GemvTouch {
    pub kernel: KernelHandle,
    /// Most bytes (values plus scales) one launch pulls into L2.
    pub bytes: u64,
    /// Most CTAs that issue the loads.
    pub ctas: u32,
}

/// The touch twin `twin` when it was resolved ([`gemv_touch_resolve`]).
pub fn gemv_touch(twin: Twin) -> Option<GemvTouch> {
    let (bytes, ctas) = settings()?;
    let kernel = RESOLVED.get()?.as_ref()?[twin as usize];
    (kernel.0 != 0).then_some(GemvTouch {
        kernel,
        bytes,
        ctas,
    })
}

impl GemvTouch {
    /// The kernels' trailing `(touch_rows, touch_ctas)` for a weight of `n`
    /// rows of `row_bytes` (values plus scales) on a grid `grid_x` CTAs wide.
    fn geometry(&self, n: u32, row_bytes: u32, grid_x: u32) -> (u32, u32) {
        let rows = (self.bytes / u64::from(row_bytes.max(1))).min(u64::from(n)) as u32;
        (rows, self.ctas.min(grid_x))
    }

    /// A `w4a16_gemv_tc{8,16,32}` (or `_ld`) launch through its touch twin
    /// ([`w4a16_tc_twin`], at least `m` rows): `weight` rows are `ld_half`
    /// packed bytes and `ld_groups` scale bytes apart (`k / 2` and `k / 16`
    /// for a whole weight).
    #[allow(clippy::too_many_arguments)]
    pub fn w4a16_tc(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ld_half: u32,
        ld_groups: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=32).contains(&m)
                && k.is_multiple_of(16)
                && ld_half >= k / 2
                && ld_groups >= k / 16,
            "w4a16 tensor-core touch GEMV: m={m} k={k} ld={ld_half}/{ld_groups}"
        );
        let grid = div_ceil(n, 16);
        let (rows, ctas) = self.geometry(n, k / 2 + k / 16, grid);
        KernelLaunch::new(gpu, self.kernel)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(ld_half)
            .arg_u32(ld_groups)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)
    }

    /// Two `w4a16_gemv_tc{8,16,32}` projections of one `input` (whole
    /// weights of `n` rows by `k`) in one launch through the pair twin
    /// (`w4a16_pair_twin`, at least `m` rows), grid z = plane: each
    /// `(weight, output)` gets the tier's unchanged body and touches its
    /// weight during the PDL wait.
    #[allow(clippy::too_many_arguments)]
    pub fn w4a16_tc_pair(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        pair: [(&QuantizedWeight, DevicePtr); 2],
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=32).contains(&m) && k.is_multiple_of(16),
            "w4a16 tensor-core pair GEMV: m={m} k={k}"
        );
        let grid = div_ceil(n, 16);
        let (rows, ctas) = self.geometry(n, k / 2 + k / 16, grid);
        let mut launch = KernelLaunch::new(gpu, self.kernel)
            .grid([grid, 1, 2])
            .block([256, 1, 1])
            .arg_ptr(input);
        for (weight, output) in pair {
            launch = launch
                .arg_ptr(weight.weight)
                .arg_ptr(weight.weight_scale)
                .arg_f32(weight.weight_scale_2)
                .arg_ptr(output);
        }
        launch
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)
    }

    /// `w4a16_gemv_batch2` / `_batch3` (`m` = 2 / 3) through their PDL touch
    /// twin ([`Twin::W4Batch2`] / [`Twin::W4Batch3`], which take `m` too).
    #[allow(clippy::too_many_arguments)]
    pub fn w4a16_batch23(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (2..=3).contains(&m),
            "w4a16 batch2/3 touch GEMV takes 2 or 3 rows, got {m}"
        );
        let grid = div_ceil(n, 4);
        let (rows, ctas) = self.geometry(n, k / 2 + k / 16, grid);
        KernelLaunch::new(gpu, self.kernel)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)
    }

    /// A `mxfp8_gemv_tc{8,16,32}` launch through its touch twin
    /// ([`mxfp8_tc_twin`] of `m`).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp8_tc(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        data: DevicePtr,
        scales: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        out_stride: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=32).contains(&m) && k.is_multiple_of(32),
            "mxfp8 touch GEMV: m={m} k={k} unsupported"
        );
        let grid = div_ceil(n, 16);
        let (rows, ctas) = self.geometry(n, k + k / 32, grid);
        KernelLaunch::new(gpu, self.kernel)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(data)
            .arg_ptr(scales)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(out_stride)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)
    }

    /// The two trailing arguments of `w4a16_gemv_batch5_qkv_touch`: the Q
    /// plane's touch geometry (`n` outputs of `k` inputs, four per CTA).
    pub fn batch5_qkv_args(&self, n: u32, k: u32) -> (u32, u32) {
        self.geometry(n, k / 2 + k / 16, div_ceil(n, 4))
    }
}

/// The touch twin of the NVFP4 verify tier for `m` (2..=32) rows, launched
/// in its place: `w4a16_gemv_batch2/3` below four rows, else `tier` when it is
/// a tensor-core tier. None when there is no twin to take (flag off, no PDL,
/// another tier or target): the caller launches the tier itself.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_verify_touch(
    gpu: &dyn GpuBackend,
    tier: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Option<Result<()>> {
    if (2..=3).contains(&m) {
        let touch = gemv_touch(if m == 2 {
            Twin::W4Batch2
        } else {
            Twin::W4Batch3
        })?;
        return Some(touch.w4a16_batch23(gpu, input, weight, output, m, n, k, stream));
    }
    let rows = crate::layers::w4a16_gemv_tiers::tc_rows(tier)?;
    let touch = gemv_touch(w4a16_tc_twin(rows)?)?;
    Some(touch.w4a16_tc(gpu, input, weight, output, m, n, k, k / 2, k / 16, stream))
}

/// Two projections of one input through the pair twin of `tier` (a
/// tensor-core tier), in one launch: see [`GemvTouch::w4a16_tc_pair`]. None
/// when there is no twin to take (flag off, no PDL, a scalar tier or another
/// target): the caller launches the two one by one.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_pair_touch(
    gpu: &dyn GpuBackend,
    tier: KernelHandle,
    input: DevicePtr,
    pair: [(&QuantizedWeight, DevicePtr); 2],
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Option<Result<()>> {
    let rows = crate::layers::w4a16_gemv_tiers::tc_rows(tier)?;
    let touch = gemv_touch(w4a16_pair_twin(rows)?)?;
    Some(touch.w4a16_tc_pair(gpu, input, pair, m, n, k, stream))
}

#[cfg(test)]
#[path = "gemv_touch_tests.rs"]
mod tests;
