// SPDX-License-Identifier: AGPL-3.0-only
//! The prefill GEMMs of [`super`]'s native FP8 block projections (more rows
//! than one GEMV launch serves): the [`Fp8bPrefill`] switch, the W8A8 block
//! FP8 GEMM on cuBLASLt's pinned kernel, the in-tree `fp8_gemm_t_blockscaled`
//! or CUTLASS's blockwise kernel, and the exact W8A16 GEMM. Split from
//! `fp8b.rs` for the file-size cap.

use anyhow::{Result, ensure};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{NativeKernels, native_kernels};
use crate::weight_map::{FP8_BLOCK, Fp8BlockWeight};

/// How FP8 block projections run prefill (more rows than one GEMV launch
/// serves), from `ATLAS_GLM_FP4L_FP8_PREFILL` (`w8a8` default, `w8a16`, `bf16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8bPrefill {
    /// Activations per token and 128 values to E4M3 + f32, block FP8 GEMM
    /// (NVIDIA's vLLM recipe for `FP8_PB_WO` / FP8_BLOCK_SCALES).
    W8a8,
    /// Exact weights on BF16 activations (`w8a16_gemm_pipelined`).
    W8a16,
    /// The BF16 copy (production's numerics; keeps the BF16 copy resident).
    Bf16,
}

impl Fp8bPrefill {
    pub const VAR: &'static str = "ATLAS_GLM_FP4L_FP8_PREFILL";

    pub fn parse(value: Option<&str>) -> Result<Self> {
        Ok(match value {
            None | Some("") | Some("w8a8") => Self::W8a8,
            Some("w8a16") => Self::W8a16,
            Some("bf16") => Self::Bf16,
            Some(other) => {
                anyhow::bail!("{} must be w8a8, w8a16 or bf16, got {other:?}", Self::VAR)
            }
        })
    }

    /// The switch as set (read once).
    pub fn from_env() -> Result<Self> {
        static V: std::sync::OnceLock<std::result::Result<Fp8bPrefill, String>> =
            std::sync::OnceLock::new();
        V.get_or_init(|| {
            Self::parse(std::env::var(Self::VAR).ok().as_deref()).map_err(|e| e.to_string())
        })
        .clone()
        .map_err(anyhow::Error::msg)
    }

    /// Whether this mode still reads the projection's BF16 copy.
    pub fn needs_bf16(self) -> bool {
        self == Self::Bf16
    }
}

/// The W8A8 prefill's activation scratch: an E4M3 image and its per-token
/// per-128 f32 scales (the arena's `fp8_act` / `fp8_act_scale`).
#[derive(Debug, Clone, Copy)]
pub struct Fp8ActScratch {
    pub act: DevicePtr,
    pub act_bytes: usize,
    pub scale: DevicePtr,
    pub scale_bytes: usize,
}

impl Fp8ActScratch {
    /// The arena's FP8 activation scratch.
    pub fn of(buffers: &BufferArena) -> Self {
        Self {
            act: buffers.fp8_act(),
            act_bytes: buffers.fp8_act_bytes(),
            scale: buffers.fp8_act_scale(),
            scale_bytes: buffers.sizes().fp8_act_scale,
        }
    }

    /// How a block FP8 W8A8 prefill of an `[n, k]` weight uses this scratch:
    /// passes of `rows` (a multiple of 16) whose E4M3 image and scales fit
    /// with 16 rows of padding headroom, and a `[16, n]` BF16 slot at the end
    /// of the E4M3 region for a pass's padded tail.
    pub fn blk_plan(&self, k: usize, n: usize) -> Result<BlkPlan> {
        let slot = ((self.act_bytes.saturating_sub(16 * n * 2)) / 256) * 256;
        let by_act = slot / k;
        let by_scale = self.scale_bytes / (k / FP8_BLOCK * 4);
        let rows = (by_act.min(by_scale).saturating_sub(16) / 16) * 16;
        ensure!(
            rows >= 16,
            "block FP8 W8A8: the FP8 scratch ({} + {} B) holds no padded 16-row [_, {k}] pass \
             beside a [16, {n}] tail",
            self.act_bytes,
            self.scale_bytes
        );
        Ok(BlkPlan {
            rows,
            tail_out: self.act.offset(slot),
        })
    }
}

/// See [`Fp8ActScratch::blk_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlkPlan {
    pub rows: usize,
    pub tail_out: DevicePtr,
}

/// Per `(n, k)`: whether cuBLASLt runs block FP8 here (probed once,
/// [`fp8b_w8a8_prepare`]).
static BLK_LT: std::sync::Mutex<Vec<((usize, usize), bool)>> = std::sync::Mutex::new(Vec::new());

/// Probe (once per shape) whether the W8A8 prefill of an `[n, k]` FP8 block
/// twin can run on cuBLASLt's pinned block FP8 kernel; when it cannot
/// (cuBLASLt 13.0 on GB10 answers the VEC128 / BLK128x128 modes with
/// NOT_SUPPORTED), the shape runs the in-tree `fp8_gemm_t_blockscaled` (the
/// same W8A8 numerics: per-token-per-128 E4M3 activations, f32 epilogue),
/// logged once. Errors only when neither can run, so a loader calling this
/// fails at load, never mid-prefill.
pub fn fp8b_w8a8_prepare(
    gpu: &dyn GpuBackend,
    w: &Fp8BlockWeight,
    n: usize,
    k: usize,
) -> Result<()> {
    if BLK_LT.lock().unwrap().iter().any(|e| e.0 == (n, k)) {
        return Ok(());
    }
    let lt = match spark_runtime::cublaslt::fp8_blkscaled_pinned_prepare(
        w.scales.0, w.scales.0, n as u32, k as u32,
    ) {
        Ok(()) => true,
        Err(e) => {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(
                    "cuBLASLt block FP8 unavailable ({e:#}): native W8A8 prefill runs \
                     fp8_gemm_t_blockscaled"
                )
            });
            false
        }
    };
    let kernels = native_kernels(gpu);
    ensure!(
        kernels.act_quant.0 != 0 && (lt || kernels.w8a8_gemm.0 != 0),
        "native W8A8 prefill of [{n}, {k}]: no block FP8 GEMM (cuBLASLt refused and \
         fp8_gemm_t_blockscaled is missing); set ATLAS_GLM_FP4L_FP8_PREFILL=w8a16"
    );
    BLK_LT.lock().unwrap().push(((n, k), lt));
    Ok(())
}

/// Whether the W8A8 prefill of `w` (`[n, k]`) runs on cuBLASLt (probing it
/// now if no loader did).
fn fp8b_w8a8_lt(w: &Fp8BlockWeight, n: usize, k: usize) -> bool {
    let known = BLK_LT
        .lock()
        .unwrap()
        .iter()
        .find(|e| e.0 == (n, k))
        .map(|e| e.1);
    known.unwrap_or_else(|| {
        spark_runtime::cublaslt::fp8_blkscaled_pinned_prepare(
            w.scales.0, w.scales.0, n as u32, k as u32,
        )
        .is_ok()
    })
}

/// `ATLAS_GLM_FP8B_CUTLASS` as set (read once): `1` runs the W8A8 block FP8
/// prefill GEMM of a shape cuBLASLt refuses on CUTLASS's SM120 blockwise
/// kernel (`spark_runtime::cutlass::fp8_blockwise_gemm_act_weight_t`, about
/// 9x `fp8_gemm_t_blockscaled`), `0` (the default) on the in-tree kernel.
/// Same W8A8 numerics and activation quantization; the group sums are added
/// in another order, so the output bits differ from the in-tree kernel's.
pub fn fp8b_cutlass_requested() -> Result<bool> {
    static ON: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(
        || match std::env::var("ATLAS_GLM_FP8B_CUTLASS").as_deref() {
            Err(_) | Ok("0") => Ok(false),
            Ok("1") => Ok(true),
            Ok(v) => Err(format!("ATLAS_GLM_FP8B_CUTLASS must be 0 or 1, got {v:?}")),
        },
    )
    .clone()
    .map_err(anyhow::Error::msg)
}

/// The activation rows of a W8A8 block FP8 prefill: BF16 rows, cast pass by
/// pass into the FP8 scratch (`per_token_group_quant_fp8`), or rows already
/// cast to that kernel's bytes: E4M3 `act` (`[m, k]`) and f32 `scale`
/// (`[m, k / 128]`), both row-major, so any row block casts on its own
/// (`ATLAS_GLM_SP_MLA_QUANT_GATHER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8bRows {
    Bf16(DevicePtr),
    Cast { act: DevicePtr, scale: DevicePtr },
}

impl Fp8bRows {
    /// The E4M3 rows and scales of rows `[first, first + rows)` of a
    /// `[_, k]` input: cast into `scratch` (from its row 0), or where the
    /// cast rows already sit.
    fn pass(
        self,
        gpu: &dyn GpuBackend,
        kernels: &NativeKernels,
        scratch: Fp8ActScratch,
        (first, rows): (usize, usize),
        k: usize,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        match self {
            Self::Bf16(input) => {
                super::super::per_token_group_quant_fp8_warp(
                    gpu,
                    kernels.act_quant,
                    input.offset(first * k * 2),
                    scratch.act,
                    scratch.scale,
                    rows as u32,
                    k as u32,
                    stream,
                )?;
                Ok((scratch.act, scratch.scale))
            }
            Self::Cast { act, scale } => Ok((
                act.offset(first * k),
                scale.offset(first * (k / FP8_BLOCK) * 4),
            )),
        }
    }
}

/// The in-tree W8A8 block FP8 prefill: per-token-group activation quant and
/// `fp8_gemm_t_blockscaled` (one fixed kernel; each output word is one
/// thread's in-order sum, so rows are chunk- and height-invariant), in row
/// passes that fit the FP8 scratch; or, `cutlass`, CUTLASS's blockwise
/// kernel on the same quantized operands (no split-K: rows are as
/// invariant). Already cast `input` rows run the same passes on their own
/// bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp8b_w8a8_hand(
    gpu: &dyn GpuBackend,
    kernels: &NativeKernels,
    scratch: Fp8ActScratch,
    w: &Fp8BlockWeight,
    input: Fp8bRows,
    output: DevicePtr,
    [m, n, k]: [usize; 3],
    cutlass: bool,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernels.act_quant.0 != 0 && (cutlass || kernels.w8a8_gemm.0 != 0),
        "per_token_group_quant_fp8 / fp8_gemm_t_blockscaled are missing"
    );
    let chunk = (scratch.act_bytes / k).min(scratch.scale_bytes / (k / FP8_BLOCK * 4)) / 64 * 64;
    ensure!(
        chunk > 0,
        "FP8 W8A8 prefill: the FP8 scratch holds no 64-row [_, {k}] tile"
    );
    for first in (0..m).step_by(chunk) {
        let rows = (m - first).min(chunk);
        let (act, scale) = input.pass(gpu, kernels, scratch, (first, rows), k, stream)?;
        let rows = rows as u32;
        if cutlass {
            spark_runtime::cutlass::fp8_blockwise_gemm_act_weight_t(
                act.0,
                scale.0,
                w.data.0,
                w.scales.0,
                output.offset(first * n * 2).0,
                [rows, n as u32, k as u32],
                stream,
            )?;
            continue;
        }
        super::super::fp8_gemm_t_blockscaled(
            gpu,
            kernels.w8a8_gemm,
            act,
            scale,
            w.data,
            w.scales,
            output.offset(first * n * 2),
            rows,
            n as u32,
            k as u32,
            stream,
        )?;
    }
    Ok(())
}

/// `output[m, n] = input[m, k] · W^T` for prefill on FP8 block twin `w`
/// (untiled), `mode` W8A8 or W8A16 (`Bf16` is the caller's BF16 GEMM).
#[allow(clippy::too_many_arguments)]
pub fn fp8b_prefill_gemm(
    gpu: &dyn GpuBackend,
    kernels: &NativeKernels,
    scratch: Fp8ActScratch,
    mode: Fp8bPrefill,
    w: &Fp8BlockWeight,
    input: DevicePtr,
    output: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        !w.tiled && n.is_multiple_of(FP8_BLOCK) && k.is_multiple_of(FP8_BLOCK),
        "fp8b_prefill_gemm: [{n}, {k}] (tiled={})",
        w.tiled
    );
    match mode {
        Fp8bPrefill::W8a16 => {
            ensure!(kernels.w8a16_gemm.0 != 0, "w8a16_gemm_pipelined is missing");
            super::super::w8a16_gemm_pipelined(
                gpu,
                kernels.w8a16_gemm,
                input,
                w.data,
                w.scales,
                output,
                m as u32,
                n as u32,
                k as u32,
                stream,
            )
        }
        Fp8bPrefill::W8a8 => fp8b_w8a8_gemm(
            gpu,
            kernels,
            scratch,
            w,
            Fp8bRows::Bf16(input),
            output,
            [m, n, k],
            stream,
        ),
        Fp8bPrefill::Bf16 => anyhow::bail!("fp8b_prefill_gemm: BF16 prefill is the caller's"),
    }
}

/// The W8A8 arm of [`fp8b_prefill_gemm`] on `input` rows, BF16 or already
/// cast: the same passes and kernels either way, so cast rows give the bytes
/// their BF16 rows would.
#[allow(clippy::too_many_arguments)]
pub fn fp8b_w8a8_gemm(
    gpu: &dyn GpuBackend,
    kernels: &NativeKernels,
    scratch: Fp8ActScratch,
    w: &Fp8BlockWeight,
    input: Fp8bRows,
    output: DevicePtr,
    [m, n, k]: [usize; 3],
    stream: u64,
) -> Result<()> {
    ensure!(
        !w.tiled && n.is_multiple_of(FP8_BLOCK) && k.is_multiple_of(FP8_BLOCK),
        "fp8b_w8a8_gemm: [{n}, {k}] (tiled={})",
        w.tiled
    );
    if !fp8b_w8a8_lt(w, n, k) {
        let cutlass = fp8b_cutlass_requested()?;
        return fp8b_w8a8_hand(
            gpu,
            kernels,
            scratch,
            w,
            input,
            output,
            [m, n, k],
            cutlass,
            stream,
        );
    }
    ensure!(
        kernels.act_quant.0 != 0,
        "per_token_group_quant_fp8 is missing"
    );
    let plan = scratch.blk_plan(k, n)?;
    let kg4 = k / FP8_BLOCK * 4;
    let gemm = |a: DevicePtr, sf: DevicePtr, out: DevicePtr, rows: usize| {
        spark_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled_pinned(
            a.0,
            sf.0,
            w.data.0,
            w.scales.0,
            out.0,
            [rows as u32, n as u32, k as u32],
            stream,
        )
    };
    for first in (0..m).step_by(plan.rows) {
        let rows = (m - first).min(plan.rows);
        let (act, act_scale) = input.pass(gpu, kernels, scratch, (first, rows), k, stream)?;
        let main = rows / 16 * 16;
        if main > 0 {
            gemm(act, act_scale, output.offset(first * n * 2), main)?;
        }
        let tail = rows - main;
        if tail > 0 {
            // The last < 16 rows as one 16-row pass of the same kernel from
            // the scratch: zero rows (E4M3 0, scale 0) pad it, its output
            // lands in the scratch's tail slot and its real rows are copied
            // out. Same kernel, row-independent: the same bytes as inside a
            // taller pass. Cast rows bring their tail into the scratch.
            let (pad_act, pad_scale) = (
                scratch.act.offset(main * k),
                scratch.scale.offset(main * kg4),
            );
            if act != scratch.act {
                gpu.copy_d2d_async(act.offset(main * k), pad_act, tail * k, stream)?;
                gpu.copy_d2d_async(act_scale.offset(main * kg4), pad_scale, tail * kg4, stream)?;
            }
            let pad = 16 - tail;
            gpu.memset_async(pad_act.offset(tail * k), 0, pad * k, stream)?;
            gpu.memset_async(pad_scale.offset(tail * kg4), 0, pad * kg4, stream)?;
            gemm(pad_act, pad_scale, plan.tail_out, 16)?;
            gpu.copy_d2d_async(
                plan.tail_out,
                output.offset((first + main) * n * 2),
                tail * n * 2,
                stream,
            )?;
        }
    }
    Ok(())
}
