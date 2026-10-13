// SPDX-License-Identifier: AGPL-3.0-only

//! The fast, bit-identical arm of the qwen4_exp mHC PREFILL collapse
//! (`ATLAS_QWEN4EXP_PREFILL_HC=1`, default off).
//!
//! Per 2048-token slab the default runs seven launches over a 40 KB/token
//! FP32 highway and two 20 KB/token BF16 intermediates (`hc_post`, stage,
//! down GEMM, silu, up GEMM, injection GEMM, mix). This arm changes two of
//! them, every output byte unchanged:
//!
//! 1. **Seam** (`hc_post_stage_bf16`): a sublayer's `hc_post` and the next
//!    site's stage in one pass, the post values kept in registers -- one
//!    highway read fewer (657 MB per seam at 16K tokens). Taken where a
//!    post is followed at once by a pre (inside every layer).
//! 2. **Up GEMM + mix** (`hc_up_mix_bf16_nt`): the up projection's tiles stay
//!    in registers and fold into `y` in the epilogue -- the 42 MB `up_pre`
//!    write and read and one launch fewer (0.77 -> 0.45 ms per slab).
//!
//! The down and injection GEMMs are the default's own calls.
//!
//! Why 2 is exact, and only on the cuBLASLt versions in
//! `UP_MIX_VERIFIED_LT`: the default's up GEMM is cuBLASLt, and
//! `hc_up_mix_bf16_nt` computes the plain in-order `mma.sync.m16n8k16`
//! k-chain. They agree byte for byte exactly when the library's heuristic
//! picks a non-split kernel that runs that chain -- a property of the library
//! version, not of the math. It held at every slab height 1..2048 on 13.0.0
//! (the runtime image) and 13.1.1 (run against each library:
//! `scripts/dev/qwen4exp_hc_prefill_bench.cu <dir> 2048 32 all`). Elsewhere
//! the seam still serves and the up GEMM + mix stay the default's.
//!
//! The first version of this arm also moved full slabs' down GEMM to
//! cuBLASLt on the same argument. That held on 13.1.1 (non-split) but not on
//! 13.0.0, whose heuristic picks split-K 3 for 2048x320x10240: on the pair,
//! `_HC_CHECK` failed the first slab (677002 of 10485760 `y` bytes). The down
//! GEMM is the default's again; on 13.0.0 that cuBLASLt kernel was only
//! 0.32 vs 0.41 ms anyway.
//!
//! `ATLAS_QWEN4EXP_PREFILL_HC_CHECK=<n>` re-runs the first `n` slabs (default
//! 16) through the default path in serving and fails on any differing byte.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_dispatch::HcVariant;
use super::hyper_connection_lowrank::{HC_DECODE_MAX_T, hc_gemm_disabled, hc_post_lowrank};
use super::hyper_connection_lowrank_prefill::{
    HC_PREFILL_SLAB, HcPrefillCall, HcSlabIo, hc_inject, hc_low, hc_pre_gemm, hc_slab_default,
    hc_slab_tail, hc_stage,
};
use super::qwen4exp_decode_fuse::HcPostFold;
use crate::layers::qwen3_attention::{HcLowRank, HcSiteWeights, HcWeights};

/// `ATLAS_QWEN4EXP_PREFILL_HC=1`.
pub fn hc_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_HC").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// `ATLAS_QWEN4EXP_PREFILL_HC_CHECK=<n>`: slabs left to cross-check (`1` or
/// `true` means 16). Synchronizing and allocating; diagnostic only.
fn check_left() -> &'static std::sync::atomic::AtomicI64 {
    static LEFT: std::sync::OnceLock<std::sync::atomic::AtomicI64> = std::sync::OnceLock::new();
    LEFT.get_or_init(|| {
        let n = match std::env::var("ATLAS_QWEN4EXP_PREFILL_HC_CHECK").as_deref() {
            Ok("1") | Ok("true") => 16,
            Ok(v) => v.parse().unwrap_or(0),
            Err(_) => 0,
        };
        std::sync::atomic::AtomicI64::new(n)
    })
}

/// cuBLASLt versions ([`spark_runtime::cublaslt::version`]) whose BF16 up GEMM
/// of the collapse (`[ts, rank] x [rank, hc*H]`, opN) was compared byte for
/// byte against `hc_up_mix_bf16_nt` at every slab height 1..2048 on GB10:
/// 13.0.0 (CUDA 13.0, the runtime image) and 13.1.1.
const UP_MIX_VERIFIED_LT: [usize; 2] = [130000, 130101];

/// The fused up GEMM + mix equals the default's up GEMM + mix: the default
/// runs the tile kernel (no cuBLASLt), or a verified cuBLASLt.
fn up_mix_exact() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        if !spark_runtime::cublaslt::available() {
            return true;
        }
        let v = spark_runtime::cublaslt::version();
        let ok = UP_MIX_VERIFIED_LT.contains(&v);
        if !ok {
            tracing::warn!(
                "ATLAS_QWEN4EXP_PREFILL_HC: cuBLASLt {v} is not one the fused mHC up+mix was \
                 verified against ({UP_MIX_VERIFIED_LT:?}); the seam serves, the up GEMM and \
                 mix stay the default's"
            );
        }
        ok
    })
}

/// The fast arm's kernels, resolved per call. `up_mix` is `None` where the
/// fused up+mix would not be exact ([`up_mix_exact`]).
pub(super) struct HcFastKernels {
    post_stage: KernelHandle,
    up_mix: Option<KernelHandle>,
}

impl HcFastKernels {
    /// Whether the slabs read `up_wt`, the `[hc*H, rank]` copy of `up_w`.
    pub(super) fn reads_up_wt(&self) -> bool {
        self.up_mix.is_some()
    }
}

/// Rows per `hc_up_mix_bf16_nt` CTA and hidden columns per CTA (`HUM_BM`,
/// `HUM_BD`); the rank tile (`HUM_BK`) is 32.
const HUM_BM: u32 = 128;
const HUM_BD: u32 = 32;
const HUM_BK: usize = 32;

/// Shortest prefill the fast arm takes. Below it the per-call `up_w`
/// transpose (0.04 ms a collapse) outweighs the savings: measured per
/// layer-shaped round (2 posts, 3 collapses) on GB10, default vs fast,
/// 0.44 / 0.62 ms at 100 tokens, 1.53 / 1.61 at 512 (`examples/
/// qwen4exp_hc_prefill_check.rs`, the first version of this arm). With the
/// default's down GEMM, against the runtime image's cuBLASLt 13.0: 3.65 /
/// 2.89 at 1024, 7.55 / 5.93 at 2048, 59.3 / 44.0 at 16046.
const HC_FAST_MIN_T: u32 = 1024;

/// The fast arm for this call, when requested and the shape fits the
/// kernels: at least [`HC_FAST_MIN_T`] tokens, four streams, `hidden` a
/// multiple of 32 and at most 4096 (the seam keeps four columns per thread of
/// a 1024-thread block), `rank` a multiple of 32.
pub(super) fn prefill_arm(
    gpu: &dyn GpuBackend,
    w: &HcLowRank,
    num_tokens: u32,
    hidden: u32,
    hc_mult: u32,
) -> Option<HcFastKernels> {
    if !hc_requested()
        || num_tokens < HC_FAST_MIN_T
        || hc_mult != 4
        || !hidden.is_multiple_of(HUM_BD)
        || hidden > 4096
        || w.rank == 0
        || !w.rank.is_multiple_of(HUM_BK)
    {
        return None;
    }
    let up_mix = crate::layers::try_kernel(gpu, "hyper_connection", "hc_up_mix_bf16_nt");
    let k = HcFastKernels {
        post_stage: crate::layers::try_kernel(gpu, "hyper_connection", "hc_post_stage_bf16"),
        up_mix: up_mix_exact().then_some(up_mix),
    };
    if k.post_stage.0 == 0 || up_mix.0 == 0 {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "ATLAS_QWEN4EXP_PREFILL_HC: this build's hyper_connection module has no \
                 hc_post_stage_bf16 / hc_up_mix_bf16_nt; the default collapse serves"
            )
        });
        return None;
    }
    Some(k)
}

/// A sublayer's `hc_post` fused with the next site's `hc_pre` at prefill
/// width (`block_out` into the highway, then the collapse of `site` into
/// `y_out` and `post`). `Ok(false)` launched nothing: the caller runs its
/// post and pre as before.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_pre_seam(
    gpu: &dyn GpuBackend,
    hc: &HcWeights,
    site: &HcSiteWeights,
    block_out: DevicePtr,
    streams: DevicePtr,
    y_out: DevicePtr,
    post: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<bool> {
    let Some(w) = &site.lowrank else {
        return Ok(false);
    };
    let hc_mult = hc.hc_mult as u32;
    if HcVariant::of(hc) != HcVariant::LowRank
        || num_tokens <= HC_DECODE_MAX_T
        || scratch.is_null()
        || hc_gemm_disabled()
        || w.inject_w.is_null()
        || prefill_arm(gpu, w, num_tokens, hidden_size, hc_mult).is_none()
    {
        return Ok(false);
    }
    hc_pre_gemm(
        gpu,
        streams,
        w,
        y_out,
        post,
        scratch,
        num_tokens,
        hidden_size,
        hc_mult,
        norm_eps,
        /* inject */ true,
        Some(HcPostFold {
            block_out,
            inj: post,
        }),
        stream,
    )?;
    Ok(true)
}

/// One slab on the fast arm (or, under `_HC_CHECK`, on both arms with a
/// byte comparison).
pub(super) fn hc_slab_fast(
    c: &HcPrefillCall,
    k: &HcFastKernels,
    io: &HcSlabIo,
    stream: u64,
) -> Result<()> {
    if check_left().fetch_sub(1, std::sync::atomic::Ordering::Relaxed) > 0 {
        return hc_slab_checked(c, k, io, stream);
    }
    slab_fast(c, k, io, stream)
}

fn slab_fast(c: &HcPrefillCall, k: &HcFastKernels, io: &HcSlabIo, stream: u64) -> Result<()> {
    let gpu = c.gpu;
    let ts = io.ts;
    // 1. normed (and, folded, the previous sublayer's post into the highway).
    match io.fold {
        Some(f) => KernelLaunch::new(gpu, k.post_stage)
            .grid([ts, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(f.block_out)
            .arg_ptr(io.streams)
            .arg_ptr(f.inj)
            .arg_ptr(c.w.norm_w)
            .arg_ptr(c.normed)
            .arg_u32(c.hidden)
            .arg_u32(c.hc_mult)
            .arg_f32(c.eps)
            .launch(stream)?,
        None => hc_stage(c, io, stream)?,
    }
    // 2. low = silu(normed x down_w^T / hc): the default's calls.
    hc_low(c, ts, stream)?;
    let Some(up_mix) = k.up_mix else {
        return hc_slab_tail(c, io, stream);
    };
    hc_inject(
        c,
        gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
        ts,
        stream,
    )?;
    // 3. y = mean_s sigmoid(low x up_w) * normed, inj, in one kernel.
    KernelLaunch::new(gpu, up_mix)
        .grid([c.hidden / HUM_BD, ts.div_ceil(HUM_BM), 1])
        .block([256, 1, 1])
        .arg_ptr(c.low)
        .arg_ptr(c.up_wt)
        .arg_ptr(c.normed)
        .arg_ptr(if c.inject { c.inj_pre } else { DevicePtr::NULL })
        .arg_ptr(io.y)
        .arg_ptr(io.inj)
        .arg_u32(ts)
        .arg_u32(c.hidden)
        .arg_u32(c.rank())
        .arg_f32(1.0f32 / c.hc_mult as f32)
        .launch(stream)
}

/// Device scratch for the check: the slab's highway and `inj` before the
/// fast arm ran, and the fast arm's highway / `y` / `inj`. Allocated once.
fn check_arena(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static ARENA: std::sync::OnceLock<(u64, usize)> = std::sync::OnceLock::new();
    let &(ptr, size) = ARENA.get_or_init(|| match gpu.alloc(bytes) {
        Ok(p) => (p.0, bytes),
        Err(_) => (0, 0),
    });
    anyhow::ensure!(
        ptr != 0 && size >= bytes,
        "ATLAS_QWEN4EXP_PREFILL_HC_CHECK: could not allocate {bytes} B of check scratch"
    );
    Ok(DevicePtr(ptr))
}

/// Run the slab on the fast arm, keep its outputs, restore the inputs, run
/// the default arm, and require every byte to match. Leaves the default's
/// (identical) outputs in place.
fn hc_slab_checked(c: &HcPrefillCall, k: &HcFastKernels, io: &HcSlabIo, stream: u64) -> Result<()> {
    let gpu = c.gpu;
    let ts = io.ts as usize;
    let max = HC_PREFILL_SLAB as usize;
    let (hd, h, hc) = (c.hc_dim() as usize, c.hidden as usize, c.hc_mult as usize);
    // [streams before | inj before | streams fast | y fast | inj fast]
    let arena = check_arena(gpu, 2 * max * hd * 4 + 2 * max * hc * 4 + max * h * 2)?;
    let (s_in, i_in) = (arena, arena.offset(max * hd * 4));
    let s_fast = i_in.offset(max * hc * 4);
    let y_fast = s_fast.offset(max * hd * 4);
    let i_fast = y_fast.offset(max * h * 2);
    let (sb, yb, ib) = (ts * hd * 4, ts * h * 2, ts * hc * 4);
    let inj_src = io.fold.map(|f| f.inj);

    gpu.copy_d2d_async(io.streams, s_in, sb, stream)?;
    if let Some(src) = inj_src {
        gpu.copy_d2d_async(src, i_in, ib, stream)?;
    }
    slab_fast(c, k, io, stream)?;
    gpu.copy_d2d_async(io.streams, s_fast, sb, stream)?;
    gpu.copy_d2d_async(io.y, y_fast, yb, stream)?;
    if c.inject {
        gpu.copy_d2d_async(io.inj, i_fast, ib, stream)?;
    }
    gpu.copy_d2d_async(s_in, io.streams, sb, stream)?;
    if let Some(src) = inj_src {
        gpu.copy_d2d_async(i_in, src, ib, stream)?;
    }
    if let Some(f) = io.fold {
        let k_post = gpu.kernel("hyper_connection", "hc_post")?;
        hc_post_lowrank(
            gpu,
            k_post,
            f.block_out,
            io.streams,
            f.inj,
            io.streams,
            io.ts,
            c.hidden,
            c.hc_mult,
            stream,
        )?;
    }
    let plain = HcSlabIo { fold: None, ..*io };
    hc_slab_default(c, &plain, stream)?;
    gpu.synchronize(stream)?;

    let mut spans = vec![("y", io.y, y_fast, yb), ("highway", io.streams, s_fast, sb)];
    if c.inject {
        spans.push(("inj", io.inj, i_fast, ib));
    }
    for (name, reference, fast, bytes) in spans {
        let (mut a, mut b) = (vec![0u8; bytes], vec![0u8; bytes]);
        gpu.copy_d2h(reference, &mut a)?;
        gpu.copy_d2h(fast, &mut b)?;
        let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        anyhow::ensure!(
            diff == 0,
            "ATLAS_QWEN4EXP_PREFILL_HC_CHECK: the fast mHC prefill arm differs from the \
             default in {diff}/{bytes} {name} bytes (slab of {ts} rows, fold={})",
            io.fold.is_some()
        );
    }
    tracing::info!(
        "ATLAS_QWEN4EXP_PREFILL_HC_CHECK: slab of {ts} rows byte-identical (fold={})",
        io.fold.is_some()
    );
    Ok(())
}
