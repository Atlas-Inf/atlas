// SPDX-License-Identifier: AGPL-3.0-only
//! Optional CUTLASS host-wrapper FFI for de-risking GB10 GEMM replacements.
//!
//! Split for the ≤500 LoC cap: this root holds the shared FFI `extern` block,
//! the workspace `Ctx`, and module wiring; the public wrappers live in the
//! `gemm` (dense BF16 + NVFP4), `grouped` (per-expert MoE), and `pack`
//! (weight pack / SFB swizzle / transpose) siblings. The public API
//! (`spark_runtime::cutlass::<fn>`) is preserved via the re-exports below.

#[cfg(atlas_cutlass)]
use anyhow::{Result, bail};

/// Marks a wrapper error raised before any kernel launch: CUTLASS rejected the
/// operands (or is not built in), the output is untouched, and the caller may
/// use another backend. An error without this marker comes from the launch
/// itself and must propagate: recomputing in another backend rounds
/// differently, so a fallback that fires on some calls only makes the output
/// irreproducible.
#[derive(Debug)]
pub struct RejectedBeforeLaunch;

impl std::fmt::Display for RejectedBeforeLaunch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CUTLASS rejected the operands before launching")
    }
}

impl std::error::Error for RejectedBeforeLaunch {}

/// Whether `error` comes from a wrapper that launched nothing.
pub fn rejected_before_launch(error: &anyhow::Error) -> bool {
    error.downcast_ref::<RejectedBeforeLaunch>().is_some()
}

/// Marks a wrapper error from the launch itself: the output may be partly
/// written, so no caller may recompute it in another backend.
#[derive(Debug)]
pub struct LaunchFailed;

impl std::fmt::Display for LaunchFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CUTLASS launch failed")
    }
}

impl std::error::Error for LaunchFailed {}

/// Whether `error` reports a failed CUTLASS launch.
pub fn launch_failed(error: &anyhow::Error) -> bool {
    error.downcast_ref::<LaunchFailed>().is_some()
}

/// The BF16 GEMM wrappers report a failed launch as this value plus the
/// CUTLASS status (`ATLAS_CUTLASS_LAUNCH_FAILED` in
/// `cuda/atlas_stale_cuda_error.h`); their lower non-zero statuses are
/// rejections before any launch. The NVFP4 and pack wrappers keep their own
/// codes and are never retried in another backend.
pub const LAUNCH_FAILED: i32 = 1000;

/// The error for a non-zero BF16 wrapper `status`: [`LaunchFailed`] when the
/// launch itself failed, else [`RejectedBeforeLaunch`].
#[cfg(any(atlas_cutlass, test))]
pub(crate) fn status_error(status: i32, what: String) -> anyhow::Error {
    if status >= LAUNCH_FAILED {
        anyhow::Error::new(LaunchFailed).context(format!(
            "{what}: launch failed, CUTLASS status {}",
            status - LAUNCH_FAILED
        ))
    } else {
        anyhow::Error::new(RejectedBeforeLaunch).context(format!("{what}: status {status}"))
    }
}

#[cfg(atlas_cutlass)]
use std::ffi::c_void;
#[cfg(atlas_cutlass)]
use std::sync::OnceLock;

mod gemm;
mod grouped;
mod pack;

pub use gemm::{
    bf16_gemm_act_weight_t, bf16_gemm_tuned, bf16_grouped_gemm_act_weight_t,
    nvfp4_gemm_bf16_act_weight_t,
};
pub use grouped::{nvfp4_grouped_down, nvfp4_grouped_gate_up, nvfp4_grouped_gate_up_fused};
pub use pack::{pack_bf16_weight_to_nvfp4_t, pack_weight_sfb, transpose_nvfp4_packed_kton};

#[cfg(all(test, atlas_cutlass))]
mod tests;

#[cfg(atlas_cutlass)]
unsafe extern "C" {
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_bf16_gemm_tuned(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldc: i32,
        config: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_bf16_grouped_gemm_act_weight_t(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        g: i32,
        n: i32,
        k: i32,
        a_stride: i32,
        c_stride: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_nvfp4_gemm_bf16_act_weight_t(
        act: *const c_void,
        weight_packed_t: *const c_void,
        weight_scale_t: *const c_void,
        weight_scale_2: f32,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_pack_bf16_weight_to_nvfp4_t(
        weight_bf16: *const c_void,
        packed_t: *mut c_void,
        scale_t: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_nvfp4_grouped_gate_up(
        a_bf16: *const c_void,
        gate_packed_ptrs: *const u64,
        gate_scale_ptrs: *const u64,
        gate_scale2_vals: *const f32,
        up_packed_ptrs: *const u64,
        up_scale_ptrs: *const u64,
        up_scale2_vals: *const f32,
        c_gate_bf16: *mut c_void,
        c_up_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_nvfp4_grouped_gate_up_fused(
        a_bf16: *const c_void,
        sorted_token_ids: *const i32,
        gate_packed_ptrs: *const u64,
        gate_sfb_ptrs: *const u64,
        gate_scale2_vals: *const f32,
        up_packed_ptrs: *const u64,
        up_sfb_ptrs: *const u64,
        up_scale2_vals: *const f32,
        c_gate_bf16: *mut c_void,
        c_up_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_nvfp4_grouped_down(
        a_bf16: *const c_void,
        packed_ptrs: *const u64,
        sfb_ptrs: *const u64,
        scale2_vals: *const f32,
        c_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_pack_weight_sfb(
        scale_in: *const c_void,
        scale_out: *mut c_void,
        n: i32,
        k: i32,
        src_n_major: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn atlas_cutlass_transpose_nvfp4_packed_kton(
        src_packed_t: *const c_void,
        dst_packed: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_128x256(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_256x128(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_64x128(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_128x64(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cutlass_bf16_gemm_act_weight_t_64x64(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn atlas_cublaslt_bf16_gemm_act_weight_t_algo(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
        algo_index: i32,
        returned_count: *mut i32,
    ) -> i32;
    pub(crate) fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn atlas_cuda_stale_error_stats(last_code: *mut i32) -> u64;
}

/// `status` of the `wrapper` call just made, after reporting any stale sticky
/// CUDA runtime error its entry drained. Such an error was left unchecked by
/// an earlier runtime call on this thread; before the wrappers drained it,
/// CUTLASS read it back as its own launch failure. Warns once with the code
/// (it names the class of call that leaves it), then logs at debug level.
#[cfg(atlas_cutlass)]
pub(crate) fn drained(wrapper: &str, status: i32) -> i32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static REPORTED: AtomicU64 = AtomicU64::new(0);
    let mut code = 0i32;
    let count = unsafe { atlas_cuda_stale_error_stats(&mut code) };
    let seen = REPORTED.fetch_max(count, Ordering::Relaxed);
    if count > seen {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("unnamed");
        if seen == 0 {
            tracing::warn!(
                "stale CUDA runtime error {code} drained before CUTLASS {wrapper} on thread \
                 {name}: an earlier runtime call left it unchecked (reported once)"
            );
        } else {
            tracing::debug!(
                "stale CUDA runtime error {code} drained before CUTLASS {wrapper} on thread \
                 {name} ({count} so far)"
            );
        }
    }
    status
}

#[cfg(atlas_cutlass)]
pub(crate) struct Ctx {
    pub(crate) workspace: u64,
    pub(crate) ws_size: usize,
}

#[cfg(atlas_cutlass)]
unsafe impl Send for Ctx {}
#[cfg(atlas_cutlass)]
unsafe impl Sync for Ctx {}

#[cfg(atlas_cutlass)]
/// STATIC, DELIBERATELY — CUDA host. This is a workspace allocated in THE
/// process CUDA context (see `atlas_core::cuda_host`, which establishes one
/// per process) and sized by a fixed budget, not by any model's shapes: the
/// bounds below are generous upper limits chosen to fit any realistic serving
/// configuration, so a swap needs no reallocation and re-allocating per model
/// would churn hundreds of megabytes for no change in what is mapped.
///
/// It survives a model swap for the same reason the context does. Nothing in
/// it is derived from a model — no token ids, no weight pointers, no shapes —
/// only scratch the library plans within.
static CTX: OnceLock<Ctx> = OnceLock::new();

#[cfg(atlas_cutlass)]
pub(crate) fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    // Shared scratch for all CUTLASS host wrappers. The grouped NVFP4 MoE path
    // (single-launch kGrouped over up to 256 experts) stages packed-A + SFA +
    // per-group arrays + the gemm workspace here; at large prefill M the 256-group
    // gemm workspace alone exceeds the old 64 MB (-> status -2 + an OOB context
    // corruption). 512 MB by default; override with ATLAS_CUTLASS_WORKSPACE_MB.
    let ws_size = std::env::var("ATLAS_CUTLASS_WORKSPACE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
        * 1024
        * 1024;
    let mut workspace = 0u64;
    let status = unsafe { cuMemAlloc_v2(&mut workspace, ws_size) };
    if status != 0 {
        bail!("cuMemAlloc CUTLASS workspace failed: {status}");
    }
    let _ = CTX.set(Ctx { workspace, ws_size });
    Ok(CTX.get().unwrap())
}

#[cfg(test)]
mod marker_tests {
    use super::*;

    #[test]
    fn only_pre_launch_statuses_are_marked_rejected() {
        // can_implement / workspace / config rejections: nothing was launched.
        for status in [-5, -2, 1, 7, LAUNCH_FAILED - 1] {
            assert!(rejected_before_launch(&status_error(status, "gemm".into())));
        }
        // The launch itself failed: never a reason to try another backend.
        for status in [LAUNCH_FAILED + 1, LAUNCH_FAILED + 7] {
            let error = status_error(status, "gemm".into());
            assert!(!rejected_before_launch(&error) && launch_failed(&error));
            assert!(format!("{error:#}").contains("launch failed"));
        }
        assert!(!rejected_before_launch(&anyhow::anyhow!("unrelated")));
        assert!(!launch_failed(&status_error(1, "gemm".into())));
    }

    /// Without CUTLASS built in nothing launches, so the caller's other
    /// backend stays available.
    #[cfg(not(atlas_cutlass))]
    #[test]
    fn a_build_without_cutlass_rejects_before_launch() {
        let error = bf16_gemm_tuned(0, 0, 0, 256, 128, 128, 128, 128, 9, 0).unwrap_err();
        assert!(rejected_before_launch(&error));
    }
}
