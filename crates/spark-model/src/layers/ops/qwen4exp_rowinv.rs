// SPDX-License-Identifier: AGPL-3.0-only

//! Row-invariant qwen4_exp prefill (`ATLAS_QWEN4EXP_PREFILL_ROWINV=1`,
//! default off): a token's per-layer outputs are bitwise independent of how
//! many rows share its prefill pass and of where the pass boundaries fall
//! (on the 64-token GDN grid, which the scheduler and the prefix cache keep
//! under the switch). That makes a prompt prefilled in one pass, in chunks,
//! or after a prefix-cache restore give the same logits, and lets several
//! prompts share one pass exactly.
//!
//! What the switch changes, each to an arithmetic that does not look at the
//! row count, at each projection's own precision:
//!
//! * BF16 projections that went to cuBLASLt's heuristic, which re-picks
//!   the kernel (and its split-K) per row count: the GDN in_proj, the QSA
//!   indexer projections and the mHC down / injection (which also switched
//!   to the k-chain at full slabs). The wide in_proj runs the in-order
//!   k-chain (the same bytes on every k-chain kernel); each narrow shape
//!   keeps ONE cuBLASLt configuration at every row count -- the heuristic's
//!   pick at a reference row count ([`try_bf16_gemm`], `cublaslt::fixed_algo`).
//! * The mHC collapse no longer takes the decode split path at <= 8 rows.
//! * Attention runs the paged path from the first chunk on (the BR64 dense
//!   kernel at every width, dense rows cut at the QSA bound), q/k/v on the
//!   FP8 x FP8 `qwen4exp_fp8_gemm_w2` at every width (the first chunk's arm)
//!   and o on the FP8 m128 arm at every width.
//! * One-row passes stay on the prefill path (no decode-layer fork), and a
//!   Marconi replay uses the chunked GDN scan, not the token-sequential one.
//!
//! `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ=1` ([`bf16_proj`], on top) moves the
//! prefill to decode precision:
//!
//! * attention q(+gate)/k/v/o on BF16 copies of the NVFP4 weights decode
//!   reads, through the k-chain above, instead of FP8 weights x E4M3
//!   activations (`qwen3_attention::prefill_weights_rowinv`);
//! * the shared expert on the routed experts' TC prefill kernels -- decode's
//!   TC numerics, decode runs it as units of that family -- instead of q38's
//!   E4M3 activations; the routed experts take that arm too (`_MOE_BF16`);
//! * the mHC collapse on the DECODE collapse's own kernels at every width --
//!   FP32 `normed`, `qwen4exp_hc_mma.cu`'s tensor-core down and finish over
//!   row groups ([`hc_collapse`]) -- byte for byte its `_HC_MMA` decode
//!   collapse; the fused prefill seams are declined.
//!
//! The pass state lives in a thread-local [`Scope`] that `forward_layers`
//! holds around the prefill layers, so decode and verify -- which share the
//! layer code -- are untouched.

use std::cell::Cell;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_lowrank_gemm::gemm_raw;
use crate::layers::qwen3_attention::HcLowRank;

/// `ATLAS_QWEN4EXP_PREFILL_ROWINV=1`. Read once.
pub fn on() -> bool {
    spark_runtime::qwen4exp_prefill_rowinv()
}

/// `ATLAS_QWEN4EXP_PREFILL_BF16_PROJ=1` (default off, applies only with
/// [`on`]): the prefill projections at decode precision -- the FP8 ones
/// (attention q/k/v/o, the shared expert) on BF16 activations, the mHC
/// collapse on decode's FP32 kernels. Read once.
pub fn bf16_proj() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let set = matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_BF16_PROJ").as_deref(),
            Ok("1") | Ok("true")
        );
        if set && !on() {
            tracing::warn!(
                "ATLAS_QWEN4EXP_PREFILL_BF16_PROJ=1 ignored: it applies on top of \
                 ATLAS_QWEN4EXP_PREFILL_ROWINV=1"
            );
        }
        set && on()
    })
}

/// Longest prompt a multi-sequence prefill pass (`ATLAS_QWEN4EXP_PREFILL_MULTI`)
/// takes: under the QSA inert bound (`budget + ratio - 1` = 2051), so its
/// attention is dense and no selection runs.
pub const MULTI_MAX_PROMPT: usize = 2048;

/// Most sequences one multi-sequence prefill pass takes (the scheduler caps
/// its passes at this, both ranks check it).
pub const MULTI_MAX_SEQS: usize = 64;

/// The GDN chunk every pass start must sit on (see the module docs).
pub const PASS_GRANULE: usize = 64;

#[derive(Clone, Copy)]
struct Pass {
    gpu: *const (dyn GpuBackend + 'static),
    /// Rows one mHC slab takes in the shared scratch (FP32 `normed` and
    /// `low`), a multiple of the row-kernel groups.
    hc_rows: u32,
    /// [`bf16_proj`]: decode precision on top of the row invariance.
    decode: bool,
    /// The pass's rows are all text (no vision pads): see [`text_only`].
    text: bool,
}

thread_local! {
    static PASS: Cell<Option<Pass>> = const { Cell::new(None) };
}

/// Ends the row-invariant prefill pass when dropped.
pub struct Scope(());

impl Drop for Scope {
    fn drop(&mut self) {
        PASS.with(|p| p.set(None));
    }
}

/// Start a row-invariant prefill pass on this thread (`None` when the switch
/// is off). `hc_scratch_bytes`, `hc_dim` and `rank` size the mHC slab;
/// `text` says every row of the pass is a text token.
pub fn enter(
    gpu: &dyn GpuBackend,
    hc_scratch_bytes: usize,
    hc_dim: usize,
    rank: usize,
    text: bool,
) -> Option<Scope> {
    if !on() {
        return None;
    }
    let per_row = (hc_dim + rank).max(1) * 4;
    let group = (HCM_ROWS_DN_T * HCM_ROWS_FN_T / 32) as usize; // 192: both groups divide it
    let rows = ((hc_scratch_bytes / per_row).min(2048) / group * group) as u32;
    // SAFETY: the pointer is only dereferenced while the returned `Scope`
    // lives, which the caller holds inside the borrow of `gpu`.
    let gpu: &'static dyn GpuBackend = unsafe { std::mem::transmute(gpu) };
    PASS.with(|p| {
        p.set(Some(Pass {
            gpu: gpu as *const _,
            hc_rows: rows,
            decode: bf16_proj(),
            text,
        }))
    });
    Some(Scope(()))
}

/// A row-invariant prefill pass is running on this thread.
pub fn active() -> bool {
    PASS.with(|p| p.get().is_some())
}

/// A row-invariant pass over text rows only is running: their three mRoPE
/// position streams are equal, so the attention takes plain RoPE (the
/// cache-skip first chunk's kernel, the same rotation at a quarter of
/// `rope_mrope_interleaved`'s cost) in every chunk.
pub fn text_only() -> bool {
    PASS.with(|p| p.get().is_some_and(|p| p.text))
}

/// A row-invariant pass at decode precision ([`bf16_proj`]) is running.
pub fn decode_active() -> bool {
    PASS.with(|p| p.get().is_some_and(|p| p.decode))
}

fn pass() -> Option<(&'static dyn GpuBackend, u32)> {
    // SAFETY: see `enter`.
    PASS.with(|p| p.get().map(|p| (unsafe { &*p.gpu }, p.hc_rows)))
}

/// How a BF16 projection of shape `n x k` stays row-invariant, measured on
/// GB10 (`examples/qwen4exp_rowinv_probe fixed`):
///
/// * wide (`n >= 2048`, the GDN in_proj; up to a chunk of rows): the
///   in-order k-chain -- the fastest cuBLASLt algo-21 kernel offered for the
///   row count (what `ATLAS_LT_KCHAIN_PIN` runs at 16K rows), else the tile
///   kernel; every k-chain kernel gives the same bytes;
/// * narrow (the QSA projections, the mHC down / injection; slabs of at most
///   2048 rows): ONE cuBLASLt configuration at every row count, the
///   heuristic's pick at a reference row count -- 2048 (the QSA slab: the
///   heuristic's own kernel there), 512 for the mHC down / injection
///   (split-K with a workspace reduction: within ~30 us of the heuristic at
///   64 rows, 1.7x the k-chain the default ran at full slabs). The 64-row
///   picks are not row-invariant: never use them.
fn reference_rows(n: u32, k: u32) -> Option<u32> {
    match (n, k) {
        (2048.., _) => None,
        (..=512, 8192..) => Some(512),
        _ => Some(2048),
    }
}

/// `out[m, n] = act[m, k] @ weight[n, k]^T` (BF16) row-invariantly
/// ([`reference_rows`]) when a row-invariant pass is running; `Ok(false)`
/// launched nothing. An error where the narrow configuration refuses the row
/// count: any other kernel would break the invariance.
pub fn try_bf16_gemm(
    act: DevicePtr,
    weight: u64,
    out: DevicePtr,
    [m, n, k]: [u32; 3],
    stream: u64,
) -> Result<bool> {
    let Some((gpu, _)) = pass() else {
        return Ok(false);
    };
    let Some(ref_m) = reference_rows(n, k) else {
        // The k-chain kernels group K alike only in whole k16 steps
        // (`bf16_gemm_matches_pipelined`).
        anyhow::ensure!(
            k.is_multiple_of(16),
            "ATLAS_QWEN4EXP_PREFILL_ROWINV: K = {k} is not a whole number of k16 steps"
        );
        if !spark_runtime::cublaslt::bf16_gemm_act_weight_t_kchain_any(
            act.0,
            weight,
            out.0,
            [m, n, k],
            stream,
        )? {
            let kernel = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;
            gemm_raw(gpu, kernel, act, DevicePtr(weight), out, m, n, k, stream)?;
        }
        return Ok(true);
    };
    anyhow::ensure!(
        spark_runtime::cublaslt::bf16_gemm_act_weight_t_fixed(
            act.0,
            weight,
            out.0,
            [m, n, k],
            ref_m,
            stream
        )?,
        "ATLAS_QWEN4EXP_PREFILL_ROWINV: cuBLASLt's {ref_m}-row configuration for \
         {n}x{k} refuses {m} rows"
    );
    Ok(true)
}

/// Rows a group of `hc_mma_down_rows` / `hc_mma_finish_rows` and their
/// weight tiles a CTA (`HCM_ROWS_*`, `HCM_DN_CL` in `qwen4exp_hc_mma.cu`).
const HCM_ROWS_DN_T: u32 = 96;
const HCM_ROWS_FN_T: u32 = 64;
const HCM_ROWS_WM: u32 = 7;
const HCM_DN_CL: u32 = 8;
/// `hc_mma_finish`'s output dims a CTA (`HCM_FN_DW`).
const HCM_FN_DW: u32 = 32;

/// The mHC collapse of `num_tokens` rows, byte for byte the decode collapse
/// (`ATLAS_QWEN4EXP_HC_FAST` + `_HC_MMA`) of each row: per slab of the
/// scratch, `hc_pre_stage_vec` (FP32 `normed`, one 1024-thread RMS a row),
/// then `hc_mma_down_rows` + `hc_mma_finish_rows` (`qwen4exp_hc_mma.cu`, the
/// decode kernels' per-row operation sequence over row groups). `inj_out`
/// NULL is the model-level head. `Ok(false)` outside a row-invariant pass;
/// an error where it cannot run, since any fallback would break the
/// invariance.
#[allow(clippy::too_many_arguments)]
pub fn hc_collapse(
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<bool> {
    let Some((gpu, slab)) = pass().filter(|_| decode_active()) else {
        return Ok(false);
    };
    let hc_dim = hc_mult * hidden_size;
    let rank = w.rank as u32;
    let k_stage = crate::layers::try_kernel(gpu, "hyper_connection", "hc_pre_stage_vec");
    let k_down = crate::layers::try_kernel(gpu, "qwen4exp_hc_mma", "hc_mma_down_rows");
    let k_fin = crate::layers::try_kernel(gpu, "qwen4exp_hc_mma", "hc_mma_finish_rows");
    anyhow::ensure!(
        slab >= HCM_ROWS_DN_T
            && !scratch.is_null()
            && hc_mult == 4
            && hc_dim.is_multiple_of(32 * HCM_DN_CL * 4)
            && rank.is_multiple_of(16 * 5)
            && hidden_size.is_multiple_of(HCM_FN_DW)
            && k_stage.0 != 0
            && k_down.0 != 0
            && k_fin.0 != 0,
        "ATLAS_QWEN4EXP_PREFILL_ROWINV: the mHC row kernels do not serve this \
         shape (hidden {hidden_size}, hc {hc_mult}, rank {rank}, slab {slab})"
    );
    let normed = scratch;
    let low = scratch.offset(slab as usize * hc_dim as usize * 4);
    let rows = rank + if inj_out.is_null() { 0 } else { hc_mult };
    let mut t0 = 0u32;
    while t0 < num_tokens {
        let ts = slab.min(num_tokens - t0);
        let t = t0 as usize;
        // Every stage block computes its row's whole RMS, so the block split
        // is launch geometry only (`hc_stage_split`): one a row once the
        // rows fill the part.
        let split = if ts >= 48 { 1 } else { 8 };
        KernelLaunch::new(gpu, k_stage)
            .grid([ts, split, 1])
            .block([1024, 1, 1])
            .arg_ptr(streams.offset(t * hc_dim as usize * 4))
            .arg_ptr(w.norm_w)
            .arg_ptr(normed)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_f32(norm_eps)
            .launch(stream)?;
        let inj = if inj_out.is_null() {
            DevicePtr::NULL
        } else {
            inj_out.offset(t * hc_mult as usize * 4)
        };
        KernelLaunch::new(gpu, k_down)
            .grid([
                rows.div_ceil(16).div_ceil(HCM_ROWS_WM),
                HCM_DN_CL,
                ts.div_ceil(HCM_ROWS_DN_T),
            ])
            .block([32 * HCM_ROWS_WM, 1, 1])
            .arg_ptr(normed)
            .arg_ptr(w.down_w)
            .arg_ptr(if inj.is_null() {
                DevicePtr::NULL
            } else {
                w.inject_w
            })
            .arg_ptr(low)
            .arg_ptr(inj)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(rank)
            .arg_u32(ts)
            .launch(stream)?;
        KernelLaunch::new(gpu, k_fin)
            .grid([hidden_size / HCM_FN_DW, 1, ts.div_ceil(HCM_ROWS_FN_T)])
            .block([128, 1, 1])
            // The stream-mean tile only: the fragments come from global.
            .shared_mem(4 * ts.min(HCM_ROWS_FN_T).div_ceil(8) * 8 * HCM_FN_DW * 4)
            .arg_ptr(normed)
            .arg_ptr(low)
            .arg_ptr(w.up_w)
            .arg_ptr(y_out.offset(t * hidden_size as usize * 2))
            .arg_u32(hidden_size)
            .arg_u32(rank)
            .arg_u32(ts)
            .launch(stream)?;
        t0 += ts;
        // ATLAS_QWEN4EXP_PREFILL_SP_PIPE: rows [0, t0) of `y_out` are set.
        crate::layers::qwen4exp_sp_pipe::slab_done(t0 as usize, stream)?;
    }
    Ok(true)
}

/// Whether a pass over `[start, ..)` keeps the GDN grid (logs once if not:
/// the rows after it are then not chunk-invariant).
pub fn check_pass_start(start: usize) {
    if on() && !start.is_multiple_of(PASS_GRANULE) {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(
                "ATLAS_QWEN4EXP_PREFILL_ROWINV: a prefill pass starts at {start}, off the \
                 {PASS_GRANULE}-token GDN grid; its rows are not chunk-invariant (logged once)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference_rows;

    #[test]
    fn wide_shapes_take_the_k_chain_and_narrow_ones_a_fixed_configuration() {
        // GDN in_proj at TP2 / TP1: the k-chain.
        assert_eq!(reference_rows(8192, 2560), None);
        assert_eq!(reference_rows(16384, 2560), None);
        // mHC down / injection: the 512-row pick (the 64-row ones are not
        // row-invariant).
        assert_eq!(reference_rows(320, 10240), Some(512));
        assert_eq!(reference_rows(4, 10240), Some(512));
        // QSA indexer q/k: the slab's own pick.
        assert_eq!(reference_rows(640, 2560), Some(2048));
    }
}
