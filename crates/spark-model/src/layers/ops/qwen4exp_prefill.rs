// SPDX-License-Identifier: AGPL-3.0-only

//! `qwen4_exp` (Qwen3.8-Flash-Next) PREFILL switches and the launches behind
//! them. Every switch is read once per process and default OFF, except
//! `_QSA_TC2R`: the TP2 default (`=0` falls back to `_g` / `_GP`).
//!
//! | switch                                   | what                                   | exact? |
//! |------------------------------------------|----------------------------------------|--------|
//! | `ATLAS_QWEN4EXP_PREFILL_QSA_TC2R=0`      | TP2 QSA attention off tensor cores     | TC2R = TP1 tc2, != `_g` |
//! | `ATLAS_QWEN4EXP_PREFILL_QSA_LEAN=1`      | TP1 tc2 -> its lean twin (2 CTAs/SM)   | = tc2  |
//! | `ATLAS_QWEN4EXP_PREFILL_QSA_GP=1`        | `_g` QSA attention, rescheduled        | = `_g` |
//! | `ATLAS_QWEN4EXP_PREFILL_BA_ROWS=1`       | GDN BA GEMM + gates, 2 tokens a CTA    | = `dense_gemm_ba_gates_prefill` |
//! | `ATLAS_QWEN4EXP_PREFILL_FP8_W2=1`        | attention q/k/v FP8 x FP8 GEMM, 2x4 warps | = `fp8_fp8_gemm_t_m128` |
//! | `ATLAS_QWEN4EXP_PREFILL_QSA_SCORE=1`     | QSA block scorer, 16-byte loads, 2 rows a thread | = `qsa_score_rows_exact` |
//! | `ATLAS_QWEN4EXP_PREFILL_GDN=1`           | GDN spine over 2 CTAs a head (TP2); wide `chunk_fwd_o` | = `..._pipe`, = `chunk_fwd_o` |
//! | `ATLAS_QWEN4EXP_PREFILL_HC=1`            | mHC collapse: seam, down, up+mix fused | = default (`qwen4exp_prefill_hc`) |
//! | `ATLAS_QWEN4EXP_PREFILL_MOE=1`           | router, shared and routed experts, unpermute | = default (`moe::forward_prefill_q38`) |
//! | `ATLAS_QWEN4EXP_PREFILL_HC_CHECK=<n>`    | cross-check the first n mHC slabs      | diagnostic |
//!
//! "= X" means byte-identical outputs to kernel X on the same inputs, checked
//! by the `scripts/dev/qwen4exp_*_bench.cu` harnesses on a GB10.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

fn flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1") | Ok("true"))
}

/// The TP2 QSA prefill attention on tensor cores (`qsa_prefill_attn_tc2r`,
/// one kv head a rank): on unless `ATLAS_QWEN4EXP_PREFILL_QSA_TC2R=0`, which
/// keeps `_g` (or `_gp` under `_QSA_GP`). Its numerics are TP1 tc2's, not
/// `_g`'s; the TP2 default since 2026-10-06.
pub fn qsa_tc2r_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_QSA_TC2R").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

/// `ATLAS_QWEN4EXP_PREFILL_QSA_LEAN=1`.
pub fn qsa_lean_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_QSA_LEAN"))
}

/// One slab of QSA selected-set attention: the arguments every arm takes.
pub struct QsaAttnSlab {
    pub q: DevicePtr,
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    pub block_table: DevicePtr,
    pub lists: DevicePtr,
    pub attn_out: DevicePtr,
    pub rows: u32,
    pub first_pos: u32,
    pub topk: u32,
    pub ratio: u32,
    pub block_size: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub inv_sqrt_d: f32,
}

/// Run the slab on a lean tensor-core arm when one is requested and serves
/// this geometry; `Ok(false)` leaves it to the caller's usual arms.
///
/// * One kv head (TP2: 24 q / 2 kv split by heads) under `_QSA_TC2R`:
///   `qsa_prefill_attn_tc2r` puts TWO ROWS in tc2's M tile, one per half.
/// * Two kv heads (TP1) under `_QSA_LEAN`, when tc2 is the arm in force:
///   `qsa_prefill_attn_tc2l`, tc2 with Q in registers and one K buffer.
///
/// Both compile from `qsa_attn_tc2.cu`, so every output byte equals tc2's
/// for the same heads (`scripts/dev/qwen4exp_qsa_tc2r_bench.cu`). Measured on
/// GB10 at 2048 rows / position 14000 / topk 512: 12 heads x 1 kv head, `_g`
/// 21.2 ms -> tc2r 5.85 ms (3.6x); 24 x 2, tc2 20.9 -> tc2l 12.0 ms (1.75x).
pub fn try_qsa_prefill_attn_lean(
    gpu: &dyn GpuBackend,
    s: &QsaAttnSlab,
    tc2_in_force: bool,
    stream: u64,
) -> Result<bool> {
    let shape_ok = s.hd == 256 && s.nkv != 0 && s.nq.is_multiple_of(s.nkv) && s.nq / s.nkv <= 16;
    let (module, entry, row_pairs) = if s.nkv == 1 && qsa_tc2r_requested() {
        ("qsa_attn_tc2r", "qsa_prefill_attn_tc2r", true)
    } else if s.nkv == 2 && tc2_in_force && qsa_lean_requested() {
        ("qsa_attn_tc2l", "qsa_prefill_attn_tc2l", false)
    } else {
        return Ok(false);
    };
    let k = crate::layers::try_kernel(gpu, module, entry);
    if !shape_ok || k.0 == 0 || s.rows == 0 {
        return Ok(false);
    }
    let ctas = if row_pairs {
        s.rows.div_ceil(2)
    } else {
        s.rows
    };
    let mut l = KernelLaunch::new(gpu, k)
        .grid([1, ctas, 1])
        .block([128, 1, 1])
        .arg_ptr(s.q)
        .arg_ptr(s.k_cache)
        .arg_ptr(s.v_cache)
        .arg_ptr(s.attn_out)
        .arg_ptr(s.block_table)
        .arg_ptr(s.lists)
        .arg_u32(s.first_pos)
        .arg_u32(s.topk)
        .arg_u32(s.ratio)
        .arg_u32(s.block_size)
        .arg_u32(s.nq)
        .arg_u32(s.nkv)
        .arg_u32(s.hd)
        .arg_f32(s.inv_sqrt_d);
    if row_pairs {
        l = l.arg_u32(s.rows);
    }
    l.launch(stream)?;
    Ok(true)
}

/// `ATLAS_QWEN4EXP_PREFILL_QSA_GP=1`.
pub fn qsa_gp_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_QSA_GP"))
}

/// `QSA_GP_DEPTH` in `qsa_attn_gp.cu`: keys in flight a warp.
const QSA_GP_DEPTH: u32 = 4;

/// Dynamic shared memory of `qsa_prefill_attn_gp`: the key loop's K/V ring
/// and slot table, or `_g`'s merge buffer that aliases them, whichever is
/// larger.
pub fn qsa_prefill_attn_gp_smem(topk: u32, ratio: u32, hd: u32) -> u32 {
    let warps = 8;
    let ring = warps * QSA_GP_DEPTH * 2 * 256 * 2;
    let slots = (topk * ratio + ratio) * 4;
    let merge = (warps * super::QSA_PA_G * hd + 2 * warps * super::QSA_PA_G) * 4;
    (ring + slots).max(merge)
}

/// Run a slab that `qsa_prefill_attn_g` would serve on its exact twin
/// `qsa_prefill_attn_gp` (`qsa_attn_gp.cu`): the same arithmetic in the same
/// order, with the K/V addresses resolved once a row, a 4-deep cp.async ring,
/// the four heads' butterflies merged into one and two keys a step -- every
/// output byte equal to `_g`'s (`scripts/dev/qwen4exp_qsa_gp_bench.cu`; GB10,
/// 2048 rows at position 14000, 12 q / 1 kv heads: 21.1 -> 9.1 ms).
/// `Ok(false)` launched nothing: the caller runs `_g`.
pub fn try_qsa_prefill_attn_gp(gpu: &dyn GpuBackend, s: &QsaAttnSlab, stream: u64) -> Result<bool> {
    if !qsa_gp_requested() || !launch_qsa_prefill_attn_gp(gpu, s, stream)? {
        return Ok(false);
    }
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!("QSA prefill attention: gp (exact twin of grouped)");
    }
    Ok(true)
}

/// [`try_qsa_prefill_attn_gp`] without the switch: launch when the kernel
/// serves the slab's geometry, else `Ok(false)`.
pub fn launch_qsa_prefill_attn_gp(
    gpu: &dyn GpuBackend,
    s: &QsaAttnSlab,
    stream: u64,
) -> Result<bool> {
    let smem = qsa_prefill_attn_gp_smem(s.topk, s.ratio, s.hd);
    let serves = s.hd == 256
        && super::qsa_prefill_attn_grouped_ok(s.nq, s.nkv, s.hd)
        && smem <= 96 * 1024
        && s.k_cache.0.is_multiple_of(16)
        && s.v_cache.0.is_multiple_of(16)
        && s.rows != 0;
    if !serves {
        return Ok(false);
    }
    let k = crate::layers::try_kernel(gpu, "qsa_attn_gp", "qsa_prefill_attn_gp");
    if k.0 == 0 {
        return Ok(false);
    }
    KernelLaunch::new(gpu, k)
        .grid([s.rows, s.nq / super::QSA_PA_G, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(s.q)
        .arg_ptr(s.k_cache)
        .arg_ptr(s.v_cache)
        .arg_ptr(s.block_table)
        .arg_ptr(s.lists)
        .arg_ptr(s.attn_out)
        .arg_u32(s.first_pos)
        .arg_u32(s.topk)
        .arg_u32(s.ratio)
        .arg_u32(s.block_size)
        .arg_u32(s.nq)
        .arg_u32(s.nkv)
        .arg_u32(s.hd)
        .arg_f32(s.inv_sqrt_d)
        .launch(stream)?;
    Ok(true)
}

/// `ATLAS_QWEN4EXP_PREFILL_QSA_SCORE=1`.
pub fn qsa_score_v4_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_QSA_SCORE"))
}

/// Rows a `qsa_score_rows_exact_v4` thread scores (`QSA_V4_RPT`).
const QSA_V4_RPT: u32 = 2;

/// The QSA prefill block scores on `qsa_score_rows_exact_v4`: the exact
/// scorer's products, tree and folds on the same values, staged with 16-byte
/// shared loads and two rows a thread -- every score byte identical
/// (`scripts/dev/qwen4exp_qsa_score_bench.cu`; GB10, 2048 rows at position
/// 14000: 6.28 -> 3.06 ms). `ptrs` = [q, block_keys, scores]; `dims` =
/// [rows, n_blocks_max, first_pos, score_stride, ratio, n_heads, hd].
/// `Ok(false)` launched nothing.
pub fn try_qsa_score_v4(
    gpu: &dyn GpuBackend,
    ptrs: [DevicePtr; 3],
    dims: [u32; 7],
    stream: u64,
) -> Result<bool> {
    let [
        rows,
        n_blocks_max,
        first_pos,
        score_stride,
        ratio,
        n_heads,
        hd,
    ] = dims;
    let [q, block_keys, scores] = ptrs;
    let (bm, bn) = (super::QSA_SE_BM * QSA_V4_RPT, super::QSA_SE_BN);
    let smem = (bm * n_heads * hd + bn * (hd + 4)) * 4;
    if !qsa_score_v4_requested()
        || !hd.is_multiple_of(32)
        || smem > 96 * 1024
        || !q.0.is_multiple_of(16)
        || rows == 0
    {
        return Ok(false);
    }
    let k = crate::layers::try_kernel(gpu, "qsa_indexer", "qsa_score_rows_exact_v4");
    if k.0 == 0 {
        return Ok(false);
    }
    KernelLaunch::new(gpu, k)
        .grid([rows.div_ceil(bm), n_blocks_max.div_ceil(bn), 1])
        .block([super::QSA_SE_BM * bn, 1, 1])
        .shared_mem(smem)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(rows)
        .arg_u32(n_blocks_max)
        .launch(stream)?;
    Ok(true)
}

/// `ATLAS_QWEN4EXP_PREFILL_GDN=1`.
pub fn gdn_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_GDN"))
}

/// The GDN prefill state spine as `gated_delta_rule_chunk_delta_h_pipe_dv64`
/// (common/gated_delta_rule_fla.cu): `..._pipe`'s per-column program with the
/// 128 state columns of a head split over two CTAs, every S_out / uc_out /
/// state byte identical (`scripts/dev/qwen4exp_gdn_spine_bench.cu`). Taken
/// only where `..._pipe` is the spine in force and the doubled grid still
/// fits one wave: at TP2 a qwen4_exp rank has 24 v-heads, so `..._pipe`
/// leaves half of GB10's 48 SMs idle (16000 tokens: 9.69 -> 6.99 ms a
/// layer); at TP1's 48 heads the split is a second wave and loses (0.70x).
/// Returns the kernel and its dynamic shared memory.
pub fn gdn_pipe_dv(
    gpu: &dyn GpuBackend,
    heads_x_batch: u32,
    kd: u32,
    vd: u32,
    is_varlen: bool,
) -> Option<(spark_runtime::gpu::KernelHandle, u32)> {
    const C: u32 = 64;
    if !gdn_requested() || kd != 128 || vd != 128 || is_varlen {
        return None;
    }
    let sms = gpu.sm_count().unwrap_or(0);
    if sms == 0 || heads_x_batch * 2 > sms {
        return None;
    }
    let k = crate::layers::try_kernel(
        gpu,
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_pipe_dv64",
    );
    // 2 x {W, K} [C, kd] + 2 x U [C, 64] BF16, then gc [2][C] and decay [2][C + 1].
    let smem = 2 * 2 * C * kd * 2 + 2 * C * 64 * 2 + 2 * C * 4 + 2 * (C + 1) * 4;
    (k.0 != 0).then_some((k, smem))
}

/// The GDN prefill output pass as `gated_delta_rule_chunk_fwd_o_wide`: the
/// default `chunk_fwd_o` with its closing per-(row, column) loop on all 512
/// threads, S_c staged as stored (16-byte copies, fragments read `[k][v]`) and
/// q / k / uc staged 16 bytes a copy -- every output byte identical
/// (`scripts/dev/qwen4exp_gdn_spine_bench.cu`; 16000 tokens at the TP2 rank
/// shape: 7.01 -> 3.57 ms a layer). Same grid, block and shared memory.
pub fn gdn_fwd_o_wide(
    gpu: &dyn GpuBackend,
    kd: u32,
    vd: u32,
    qk_stride: u32,
    q: DevicePtr,
    k: DevicePtr,
) -> Option<spark_runtime::gpu::KernelHandle> {
    if !gdn_requested()
        || kd != 128
        || vd != 128
        || !qk_stride.is_multiple_of(8)
        || !q.0.is_multiple_of(16)
        || !k.0.is_multiple_of(16)
    {
        return None;
    }
    let h = crate::layers::try_kernel(
        gpu,
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_fwd_o_wide",
    );
    (h.0 != 0).then_some(h)
}

/// `ATLAS_QWEN4EXP_PREFILL_BA_ROWS=1`.
fn ba_rows_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_BA_ROWS"))
}

/// `QBA_TOK` / `QBA_MAX_GROUPS` in `qwen4exp_gdn_prefill.cu`.
const QBA_TOK: u32 = 2;
const QBA_MAX_OUTPUTS: u32 = 48;

/// The GDN prefill BA GEMM + gates on `qwen4exp_ba_gates_prefill_rows`: the
/// default `dense_gemm_ba_gates_prefill`'s lanes, order and transforms for
/// every output, with two tokens and every output in one CTA (the weight was
/// re-read from L2 for every token) -- every gate byte identical
/// (`scripts/dev/qwen4exp_ba_gates_bench.cu`; GB10, 16016 tokens at the TP2
/// shape: 2.25 -> 0.97 ms). `dims` = [m, n, k, k_stride, gate_stride, nv,
/// vheads_per_group] as the default takes them; `ptrs` = [input, weight,
/// a_log, dt_bias, gate_out]. `Ok(false)` launched nothing.
pub fn try_ba_gates_rows(
    gpu: &dyn GpuBackend,
    ptrs: [DevicePtr; 5],
    dims: [u32; 7],
    stream: u64,
) -> Result<bool> {
    if !ba_rows_requested() {
        return Ok(false);
    }
    launch_ba_gates_rows(gpu, ptrs, dims, stream)
}

/// [`try_ba_gates_rows`] without the switch.
pub fn launch_ba_gates_rows(
    gpu: &dyn GpuBackend,
    ptrs: [DevicePtr; 5],
    dims: [u32; 7],
    stream: u64,
) -> Result<bool> {
    let [m, n, k, k_stride, gate_stride, nv, vpg] = dims;
    if n > QBA_MAX_OUTPUTS || !k.is_multiple_of(8) || m == 0 {
        return Ok(false);
    }
    let kernel = crate::layers::try_kernel(
        gpu,
        "qwen4exp_gdn_prefill",
        "qwen4exp_ba_gates_prefill_rows",
    );
    if kernel.0 == 0 {
        return Ok(false);
    }
    let [input, weight, a_log, dt_bias, gate_out] = ptrs;
    KernelLaunch::new(gpu, kernel)
        .grid([m.div_ceil(QBA_TOK), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_stride)
        .arg_u32(gate_stride)
        .arg_u32(nv)
        .arg_u32(vpg)
        .launch(stream)?;
    Ok(true)
}

/// `ATLAS_QWEN4EXP_PREFILL_FP8_W2=1`.
fn fp8_w2_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_FP8_W2"))
}

/// `QF_BK` / `QF_STAGES` in `qwen4exp_fp8_gemm.cu`.
const QF_BK: u32 = 128;
const QF_STAGES: u32 = 2;

/// An attention prefill projection (`fp8_fp8_gemm_t_m128`: E4M3 activations
/// x E4M3 weight, BF16 out) on `qwen4exp_fp8_gemm_w2`: the same k32 MMA
/// chain per output on 8 warps in a 2 x 4 grid with 128-wide K steps --
/// every byte identical (`scripts/dev/qwen4exp_fp8_gemm_bench.cu`; GB10,
/// TP2 q+gate 16016 x 6144 x 2560: 8.46 -> 5.16 ms; k/v: 0.54 -> 0.30).
/// `ptrs` = [a_fp8, b_fp8, out], `dims` = [m, n, k]. `Ok(false)` launched
/// nothing.
pub fn try_fp8_gemm_w2(
    gpu: &dyn GpuBackend,
    ptrs: [DevicePtr; 3],
    dims: [u32; 3],
    stream: u64,
) -> Result<bool> {
    let [m, n, k] = dims;
    if !fp8_w2_requested() || !k.is_multiple_of(QF_BK) || m == 0 || n == 0 {
        return Ok(false);
    }
    launch_fp8_gemm_w2(gpu, ptrs, dims, stream)
}

/// [`try_fp8_gemm_w2`] without the switch.
pub fn launch_fp8_gemm_w2(
    gpu: &dyn GpuBackend,
    [a, b, out]: [DevicePtr; 3],
    [m, n, k]: [u32; 3],
    stream: u64,
) -> Result<bool> {
    let kernel = crate::layers::try_kernel(gpu, "qwen4exp_fp8_gemm", "qwen4exp_fp8_gemm_w2");
    if kernel.0 == 0 || !k.is_multiple_of(QF_BK) {
        return Ok(false);
    }
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(128), m.div_ceil(128), 1])
        .block([256, 1, 1])
        .shared_mem(QF_STAGES * 256 * (QF_BK + 16))
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
#[path = "qwen4exp_prefill_gpu_tests.rs"]
mod gpu_tests;
