// SPDX-License-Identifier: AGPL-3.0-only

//! `qwen4_exp` (Qwen3.8-Flash-Next) PREFILL switches and the launches behind
//! them. Every switch is default OFF and read once per process.
//!
//! | switch                                   | what                                   | exact? |
//! |------------------------------------------|----------------------------------------|--------|
//! | `ATLAS_QWEN4EXP_PREFILL_QSA_TC2R=1`      | TP2 QSA attention on tensor cores      | = TP1 tc2, != `_g` |
//! | `ATLAS_QWEN4EXP_PREFILL_QSA_LEAN=1`      | TP1 tc2 -> its lean twin (2 CTAs/SM)   | = tc2  |
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

/// `ATLAS_QWEN4EXP_PREFILL_QSA_TC2R=1`.
pub fn qsa_tc2r_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_QWEN4EXP_PREFILL_QSA_TC2R"))
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
