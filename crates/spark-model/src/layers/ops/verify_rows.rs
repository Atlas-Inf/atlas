// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-width (M <= 32 rows) launch ops for the Ricursive target's DFlash
//! verify and drafter: small-M BF16 / FP8 / W4A4 GEMVs and the graph-safe
//! verify-row attention (kernels in `kernels/gb10/ricursive-ripple-65b/`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Rows [`bf16_gemv_m32`] serves (two 16-row MMA fragments).
pub const BF16_GEMV_MAX_M: u32 = 32;

/// Small-M BF16 GEMV (`bf16_gemv_m32`, Ricursive target): `out[m, n]` (row
/// stride `ldc`) = `act[m, k] . weight[n, k]^T` for `m <= BF16_GEMV_MAX_M`;
/// `qg_head_dim` > 0 stores a gated q_proj's `[Q_h | G_h]` heads split.
/// 16 weight rows per CTA streamed once through shared memory.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemv_m32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    act: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    [m, n, k]: [u32; 3],
    ldc: u32,
    qg_head_dim: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=BF16_GEMV_MAX_M).contains(&m) && k.is_multiple_of(512) && ldc >= n,
        "bf16_gemv_m32: m={m} n={n} k={k} ldc={ldc} unsupported"
    );
    ensure!(
        qg_head_dim == 0 || n.is_multiple_of(2 * qg_head_dim),
        "bf16_gemv_m32: n={n} is not whole [Q|G] heads of {qg_head_dim}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), 1, 1]) // B16G_BN
        .block([128, 1, 1])
        .arg_ptr(act)
        .arg_ptr(weight)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(ldc)
        .arg_u32(qg_head_dim)
        .launch(stream)
}

/// Small-M FP8-weight GEMV (`fp8_gemv_m32`, Ricursive's drafter):
/// `out[m, n] = row_scale[n] * act[m, k] . w8[n, k]^T` for `m <= BF16_GEMV_MAX_M`
/// on [`crate::weight_map::Fp8DenseWeight`] (E4M3 rows, f32 per-row scale).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemv_m32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    act: DevicePtr,
    weight: &crate::weight_map::Fp8DenseWeight,
    out: DevicePtr,
    [m, n, k]: [u32; 3],
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=BF16_GEMV_MAX_M).contains(&m) && k.is_multiple_of(512),
        "fp8_gemv_m32: m={m} k={k} unsupported"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(act)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Rows [`w4a4_gemv_m32`] serves (two 16-row MMA fragments).
pub const W4A4_GEMV_MAX_M: u32 = 32;

/// Small-M W4A4 GEMV (`w4a4_gemv_m32_bn{32,16}`, Ricursive target): the
/// operands and scales of [`w4a4_gemm_mfast_scaled`] for
/// `m <= W4A4_GEMV_MAX_M` rows. `kernels` = (32-row, 16-row) weight-panel
/// variants; the 16-row panel keeps narrow N (the down projection) at full
/// occupancy.
#[allow(clippy::too_many_arguments)]
pub fn w4a4_gemv_m32(
    gpu: &dyn GpuBackend,
    kernels: (KernelHandle, KernelHandle),
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    scale_a2: f32,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=W4A4_GEMV_MAX_M).contains(&m) && k.is_multiple_of(512),
        "w4a4_gemv_m32: m={m} k={k} unsupported"
    );
    let (kernel, bn) = if n >= 16384 {
        (kernels.0, 32)
    } else {
        (kernels.1, 16)
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, bn), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(output)
        .arg_f32(scale_a2)
        .arg_f32(weight.weight_scale_2)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// DFlash verify attention for one sequence's `q_len` rows
/// (`ricursive_verify_attn`): [`prefill_attention_paged`]'s BF16 kernel with
/// the KV extent read from `dev_kv_len` (the last row's device `seq_len`),
/// so the launch carries no per-step host values and can be graph-captured.
#[allow(clippy::too_many_arguments)]
pub fn verify_rows_attention_paged(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    dev_kv_len: DevicePtr,
    [q_len, num_q_heads, num_kv_heads, head_dim]: [u32; 4],
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(q_len)
        .arg_u32(0) // kv_len: overwritten from dev_kv_len in-kernel
        .arg_u32(0) // q_offset: kv_len - q_len in-kernel
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32) // causal
        .arg_f32(inv_sqrt_d)
        .arg_ptr(dev_kv_len)
        .launch(stream)
}
