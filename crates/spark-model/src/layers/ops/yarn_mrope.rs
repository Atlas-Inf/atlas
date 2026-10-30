// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Static YaRN over partial, interleaved MRoPE; both target and MTP use this.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn_mrope(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    seq_len: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    rot: u32,
    inv_freq: DevicePtr,
    attention_factor: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        rot > 0 && rot <= 256 && rot.is_multiple_of(2),
        "invalid YaRN rotary width"
    );
    anyhow::ensure!(
        !inv_freq.is_null() && kernel.0 != 0,
        "Qwen YaRN kernel/table missing"
    );
    let per_block = 128 / (rot / 2);
    KernelLaunch::new(gpu, kernel)
        .grid([nq + nkv, seq_len.div_ceil(per_block), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(pos_t)
        .arg_ptr(pos_h)
        .arg_ptr(pos_w)
        .arg_u32(seq_len)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_ptr(inv_freq)
        .arg_f32(attention_factor)
        .launch(stream)
}
