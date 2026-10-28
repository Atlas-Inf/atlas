// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

pub fn kda_pack_qkv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    planes: DevicePtr,
    packed: DevicePtr,
    tokens: u32,
    dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(tokens * 3 * dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(planes)
        .arg_ptr(packed)
        .arg_u32(tokens)
        .arg_u32(dim)
        .launch(stream)
}

/// The KDA prefill convolution of `m` rows into `output` (`[m, 3p]`) from
/// packed `[m, 3p]` q|k|v rows or, `planar`, the three `[m, p]` planes the
/// projection wrote (`ATLAS_GLM_KDA_CONV_PLANES`, no `kda_pack_qkv` copy):
/// one [`super::conv1d_update_prefill`] per plane over that plane's channels,
/// state rows and weights. Every (channel, token) output and every channel's
/// state is the same arithmetic either way, so the bytes are identical.
#[allow(clippy::too_many_arguments)]
pub fn kda_conv_prefill(
    gpu: &dyn GpuBackend,
    [kernel, tp_kernel]: [KernelHandle; 2],
    conv_state: DevicePtr,
    (input, planar): (DevicePtr, bool),
    weight: &DenseWeight,
    output: DevicePtr,
    [m, p, d_conv]: [u32; 3],
    stream: u64,
) -> Result<()> {
    let (planes, in_stride) = if planar { (3, p) } else { (1, 3 * p) };
    let channels = 3 * p / planes;
    for plane in 0..planes {
        let ch0 = (plane * channels) as usize;
        super::conv1d_update_prefill(
            gpu,
            kernel,
            tp_kernel,
            conv_state.offset(ch0 * d_conv as usize * 4),
            input.offset(plane as usize * m as usize * channels as usize * 2),
            &DenseWeight {
                weight: weight.weight.offset(ch0 * d_conv as usize * 2),
            },
            DevicePtr::NULL,
            output.offset(ch0 * 2),
            channels,
            d_conv,
            m,
            in_stride,
            3 * p,
            stream,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kda_recurrent(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qkv: DevicePtr,
    raw_gate: DevicePtr,
    raw_beta: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    state: DevicePtr,
    output: DevicePtr,
    tokens: u32,
    heads: u32,
    dim: u32,
    lower_bound: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([heads, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(qkv)
        .arg_ptr(raw_gate)
        .arg_ptr(raw_beta)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(state)
        .arg_ptr(output)
        .arg_u32(tokens)
        .arg_u32(heads)
        .arg_u32(dim)
        .arg_f32(lower_bound)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn kda_recurrent_verify_snap(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qkv: DevicePtr,
    raw_gate: DevicePtr,
    raw_beta: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    state: DevicePtr,
    output: DevicePtr,
    state_inter: DevicePtr,
    inter_stride: usize,
    tokens: u32,
    heads: u32,
    dim: u32,
    lower_bound: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([heads, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(qkv)
        .arg_ptr(raw_gate)
        .arg_ptr(raw_beta)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(state)
        .arg_ptr(output)
        .arg_ptr(state_inter)
        .arg_u64(inter_stride as u64)
        .arg_u32(tokens)
        .arg_u32(heads)
        .arg_u32(dim)
        .arg_f32(lower_bound)
        .launch(stream)
}

/// `kda_recurrent_verify_snap` for up to four owners in one launch: owner
/// `o` advances `states[o]` over its `tokens` rows at row `o * tokens`,
/// writing its rollback slab `inters[o]`; bit-identical per owner. With
/// `kda_recurrent_bf16_verify_rec_owners` the states are only read and
/// `inters[o]` receives the fold records [`kda_commit_records`] consumes.
#[allow(clippy::too_many_arguments)]
pub fn kda_recurrent_verify_snap_owners(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qkv: DevicePtr,
    raw_gate: DevicePtr,
    raw_beta: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    states: &[DevicePtr],
    inters: &[DevicePtr],
    inter_stride: usize,
    tokens: u32,
    heads: u32,
    dim: u32,
    lower_bound: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        (1..=4).contains(&states.len()) && inters.len() == states.len(),
        "KDA owner-batched verify takes 1..=4 owners"
    );
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid([heads, states.len() as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(qkv)
        .arg_ptr(raw_gate)
        .arg_ptr(raw_beta)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(output);
    for list in [states, inters] {
        for o in 0..4 {
            l = l.arg_ptr(list.get(o).copied().unwrap_or(DevicePtr::NULL));
        }
    }
    l.arg_u64(inter_stride as u64)
        .arg_u32(tokens)
        .arg_u32(heads)
        .arg_u32(dim)
        .arg_f32(lower_bound)
        .launch(stream)
}

/// Floats per head in one row of KDA fold records (decay, key, correction).
pub const KDA_RECORD_FLOATS: usize = 384;

/// Key-row blocks per head in [`kda_commit_records`] (`KDA_COMMIT_KSPLIT`).
const KDA_COMMIT_KSPLIT: u32 = 4;

/// `ATLAS_GLM_KDA_COMMIT_CTAS`: unset or `0` commits over the full grid
/// (`kda_commit_records`, one CTA per key-row block of every layer), `N` in
/// `1..=4096` over `N` CTAs (`kda_commit_records_stride`, same words). Read
/// once.
pub fn kda_commit_ctas() -> Result<u32> {
    static CTAS: std::sync::OnceLock<std::result::Result<u32, String>> = std::sync::OnceLock::new();
    CTAS.get_or_init(
        || match std::env::var("ATLAS_GLM_KDA_COMMIT_CTAS").as_deref() {
            Err(_) => Ok(0),
            Ok(value) => value
                .parse::<u32>()
                .ok()
                .filter(|&n| n <= 4096)
                .ok_or_else(|| {
                    format!("ATLAS_GLM_KDA_COMMIT_CTAS requires 0..=4096, got {value:?}")
                }),
        },
    )
    .clone()
    .map_err(anyhow::Error::msg)
}

/// The commit kernel for [`kda_commit_ctas`] CTAs.
pub fn kda_commit_kernel(gpu: &dyn GpuBackend, ctas: u32) -> Result<KernelHandle> {
    let name = match ctas {
        0 => "kda_commit_records",
        _ => "kda_commit_records_stride",
    };
    gpu.kernel("kda", name)
}

/// Advance `layers` FP32 states (`heads` x 128 x 128 each, layer `l` at
/// `state + l * state_layer_stride` floats) over the first `rows` fold
/// records `record_stride` floats apart (layer `l`'s at `records + l *
/// records_layer_stride` floats): the accepted prefix of a records verify,
/// bit-identical to the state that verify reached at that row. One launch
/// for every layer; a single layer passes `layers == 1` (strides unused).
/// `kernel` is [`kda_commit_kernel`]`(ctas)`: the full grid at `ctas == 0`,
/// else `ctas` CTAs striding over it.
#[allow(clippy::too_many_arguments)]
pub fn kda_commit_records(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    state: DevicePtr,
    state_layer_stride: usize,
    records: DevicePtr,
    records_layer_stride: usize,
    record_stride: usize,
    rows: u32,
    heads: u32,
    layers: u32,
    ctas: u32,
    stream: u64,
) -> Result<()> {
    let grid = match ctas {
        0 => [heads * KDA_COMMIT_KSPLIT, layers, 1],
        _ => [ctas.min(heads * KDA_COMMIT_KSPLIT * layers), 1, 1],
    };
    let launch = KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block([128, 1, 1])
        .arg_ptr(state)
        .arg_u64(state_layer_stride as u64)
        .arg_ptr(records)
        .arg_u64(records_layer_stride as u64)
        .arg_u64(record_stride as u64)
        .arg_u32(rows)
        .arg_u32(heads);
    match ctas {
        0 => launch.launch(stream),
        _ => launch.arg_u32(layers).launch(stream),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn kda_recurrent_regresident(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qkv: DevicePtr,
    q_norm: DevicePtr,
    k_norm: DevicePtr,
    decay: DevicePtr,
    beta: DevicePtr,
    state: DevicePtr,
    output: DevicePtr,
    tokens: u32,
    heads: u32,
    dim: u32,
    stream: u64,
) -> Result<()> {
    const WARPS_PER_BLOCK: u32 = 4;
    KernelLaunch::new(gpu, kernel)
        .grid([heads, div_ceil(dim, WARPS_PER_BLOCK), 1])
        .block([32 * WARPS_PER_BLOCK, 1, 1])
        .arg_ptr(qkv)
        .arg_ptr(q_norm)
        .arg_ptr(k_norm)
        .arg_ptr(decay)
        .arg_ptr(beta)
        .arg_ptr(state)
        .arg_ptr(output)
        .arg_u32(tokens)
        .arg_u32(heads)
        .arg_u32(dim)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn kda_preprocess_regresident(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qkv: DevicePtr,
    raw_gate: DevicePtr,
    raw_beta: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    q_norm: DevicePtr,
    k_norm: DevicePtr,
    decay: DevicePtr,
    beta: DevicePtr,
    tokens: u32,
    heads: u32,
    dim: u32,
    lower_bound: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([heads, tokens, 1])
        .block([dim, 1, 1])
        .arg_ptr(qkv)
        .arg_ptr(raw_gate)
        .arg_ptr(raw_beta)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(q_norm)
        .arg_ptr(k_norm)
        .arg_ptr(decay)
        .arg_ptr(beta)
        .arg_u32(tokens)
        .arg_u32(heads)
        .arg_u32(dim)
        .arg_f32(lower_bound)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn kda_sigmoid_gated_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    tokens: u32,
    heads: u32,
    dim: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([tokens * heads, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(heads)
        .arg_u32(dim)
        .arg_f32(eps)
        .launch(stream)
}

#[cfg(all(test, feature = "cuda"))]
#[path = "kda_conv_planes_test_gpu.rs"]
mod conv_planes_gpu_tests;
