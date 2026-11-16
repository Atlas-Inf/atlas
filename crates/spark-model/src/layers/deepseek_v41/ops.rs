// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the DeepSeek-V4.1 attention kernels (`dsv41_attn.cu`,
//! `dsv41_cache.cu`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Values in one key, and in one head's query.
pub const HEAD_DIM: usize = 512;
/// The most keys one query attends to: the window and the selected
/// compressed positions.
pub const MAX_KEYS: usize = 640;

const THREADS: u32 = 256;
/// `dsv41_sparse_attn` runs a head on one block this wide: a thread a key.
const ATTN_THREADS: u32 = 640;
/// Heads a block of the shared form takes together, and the FP32 values it
/// keeps of a (query, head) between its launches (`dsv41_attn.cu`).
const ATTN_GROUP: usize = 16;
const ATTN_RECORD: usize = 656;

/// The kernels of the V4.1 attention cache.
#[derive(Clone, Copy)]
pub struct Kernels {
    pub front: KernelHandle,
    /// The attention with a block a head, and with the heads sharing the
    /// keys they read, in three launches.
    pub sparse_attn: KernelHandle,
    attn_logits: KernelHandle,
    attn_schedule: KernelHandle,
    attn_output: KernelHandle,
    pub rotate: KernelHandle,
    pub fp8_quant_dequant: KernelHandle,
    pub fp4_quant_dequant: KernelHandle,
}

impl Kernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            front: gpu.kernel("dsv41_front", "dsv41_attn_front")?,
            sparse_attn: gpu.kernel("dsv41_attn", "dsv41_sparse_attn")?,
            attn_logits: gpu.kernel("dsv41_attn", "dsv41_attn_logits")?,
            attn_schedule: gpu.kernel("dsv41_attn", "dsv41_attn_schedule")?,
            attn_output: gpu.kernel("dsv41_attn", "dsv41_attn_output")?,
            rotate: gpu.kernel("dsv41_cache", "dsv41_rotate")?,
            fp8_quant_dequant: gpu.kernel("dsv41_cache", "dsv41_fp8_quant_dequant")?,
            fp4_quant_dequant: gpu.kernel("dsv41_cache", "dsv41_fp4_quant_dequant")?,
        })
    }
}

/// A turn of rotary pairs: `positions` (`[rows]` u32), `freqs` (`[rope / 2]`
/// FP32, radians per position), over the last `rope` values of a vector.
#[derive(Clone, Copy)]
pub struct Turn {
    pub positions: DevicePtr,
    pub freqs: DevicePtr,
    pub rope: usize,
}

/// What each of `rows` tokens' attention does between its first two
/// projections and its heads, in one launch (`dsv41_front.cu`): `q_normed`
/// (`[rows, q_rank]` BF16) = the norm of `q_low` under `q_weight`, and `slot`
/// (`[rows, 512]` BF16) = the FP8 form of the norm of `key` under
/// `key_weight` turned to the token's position (`turn.positions`, a row a
/// token). The bytes of [`rotate`], [`fp8_quant_dequant`] and the RMS norm
/// launched one at a time.
#[allow(clippy::too_many_arguments)]
pub fn front(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_low: DevicePtr,
    q_weight: DevicePtr,
    q_normed: DevicePtr,
    key: DevicePtr,
    key_weight: DevicePtr,
    slot: DevicePtr,
    turn: Turn,
    q_rank: usize,
    rows: usize,
    eps: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        q_rank.is_multiple_of(2) && q_rank <= 2048 && turn.rope <= HEAD_DIM,
        "dsv41_attn_front: q_rank {q_rank}, rope {}",
        turn.rope
    );
    KernelLaunch::new(gpu, kernel)
        .grid([rows as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(q_low)
        .arg_ptr(q_weight)
        .arg_ptr(q_normed)
        .arg_ptr(key)
        .arg_ptr(key_weight)
        .arg_ptr(slot)
        .arg_ptr(turn.positions)
        .arg_ptr(turn.freqs)
        .arg_u32(q_rank as u32)
        .arg_u32(turn.rope as u32)
        .arg_f32(eps)
        .launch(stream)
}

/// Attention of `queries` queries of `heads` heads (`q`, `[queries, heads,
/// 512]` BF16) over gathered keys, into `out` of the same shape, in the plain
/// form: a block a head. `idx` (`[queries, keys]` i32) holds row numbers, -1
/// for none: columns below `split` read `keys_a`, the rest `keys_b`. `sink`
/// is `[heads]` FP32.
///
/// A query's bytes depend only on its own q and on its index row, not on the
/// other queries of the launch.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_plain(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    keys_a: DevicePtr,
    keys_b: DevicePtr,
    idx: DevicePtr,
    sink: DevicePtr,
    out: DevicePtr,
    queries: usize,
    heads: usize,
    keys: usize,
    split: usize,
    stream: u64,
) -> Result<()> {
    ensure!(keys <= MAX_KEYS, "dsv41_sparse_attn: {keys} keys");
    KernelLaunch::new(gpu, kernel)
        .grid([queries as u32, heads as u32, 1])
        .block([ATTN_THREADS, 1, 1])
        .arg_ptr(q)
        .arg_ptr(keys_a)
        .arg_ptr(keys_b)
        .arg_ptr(idx)
        .arg_ptr(sink)
        .arg_ptr(out)
        .arg_u32(heads as u32)
        .arg_u32(keys as u32)
        .arg_u32(split as u32)
        .arg_f32((HEAD_DIM as f32).powf(-0.5))
        .launch(stream)
}

/// Bytes of the scratch [`sparse_attn`] keeps between its launches.
pub fn attn_records(queries: usize, heads: usize) -> usize {
    queries * heads * ATTN_RECORD * 4
}

/// [`sparse_attn_plain`] with the heads of a query sharing the keys they
/// read, in three launches: the same bytes, and at one token a fraction of
/// the time, for the plain form reads every key twice a head. `records` is
/// scratch of [`attn_records`] bytes, and `heads` a multiple of 16.
///
/// With `turn`, `q` is turned to each query's position before it scores
/// and the output turned back, as [`rotate`] would in place before and
/// after.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn(
    gpu: &dyn GpuBackend,
    kernels: &Kernels,
    q: DevicePtr,
    keys_a: DevicePtr,
    keys_b: DevicePtr,
    idx: DevicePtr,
    sink: DevicePtr,
    out: DevicePtr,
    records: DevicePtr,
    queries: usize,
    heads: usize,
    keys: usize,
    split: usize,
    turn: Option<Turn>,
    stream: u64,
) -> Result<()> {
    let turn = turn.unwrap_or(Turn {
        positions: DevicePtr::NULL,
        freqs: DevicePtr::NULL,
        rope: 0,
    });
    ensure!(
        keys <= MAX_KEYS && heads.is_multiple_of(ATTN_GROUP) && turn.rope.is_multiple_of(32),
        "dsv41_attn: {keys} keys, {heads} heads, rope {}",
        turn.rope
    );
    // A block is 16 heads of a query, by 16 keys and then by 16 values.
    let groups = (queries * heads / ATTN_GROUP) as u32;
    let block = [(ATTN_GROUP * ATTN_GROUP) as u32, 1, 1];
    let steps = keys.div_ceil(64) as u32;
    let turned = |launch: KernelLaunch<'_>| {
        launch
            .arg_ptr(turn.positions)
            .arg_ptr(turn.freqs)
            .arg_u32(turn.rope as u32)
            .launch(stream)
    };
    turned(
        KernelLaunch::new(gpu, kernels.attn_logits)
            .grid([steps * (64 / ATTN_GROUP) as u32, groups, 1])
            .block(block)
            .arg_ptr(q)
            .arg_ptr(keys_a)
            .arg_ptr(keys_b)
            .arg_ptr(idx)
            .arg_ptr(records)
            .arg_u32(heads as u32)
            .arg_u32(keys as u32)
            .arg_u32(split as u32)
            .arg_f32((HEAD_DIM as f32).powf(-0.5)),
    )?;
    KernelLaunch::new(gpu, kernels.attn_schedule)
        .grid([queries as u32, heads as u32, 1])
        .block([ATTN_THREADS, 1, 1])
        .arg_ptr(records)
        .arg_ptr(sink)
        .arg_u32(heads as u32)
        .arg_u32(keys as u32)
        .launch(stream)?;
    turned(
        KernelLaunch::new(gpu, kernels.attn_output)
            .grid([(HEAD_DIM / ATTN_GROUP) as u32, groups, 1])
            .block(block)
            .arg_ptr(keys_a)
            .arg_ptr(keys_b)
            .arg_ptr(idx)
            .arg_ptr(records)
            .arg_ptr(out)
            .arg_u32(heads as u32)
            .arg_u32(keys as u32)
            .arg_u32(split as u32),
    )
}

/// Rotate the last `rope` values of every vector of `x` (`[rows, vectors,
/// dim]` BF16) to its row's position (`positions`, `[rows]` u32), in place.
/// `freqs` is `[rope / 2]` FP32, radians per position. `inverse` undoes it.
#[allow(clippy::too_many_arguments)]
pub fn rotate(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    positions: DevicePtr,
    freqs: DevicePtr,
    rows: usize,
    vectors: usize,
    dim: usize,
    rope: usize,
    inverse: bool,
    stream: u64,
) -> Result<()> {
    ensure!(
        rope <= dim && rope.is_multiple_of(2),
        "dsv41_rotate: rope {rope} of {dim}"
    );
    let pairs = (rows * vectors * rope / 2) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(pairs, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_ptr(positions)
        .arg_ptr(freqs)
        .arg_u32(rows as u32)
        .arg_u32(vectors as u32)
        .arg_u32(dim as u32)
        .arg_u32(rope as u32)
        .arg_u32(u32::from(inverse))
        .launch(stream)
}

/// FP8 under a power-of-two scale per 32 values, as BF16: `values` BF16
/// values of `src` into `dst`, which may be `src`.
pub fn fp8_quant_dequant(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    values: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        values.is_multiple_of(32),
        "dsv41_fp8_quant_dequant: {values} values"
    );
    let blocks = (values / 32) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(blocks)
        .launch(stream)
}

/// The scale an FP4 group is quantized under.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp4Scale {
    /// A power of two per 32 values: index keys.
    PowerOfTwoPer32,
    /// An E4M3 value per 16 values: compressed keys.
    E4m3Per16,
}

/// FP4 under `scale`, back as BF16, over `values` BF16 values of `x` in place.
pub fn fp4_quant_dequant(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    values: usize,
    scale: Fp4Scale,
    stream: u64,
) -> Result<()> {
    let width = match scale {
        Fp4Scale::PowerOfTwoPer32 => 32,
        Fp4Scale::E4m3Per16 => 16,
    };
    ensure!(
        values.is_multiple_of(width),
        "dsv41_fp4_quant_dequant: {values} values"
    );
    let groups = (values / width) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(groups, THREADS), 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(x)
        .arg_u32(groups)
        .arg_u32(u32::from(scale == Fp4Scale::E4m3Per16))
        .launch(stream)
}

#[cfg(test)]
#[path = "ops_test_gpu.rs"]
mod test_gpu;

#[cfg(test)]
#[path = "proj_test_gpu.rs"]
mod proj_test_gpu;

#[cfg(test)]
#[path = "front_test_gpu.rs"]
mod front_test_gpu;
