// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the DeepSeek-V4.1 attention kernels (`dsv41_attn.cu`,
//! `dsv41_cache.cu`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::profile;
use crate::layers::ops::{MxAct, mx_lt_inner_pad};

/// Values in one key, and in one head's query.
pub const HEAD_DIM: usize = 512;
/// The most keys one query attends to: the window and the selected
/// compressed positions.
pub const MAX_KEYS: usize = 640;

const THREADS: u32 = 256;
/// `dsv41_sparse_attn` runs a head on one block this wide: a thread a key.
const ATTN_THREADS: u32 = 640;
/// The queries from which the tensor-core form runs a block a (query,
/// heads); fewer are spread over more blocks (`dsv41_attn_tc.cu`), with the
/// same bytes: the output over `FEW_ACROSS` blocks of `FEW_THREADS` threads
/// and `FEW_SHARED` bytes of shared memory each.
const SPREAD_BELOW: usize = 64;
const FEW_ACROSS: u32 = 16;
/// How many (query, 16 heads) of a few queries a block of the output takes
/// in turn, and how many queries a block of the schedule. A block a query
/// is more blocks than the GPU's 48 multiprocessors hold at once from two
/// queries on (one of the output's blocks each, for their shared memory),
/// and the kernels queued behind, which read their weights ahead while they
/// wait, then start only as these end; every query through one set of
/// blocks leaves most of the GPU idle at six. Measured on a verify step of
/// two rows and of six: a block a query 33.8 and 47.4 ms, one set of blocks
/// 31.6 and 50.2.
const FEW_IN_TURN: u32 = 3;
const SCHEDULED_IN_TURN: u32 = 2;
const FEW_THREADS: u32 = 128;
const FEW_SHARED: u32 = ((MAX_KEYS * HEAD_DIM / 16 + ATTN_GROUP * MAX_KEYS) * 2) as u32;
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
    /// The first of those on tensor cores (`dsv41_attn_tc.cu`).
    attn_logits_tc: KernelHandle,
    /// The third on tensor cores for a few queries, with the batch's bytes.
    attn_output_few: KernelHandle,
    /// A batch's logits, schedule and output in one launch.
    attn_batch: KernelHandle,
    pub rotate: KernelHandle,
    pub fp8_quant_dequant: KernelHandle,
    /// A batch's window out of the ring, and a verify's refused rows back.
    pub window_unroll: KernelHandle,
    pub window_restore: KernelHandle,
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
            attn_logits_tc: gpu.kernel("dsv41_attn_tc", "dsv41_attn_logits_tc")?,
            attn_output_few: gpu.kernel("dsv41_attn_tc", "dsv41_attn_output_tc_few")?,
            attn_batch: gpu.kernel("dsv41_attn_tc", "dsv41_attn_batch_tc")?,
            rotate: gpu.kernel("dsv41_cache", "dsv41_rotate")?,
            fp8_quant_dequant: gpu.kernel("dsv41_cache", "dsv41_fp8_quant_dequant")?,
            window_unroll: gpu.kernel("dsv41_cache", "dsv41_window_unroll")?,
            window_restore: gpu.kernel("dsv41_cache", "dsv41_window_restore")?,
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
/// (`[rows, q_rank]` BF16) = the FP8 form of the norm of `q_low` under
/// `q_weight`, which is what the heads' projection reads, and `slot`
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
        q_rank.is_multiple_of(64) && q_rank <= 2048 && turn.rope <= HEAD_DIM,
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

/// Whether [`sparse_attn`] takes `queries` queries in one launch
/// (`dsv41_attn_batch_tc`), which is the launch that can leave its output
/// in codes ([`Coded`]).
pub fn in_one_launch(queries: usize, batch: bool) -> bool {
    batch && queries >= SPREAD_BELOW
}

/// Where a batch's attention leaves its output for the projection after
/// it, in place of BF16 values: by groups of `group_heads` heads, a
/// group's E4M3 codes (`[queries, group_heads * 512]`, a group
/// `code_stride` bytes after the last) and its scales tiled as the GEMM
/// reads them (a group `scale_stride` bytes after the last). The codes
/// `dsv41_mx_quant` makes of the values.
#[derive(Clone, Copy)]
pub struct Coded {
    pub codes: DevicePtr,
    pub code_stride: usize,
    pub scales: DevicePtr,
    pub scale_stride: usize,
    pub group_heads: usize,
}

impl Coded {
    /// Group `group`'s output as a GEMM operand.
    pub fn operand(&self, group: usize) -> MxAct {
        MxAct {
            q: self.codes.offset(group * self.code_stride),
            sf: self.scales.offset(group * self.scale_stride),
        }
    }
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
///
/// `batch` takes the first and third launches on tensor cores: the same
/// products with their FP32 sums in another order. A batch runs a block a
/// query, the three launches' work in one that never leaves the GPU's
/// cache; a few queries are spread over more blocks, with the batch's bytes
/// (`dsv41_attn_tc.cu`), so a query's do not depend on how many share its
/// launch. `records` is the few queries' and the row-at-a-time order's.
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
    batch: bool,
    coded: Option<Coded>,
    stream: u64,
) -> Result<()> {
    ensure!(
        coded.is_none() || in_one_launch(queries, batch),
        "dsv41_attn: {queries} queries do not leave their output in codes"
    );
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
    // The turn's arguments, which end every launch here but one.
    fn turned<'a>(launch: KernelLaunch<'a>, turn: &Turn) -> KernelLaunch<'a> {
        launch
            .arg_ptr(turn.positions)
            .arg_ptr(turn.freqs)
            .arg_u32(turn.rope as u32)
    }
    // A block a (query, heads) fills the GPU with a batch; a few queries
    // are spread over more blocks.
    let few = batch && queries < SPREAD_BELOW;
    if batch && !few {
        // A batch in one launch and one pass over its keys, with the three
        // launches' bytes (`dsv41_attn_batch_tc`).
        let began = profile::begin(gpu, stream)?;
        turned(
            KernelLaunch::new(gpu, kernels.attn_batch)
                .grid([1, groups, 1])
                .block(block)
                .arg_ptr(q)
                .arg_ptr(keys_a)
                .arg_ptr(keys_b)
                .arg_ptr(idx)
                .arg_ptr(sink)
                .arg_ptr(out)
                .arg_u32(heads as u32)
                .arg_u32(keys as u32)
                .arg_u32(split as u32)
                .arg_f32((HEAD_DIM as f32).powf(-0.5)),
            &turn,
        )
        .arg_ptr(coded.map_or(DevicePtr::NULL, |c| c.codes))
        .arg_u64(coded.map_or(0, |c| c.code_stride) as u64)
        .arg_ptr(coded.map_or(DevicePtr::NULL, |c| c.scales))
        .arg_u64(coded.map_or(0, |c| c.scale_stride) as u64)
        .arg_u32(coded.map_or(1, |c| c.group_heads) as u32)
        .arg_u32(coded.map_or(0, |c| mx_lt_inner_pad(c.group_heads * HEAD_DIM)) as u32)
        .launch(stream)?;
        return profile::end(gpu, stream, "attention", began);
    }
    // What is left is a few queries on tensor cores, or the row-at-a-time
    // order.
    let (logits, across) = match few {
        true => (kernels.attn_logits_tc, steps.div_ceil(2)),
        false => (kernels.attn_logits, steps * (64 / ATTN_GROUP) as u32),
    };
    let output = match few {
        // A few queries a block, one after another: see the kernel.
        true => KernelLaunch::new(gpu, kernels.attn_output_few)
            .grid([FEW_ACROSS, groups.div_ceil(FEW_IN_TURN), 1])
            .block([FEW_THREADS, 1, 1])
            .shared_mem(FEW_SHARED),
        false => KernelLaunch::new(gpu, kernels.attn_output)
            .grid([(HEAD_DIM / ATTN_GROUP) as u32, groups, 1])
            .block(block),
    };
    let began = profile::begin(gpu, stream)?;
    turned(
        KernelLaunch::new(gpu, logits)
            .grid([across, groups, 1])
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
        &turn,
    )
    .launch(stream)?;
    profile::end(gpu, stream, "attention: logits", began)?;
    let began = profile::begin(gpu, stream)?;
    KernelLaunch::new(gpu, kernels.attn_schedule)
        .grid([
            (queries as u32).div_ceil(if few { SCHEDULED_IN_TURN } else { 1 }),
            heads as u32,
            1,
        ])
        .block([ATTN_THREADS, 1, 1])
        .arg_ptr(records)
        .arg_ptr(sink)
        .arg_u32(heads as u32)
        .arg_u32(keys as u32)
        .arg_u32(queries as u32)
        .launch(stream)?;
    profile::end(gpu, stream, "attention: schedule", began)?;
    let began = profile::begin(gpu, stream)?;
    let output = turned(
        output
            .arg_ptr(keys_a)
            .arg_ptr(keys_b)
            .arg_ptr(idx)
            .arg_ptr(records)
            .arg_ptr(out)
            .arg_u32(heads as u32)
            .arg_u32(keys as u32)
            .arg_u32(split as u32),
        &turn,
    );
    match few {
        true => output.arg_u32(groups),
        false => output,
    }
    .launch(stream)?;
    profile::end(gpu, stream, "attention: output", began)
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
    // A thread two values.
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks * 16, THREADS), 1, 1])
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
