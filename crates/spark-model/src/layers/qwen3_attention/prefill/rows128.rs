// SPDX-License-Identifier: AGPL-3.0-only

//! Batched paged prefill attention on 128-row tiles (`prefill_paged_rows128.cu`,
//! lever `ATLAS_ATTN_ROWS128=1`): the BF16-KV, HDIM-256 replacement of
//! `inferspark_prefill_paged_batched_64`, same arguments and bit-identical
//! output, FlashAttention-2 layout (every warp owns its rows end to end, K
//! and V tiles shared by 128 rows).
//!
//! Score passes take the `_gqa` entries instead, lever or not ([`Tiles::Gqa`]):
//! each tile is one KV head's query heads over a stream's rows, so the short
//! streams of a score pass fill tiles, and a long KV range (a knowledge pack)
//! is split across a 2-CTA cluster.
//!
//! The handles are resolved on first use, the per-head ones only with the
//! lever set (only the 27B dense targets ship the module), so other boot
//! audits are unchanged. They are process-wide: the lever is read once and a
//! process drives one GPU backend.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Query rows a CTA (`rows128::BR` in the kernel).
const ROWS: u32 = 128;
/// The kernel's `stream_splits`: a GQA stream with more keys than this, and
/// at least `SPLIT_KEYS_PER_ROW` keys a row, splits its tiles' KV ranges.
const SPLIT_MIN_KEYS: i32 = 4096;
const SPLIT_KEYS_PER_ROW: i32 = 4;
/// CTAs a split entry's cluster.
const SPLIT_CTAS: u32 = 2;
const MODULE: &str = "prefill_paged_rows128";

/// The tile layout of a launch.
#[derive(Clone, Copy)]
pub(super) enum Tiles {
    /// 128 rows of one query head (the lever's replacement of `_64`).
    PerHead,
    /// 128 rows of one KV head's query heads (score passes); the split entry
    /// when any stream splits (it computes the others as the unsplit entry).
    Gqa { split: bool },
}

impl Tiles {
    /// A VARLEN pass's GQA layout, from its streams' KV extents and
    /// `cu_seqlens` (host copies). Each stream's entry is fixed by the stream
    /// alone, so a merged pass scores as each request alone.
    pub(super) fn gqa(kv_lens: &[i32], cu_seqlens: &[i32]) -> Self {
        Tiles::Gqa {
            split: kv_lens
                .iter()
                .zip(cu_seqlens.windows(2))
                .any(|(&k, c)| k > SPLIT_MIN_KEYS && k >= SPLIT_KEYS_PER_ROW * (c[1] - c[0])),
        }
    }
}

/// `[cp.async, TMA]` kernels when the lever is set and the target ships
/// them (handle 0 where absent).
fn kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 2] {
    static K: OnceLock<[KernelHandle; 2]> = OnceLock::new();
    *K.get_or_init(|| {
        let on = std::env::var("ATLAS_ATTN_ROWS128").as_deref() == Ok("1");
        [
            "prefill_paged_bf16_rows128",
            "prefill_paged_bf16_rows128_tma",
        ]
        .map(|f| crate::layers::try_kernel_gated(on, gpu, MODULE, f))
    })
}

/// `[[cp.async, TMA]; unsplit, split]` GQA entries where the target ships them.
fn gqa_kernels(gpu: &dyn GpuBackend) -> [[KernelHandle; 2]; 2] {
    static K: OnceLock<[[KernelHandle; 2]; 2]> = OnceLock::new();
    *K.get_or_init(|| {
        [
            [
                "prefill_paged_bf16_rows128_gqa",
                "prefill_paged_bf16_rows128_gqa_tma",
            ],
            [
                "prefill_paged_bf16_rows128_gqa_split2",
                "prefill_paged_bf16_rows128_gqa_split2_tma",
            ],
        ]
        .map(|p| p.map(|f| crate::layers::try_kernel(gpu, MODULE, f)))
    })
}

/// Whether `tiles` can launch: the per-head entries need the lever, and every
/// entry needs a target that ships it.
pub(super) fn available(gpu: &dyn GpuBackend, tiles: Tiles) -> bool {
    match tiles {
        Tiles::PerHead => kernels(gpu)[0].0 != 0,
        Tiles::Gqa { .. } => gqa_kernels(gpu).iter().all(|k| k[0].0 != 0),
    }
}

/// A layer's BF16 KV pools as the TMA kernel reads them: K and V base,
/// `blocks` pages of 16 rows, `block_stride` bytes a page.
pub(super) struct Pools {
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub blocks: usize,
    pub block_stride: usize,
}

/// `ops::prefill_attention_paged_batched_64`'s launch, on 128-row tiles laid
/// out as `tiles`.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch(
    gpu: &dyn GpuBackend,
    tiles: Tiles,
    pools: Pools,
    [q, output, block_table_ptrs]: [DevicePtr; 3],
    batch_size: u32,
    [cu_seqlens, kv_lens, state_end]: [DevicePtr; 3],
    [q_len, kv_len, q_offset]: [u32; 3],
    [
        num_q_heads,
        num_kv_heads,
        head_dim,
        cache_block_size,
        sliding_window,
    ]: [u32; 5],
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        head_dim == 256,
        "prefill_paged_rows128: head_dim {head_dim} != 256"
    );
    let [plain, tma] = match tiles {
        Tiles::PerHead => kernels(gpu),
        Tiles::Gqa { split } => gqa_kernels(gpu)[split as usize],
    };
    let grid = match tiles {
        Tiles::PerHead => [num_q_heads, div_ceil(q_len, ROWS), batch_size],
        Tiles::Gqa { split } => {
            let ctas = if split { SPLIT_CTAS } else { 1 };
            let rows = q_len * (num_q_heads / num_kv_heads);
            [num_kv_heads * ctas, div_ceil(rows, ROWS), batch_size]
        }
    };
    // The TMA kernel's boxes are 16-row pages of [num_kv_heads * 256] BF16 rows.
    let row = num_kv_heads as usize * 256;
    let maps = (tma.0 != 0 && cache_block_size == 16 && pools.block_stride == 16 * row * 2)
        .then(|| tensor_maps(&pools, row))
        .transpose()?
        .flatten();
    let kernel = if maps.is_some() { tma } else { plain };
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(pools.k)
        .arg_ptr(pools.v)
        .arg_ptr(output)
        .arg_ptr(block_table_ptrs)
        .arg_u32(batch_size)
        .arg_ptr(cu_seqlens)
        .arg_ptr(kv_lens)
        .arg_ptr(state_end)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d);
    if let Some([km, vm]) = &maps {
        l = l.arg_tensormap(km).arg_tensormap(vm);
    }
    l.launch(stream)
}

/// The K and V tensor maps (`[blocks * 16] x [row]` BF16, 16 x 64 boxes,
/// 128-byte swizzle); `None` on a build without CUDA tensor maps.
#[cfg(feature = "cuda")]
fn tensor_maps(p: &Pools, row: usize) -> Result<Option<[[u8; 128]; 2]>> {
    use spark_runtime::cuda_backend::tensormap::TensorMap;
    let rows = (p.blocks * 16) as u64;
    let map = |base| TensorMap::tiled_2d_bf16_swizzle128(base, rows, row as u64, row as u64, 16);
    Ok(Some([*map(p.k)?.bytes(), *map(p.v)?.bytes()]))
}
#[cfg(not(feature = "cuda"))]
fn tensor_maps(_: &Pools, _: usize) -> Result<Option<[[u8; 128]; 2]>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::Tiles;

    fn splits(kv_lens: &[i32], cu_seqlens: &[i32]) -> bool {
        matches!(Tiles::gqa(kv_lens, cu_seqlens), Tiles::Gqa { split: true })
    }

    #[test]
    fn long_pack_under_a_short_state_splits() {
        // 12K pack: head of 237 state rows, three 20-row branches.
        assert!(splits(
            &[12291, 12308, 12308, 12308],
            &[0, 237, 257, 277, 297]
        ));
        // 6K pack, branches alone.
        assert!(splits(&[6257], &[0, 20]));
    }

    #[test]
    fn short_packs_and_long_states_do_not_split() {
        // 1.2K and 3.2K packs: at most 4,096 keys.
        assert!(!splits(&[1419, 1436], &[0, 229, 249]));
        assert!(!splits(&[3512, 3529], &[0, 232, 252]));
        // A 2,450-token state without a pack: its rows fill the GPU.
        assert!(!splits(&[2450, 2467], &[0, 2448, 2468]));
        // 16K of state: fewer than 4 keys a row.
        assert!(!splits(&[16384], &[0, 16384]));
    }
}
