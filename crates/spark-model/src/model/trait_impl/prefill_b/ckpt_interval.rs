// SPDX-License-Identifier: AGPL-3.0-only

//! When a non-last prefill chunk saves an interval SSM checkpoint.
//!
//! A checkpoint fires at the end of any chunk that CROSSES an interval
//! boundary (`--ssm-checkpoint-interval` blocks), not only at one that ends
//! exactly on it. Chunk sizes vary: an idle server's first chunk is the whole
//! arena, the prefill budget plus one row per decode slot (16,384 + 64 on
//! Qwen3.8-27B with 64 sequences), so every later chunk end sat 4 blocks past
//! a 1,024-block multiple and a cold 180K prefill saved no interval checkpoint
//! at all (2026-11-09, one DGX Spark). A later match behind the conversation's
//! tail then had no SSM anchor and recomputed the whole prompt. The busy
//! server's 16,384-token chunks saved one every chunk; now both do.

/// True when the chunk `[chunk_start, end_token)` crosses a multiple of
/// `interval_blocks` blocks of `block_size` tokens. Equal to the old
/// "ends on a multiple" rule whenever chunks are interval-aligned.
pub(super) fn crosses_interval(
    chunk_start: usize,
    end_token: usize,
    block_size: usize,
    interval_blocks: usize,
) -> bool {
    if interval_blocks == 0 || block_size == 0 {
        return false;
    }
    let span = interval_blocks * block_size;
    end_token / span > chunk_start / span
}

#[cfg(test)]
mod tests {
    use super::crosses_interval;

    const BS: usize = 16;
    const EVERY: usize = 1024; // blocks: 16,384 tokens

    /// The production shape: an idle server prefills 180,033 tokens with a
    /// 16,448-token first chunk, then 16,384-token chunks (log: 16448, 32832,
    /// 49216, ...). Every non-last chunk end saves a checkpoint.
    #[test]
    fn an_idle_first_chunk_still_checkpoints_every_chunk() {
        let mut ends = vec![16_448];
        while ends[ends.len() - 1] + 16_384 < 180_033 {
            ends.push(ends[ends.len() - 1] + 16_384);
        }
        let starts = std::iter::once(0).chain(ends.iter().copied());
        let fired = starts
            .zip(&ends)
            .filter(|&(s, &e)| crosses_interval(s, e, BS, EVERY))
            .count();
        assert_eq!((ends.len(), fired), (10, 10));
        // The old rule (end block a multiple of the interval) fired on none.
        assert!(ends.iter().all(|&e| !(e / BS).is_multiple_of(EVERY)));
    }

    /// Aligned chunks keep the old cadence exactly.
    #[test]
    fn aligned_chunks_are_unchanged() {
        for k in 1..12 {
            let end = k * 16_384;
            assert!(crosses_interval(end - 16_384, end, BS, EVERY));
        }
        // Smaller aligned chunks fire only on the multiple.
        assert!(!crosses_interval(0, 8_192, BS, EVERY));
        assert!(crosses_interval(8_192, 16_384, BS, EVERY));
    }

    /// A short warm replay that stays inside one interval saves nothing.
    #[test]
    fn a_chunk_inside_one_interval_saves_nothing() {
        assert!(!crosses_interval(181_968, 181_995, BS, EVERY));
    }

    /// Interval 0 turns interval checkpoints off.
    #[test]
    fn zero_interval_is_off() {
        assert!(!crosses_interval(0, 1 << 20, BS, 0));
    }
}
