// SPDX-License-Identifier: AGPL-3.0-only
// Ported from Enntity/atlas 5eeeb1a, dc896cc (AGPL-3.0-only, sparkqwen queued-first-chunk series).

//! `ATLAS_PREFILL_QUEUE_FIRST_CHUNK=1` (default off): a new prompt admitted
//! behind a prefill in flight waits in `prefilling` for its first chunk.
//!
//! Without it every new prompt runs its first chunk at admission, ahead of
//! the remaining chunks of the prompts already prefilling. Four ~18K-token
//! prompts arriving a second apart at 16K-token chunks (qwen4_exp TP2,
//! 2026-10-16) ran all four first chunks before any 1.7K-token tail, so
//! their first tokens came out together, 16-18.5 s after the first arrival,
//! where a FIFO prefill gives 5.6/10.2/14.7/19.3 s. Prompts that fit one
//! chunk finish at admission and were never affected.
//!
//! With it, such a prompt is set up at admission (grammar, sequence, its
//! worker's MTP fence, vision state and shared-prefix plant) and joins the
//! queue at its `queue_order` position with no chunk run and no prefill
//! command sent; the continue phase runs the chunk when the prompt reaches
//! the head (`phase_continue_prefills::queued_first`), sending the worker
//! its command then. The worker runs commands in the order the head sends
//! them, and the head now sends each chunk command right before it runs
//! that chunk, so both ranks run the chunks in one order. With
//! `ATLAS_PREFILL_SRPT=1` a short tail then goes ahead of a new long prompt,
//! which it could not while the new prompt's first chunk ran at admission.
//!
//! Only on multi-rank worlds (`Model::is_ep`), where every prefill chunk is
//! the queue head's: one GPU advances all prefilling prompts together in its
//! batched prefill steps, and co-dispatch, VARLEN and the always-mixed fuse
//! (none of them on multi-rank worlds) already defer the chunk there. Those
//! deferrals and the multi-prompt prefill of short prompts keep precedence.
//! Image prompts keep the inline chunk, which encodes their pixels, as do
//! beam searches, which run whole at admission. A prompt admitted with
//! nothing in flight runs its first chunk at admission as before, and so
//! does one that fits a single chunk: it finishes there, while queued it
//! would wait behind every remaining chunk of a long prompt in flight (tens
//! of seconds behind a ~200K-token prefill with FIFO order).

use super::FirstChunk;

/// What a new request does with its first chunk. `defer`: a batched step
/// (co-dispatch, VARLEN, always-mixed fuse, multi-prompt prefill) takes it;
/// `queue_lever`: `ATLAS_PREFILL_QUEUE_FIRST_CHUNK`; `in_flight`: a prefill is
/// in `prefilling` ahead of it, admitted this tick or before; `one_chunk`:
/// the whole prompt fits its first chunk.
pub(super) fn first_chunk(
    defer: bool,
    queue_lever: bool,
    multi_rank: bool,
    in_flight: bool,
    vision: bool,
    beams: bool,
    one_chunk: bool,
) -> FirstChunk {
    if defer {
        FirstChunk::Defer
    } else if queue_lever && multi_rank && in_flight && !vision && !beams && !one_chunk {
        FirstChunk::Queue
    } else {
        FirstChunk::Inline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (defer, lever, multi-rank, in flight, vision, beams), a prompt longer
    /// than one chunk.
    fn decide(flags: (bool, bool, bool, bool, bool, bool)) -> FirstChunk {
        let (defer, lever, ep, in_flight, vision, beams) = flags;
        first_chunk(defer, lever, ep, in_flight, vision, beams, false)
    }

    #[test]
    fn a_prompt_that_fits_one_chunk_runs_at_admission() {
        assert_eq!(
            first_chunk(false, true, true, true, false, false, true),
            FirstChunk::Inline
        );
        // A deferral still takes it.
        assert_eq!(
            first_chunk(true, true, true, true, false, false, true),
            FirstChunk::Defer
        );
    }

    #[test]
    fn a_prompt_behind_a_prefill_in_flight_queues_on_a_multi_rank_world() {
        assert_eq!(
            decide((false, true, true, true, false, false)),
            FirstChunk::Queue
        );
    }

    #[test]
    fn without_the_lever_nothing_queues() {
        for ep in [false, true] {
            for in_flight in [false, true] {
                assert_eq!(
                    decide((false, false, ep, in_flight, false, false)),
                    FirstChunk::Inline
                );
                assert_eq!(
                    decide((true, false, ep, in_flight, false, false)),
                    FirstChunk::Defer
                );
            }
        }
    }

    /// Co-dispatch with nothing in flight is untouched, and every deferral
    /// keeps precedence over the queue.
    #[test]
    fn a_deferral_wins_and_nothing_in_flight_runs_inline() {
        assert_eq!(
            decide((true, true, true, true, false, false)),
            FirstChunk::Defer
        );
        assert_eq!(
            decide((true, true, false, false, false, false)),
            FirstChunk::Defer
        );
        assert_eq!(
            decide((false, true, true, false, false, false)),
            FirstChunk::Inline
        );
    }

    #[test]
    fn one_gpu_image_prompts_and_beams_keep_the_inline_chunk() {
        assert_eq!(
            decide((false, true, false, true, false, false)),
            FirstChunk::Inline
        );
        assert_eq!(
            decide((false, true, true, true, true, false)),
            FirstChunk::Inline
        );
        assert_eq!(
            decide((false, true, true, true, false, true)),
            FirstChunk::Inline
        );
    }
}
