// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): every recorded block, token by token, on the
//! release's weights. Each token starts from DeepSeek's recorded stream into
//! the block and must leave the stream DeepSeek's block left. The caches the
//! block reads are the ones it built itself from the tokens before; what a
//! layer below hands it (keys, the positions chosen) is DeepSeek's.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::block -- --ignored --nocapture --test-threads=1

use std::sync::{Arc, Mutex};

use anyhow::Result;

use super::*;
use crate::layers::deepseek_v41::cache_recipe::{completed_group, visible_groups};
use crate::layers::deepseek_v41::index::INDEX_DIM;
use crate::layers::deepseek_v41::ops::HEAD_DIM;
use crate::layers::deepseek_v41::test_support::{
    Recording, SHAPE, backend, bf16s, block, chosen, closeness, download, free_store, hand_choice,
    hand_keys, role, rows_of, source_keys,
};

/// 200 tokens: past the 128 the window holds, so the ring has wrapped.
const TOKENS: usize = 200;

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn blocks_replay_the_reference_token_by_token() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let shared = Shared {
        shape: SHAPE,
        kernels: BlockKernels::load(gpu)?,
        scratch: Arc::new(Scratch::alloc(gpu, &SHAPE, 1024)?),
        handoff: Arc::new(Mutex::new(Handoff::default())),
    };
    let (copies, hidden) = (SHAPE.hc.copies, SHAPE.hidden);
    let positions: Vec<usize> = (0..TOKENS).collect();
    let streams = gpu.alloc(copies * hidden * 2)?;
    let pre = gpu.alloc(copies * 4)?;
    // The worst layer of each measure, to state what the sweep found.
    let (mut worst_attn, mut worst_same, mut worst_distance, mut worst_pre) =
        (0.0f64, 1.0f64, 0.0f64, 0.0f32);
    for layer in 0..rec.layers() {
        let role = role(layer);
        let ratio = role.ratio;
        let (block, store) = block(gpu, layer, true, shared.clone())?;
        let mut state = SeqState::alloc(gpu, role.compresses(), 2048)?;
        let l = format!("L{layer}.");
        let recorded = |name: &str, width: usize| -> Result<Vec<u8>> {
            Ok(rows_of(
                &rec.prefill(&format!("{l}{name}"))?.1,
                width,
                &positions,
            ))
        };
        let inputs = recorded("block_in", copies * hidden)?;
        let carried = recorded("pre_mix_in", copies * 2)?;
        let want = bf16s(&recorded("block_out", copies * hidden)?);
        let want_pre = recorded("pre_mix_out", copies * 2)?;
        let attention = recorded("attn_out", hidden)?;
        // What DeepSeek's layers below hand this one, and its own keys to
        // compare with.
        let (keys, index_keys, rows, row) = match ratio {
            0 => (Vec::new(), Vec::new(), Vec::new(), 0),
            _ => {
                let (keys, index_keys) = source_keys(&rec, &role)?;
                let (shape, rows) = rec.prefill(&format!("{l}sa_idxs"))?;
                (keys, index_keys, rows, shape[2] * 4)
            }
        };
        hand_keys(gpu, &block, &role, &keys, &index_keys)?;
        let scratch = &block.shared.scratch;
        let (mut got, mut attended, mut pre_gap) =
            (Vec::with_capacity(want.len()), Vec::new(), 0.0f32);
        let (mut key_differ, mut index_key_differ, mut groups) = (0usize, 0usize, 0usize);
        for position in 0..TOKENS {
            let stream_in =
                &inputs[position * copies * hidden * 2..(position + 1) * copies * hidden * 2];
            gpu.copy_h2d(stream_in, streams)?;
            gpu.copy_h2d(
                &carried[position * copies * 4..(position + 1) * copies * 4],
                pre,
            )?;
            if ratio > 0 {
                let theirs = chosen(
                    &rows[position * row..(position + 1) * row],
                    rec.prefill as i32,
                );
                hand_choice(&block, &role, &theirs, visible_groups(position, ratio));
            }
            block.attend(gpu, 0, streams, pre, &mut state, position, stream)?;
            attended.extend(download(gpu, scratch.sublayer_out, hidden)?);
            // A key is FP4: one rounding step in it is a large step, and
            // every later token reads it. So once a group's keys have been
            // compared, the tokens after it read DeepSeek's.
            if let (Some(held), Some(group)) =
                (&state.compressed, completed_group(position, ratio.max(1)))
            {
                for (mine, theirs, width, differ) in [
                    (held.keys, &keys, HEAD_DIM, &mut key_differ),
                    (
                        held.index_keys,
                        &index_keys,
                        INDEX_DIM,
                        &mut index_key_differ,
                    ),
                ] {
                    let theirs = &theirs[group * width * 2..(group + 1) * width * 2];
                    let slot = mine.offset(group * width * 2);
                    let mine = download(gpu, slot, width)?;
                    *differ += mine
                        .iter()
                        .zip(bf16s(theirs))
                        .filter(|(a, b)| *a != b)
                        .count();
                    gpu.copy_h2d(theirs, slot)?;
                }
                groups += 1;
            }
            // The rest of the block from DeepSeek's own attention output, so
            // the attention's rounding is not read as the FFN's error.
            gpu.copy_h2d(
                &attention[position * hidden * 2..(position + 1) * hidden * 2],
                scratch.sublayer_out,
            )?;
            block.finish(gpu, streams, pre, None, stream)?;
            got.extend(download(gpu, streams, copies * hidden)?);
            let mut out = vec![0u8; copies * 4];
            gpu.copy_d2h(pre, &mut out)?;
            let recorded = want_pre[position * copies * 4..].chunks_exact(4);
            for (a, b) in out.chunks_exact(4).zip(recorded) {
                let a = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                let b = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                pre_gap = pre_gap.max((a - b).abs() / b.abs().max(1e-6));
            }
        }
        let (attn_same, attn_distance) = closeness(&attended, &bf16s(&attention));
        let (same, distance) = closeness(&got, &want);
        println!(
            "layer {layer}: attention {attn_distance:.2e} from the reference ({:.1}% of values equal); the block after it {:.3}% of stream values equal, {distance:.2e} away; next pre within {pre_gap:.1e}",
            100.0 * attn_same,
            100.0 * same
        );
        if groups > 0 {
            println!(
                "layer {layer}: {key_differ} of {} compressed-key values and {index_key_differ} of {} index-key values differ from the reference's",
                groups * HEAD_DIM,
                groups * INDEX_DIM
            );
            assert!(
                key_differ * 1000 < groups * HEAD_DIM
                    && index_key_differ * 1000 < groups * INDEX_DIM,
                "layer {layer}: {key_differ} key values and {index_key_differ} index-key values differ"
            );
        }
        worst_attn = worst_attn.max(attn_distance);
        worst_same = worst_same.min(same);
        worst_distance = worst_distance.max(distance);
        worst_pre = worst_pre.max(pre_gap);
        state.free(gpu)?;
        free_store(gpu, &store)?;
        gpu.free(block.freqs)?;
    }
    println!(
        "{} layers x {TOKENS} tokens; worst layer: attention {worst_attn:.2e} from the reference, the block after it {:.3}% of stream values equal and {worst_distance:.2e} away, next pre within {worst_pre:.1e}",
        rec.layers(),
        100.0 * worst_same
    );
    // The attention output is a projection of FP8-quantized values, so the
    // few values of the attention that differ by a rounding step move many
    // outputs by one BF16 step. Measured: 1.1e-3 to 7.5e-3 over the 40
    // layers, the worst at layers 20, 21 and 35.
    assert!(
        worst_attn < 1e-2,
        "attention {worst_attn:e} from the reference"
    );
    assert!(
        worst_same > 0.97 && worst_distance < 1e-3 && worst_pre < 2e-3,
        "{worst_same} equal, {worst_distance:e} away, pre {worst_pre:e}"
    );
    Ok(())
}
