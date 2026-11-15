// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): whole blocks, token by token, on the release's
//! weights. Each token starts from DeepSeek's recorded stream into the block
//! and must leave the stream DeepSeek's block left; the caches the block
//! reads are the ones it built itself from the tokens before.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::block -- --ignored --nocapture --test-threads=1

use std::sync::{Arc, Mutex};

use anyhow::Result;

use super::*;
use crate::layers::deepseek_v41::cache_recipe::rope_freqs;
use crate::layers::deepseek_v41::hc::HcShape;
use crate::layers::deepseek_v41::test_support::{
    Recording, backend, bf16s, closeness, download, layer_store, le_bytes, rows_of, upload,
};

const SHAPE: Shape = Shape {
    hidden: 5120,
    heads: 64,
    q_rank: 1280,
    o_groups: 8,
    o_rank: 1024,
    rope: 64,
    inter: 2304,
    experts: 384,
    top_k: 6,
    route_scale: 1.5,
    swiglu_limit: 10.0,
    norm_eps: 1e-20,
    index_topk: 512,
    hc: HcShape {
        hidden: 5120,
        copies: 4,
        sinkhorn_iters: 20,
        norm_eps: 1e-20,
        hc_eps: 1e-6,
    },
};

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn blocks_replay_the_reference_token_by_token() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let shared = Shared {
        shape: SHAPE,
        kernels: BlockKernels::load(gpu)?,
        scratch: Arc::new(Scratch::alloc(gpu, &SHAPE)?),
        handoff: Arc::new(Mutex::new(Handoff::default())),
    };
    let (copies, hidden) = (SHAPE.hc.copies, SHAPE.hidden);
    // 200 tokens: past the 128 the window holds, so the ring has wrapped.
    let tokens = 200;
    let positions: Vec<usize> = (0..tokens).collect();
    let streams = gpu.alloc(copies * hidden * 2)?;
    let pre = gpu.alloc(copies * 4)?;
    for (layer, ratio) in [(0usize, 0usize), (1, 0), (2, 2)] {
        let store = layer_store(gpu, layer)?;
        let ladder = if ratio == 0 {
            rope_freqs(SHAPE.rope, 0, 10_000.0, 16.0, 32.0, 1.0)
        } else {
            rope_freqs(SHAPE.rope, 65536, 160_000.0, 16.0, 32.0, 1.0)
        };
        let block = Block {
            index: layer,
            ratio,
            freqs: upload(gpu, &le_bytes(&ladder, f32::to_le_bytes))?,
            weights: BlockWeights::load(&store, gpu, layer, ratio, 0..SHAPE.experts)?,
            shared: shared.clone(),
        };
        let mut state = SeqState::alloc(gpu, ratio, 2048)?;
        let l = format!("L{layer}.");
        let inputs = rows_of(
            &rec.prefill(&format!("{l}block_in"))?.1,
            copies * hidden,
            &positions,
        );
        let carried = rows_of(
            &rec.prefill(&format!("{l}pre_mix_in"))?.1,
            copies * 2,
            &positions,
        );
        let want = bf16s(&rows_of(
            &rec.prefill(&format!("{l}block_out"))?.1,
            copies * hidden,
            &positions,
        ));
        let want_pre = rows_of(
            &rec.prefill(&format!("{l}pre_mix_out"))?.1,
            copies * 2,
            &positions,
        );
        let attention = rows_of(&rec.prefill(&format!("{l}attn_out"))?.1, hidden, &positions);
        let scratch = &block.shared.scratch;
        let (mut got, mut attended, mut pre_gap) =
            (Vec::with_capacity(want.len()), Vec::new(), 0.0f32);
        // DeepSeek's compressed keys for this prompt, and how many values of
        // the block's own keys differ from them.
        let theirs = match ratio {
            0 => Vec::new(),
            _ => {
                rec.caches(&format!("after_prefill.{l}compress_kv_cache"))?
                    .1
            }
        };
        let mut key_values_differing = 0usize;
        for position in 0..tokens {
            let stream_in =
                &inputs[position * copies * hidden * 2..(position + 1) * copies * hidden * 2];
            gpu.copy_h2d(stream_in, streams)?;
            gpu.copy_h2d(
                &carried[position * copies * 4..(position + 1) * copies * 4],
                pre,
            )?;
            block.attend(gpu, 0, streams, pre, &mut state, position, stream)?;
            attended.extend(download(gpu, scratch.sublayer_out, hidden)?);
            // A compressed key is FP4: one rounding step in it is a large
            // step, and every later token reads it. So once a group's key has
            // been compared, the tokens after it read DeepSeek's.
            if let (Some(held), Some(group)) = (
                &state.compressed,
                crate::layers::deepseek_v41::cache_recipe::completed_group(position, ratio.max(1)),
            ) {
                let row = &theirs[group * 512 * 2..(group + 1) * 512 * 2];
                let mine = download(gpu, held.keys.offset(group * 512 * 2), 512)?;
                key_values_differing +=
                    mine.iter().zip(bf16s(row)).filter(|(a, b)| *a != b).count();
                gpu.copy_h2d(row, held.keys.offset(group * 512 * 2))?;
            }
            // The rest of the block from DeepSeek's own attention output, so
            // the attention's rounding is not read as the FFN's error.
            gpu.copy_h2d(
                &attention[position * hidden * 2..(position + 1) * hidden * 2],
                scratch.sublayer_out,
            )?;
            block.finish(gpu, streams, pre, stream)?;
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
            "layer {layer}: {tokens} tokens; attention {attn_distance:.2e} from the reference ({:.1}% of values equal); the block after it {:.3}% of stream values equal, {distance:.2e} away; next pre within {pre_gap:.1e}",
            100.0 * attn_same,
            100.0 * same
        );
        if ratio > 0 {
            let values = tokens / ratio * 512;
            println!(
                "layer {layer}: {key_values_differing} of {values} compressed-key values differ from the reference's"
            );
            assert!(
                key_values_differing * 1000 < values,
                "layer {layer}: {key_values_differing} key values differ"
            );
        }
        // The attention output is a projection of FP8-quantized values, so the
        // few values of the attention that differ by a rounding step move many
        // outputs by one BF16 step: about 1e-3, and one in ten values.
        assert!(
            attn_distance < 3e-3 && attn_same > 0.8,
            "layer {layer}: attention {attn_distance:e} from the reference, {attn_same} equal"
        );
        assert!(
            same > 0.98 && distance < 5e-4 && pre_gap < 1e-3,
            "layer {layer}: {same} equal, pre {pre_gap:e}"
        );
        state.free(gpu)?;
    }
    Ok(())
}
