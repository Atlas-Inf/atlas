// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): layer 2's attention sublayer over the whole
//! recorded prompt, token by token, on the release's weights. From position
//! 1025 a query sees more compressed positions than it may read, and the
//! indexer chooses 512.
//!
//! Each token starts from DeepSeek's recorded attention input. The keys it
//! reads are the ones the block built from the tokens before, replaced by
//! DeepSeek's once compared.
//!
//!   ATLAS_DSV41_REC=<recording with --ops and the index replay> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::block_index -- --ignored --nocapture --test-threads=1

use std::sync::{Arc, Mutex};

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use crate::layers::deepseek_v41::block::{
    Block, BlockKernels, Candidates, Handoff, Scratch, SeqState, Shared,
};
use crate::layers::deepseek_v41::cache_recipe::{completed_group, visible_groups};
use crate::layers::deepseek_v41::index::INDEX_DIM;
use crate::layers::deepseek_v41::ops::HEAD_DIM;
use crate::layers::deepseek_v41::test_support::{
    Recording, SHAPE, backend, bf16s, closeness, compressed_ladder, download, layer_store,
    le_bytes, upload,
};
use crate::layers::deepseek_v41::weights::BlockWeights;

const LAYER: usize = 2;
const RATIO: usize = 2;
const WINDOW: i32 = 128;

/// The compressed positions of a recorded index row.
fn chosen(row: &[u8], offset: i32) -> Vec<i32> {
    row.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .filter(|&i| i >= 0)
        .map(|i| i - offset)
        .collect()
}

/// Queries that chose among more positions than they may read, and how
/// their choices compare with the reference's.
#[derive(Default)]
struct Choices {
    queries: usize,
    same: usize,
    least_shared: Option<usize>,
}

impl Choices {
    fn add(&mut self, mine: &[i32], theirs: &[i32]) {
        let shared = mine.iter().filter(|p| theirs.contains(p)).count();
        self.queries += 1;
        self.same += usize::from(mine == theirs);
        self.least_shared = Some(self.least_shared.map_or(shared, |least| least.min(shared)));
    }
}

fn equal(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x == y).count()
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops, with the index replay) and ATLAS_DSV41_CKPT"]
fn layer_2_chooses_the_references_compressed_positions() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let (hidden, topk, heads) = (SHAPE.hidden, SHAPE.index_topk, SHAPE.index_heads);
    let total = rec.prefill + rec.decode;
    let store = layer_store(gpu, LAYER)?;
    let block = Block {
        index: LAYER,
        ratio: RATIO,
        candidates: Candidates::None,
        freqs: upload(gpu, &le_bytes(&compressed_ladder(), f32::to_le_bytes))?,
        weights: BlockWeights::load(&store, gpu, LAYER, RATIO, true, 0..0)?,
        shared: Shared {
            shape: SHAPE,
            kernels: BlockKernels::load(gpu)?,
            scratch: Arc::new(Scratch::alloc(gpu, &SHAPE, total / RATIO)?),
            handoff: Arc::new(Mutex::new(Handoff::default())),
        },
    };
    let sc = block.shared.scratch.clone();
    let mut state = SeqState::alloc(gpu, RATIO, total)?;

    let (_, inputs) = rec.prefill("L2.attn_in")?;
    let (_, outputs) = rec.prefill("L2.attn_out")?;
    let (_, recorded) = rec.prefill("L2.indexer_topk")?;
    let (_, head_weights) = rec.prefill("L2.op.attn.indexer.weights_proj.out")?;
    let (_, queries) = rec.index("prefill.q")?;
    let (shape, scores) = rec.index("prefill.score")?;
    let (score_width, scores) = (shape[1], bf16s(&scores));
    let (_, keys) = rec.caches("after_decode.L2.compress_kv_cache")?;
    let (_, index_keys) = rec.caches("after_decode.L2.indexer_k_cache")?;

    let (mut prefill, mut even, mut odd) =
        (Choices::default(), Choices::default(), Choices::default());
    let (mut key_differ, mut index_key_differ, mut groups) = (0usize, 0usize, 0usize);
    // Of the indexer's own steps on prefill queries that choose: values equal
    // to the reference's, and values compared.
    let (mut q_same, mut weight_same, mut score_same, mut scored) = (0usize, 0usize, 0usize, 0);
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for position in 0..total {
        let step = position.checked_sub(rec.prefill);
        let (input, output, theirs) = match step {
            None => (
                inputs[position * hidden * 2..(position + 1) * hidden * 2].to_vec(),
                outputs[position * hidden * 2..(position + 1) * hidden * 2].to_vec(),
                chosen(
                    &recorded[position * topk * 4..(position + 1) * topk * 4],
                    rec.prefill as i32,
                ),
            ),
            Some(step) => (
                rec.step("L2.attn_in", step)?.1,
                rec.step("L2.attn_out", step)?.1,
                chosen(&rec.step("L2.indexer_topk", step)?.1, WINDOW),
            ),
        };
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&input, sc.normed)?;
        block.attention(gpu, 0, &mut state, position, stream)?;
        got.extend(download(gpu, sc.sublayer_out, hidden)?);
        want.extend(bf16s(&output));

        let visible = visible_groups(position, RATIO);
        let mine = block.shared.handoff.lock().expect("handoff").rows[0]
            .chosen
            .clone();
        if visible <= topk {
            assert_eq!(mine, theirs, "position {position}");
        } else {
            match step {
                None => prefill.add(&mine, &theirs),
                Some(_) if visible.is_multiple_of(2) => even.add(&mine, &theirs),
                Some(_) => odd.add(&mine, &theirs),
            }
        }
        if step.is_none() && visible > topk {
            let q = heads * INDEX_DIM;
            q_same += equal(
                &download(gpu, sc.index_q, q)?,
                &bf16s(&queries[position * q * 2..(position + 1) * q * 2]),
            );
            weight_same += equal(
                &download(gpu, sc.index_weights, heads)?,
                &bf16s(&head_weights[position * heads * 2..(position + 1) * heads * 2]),
            );
            score_same += equal(
                &download(gpu, sc.scores, visible)?,
                &scores[position * score_width..position * score_width + visible],
            );
            scored += visible;
        }

        // A key is FP4, so one rounding step in it is a large one, and every
        // later token reads it: once compared, it is replaced by DeepSeek's.
        if let (Some(held), Some(group)) = (&state.compressed, completed_group(position, RATIO)) {
            let theirs = &keys[group * HEAD_DIM * 2..(group + 1) * HEAD_DIM * 2];
            let slot = held.keys.offset(group * HEAD_DIM * 2);
            key_differ += HEAD_DIM - equal(&download(gpu, slot, HEAD_DIM)?, &bf16s(theirs));
            gpu.copy_h2d(theirs, slot)?;
            let theirs = &index_keys[group * INDEX_DIM * 2..(group + 1) * INDEX_DIM * 2];
            let slot = held.index_keys.offset(group * INDEX_DIM * 2);
            index_key_differ += INDEX_DIM - equal(&download(gpu, slot, INDEX_DIM)?, &bf16s(theirs));
            gpu.copy_h2d(theirs, slot)?;
            groups += 1;
        }
    }

    let share = |same: usize, of: usize| 100.0 * same as f64 / of.max(1) as f64;
    let split = rec.prefill * hidden;
    let (prefill_same, prefill_distance) = closeness(&got[..split], &want[..split]);
    let (decode_same, decode_distance) = closeness(&got[split..], &want[split..]);
    // Decode steps by whether the reference scored an even number of keys.
    let steps = |even: bool| -> (Vec<u16>, Vec<u16>) {
        let mut pair = (Vec::new(), Vec::new());
        for position in rec.prefill..total {
            if visible_groups(position, RATIO).is_multiple_of(2) == even {
                let at = position * hidden..(position + 1) * hidden;
                pair.0.extend_from_slice(&got[at.clone()]);
                pair.1.extend_from_slice(&want[at]);
            }
        }
        pair
    };
    let (even_got, even_want) = steps(true);
    let (odd_got, odd_want) = steps(false);
    let (even_distance, odd_distance) = (
        closeness(&even_got, &even_want).1,
        closeness(&odd_got, &odd_want).1,
    );
    println!(
        "layer {LAYER}: {total} tokens, {groups} groups; {key_differ} of {} compressed-key values and {index_key_differ} of {} index-key values differ from the reference's",
        groups * HEAD_DIM,
        groups * INDEX_DIM
    );
    println!(
        "indexer on the {} prefill queries that choose: query heads {:.3}% of values equal, head weights {:.2}%, scores {:.2}%",
        prefill.queries,
        share(q_same, prefill.queries * heads * INDEX_DIM),
        share(weight_same, prefill.queries * heads),
        share(score_same, scored)
    );
    for (name, c) in [
        ("prefill", &prefill),
        ("decode, even keys", &even),
        ("decode, odd keys", &odd),
    ] {
        println!(
            "{name}: {} of {} queries choose exactly the reference's {topk}; the least shared is {}",
            c.same,
            c.queries,
            c.least_shared.unwrap_or(topk)
        );
    }
    println!(
        "attention output: prefill {prefill_distance:.2e} from the reference, {:.1}% of values equal; decode {decode_distance:.2e}, {:.1}% ({even_distance:.2e} on even key counts, {odd_distance:.2e} on odd)",
        100.0 * prefill_same,
        100.0 * decode_same
    );
    // A token that reads up to 640 keys has more BF16 weights to round than
    // one that reads the window, and decode steps that chose other positions
    // read other keys.
    assert!(prefill_distance < 5e-3 && decode_distance < 2e-2);
    assert!(key_differ * 1000 < groups * HEAD_DIM && index_key_differ * 1000 < groups * INDEX_DIM);
    for c in [&prefill, &even, &odd] {
        assert!(c.queries > 0 && c.least_shared.unwrap_or(0) * 100 >= topk * 95);
    }
    state.free(gpu)?;
    Ok(())
}
