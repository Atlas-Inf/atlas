// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`): the attention sublayer of one layer over the
//! whole recorded prompt, token by token, on the release's weights. From
//! the position where a query sees more compressed positions than it may
//! read, the indexer chooses 512.
//!
//! Each token starts from DeepSeek's recorded attention input. What the
//! layer builds itself (compressed keys, index keys) it reads back from the
//! tokens before, replaced by DeepSeek's once compared. What it is handed by
//! a layer below (keys, the positions chosen, candidates) is DeepSeek's.
//!
//!   ATLAS_DSV41_REC=<full-depth recording with the index replay> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::block_index -- --ignored --nocapture --test-threads=1

use std::sync::{Arc, Mutex};

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use crate::layers::deepseek_v41::block::{BlockKernels, Handoff, Scratch, SeqState, Shared};
use crate::layers::deepseek_v41::cache_recipe::{WINDOW, completed_group, visible_groups};
use crate::layers::deepseek_v41::index::INDEX_DIM;
use crate::layers::deepseek_v41::ops::{self, HEAD_DIM};
use crate::layers::deepseek_v41::parts::Held;
use crate::layers::deepseek_v41::test_support::{
    Recording, SHAPE, backend, bf16s, block, chosen, closeness, download, free_store, hand_choice,
    hand_keys, role, source_keys,
};

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

/// What a run found.
struct Report {
    prefill: Choices,
    even: Choices,
    odd: Choices,
    /// Relative distance of the sublayer's output from the reference's:
    /// prefill, decode steps on an even number of keys, on an odd number.
    distance: [f64; 3],
}

fn run(layer: usize) -> Result<Report> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let (hidden, topk, heads) = (SHAPE.hidden, SHAPE.index_topk, SHAPE.index_heads);
    let role = role(layer);
    let ratio = role.ratio;
    let total = rec.prefill + rec.decode;
    let shared = Shared {
        shape: SHAPE,
        held: Held::ALL,
        kernels: BlockKernels::load(gpu)?,
        scratch: Arc::new(Scratch::alloc(gpu, &SHAPE, total)?),
        handoff: Arc::new(Mutex::new(Handoff::default())),
    };
    let (block, store) = block(gpu, layer, false, shared)?;
    let sc = block.shared.scratch.clone();
    let mut state = SeqState::alloc(gpu, role.compresses(), total)?;

    let l = format!("L{layer}.");
    let (_, inputs) = rec.prefill(&format!("{l}attn_in"))?;
    let (_, outputs) = rec.prefill(&format!("{l}attn_out"))?;
    let (shape, rows) = rec.prefill(&format!("{l}sa_idxs"))?;
    let row = shape[2] * 4;
    let (keys, index_keys) = source_keys(&rec, &role)?;
    hand_keys(gpu, &block, &role, &keys, &index_keys)?;
    // Layer 2 has the reference's own scores replayed (`index_replay.py`).
    let replay = match layer {
        2 => Some((
            rec.prefill("L2.op.attn.indexer.weights_proj.out")?.1,
            rec.index("prefill.q")?.1,
            rec.index("prefill.score")?,
        )),
        _ => None,
    };

    let (mut prefill, mut even, mut odd) =
        (Choices::default(), Choices::default(), Choices::default());
    let (mut key_differ, mut index_key_differ, mut groups) = (0usize, 0usize, 0usize);
    let (mut q_same, mut weight_same, mut score_same, mut scored) = (0usize, 0usize, 0usize, 0);
    let mut pairs: [(Vec<u16>, Vec<u16>); 3] = Default::default();
    for position in 0..total {
        let step = position.checked_sub(rec.prefill);
        let visible = visible_groups(position, ratio);
        let (input, output, theirs) = match step {
            None => (
                inputs[position * hidden * 2..(position + 1) * hidden * 2].to_vec(),
                outputs[position * hidden * 2..(position + 1) * hidden * 2].to_vec(),
                chosen(
                    &rows[position * row..(position + 1) * row],
                    rec.prefill as i32,
                ),
            ),
            Some(step) => (
                rec.step(&format!("{l}attn_in"), step)?.1,
                rec.step(&format!("{l}attn_out"), step)?.1,
                chosen(&rec.step(&format!("{l}sa_idxs"), step)?.1, WINDOW as i32),
            ),
        };
        hand_choice(&block, &role, &theirs, visible);
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&input, sc.normed)?;
        ops::fp8_quant_dequant(
            gpu,
            block.shared.kernels.cache.fp8_quant_dequant,
            sc.normed,
            sc.quantized,
            hidden,
            stream,
        )?;
        block.attention(gpu, 0, &mut state, position, None, stream)?;
        let kind = match step {
            None => 0,
            Some(_) if visible.is_multiple_of(2) => 1,
            Some(_) => 2,
        };
        pairs[kind]
            .0
            .extend(download(gpu, block.attention_out(gpu, stream)?, hidden)?);
        pairs[kind].1.extend(bf16s(&output));

        let mine = block.shared.handoff.lock().expect("handoff").rows[0]
            .chosen
            .clone();
        if visible <= topk {
            assert_eq!(mine, theirs, "position {position}");
        } else {
            [&mut prefill, &mut even, &mut odd][kind].add(&mine, &theirs);
        }
        if let (Some((weights, queries, (shape, scores))), None, true) =
            (&replay, step, visible > topk)
        {
            let q = heads * INDEX_DIM;
            q_same += equal(
                &download(gpu, sc.index_q, q)?,
                &bf16s(&queries[position * q * 2..(position + 1) * q * 2]),
            );
            weight_same += equal(
                &download(gpu, sc.index_weights, heads)?,
                &bf16s(&weights[position * heads * 2..(position + 1) * heads * 2]),
            );
            let at = position * shape[1] * 2;
            score_same += equal(
                &download(gpu, sc.scores, visible)?,
                &bf16s(&scores[at..at + visible * 2]),
            );
            scored += visible;
        }

        // A key is FP4, so one rounding step in it is a large one, and every
        // later token reads it: once compared, it is replaced by DeepSeek's.
        if let (Some(held), Some(group)) = (&state.compressed, completed_group(position, ratio)) {
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
    if role.key_source {
        println!(
            "layer {layer}: {total} tokens, {groups} groups; {key_differ} of {} compressed-key values and {index_key_differ} of {} index-key values differ from the reference's",
            groups * HEAD_DIM,
            groups * INDEX_DIM
        );
        assert!(
            key_differ * 1000 < groups * HEAD_DIM && index_key_differ * 1000 < groups * INDEX_DIM
        );
    }
    if replay.is_some() {
        println!(
            "layer {layer} indexer on the {} prefill queries that choose: query heads {:.3}% of values equal, head weights {:.2}%, scores {:.2}%",
            prefill.queries,
            share(q_same, prefill.queries * heads * INDEX_DIM),
            share(weight_same, prefill.queries * heads),
            share(score_same, scored)
        );
    }
    if role.index_source {
        for (name, c) in [
            ("prefill", &prefill),
            ("decode, even keys", &even),
            ("decode, odd keys", &odd),
        ] {
            println!(
                "layer {layer} {name}: {} of {} queries choose exactly the reference's {topk}; the least shared is {}",
                c.same,
                c.queries,
                c.least_shared.unwrap_or(topk)
            );
        }
    }
    let distance = [0, 1, 2].map(|kind| closeness(&pairs[kind].0, &pairs[kind].1).1);
    println!(
        "layer {layer} attention output from the reference: prefill {:.2e} ({:.1}% of values equal); decode {:.2e} on even key counts, {:.2e} on odd",
        distance[0],
        100.0 * closeness(&pairs[0].0, &pairs[0].1).0,
        distance[1],
        distance[2]
    );
    state.free(gpu)?;
    free_store(gpu, &store)?;
    block.weights.free_parts(gpu)?;
    Ok(Report {
        prefill,
        even,
        odd,
        distance,
    })
}

/// An index source shares at least 95% of the chosen positions with the
/// reference on every query; where the reference's scores are exact (prefill
/// and even key counts) the sublayer's output is BF16 rounding steps away.
fn check(report: &Report) {
    for c in [&report.prefill, &report.even, &report.odd] {
        assert!(c.queries > 0 && c.least_shared.unwrap_or(0) * 100 >= SHAPE.index_topk * 95);
    }
    assert!(report.distance[0] < 5e-3 && report.distance[1] < 1e-2);
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (with the index replay) and ATLAS_DSV41_CKPT"]
fn layer_2_chooses_the_references_compressed_positions() -> Result<()> {
    let report = run(2)?;
    check(&report);
    // Measured: 375 of 376, the other sharing 510.
    assert!(report.prefill.same * 100 >= report.prefill.queries * 98);
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (full depth) and ATLAS_DSV41_CKPT"]
fn layer_3_reads_what_layer_2_made_and_chose() -> Result<()> {
    assert!(run(3)?.distance[0] < 5e-3);
    Ok(())
}

/// Layer 20: the first at ratio 1, where every token is a group, and the
/// candidate source.
#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (full depth) and ATLAS_DSV41_CKPT"]
fn layer_20_compresses_every_token_and_chooses() -> Result<()> {
    check(&run(20)?);
    Ok(())
}

/// Layer 24: an index source that scores layer 20's index keys, inside the
/// candidates layer 20 left.
#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (full depth) and ATLAS_DSV41_CKPT"]
fn layer_24_chooses_among_layer_20s_keys() -> Result<()> {
    check(&run(24)?);
    Ok(())
}
