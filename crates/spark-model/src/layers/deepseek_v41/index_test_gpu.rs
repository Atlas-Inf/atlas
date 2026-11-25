// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): the index scores and the positions chosen from
//! them, against DeepSeek's reference on layer 2 of the recorded prompt.
//! `A_index.safetensors` is the replay of the reference's own scoring lines
//! on the recording (`index_replay.py`), which gives the recorded top-k on
//! every query.
//!
//!   ATLAS_DSV41_REC=<recording with --ops and the index replay> \
//!     cargo test --release -p spark-model --lib deepseek_v41::index -- --ignored --nocapture --test-threads=1

use anyhow::Result;

use super::*;
use crate::layers::deepseek_v41::cache_recipe::bf16_to_f32;
use crate::layers::deepseek_v41::index_recipe::top_positions;
use crate::layers::deepseek_v41::moe;
use crate::layers::deepseek_v41::test_support::{Recording, backend, bf16s, download, upload};

const HEADS: usize = 32;
const RATIO: usize = 2;
const TOPK: usize = 512;
const WINDOW: i32 = 128;

fn floats(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| bf16_to_f32(b)).collect()
}

/// The compressed positions of a recorded index row: -1 is none, and the
/// rest sit `offset` up, after the window's keys.
fn chosen(row: &[u8], offset: i32) -> Vec<i32> {
    row.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .filter(|&i| i >= 0)
        .map(|i| i - offset)
        .collect()
}

#[test]
fn the_head_weight_scale_is_the_references_scalar() {
    assert_eq!(head_weight_scale(32), 1.0 / 64.0);
    // 64 heads: 2^-6.5 rounded to BF16's eight bits.
    assert_eq!(head_weight_scale(64), 181.0 / 16384.0);
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target and ATLAS_DSV41_REC (--ops, with the index replay)"]
fn index_scores_and_choices_replay_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let kernels = IndexKernels::load(gpu)?;
    let round = moe::MoeKernels::load(gpu)?.round_bf16;
    let (_, keys) = rec.caches("after_decode.L2.indexer_k_cache")?;
    let groups = keys.len() / (INDEX_DIM * 2);
    let keys = upload(gpu, &keys)?;
    let (partial, rounded) = (gpu.alloc(groups * 4)?, gpu.alloc(groups * 2)?);
    let (q_dev, w_dev) = (gpu.alloc(HEADS * INDEX_DIM * 2)?, gpu.alloc(HEADS * 2)?);
    // The recorded head weights are already scaled.
    let scores = |q: &[u8], w: &[u8], visible: usize| -> Result<Vec<f32>> {
        gpu.synchronize(stream)?;
        gpu.copy_h2d(q, q_dev)?;
        gpu.copy_h2d(w, w_dev)?;
        score(
            gpu,
            kernels.score,
            q_dev,
            keys,
            w_dev,
            partial,
            HEADS,
            visible,
            1.0,
            stream,
        )?;
        moe::round_bf16(gpu, round, partial, rounded, visible, stream)?;
        Ok(floats(&download(gpu, rounded, visible)?))
    };
    let (q_row, w_row) = (HEADS * INDEX_DIM * 2, HEADS * 2);

    // Prefill: the reference's arithmetic is exact, and so are the scores.
    let (_, q) = rec.index("prefill.q")?;
    let (_, w) = rec.index("prefill.weights")?;
    let (shape, want) = rec.index("prefill.score")?;
    let (width, want) = (shape[1], bf16s(&want));
    let (_, topk) = rec.prefill("L2.indexer_topk")?;
    let (mut values, mut cut) = (0usize, 0usize);
    for query in 0..rec.prefill {
        let visible = (query + 1) / RATIO;
        let theirs = floats(&want[query * width..query * width + visible]);
        let recorded = chosen(
            &topk[query * TOPK * 4..(query + 1) * TOPK * 4],
            rec.prefill as i32,
        );
        assert_eq!(top_positions(&theirs, TOPK), recorded, "query {query}");
        if visible == 0 {
            continue;
        }
        let mine = scores(
            &q[query * q_row..(query + 1) * q_row],
            &w[query * w_row..(query + 1) * w_row],
            visible,
        )?;
        assert!(mine == theirs, "query {query}: scores differ");
        values += visible;
        cut += usize::from(visible > TOPK);
    }
    println!(
        "prefill: {values} scores of {} queries equal the reference's; the positions chosen equal the recorded ones on all, and {cut} queries choose {TOPK} of more than {TOPK}",
        rec.prefill
    );

    // Decode. With an odd number of keys the reference's scores carry its
    // library's rounding, and what it chose follows them.
    let (_, q) = rec.index("decode.q")?;
    let (_, w) = rec.index("decode.weights")?;
    let (shape, want) = rec.index("decode.score")?;
    let (width, want) = (shape[1], bf16s(&want));
    let (mut even, mut odd, mut odd_equal, mut odd_values, mut least) =
        (0, 0, 0usize, 0usize, TOPK);
    for step in 0..rec.decode {
        let visible = (rec.prefill + step + 1) / RATIO;
        let theirs = floats(&want[step * width..step * width + visible]);
        let recorded = chosen(&rec.step("L2.indexer_topk", step)?.1, WINDOW);
        assert_eq!(top_positions(&theirs, TOPK), recorded, "step {step}");
        let mine = scores(
            &q[step * q_row..(step + 1) * q_row],
            &w[step * w_row..(step + 1) * w_row],
            visible,
        )?;
        if visible.is_multiple_of(2) {
            assert!(mine == theirs, "step {step}: scores differ");
            even += 1;
            continue;
        }
        odd += 1;
        odd_equal += mine.iter().zip(&theirs).filter(|(a, b)| a == b).count();
        odd_values += visible;
        let kept = top_positions(&mine, TOPK);
        least = least.min(kept.iter().filter(|p| recorded.contains(p)).count());
    }
    println!(
        "decode: {even} steps with an even number of keys equal the reference's scores; on the {odd} with an odd number {:.1}% of scores are equal and at least {least} of {TOPK} chosen positions are the reference's",
        100.0 * odd_equal as f64 / odd_values as f64
    );
    assert!(even > 0 && odd > 0 && least * 100 >= TOPK * 95);
    Ok(())
}

/// The device's two-level choice against the host's rules
/// (`index_recipe.rs`), on scores that tie where the cuts fall: few distinct
/// BF16 values over many positions, and queries that see fewer positions
/// than there are to keep, a few more, and many more.
#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn the_device_chooses_what_the_hosts_rules_choose() -> Result<()> {
    use crate::layers::deepseek_v41::index_recipe::{candidate_blocks, keep_candidates};
    use crate::layers::deepseek_v41::test_support::le_bytes;
    const KEEP: usize = 16;
    const BLOCK: usize = 8;
    const TOP: usize = 48;
    const QUERIES: usize = 300;
    let gpu = backend()?;
    let (k, stream) = (IndexKernels::load(gpu)?, gpu.default_stream());
    // Query y sees y + 1 positions.
    let rows = Queries {
        count: QUERIES,
        position: 0,
        ratio: 1,
        stride: QUERIES,
    };
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Negative, zero and positive, an eighth apart: BF16 holds each.
    let scores: Vec<f32> = (0..QUERIES * QUERIES)
        .map(|_| (next() % 41) as f32 / 8.0 - 2.0)
        .collect();
    let partial = upload(gpu, &le_bytes(&scores, f32::to_le_bytes))?;
    let cols = QUERIES.div_ceil(BLOCK);
    let flags = Flags {
        flags: gpu.alloc(QUERIES * cols)?,
        cols,
        block: BLOCK,
    };
    let (source, reader) = (gpu.alloc(QUERIES * TOP * 4)?, gpu.alloc(QUERIES * TOP * 4)?);
    candidate_rows(gpu, k.candidates, partial, flags, KEEP, rows, stream)?;
    choose_rows(gpu, k.choose, partial, source, TOP, rows, None, stream)?;
    choose_rows(
        gpu,
        k.choose,
        partial,
        reader,
        TOP,
        rows,
        Some(flags),
        stream,
    )?;
    gpu.synchronize(stream)?;
    let mut got_flags = vec![0u8; QUERIES * cols];
    gpu.copy_d2h(flags.flags, &mut got_flags)?;
    let read = |p| -> Result<Vec<i32>> {
        let mut bytes = vec![0u8; QUERIES * TOP * 4];
        gpu.copy_d2h(p, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let (got_source, got_reader) = (read(source)?, read(reader)?);
    let padded = |mut chosen: Vec<i32>| {
        chosen.resize(TOP, -1);
        chosen
    };
    for y in 0..QUERIES {
        let visible = y + 1;
        let mine = &scores[y * QUERIES..y * QUERIES + visible];
        let kept = candidate_blocks(mine, KEEP, BLOCK);
        let want_flags: Vec<u8> = kept.chunks(BLOCK).map(|b| u8::from(b[0])).collect();
        assert_eq!(
            &got_flags[y * cols..y * cols + want_flags.len()],
            &want_flags[..],
            "query {y}: candidate blocks"
        );
        // The candidate source chooses among every position.
        let want = match visible <= TOP {
            true => (0..visible as i32).collect(),
            false => top_positions(mine, TOP),
        };
        assert_eq!(
            got_source[y * TOP..(y + 1) * TOP],
            padded(want)[..],
            "query {y}: the source's choice"
        );
        // The layers after it, among the kept blocks.
        let want = match visible <= TOP {
            true => (0..visible as i32).collect(),
            false => {
                let mut masked = mine.to_vec();
                keep_candidates(&mut masked, &kept);
                top_positions(&masked, TOP)
            }
        };
        assert_eq!(
            got_reader[y * TOP..(y + 1) * TOP],
            padded(want)[..],
            "query {y}: a reader's choice"
        );
    }
    println!(
        "{QUERIES} queries over up to {QUERIES} positions: candidate blocks and both choices are the host's"
    );
    for p in [partial, flags.flags, source, reader] {
        gpu.free(p)?;
    }
    Ok(())
}
