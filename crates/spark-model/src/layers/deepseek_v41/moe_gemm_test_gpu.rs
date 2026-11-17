// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): a batch's routed experts on the block-scaled
//! tensor cores (`dsv41_mxg.cu`) against the grouped tile kernels, which
//! leave a row the bytes of the row alone: on the release's weights and
//! DeepSeek's recorded MoE inputs, under a routing that gives every expert
//! a full tile and a part of one.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::moe_gemm -- --ignored --nocapture --test-threads=1

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;

use super::{ExpertGemm, Runs, Sent, Span};
use crate::layers::deepseek_v41::block::{BlockKernels, Handoff, Scratch, Shared};
use crate::layers::deepseek_v41::moe_grouped::{self, GateUp, Plan};
use crate::layers::deepseek_v41::ops;
use crate::layers::deepseek_v41::parts::{Held, PARTS};
use crate::layers::deepseek_v41::test_support::{
    Recording, SHAPE, backend, block, closeness, download, free_store, picks, rows_of, upload,
};

const LAYER: usize = 1;
const ROWS: usize = 320;
/// The experts the rows choose among: 160 pairs each, a tile and a quarter.
const AMONG: usize = 12;

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn the_tensor_cores_leave_what_the_tile_kernels_leave() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let shared = Shared {
        shape: SHAPE,
        held: Held::ALL,
        kernels: BlockKernels::load(gpu)?,
        scratch: Arc::new(Scratch::alloc(gpu, &SHAPE, ROWS)?),
        handoff: Arc::new(Mutex::new(Handoff::default())),
    };
    let (block, store) = block(gpu, LAYER, true, shared)?;
    let (s, k, sc) = (
        &block.shared.shape,
        &block.shared.kernels,
        &block.shared.scratch,
    );
    let slab = block.weights.moe.experts[0].context("the routed experts' first part")?;
    let (slots, part_rows, n) = (s.top_k + 1, s.inter / PARTS, s.hidden / PARTS);

    // The rows' inputs, rounded as the MoE reads them.
    let inputs = rows_of(
        &rec.prefill(&format!("L{LAYER}.op.ffn.in"))?.1,
        s.hidden,
        &picks(&rec, ROWS),
    );
    let (x, quantized) = (upload(gpu, &inputs)?, gpu.alloc(ROWS * s.hidden * 2)?);
    let quant = k.cache.fp8_quant_dequant;
    ops::fp8_quant_dequant(gpu, quant, x, quantized, ROWS * s.hidden, stream)?;

    // Every row six of the twelve, ascending as the router leaves them.
    let mut chosen = Vec::with_capacity(ROWS * s.top_k);
    for row in 0..ROWS {
        let mut mine: Vec<u32> = (0..s.top_k)
            .map(|j| ((row + 2 * j) % AMONG) as u32)
            .collect();
        mine.sort_unstable();
        chosen.extend(mine);
    }
    let weights: Vec<f32> = (0..ROWS * s.top_k)
        .map(|i| 0.1 + 0.01 * (i % 7) as f32)
        .collect();
    let plan = Plan::new(&chosen, &weights, s.top_k, slots, s.experts);
    let pairs = plan.pairs();
    sc.grouped.upload(gpu, &plan)?;
    let big = ExpertGemm::alloc(gpu, s, ROWS)?;
    let runs = Runs::new(&plan, ROWS, s.top_k, slots);
    ensure!(
        runs.tile_count() == 2 * AMONG,
        "{} tiles",
        runs.tile_count()
    );
    big.upload(gpu, &runs)?;

    // The way in. The tile kernels leave a pair's gate values and then its
    // up values; the tensor cores every pair's gate values and then every
    // pair's up values.
    let inner = gpu.alloc(PARTS * pairs * part_rows * 2)?;
    let limit = s.swiglu_limit;
    moe_grouped::experts_in(
        gpu,
        &k.moe,
        &slab,
        plan.counts(),
        &sc.grouped,
        quantized,
        inner,
        part_rows,
        s.hidden,
        limit,
        None,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let tiled = download(gpu, sc.grouped.gate_up, pairs * 2 * part_rows)?;
    let gate_up = gpu.alloc(pairs * 2 * part_rows * 2)?;
    let row_of = sc.grouped.row_of;
    big.round_rows(gpu, quantized, 0, ROWS, s.hidden, stream)?;
    let all = runs.whole();
    big.project_in(
        gpu,
        &slab,
        &runs,
        all.tiles.clone(),
        row_of,
        gate_up,
        part_rows,
        s.hidden,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let planes = download(gpu, gate_up, pairs * 2 * part_rows)?;
    let want: Vec<u16> = (0..2 * pairs * part_rows)
        .map(|i| {
            let (plane, pair, j) = (
                i / (pairs * part_rows),
                i / part_rows % pairs,
                i % part_rows,
            );
            tiled[(pair * 2 + plane) * part_rows + j]
        })
        .collect();
    let (same, distance) = closeness(&planes, &want);
    println!(
        "in: {:.3}% of {} values equal, distance {distance:.2e}",
        same * 100.0,
        want.len()
    );
    ensure!(same > 0.99 && distance < 1e-3, "the way in differs");

    // The way out, every part of the hidden activations being the first's:
    // as values for the tile kernels, and for the tensor cores as the codes
    // and scales the ranks would send.
    let piece_stride = pairs * part_rows;
    let sent = Sent::new(pairs, ROWS, part_rows);
    let coded = gpu.alloc(PARTS * sent.bytes)?;
    for part in 0..PARTS {
        if part > 0 {
            gpu.copy_d2d(
                inner,
                inner.offset(part * piece_stride * 2),
                piece_stride * 2,
            )?;
        }
        let to = coded.offset(part * sent.bytes);
        moe_grouped::activate_codes(
            gpu,
            &k.moe,
            GateUp::paired(sc.grouped.gate_up, part_rows),
            sc.grouped.weights,
            to,
            to.offset(sent.scales),
            pairs,
            part_rows,
            limit,
            stream,
        )?;
    }
    let out = gpu.alloc(ROWS * slots * n * 2)?;
    moe_grouped::experts_out(
        gpu,
        &k.moe,
        &slab,
        plan.counts(),
        &sc.grouped,
        inner,
        piece_stride,
        out,
        part_rows,
        n,
        None,
        stream,
    )?;
    big.project_out(gpu, &slab, &all, coded, sent, part_rows, s.hidden, stream)?;
    gpu.synchronize(stream)?;
    let (tiled, got) = (
        download(gpu, out, ROWS * slots * n)?,
        download(gpu, big.pair_out, pairs * n)?,
    );
    let want: Vec<u16> = plan
        .out_row
        .iter()
        .flat_map(|&row| {
            tiled[row as usize * n..(row as usize + 1) * n]
                .iter()
                .copied()
        })
        .collect();
    let (same, distance) = closeness(&got, &want);
    println!(
        "out: {:.3}% of {} values equal, distance {distance:.2e}",
        same * 100.0,
        want.len()
    );
    ensure!(same > 0.99 && distance < 1e-3, "the way out differs");

    for p in [x, quantized, inner, gate_up, out, coded] {
        gpu.free(p)?;
    }
    free_store(gpu, &store)
}

/// Runs of the experts' pairs as tiles: `(expert, pairs)` in order.
fn runs_of(experts: &[(u32, usize)]) -> Runs {
    let (mut tiles, mut first) = (Vec::new(), 0usize);
    for &(expert, pairs) in experts {
        for at in (0..pairs).step_by(super::TILE) {
            tiles.extend([
                expert,
                (first + at) as u32,
                (pairs - at).min(super::TILE) as u32,
            ]);
        }
        first += pairs;
    }
    Runs {
        tiles,
        pairs: first,
        pair_of: Vec::new(),
    }
}

#[test]
fn a_launchs_tiles_are_cut_where_a_pairs_scales_start_a_word() {
    // Tiles start at pairs 0, 128, 256, 300, 364 and 492: the first three
    // on a multiple of eight.
    let runs = runs_of(&[(3, 300), (7, 64), (9, 200)]);
    let span = |tiles: std::ops::Range<usize>, pairs: std::ops::Range<usize>| Span { tiles, pairs };
    assert_eq!(runs.whole(), span(0..6, 0..564));
    assert_eq!(
        runs.halves(5),
        Some([span(0..2, 0..256), span(2..6, 256..564)])
    );
    assert_eq!(
        runs.halves(1),
        Some([span(0..1, 0..128), span(1..6, 128..564)])
    );
    // No tile inside starts on one; and one tile is not cut.
    assert_eq!(runs_of(&[(0, 100), (1, 100)]).halves(4), None);
    assert_eq!(runs_of(&[(0, 100)]).halves(4), None);
    assert_eq!(runs_of(&[]).halves(4), None);
}
