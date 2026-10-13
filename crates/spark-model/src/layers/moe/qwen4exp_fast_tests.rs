// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_QWEN4EXP_MOE_FAST` dispatch on the recording backend: which launches
//! each row count makes, with what grid, shared memory, rows and buffers, and
//! where the EP reductions fall. Plumbing only; the kernels' output bytes are
//! checked against the replaced kernels by scripts/dev/qwen4exp_moe_decode_bench.cu.
use super::{
    arena_tests::ContextResources,
    recording::{Arg, Event, Gpu},
};
use crate::layers::{moe::MoeLayer, ops};
use crate::weight_map::{DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight};
use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::{
    buffers::BufferArena,
    gpu::{DevicePtr, GpuBackend},
};
use std::sync::Mutex;

const H: usize = 2560;
const I: usize = 640;
const E: usize = 512;
const K: usize = 10;

fn projection(gpu: &Gpu, n: usize, k: usize) -> QuantizedWeight {
    QuantizedWeight {
        weight: gpu.alloc(n * k / 2).unwrap(),
        weight_scale: gpu.alloc(n * k / 16).unwrap(),
        weight_scale_2: 1.25,
        ..QuantizedWeight::null()
    }
}

fn expert(gpu: &Gpu) -> ExpertWeight {
    ExpertWeight {
        gate_proj: projection(gpu, I, H),
        up_proj: projection(gpu, I, H),
        down_proj: projection(gpu, H, I),
    }
}

/// A qwen4_exp-shaped MoE: 512 experts (the upper half remote at EP2),
/// top-10 softmax routing, a gated 640-wide shared expert.
fn qwen4exp_layer(gpu: &Gpu, ep: usize) -> (atlas_core::config::ModelConfig, MoeLayer) {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen4_exp".into();
    config.hidden_size = H;
    config.moe_intermediate_size = I;
    config.shared_expert_intermediate_size = I;
    config.num_experts = E;
    config.num_experts_per_tok = K;
    config.scoring_func = "softmax".into();
    config.tp_world_size = ep;
    config.ep_world_size = ep;
    config.tp_rank = 0;
    config.ep_rank = 0;
    config.adapter_max_rank = 0;
    let mut weights = MoeWeights::empty(E);
    for (e, w) in weights.experts.iter_mut().enumerate() {
        *w = if config.is_local_expert(e) {
            expert(gpu)
        } else {
            ExpertWeight::null()
        };
    }
    weights.shared_expert = expert(gpu);
    weights.gate = DenseWeight {
        weight: gpu.alloc(E * H * 2).unwrap(),
    };
    weights.shared_expert_gate = DenseWeight {
        weight: gpu.alloc(H * 2).unwrap(),
    };
    let layer = MoeLayer::new(weights, E, None, gpu, &config).unwrap();
    (config, layer)
}

fn fast(gpu: &Gpu) -> ops::Qwen4ExpMoeFast {
    ops::Qwen4ExpMoeFast {
        gate_up: gpu
            .kernel("qwen4exp_moe_decode", "qwen4exp_moe_gate_up_t")
            .unwrap(),
        silu_down: gpu
            .kernel("qwen4exp_moe_decode", "qwen4exp_moe_silu_down_t")
            .unwrap(),
    }
}

type Launch = (u64, [u32; 3], [u32; 3], u32, Vec<Arg>, usize);

fn launches(gpu: &Gpu) -> Vec<Launch> {
    gpu.trace()
        .into_iter()
        .enumerate()
        .filter_map(|(at, e)| match e {
            Event::Launch(h, g, b, m, _, a) => Some((h, g, b, m, a, at)),
            _ => None,
        })
        .collect()
}

fn word(v: usize) -> Arg {
    Arg::Bytes((v as u32).to_ne_bytes().to_vec())
}

fn run(layer: &MoeLayer, rows: usize, input: DevicePtr, ctx: &crate::layer::ForwardContext) {
    match rows {
        1 => layer.forward(input, ctx, 7).map(|_| ()),
        2 => layer.forward_k2(input, ctx, 7),
        3 => layer.forward_k3(input, ctx, 7),
        _ => layer.forward_batched(input, rows, ctx, 7),
    }
    .unwrap();
}

/// The fast pairs for `rows` rows, one per chunk of up to four: grids,
/// block, shared memory, the rows and routing slots each chunk reads, the
/// buffers it writes, and its trailing `rows` argument. Returns the trace
/// positions of each chunk's (gate/up, down) launches.
fn assert_fast_pairs(
    calls: &[Launch],
    f: ops::Qwen4ExpMoeFast,
    arena: &BufferArena,
    rows: usize,
) -> Vec<(usize, usize)> {
    let gate_up: Vec<_> = calls.iter().filter(|c| c.0 == f.gate_up.0).collect();
    let down: Vec<_> = calls.iter().filter(|c| c.0 == f.silu_down.0).collect();
    let chunks = rows.div_ceil(4);
    assert_eq!((gate_up.len(), down.len()), (chunks, chunks), "rows={rows}");
    (0..chunks)
        .map(|c| {
            let n = (rows - 4 * c).min(4);
            let slots = arena.scratch().offset(4 * c * K * 4);
            let (_, grid, block, smem, args, at_gate_up) = gate_up[c];
            assert_eq!(*grid, [2, (n * K + 1) as u32, 2]);
            assert_eq!((*block, *smem), ([160, 1, 1], (n * H * 4) as u32));
            assert_eq!(args[0], Arg::Ptr(arena.norm_output().offset(4 * c * H * 2)));
            assert_eq!(args[4], Arg::Ptr(arena.expert_gate_out()));
            assert_eq!(args[8], Arg::Ptr(arena.expert_up_out()));
            assert_eq!(args[9], Arg::Ptr(slots));
            assert_eq!(args[18..], [word(I), word(H), word(K), word(n)]);
            let (_, grid, block, smem, args, at_down) = down[c];
            assert_eq!(*grid, [8, (n * K + 1) as u32, 1]);
            assert_eq!((*block, *smem), ([160, 1, 1], (n * I * 4) as u32));
            assert_eq!(args[5], Arg::Ptr(arena.expert_down_out()));
            assert_eq!(args[6], Arg::Ptr(slots));
            assert_eq!(args[13..], [word(H), word(I), word(K), word(n)]);
            assert!(at_gate_up < at_down);
            (*at_gate_up, *at_down)
        })
        .collect()
}

#[test]
fn qwen4exp_moe_fast_takes_every_unified_row_count_in_one_pair() {
    let gpu = Gpu::new();
    let (config, mut layer) = qwen4exp_layer(&gpu, 1);
    layer.transpose_for_prefill_unified(&gpu, &config).unwrap();
    assert!(layer.use_t_layout_for_decode());
    let f = fast(&gpu);
    let replaced = [
        layer.moe_expert_gate_up_shared_t_k.0,
        layer.moe_expert_silu_down_shared_t_k.0,
        layer.moe_expert_gate_up_shared_batch2_t_k.0,
        layer.moe_expert_silu_down_shared_batch2_t_k.0,
        layer.moe_expert_gate_up_shared_batch3_t_k.0,
        layer.moe_expert_silu_down_shared_batch3_t_k.0,
    ];
    let arena = BufferArena::new(&config, 4, 2048, 16, 4, &gpu).unwrap();
    let resources = ContextResources::new();
    let ctx = resources.view(&arena, &config, &gpu);
    for rows in 1..=4 {
        // Off (the default): the replaced kernels, one pair per row at 4.
        layer.qwen4exp_moe_fast = ops::Qwen4ExpMoeFast::OFF;
        gpu.clear();
        run(&layer, rows, arena.norm_output(), &ctx);
        let calls = launches(&gpu);
        assert!(
            calls
                .iter()
                .all(|c| c.0 != f.gate_up.0 && c.0 != f.silu_down.0)
        );
        let old = calls.iter().filter(|c| replaced.contains(&c.0)).count();
        assert_eq!(old, if rows == 4 { 8 } else { 2 }, "rows={rows}");

        layer.qwen4exp_moe_fast = f;
        gpu.clear();
        run(&layer, rows, arena.norm_output(), &ctx);
        let calls = launches(&gpu);
        assert!(
            calls.iter().all(|c| !replaced.contains(&c.0)),
            "rows={rows}"
        );
        assert_eq!(assert_fast_pairs(&calls, f, &arena, rows).len(), 1);
    }
}

/// Records each reduction with the trace length when it was issued.
struct Comm<'a> {
    gpu: &'a Gpu,
    reductions: Mutex<Vec<(usize, usize)>>,
}
impl CommBackend for Comm<'_> {
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_reduce(&self, _: u64, n: usize) -> Result<()> {
        self.all_reduce_async(0, n, 0)
    }
    fn all_reduce_async(&self, _: u64, n: usize, _: u64) -> Result<()> {
        self.reductions
            .lock()
            .unwrap()
            .push((n, self.gpu.trace().len()));
        Ok(())
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected receive")
    }
}

/// TP2's hybrid layout keeps single-row decode on the originals and hands
/// only `forward_batched` to the fast pair, four rows a chunk, whose routing,
/// blends and EP reductions stay per row, in the loop's order, and whose next
/// chunk reuses the expert buffers only after the previous chunk's blends.
#[test]
fn qwen4exp_moe_fast_hybrid_batched_rows_keep_per_row_routing_blends_and_reductions() {
    let gpu = Gpu::new();
    let (config, mut layer) = qwen4exp_layer(&gpu, 2);
    layer.transpose_for_prefill_hybrid(&gpu, &config).unwrap();
    assert!(!layer.use_t_layout_for_decode() && layer.use_t_layout_for_prefill());
    let f = fast(&gpu);
    layer.qwen4exp_moe_fast = f;
    let arena = BufferArena::new(&config, 8, 2048, 16, 8, &gpu).unwrap();
    let resources = ContextResources::new();
    let comm = Comm {
        gpu: &gpu,
        reductions: Mutex::new(vec![]),
    };
    let mut ctx = resources.view(&arena, &config, &gpu);
    ctx.comm = Some(&comm);

    gpu.clear();
    run(&layer, 1, arena.norm_output(), &ctx);
    let calls = launches(&gpu);
    assert!(
        calls
            .iter()
            .all(|c| c.0 != f.gate_up.0 && c.0 != f.silu_down.0)
    );
    assert_eq!(
        calls
            .iter()
            .filter(|c| c.0 == layer.moe_expert_gate_up_shared.0)
            .count(),
        1
    );

    for rows in 1..=6 {
        comm.reductions.lock().unwrap().clear();
        gpu.clear();
        layer
            .forward_batched(arena.norm_output(), rows, &ctx, 7)
            .unwrap();
        let calls = launches(&gpu);
        let pairs = assert_fast_pairs(&calls, f, &arena, rows);
        let scratch = arena.scratch();
        let topk: Vec<_> = calls.iter().filter(|c| c.0 == layer.moe_topk.0).collect();
        let blend: Vec<_> = calls
            .iter()
            .filter(|c| c.0 == layer.moe_weighted_sum_blend.0)
            .collect();
        let reductions = comm.reductions.lock().unwrap().clone();
        assert_eq!(
            (topk.len(), blend.len(), reductions.len()),
            (rows, rows, rows)
        );
        let weights = scratch.offset(rows * K * 4);
        for r in 0..rows {
            let (c, t) = (r / 4, r % 4);
            assert!(topk[r].5 < pairs[0].0, "every row routes first");
            assert_eq!(topk[r].4[1], Arg::Ptr(scratch.offset(r * K * 4)));
            assert_eq!(topk[r].4[2], Arg::Ptr(weights.offset(r * K * 4)));
            let b = &blend[r].4;
            assert_eq!(b[0], Arg::Ptr(arena.moe_output().offset(r * H * 2)));
            assert_eq!(
                b[1],
                Arg::Ptr(arena.expert_down_out().offset(t * K * H * 2))
            );
            assert_eq!(b[2], Arg::Ptr(weights.offset(r * K * 4)));
            assert_eq!(b[3], Arg::Ptr(arena.attn_output().offset(t * H * 2)));
            assert_eq!(b[4], Arg::Ptr(arena.norm_output().offset(r * H * 2)));
            assert!(
                blend[r].5 > pairs[c].1,
                "row {r} blends after its chunk's down"
            );
            if let Some(next) = pairs.get(c + 1) {
                assert!(blend[r].5 < next.0, "chunk {c} is blended before reuse");
            }
            // Row r reduces after its blend and before the next one.
            assert_eq!(reductions[r].0, H * 2);
            assert!(reductions[r].1 > blend[r].5);
            assert!(r + 1 == rows || reductions[r].1 <= blend[r + 1].5);
        }
    }
}
