// SPDX-License-Identifier: AGPL-3.0-only
//!
//! Ported from Enntity/atlas (AGPL-3.0-only, sparkqwen drafter series).

//! GPU: `decode_rows_kv_only` (the absorb's and the prompt rows' KV writes)
//! writes the drafter KV rows a full multi-row decode of the same rows
//! (`decode_multi_seq_rows`, the shape a verify drives) writes, on the REAL
//! drafter. Inputs are synthetic (random stream rows, tokens, positions);
//! only the drafter's arithmetic is under test. Under the exact batching
//! lane (`ATLAS_QWEN4EXP_BATCH_FAST`) from 4 rows up both project K/V with
//! the same row tiers, so their rows must be byte-equal; below 4 rows the
//! full decode takes other K/V arms, and the difference is reported.
//!
//! `#[ignore]`; `ATLAS_Q38_MTP_DIR` as in `qwen4exp_mtp_batch_gpu_tests.rs`:
//! ```text
//! ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
//!   ATLAS_Q38_MTP_DIR=/models/q38-mtp \
//!   cargo test -p spark-model --release --lib qwen4exp_mtp_kv_only_gpu -- --ignored --nocapture
//! ```

use super::*;

use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use spark_runtime::buffers::BufferArena;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const MAX_SEQ: usize = 4096;

/// SplitMix64.
fn next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn bf16(b: &[u8]) -> f32 {
    f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)
}

#[test]
#[ignore]
fn qwen4exp_mtp_kv_only_gpu_matches_the_full_decode() {
    let dir = std::path::PathBuf::from(
        std::env::var("ATLAS_Q38_MTP_DIR").expect("ATLAS_Q38_MTP_DIR names the drafter's dir"),
    );
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config = atlas_core::config::parse_config(&json).unwrap();
    let loader = SafetensorsLoader {
        peak_memory_multiplier: Some(0.05),
        ..SafetensorsLoader::new()
    };
    let store = loader.load(&dir, g, 0).unwrap();
    let module = crate::weight_loader::qwen4_exp::load_qwen4exp_mtp_module(
        &store,
        &config,
        g,
        &[KvCacheDtype::Bf16],
    )
    .unwrap()
    .expect("mtp.* tensors");
    let embed =
        crate::weight_map::dense(&store, "model.language_model.embed_tokens.weight").unwrap();
    let lm_head = crate::weight_map::dense(&store, "lm_head.weight").unwrap();
    let head = Qwen4ExpMtpHead::new(module, embed, lm_head, g, 100_000, MAX_SEQ).unwrap();

    let buffers = BufferArena::new(&config, 64, MAX_SEQ, 16, 8, g).unwrap();
    let (dispatch, derived, stats) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelStats::new(),
    );
    let mut levers = ModelLevers::defaults();
    levers.qwen4exp_batch_fast = true;
    levers.qwen4exp_exact_verify = true;
    let ctx = ForwardContext {
        ssm_batch: None,
        buffers: &buffers,
        gpu: g,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Skip,
    };
    let s0 = g.default_stream();
    let rb = head.absorb.row_bytes();
    let body = &head.module.config;
    let row_kv = body.num_key_value_heads * body.head_dim * 2;
    let layer = body
        .layer_types
        .iter()
        .filter(|t| matches!(t, atlas_core::config::LayerType::FullAttention))
        .count();

    let mut seed = 0x6b76u64;
    let mut exact_from_four = true;
    for n in [1usize, 2, 3, 4, 7, 8] {
        let mut st = head.alloc_state_inner(g).unwrap();
        let mut kv = head.kv_cache.lock();
        let bs = kv.block_size();
        st.block_table.push(kv.alloc_block().unwrap());
        let tokens: Vec<u32> = (0..n).map(|_| (next(&mut seed) % 100_000) as u32).collect();
        let positions: Vec<usize> = (0..n).map(|i| 100 + i).collect();
        let kv_rows: Vec<usize> = (0..n).collect();
        let tables = vec![st.block_table.clone(); n];
        let streams: Vec<u8> = (0..n * rb / 4)
            .flat_map(|_| {
                (((next(&mut seed) >> 40) as f32 / (1u64 << 23) as f32) - 1.0).to_le_bytes()
            })
            .collect();
        let slab = qwen4exp_mtp_batch::pack_slab(
            &tokens,
            &positions,
            &kv_rows,
            &tables,
            1,
            bs,
            head.batch_slab_blocks,
        )
        .unwrap();
        let meta = qwen4exp_mtp_batch::slab_meta(head.batch_slab, 0, n, slab.max_blocks);
        let block = st.block_table[0] as usize;
        let read = |kv: &PagedKvCache| -> (Vec<u8>, Vec<u8>) {
            g.synchronize(s0).unwrap();
            let stride = kv.k_block_stride_bytes_for_layer(layer);
            let mut k = vec![0u8; n * row_kv];
            let mut v = vec![0u8; n * row_kv];
            g.copy_d2h(kv.k_pool_ptr(layer).offset(block * stride), &mut k)
                .unwrap();
            g.copy_d2h(kv.v_pool_ptr(layer).offset(block * stride), &mut v)
                .unwrap();
            (k, v)
        };
        let mut run = |kv_only: bool, kv: &mut PagedKvCache| {
            g.copy_h2d(&slab.bytes, head.batch_slab).unwrap();
            g.copy_h2d(&streams, buffers.hc_streams()).unwrap();
            head.stage_rows_input(&head.own_scratch(), head.batch_slab, n, &ctx, s0)
                .unwrap();
            let body_ctx = head.body_ctx(&ctx, meta);
            if kv_only {
                let rows = crate::layer::KvRows {
                    hidden: buffers.hidden_states(),
                    residual: buffers.residual(),
                    n,
                    seq_lens: &kv_rows,
                    stream: s0,
                };
                assert!(
                    head.module
                        .body
                        .decode_rows_kv_only(rows, kv, &body_ctx)
                        .unwrap()
                );
            } else {
                let mut states: Vec<&mut (dyn LayerState + 'static)> = vec![st.body_state.as_mut()];
                head.module
                    .body
                    .decode_multi_seq_rows(
                        buffers.hidden_states(),
                        buffers.residual(),
                        n,
                        &mut states,
                        &vec![0; n],
                        kv,
                        &kv_rows,
                        &tables,
                        &body_ctx,
                        s0,
                    )
                    .unwrap();
            }
        };
        run(false, &mut kv);
        let (k_full, v_full) = read(&kv);
        let stride = kv.k_block_stride_bytes_for_layer(layer);
        g.memset(kv.k_pool_ptr(layer).offset(block * stride), 0, stride)
            .unwrap();
        g.memset(kv.v_pool_ptr(layer).offset(block * stride), 0, stride)
            .unwrap();
        run(true, &mut kv);
        let (k_only, v_only) = read(&kv);
        let (mut differ, mut max_abs, mut max_ref) = (0usize, 0f32, 0f32);
        for (a, b) in [(&k_full, &k_only), (&v_full, &v_only)] {
            for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
                let (x, y) = (bf16(x), bf16(y));
                differ += usize::from(x.to_bits() != y.to_bits());
                max_abs = max_abs.max((x - y).abs());
                max_ref = max_ref.max(x.abs());
            }
        }
        eprintln!(
            "n={n}: {differ} of {} K/V elements differ, max |diff| {max_abs:.3e} (max |K/V| {max_ref:.3e})",
            2 * n * row_kv / 2
        );
        if n >= 4 {
            exact_from_four &= differ == 0;
        } else {
            assert!(
                max_abs <= 0.02 * max_ref.max(1e-6),
                "n={n}: K/V far from the full decode"
            );
        }
        kv.free_blocks(&st.block_table);
    }
    assert!(
        exact_from_four,
        "the KV-only rows differ from the full decode at 4+ rows"
    );
}
