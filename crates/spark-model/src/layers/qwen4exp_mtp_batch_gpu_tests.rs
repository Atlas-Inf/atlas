// SPDX-License-Identifier: AGPL-3.0-only

//! GPU: the batched qwen4_exp propose drafts what the per-sequence propose
//! drafts, on the REAL drafter (`mtp.*`, the shared embedding and head), for
//! the same inputs — and the last position's draft logits agree byte for
//! byte, which is the stronger statement (an argmax can hide a last-bit
//! difference; the logits cannot).
//!
//! Inputs are synthetic: each sequence gets a few drafter KV rows of history
//! and a random target stream row, so only the drafter's arithmetic is under
//! test. `#[ignore]` per repo convention. `ATLAS_Q38_MTP_DIR` names a model
//! directory holding (at least) the `mtp.*`, embedding and `lm_head` tensors
//! with the checkpoint's `config.json`; a filtered `model.safetensors.index.json`
//! over the real shards keeps the load to the drafter. On a GB10:
//! ```text
//! ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
//!   ATLAS_Q38_MTP_DIR=/models/q38-mtp \
//!   cargo test -p spark-model --release --lib qwen4exp_mtp_batch_gpu -- --ignored --nocapture
//! ```
//! `ATLAS_QWEN4EXP_BATCH_SMALL=1`, `ATLAS_QWEN4EXP_DRAFT_HEAD_NVFP4=1` and
//! `ATLAS_QWEN4EXP_MTP_CONFIDENCE=<p>` select those variants.

use super::*;
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use spark_runtime::buffers::BufferArena;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

const DRAFTS: usize = 3;
const MAX_SEQ: usize = 4096;

/// SplitMix64: a reproducible fixture from the seed alone.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A target stream row: `hc * hidden` FP32 in `[-1, 1)`.
    fn stream_row(&mut self, len: usize) -> Vec<u8> {
        (0..len)
            .flat_map(|_| (((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0).to_le_bytes())
            .collect()
    }
}

#[test]
#[ignore]
fn qwen4exp_mtp_batch_gpu_drafts_like_the_per_sequence_path() {
    let dir = std::path::PathBuf::from(
        std::env::var("ATLAS_Q38_MTP_DIR")
            .expect("ATLAS_Q38_MTP_DIR names the drafter's model dir"),
    );
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let config = atlas_core::config::parse_config(&json).unwrap();
    // The pre-flight sizes whole shard files, and the drafter shares its
    // shard with the PLE tables; the load itself reads only indexed tensors.
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
    levers.qwen4exp_batch_small = std::env::var("ATLAS_QWEN4EXP_BATCH_SMALL").as_deref() == Ok("1");
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
    let h = config.hidden_size;
    let row_bytes = config.hc_mult * h * 4;
    let rows = head.draft.rows() as usize;
    let width = head.batch_width(&buffers, &config);
    eprintln!(
        "batch width {width}, draft rows {rows}, conf stop {}",
        head.conf_stop
    );
    assert!(width >= 2);

    let mut rng = Rng(0x5eed);
    let mut all_equal = true;
    for n in [2usize, 3, 4, width] {
        // History: sequence i gets 3 + 5i drafter rows (one row a propose).
        let mut owned: Vec<Box<dyn ProposerState>> =
            (0..n).map(|_| head.alloc_state(g).unwrap()).collect();
        for (i, s) in owned.iter_mut().enumerate() {
            for p in 0..3 + 5 * i {
                g.copy_h2d(&rng.stream_row(row_bytes / 4), buffers.hc_streams())
                    .unwrap();
                let tok = (rng.next() % 100_000) as u32;
                head.propose(
                    tok,
                    DevicePtr::NULL,
                    20 + p,
                    1,
                    s.as_mut(),
                    None,
                    &ctx,
                    s0,
                    None,
                    None,
                    None,
                )
                .unwrap();
            }
        }
        let starts: Vec<usize> = owned.iter_mut().map(|s| st(s).seq_len).collect();
        let tokens: Vec<u32> = (0..n).map(|_| (rng.next() % 100_000) as u32).collect();
        let positions: Vec<usize> = starts.iter().map(|&l| l + 40).collect();
        let streams: Vec<Vec<u8>> = (0..n).map(|_| rng.stream_row(row_bytes / 4)).collect();

        // Batched: stream row i <- sequence i.
        for (i, row) in streams.iter().enumerate() {
            g.copy_h2d(row, buffers.hc_streams().offset(i * row_bytes))
                .unwrap();
        }
        g.synchronize(s0).unwrap();
        let t = std::time::Instant::now();
        let batched = {
            let mut states: Vec<&mut dyn ProposerState> = Vec::new();
            for s in owned.iter_mut() {
                states.push(s.as_mut());
            }
            head.propose_batch(
                &tokens,
                &vec![DevicePtr::NULL; n],
                &positions,
                DRAFTS,
                &mut states,
                None,
                &ctx,
                s0,
                None,
                None,
            )
            .unwrap()
            .expect("batched propose admitted")
        };
        // The call ends in its blocking readback: the wall time is complete.
        let batched_us = t.elapsed().as_micros();
        let mut serial_us = 0u128;
        let mut batched_logits = vec![0u8; n * rows * 2];
        g.copy_d2h(buffers.logits(), &mut batched_logits).unwrap();
        let batched_lens: Vec<usize> = owned.iter_mut().map(|s| st(s).seq_len).collect();

        // Per-sequence, from the same drafter state (rows past `seq_len` are
        // rewritten before they are read).
        for (i, s) in owned.iter_mut().enumerate() {
            st(s).seq_len = starts[i];
            g.copy_h2d(&streams[i], buffers.hc_streams()).unwrap();
            g.synchronize(s0).unwrap();
            let t = std::time::Instant::now();
            let serial = head
                .propose(
                    tokens[i],
                    DevicePtr::NULL,
                    positions[i],
                    DRAFTS,
                    s.as_mut(),
                    None,
                    &ctx,
                    s0,
                    None,
                    None,
                    None,
                )
                .unwrap();
            serial_us += t.elapsed().as_micros();
            let mut logits = vec![0u8; rows * 2];
            g.copy_d2h(buffers.logits(), &mut logits).unwrap();
            // The logits rows compare only when both chains ran the full
            // depth (a confidence stop leaves an earlier position's there).
            let full = serial.len() == DRAFTS && batched[i].len() == DRAFTS;
            let logits_equal = !full || logits == batched_logits[i * rows * 2..(i + 1) * rows * 2];
            let same = serial == batched[i] && st(s).seq_len == batched_lens[i] && logits_equal;
            eprintln!(
                "n={n} seq {i}: kv {} batched {:?} serial {:?} last-logits {}",
                starts[i],
                batched[i],
                serial,
                if !full {
                    "n/a (stopped)"
                } else if logits_equal {
                    "bit-equal"
                } else {
                    "DIFFER"
                },
            );
            all_equal &= same;
        }
        eprintln!("n={n}: {DRAFTS} drafts batched {batched_us} us, per-sequence {serial_us} us");
        for s in owned.iter_mut() {
            head.free_state(g, None, s.as_mut()).unwrap();
        }
    }
    assert!(
        all_equal,
        "batched drafts differ from the per-sequence path"
    );
}

fn st(s: &mut Box<dyn ProposerState>) -> &mut Qwen4ExpMtpProposerState {
    s.as_any_mut().downcast_mut().unwrap()
}
