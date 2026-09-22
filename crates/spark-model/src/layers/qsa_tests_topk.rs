// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention): the device top-k writes EVERY
//! id the gather will read, whatever the scores.
//!
//! `n_sel` depends on the position only, and `qsa_gather` indexes the block
//! table with whatever `sel_dev` holds — layer-owned scratch shared by every
//! sequence. So "fewer blocks selected than `block_topk`" is not a quality
//! problem, it is a read of stale ids. NaN scores are not reachable today
//! (the scoring kernels' `fmaxf` relu swallows them), which is exactly why
//! this is pinned by a test rather than by luck.
//!
//!   cargo test -p spark-model --release device_topk_writes -- --ignored --nocapture

use super::super::qsa_decode_select::{expand_selection, select_blocks};
use super::*;

fn dl_i32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<i32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore]
fn device_topk_writes_every_id_the_gather_reads() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let kernel = g.kernel("qsa_indexer", "qsa_select_topk").unwrap();

    let (ratio, topk, tail) = (4usize, 8usize, 2usize);
    let nan = f32::NAN;
    let cases: [(&str, Vec<f32>); 6] = [
        ("distinct", (0..12).map(|b| ((b * 7) % 12) as f32).collect()),
        ("all tied at zero", vec![0.0; 12]),
        (
            "signed zeros",
            vec![
                -0.0, 0.0, -0.0, 0.0, 1.0, 1.0, -0.0, 0.0, 2.0, 0.0, -0.0, 0.0,
            ],
        ),
        (
            "+inf",
            vec![
                1.0,
                f32::INFINITY,
                0.5,
                f32::INFINITY,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
            ],
        ),
        // 5 real scores for 8 slots: three NaN blocks MUST be taken, by index.
        (
            "some NaN",
            vec![nan, 0.5, nan, 0.25, nan, 1.0, nan, nan, 0.75, nan, 2.0, nan],
        ),
        ("all NaN", vec![nan; 12]),
    ];
    for (name, scores) in cases {
        let complete = scores.len();
        let tail_start = complete * ratio;
        let visible = tail_start + tail;
        let n_sel = topk * ratio + tail;
        let bytes: Vec<u8> = scores.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scores_dev = upload(g, &bytes);
        // Every entry starts as -1: one the kernel does not write stays -1.
        let sel_dev = upload(g, &vec![0xFFu8; n_sel * 4]);
        crate::layers::ops::qsa_select_topk(
            g,
            kernel,
            scores_dev,
            sel_dev,
            complete as u32,
            topk as u32,
            ratio as u32,
            tail_start as u32,
            visible as u32,
            stream,
        )
        .unwrap();
        let got = dl_i32(g, sel_dev, n_sel);
        assert!(
            got.iter().all(|t| *t >= 0 && (*t as usize) < visible),
            "{name}: the gather would read an unwritten or out-of-range id: {got:?}"
        );
        let want = expand_selection(&select_blocks(&scores, topk), ratio, tail_start, visible);
        assert_eq!(got, want, "{name}: device and host arms disagree");
    }
}
