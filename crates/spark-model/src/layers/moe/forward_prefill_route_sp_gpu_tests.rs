// SPDX-License-Identifier: AGPL-3.0-only

//! The two halves of a split routing must equal the whole chunk's: q38
//! router logits (`dense_gemm_bf16_pipelined`) and the softmax top-k ids and
//! weights, byte for byte, at even and uneven SP splits.

use spark_runtime::gpu::{DevicePtr, GpuBackend};

fn bf16s(seed: u32, n: usize, scale: f32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 2.0 * scale;
            ((x.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

#[test]
#[ignore]
fn split_routes_match_the_whole_chunk() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_gemm = g.kernel("gemm", "dense_gemm_bf16_pipelined").unwrap();
    let k_topk = g.kernel("moe_topk", "moe_topk_softmax_batched").unwrap();
    let (h, e, top_k) = (2560usize, 512usize, 10usize);
    let up = |b: Vec<u8>| {
        let p = g.alloc(b.len()).unwrap();
        g.copy_h2d(&b, p).unwrap();
        p
    };
    let w = up(bf16s(1, e * h, 0.05));
    for (total, split) in [
        (16016usize, 8192usize),
        (12408, 6144),
        (5000, 2048),
        (300, 150),
    ] {
        let x = up(bf16s(total as u32, total * h, 1.0));
        let run = |ranges: &[(usize, usize)]| -> Vec<u8> {
            let logits = g.alloc(total * e * 2).unwrap();
            let routes = g.alloc(total * top_k * 8).unwrap();
            let weights = routes.offset(total * top_k * 4);
            for &(r0, n) in ranges {
                super::q38_router_gemm(
                    g,
                    k_gemm,
                    [x.offset(r0 * h * 2), w, logits.offset(r0 * e * 2)],
                    [n, e, h].map(|v| v as u32),
                    stream,
                )
                .unwrap();
                crate::layers::ops::moe_topk_softmax_batched(
                    g,
                    k_topk,
                    logits.offset(r0 * e * 2),
                    routes.offset(r0 * top_k * 4),
                    weights.offset(r0 * top_k * 4),
                    e as u32,
                    top_k as u32,
                    true,
                    n as u32,
                    stream,
                )
                .unwrap();
            }
            g.synchronize(stream).unwrap();
            let mut out = vec![0u8; total * (e * 2 + top_k * 8)];
            let (l, r) = out.split_at_mut(total * e * 2);
            g.copy_d2h(logits, l).unwrap();
            g.copy_d2h(routes, r).unwrap();
            for p in [logits, routes] {
                g.free(p).unwrap();
            }
            out
        };
        let whole = run(&[(0, total)]);
        let halves = run(&[(split, total - split), (0, split)]);
        let diff = whole.iter().zip(&halves).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{total} split at {split}: {diff} bytes differ");
        g.free(DevicePtr(x.0)).unwrap();
    }
}
