// SPDX-License-Identifier: AGPL-3.0-only
//! Actual layer/projection dispatch; CPU recording does not emulate cuBLAS math.
use super::*;

#[test]
fn actual_paged_glm_projection_dispatch() {
    for rank in 0..2 {
        with_mla(rank, |gpu, config, layer| {
            let arena = BufferArena::new(config, 32, 4096, 16, 4, gpu).unwrap();
            let resources = ContextResources::new();
            let mut dispatch = ops::GemmDispatch::defaults();
            // Native cuBLAS bypasses GpuBackend. Exercise the existing helper's
            // real TC fallback here; root separately qualifies cuBLAS natively.
            dispatch.cublas_gemm = false;
            let mut ctx = resources.view(&arena, config, gpu);
            ctx.dispatch = &dispatch;
            let scalar = gpu.kernel("gemm", "dense_gemm_bf16").unwrap().0;
            let tc = gpu.kernel("gemm_tc", "dense_gemm_tc").unwrap().0;
            let mla = layer.mla.as_ref().unwrap();
            for accelerated in [false, true] {
                for (weight, n, k) in [
                    (&mla.wq_a, 1536u32, 4096u32),
                    (&mla.wq_b, 8192, 1536),
                    (&mla.wkv_a, 512, 4096),
                    (&mla.wo, 4096, 8192),
                ] {
                    let input = gpu.alloc(17 * k as usize * 2).unwrap();
                    let output = gpu.alloc(17 * n as usize * 2).unwrap();
                    gpu.clear();
                    layer
                        .paged_glm_projection(
                            input,
                            weight,
                            output,
                            17,
                            n,
                            k,
                            &ctx,
                            73,
                            accelerated,
                        )
                        .unwrap();
                    let expected = Event::Launch(
                        if accelerated { tc } else { scalar },
                        [n.div_ceil(if accelerated { 64 } else { 16 }), 2, 1],
                        if accelerated {
                            [128, 1, 1]
                        } else {
                            [16, 16, 1]
                        },
                        0,
                        73,
                        vec![
                            Arg::Ptr(input),
                            Arg::Ptr(weight.weight),
                            Arg::Ptr(output),
                            Arg::Bytes(17u32.to_le_bytes().to_vec()),
                            Arg::Bytes(n.to_le_bytes().to_vec()),
                            Arg::Bytes(k.to_le_bytes().to_vec()),
                        ],
                    );
                    assert_eq!(
                        gpu.trace(),
                        vec![expected],
                        "rank={rank} accelerated={accelerated} n={n} k={k}"
                    );
                    gpu.free(output).unwrap();
                    gpu.free(input).unwrap();
                }
            }
            // Even an explicitly requested adapter cannot alter other models.
            let mut foreign = config.clone();
            foreign.model_type = "qwen3_next".into();
            let mut foreign_ctx = resources.view(&arena, &foreign, gpu);
            foreign_ctx.dispatch = &dispatch;
            let input = arena.hidden_states();
            let output = arena.norm_output();
            gpu.clear();
            layer
                .paged_glm_projection(
                    input,
                    &mla.wq_a,
                    output,
                    17,
                    1536,
                    4096,
                    &foreign_ctx,
                    73,
                    true,
                )
                .unwrap();
            assert!(
                matches!(gpu.trace().as_slice(), [Event::Launch(kernel, _, _, _, 73, _)] if *kernel == scalar)
            );
        });
    }
}
