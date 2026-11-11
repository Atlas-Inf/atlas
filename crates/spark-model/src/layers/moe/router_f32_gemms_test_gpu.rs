// SPDX-License-Identifier: AGPL-3.0-only

//! GPU launch matrix shared by the FP32 router reference tests.

use super::*;

/// Every FP32 router GEMM that serves `rows` rows of `x` into fresh logits.
pub(super) fn gemms(
    gpu: &dyn GpuBackend,
    router: &Router,
    x: DevicePtr,
    rows: u32,
    mut each: impl FnMut(&str, DevicePtr) -> Result<()>,
) -> Result<()> {
    let s = gpu.default_stream();
    let gate = DenseWeight {
        weight: router.gate,
    };
    let out = gpu.alloc((rows * E) as usize * 4)?;
    let mut run = |name: &str, launch: &dyn Fn() -> Result<()>| -> Result<()> {
        gpu.memset(out, 0xa5, (rows * E) as usize * 4)?;
        launch()?;
        each(name, out)
    };
    let gemm = |module: &str, name: &str| gpu.kernel(module, name);
    if rows == 1 {
        let k = gemm("gemv", "dense_gemv_bf16_fp32out")?;
        run("dense_gemv_bf16_fp32out", &|| {
            ops::dense_gemv(gpu, k, x, &gate, out, E, H, s)
        })?;
    }
    if rows <= 8 {
        for name in [
            "dense_gemv_bf16_batchm_f32",
            "dense_gemv_bf16_batchm_ahead_f32",
        ] {
            let k = gemm("dense_gemv_bf16_batchm", name)?;
            run(name, &|| {
                ops::dense_gemv_batchm(gpu, k, x, &gate, out, rows, E, H, E, s)
            })?;
        }
    }
    if rows <= crate::layers::canonical_verify::MAX_ROWS {
        run("canonical dense_f32", &|| {
            crate::layers::canonical_verify::dense_f32(gpu, x, &gate, out, rows, E, H, E, s)
        })?;
    }
    let k = gemm("gemm", "dense_gemm_bf16_f32out")?;
    run("dense_gemm_bf16_f32out", &|| {
        ops::dense_gemm(gpu, k, x, &gate, out, rows, E, H, s)
    })?;
    let k = gemm("gemm", "dense_gemm_bf16_router_f32out")?;
    run("dense_gemm_bf16_router_f32out", &|| {
        ops::dense_gemm_router(gpu, k, x, &gate, out, rows, E, H, s)
    })?;
    if rows <= 32 {
        let k = gemm("gemm", "dense_gemm_bf16_router_rows")?;
        run("dense_gemm_bf16_router_rows", &|| {
            ops::dense_gemm_router_rows(gpu, k, x, &gate, out, rows, E, H, s)
        })?;
    }
    if rows == 5 {
        let k = gemm("gemm", "dense_gemm_bf16_router_m5")?;
        run("dense_gemm_bf16_router_m5", &|| {
            ops::dense_gemm_router_m5(gpu, k, x, &gate, out, rows, E, H, s)
        })?;
        let k = gemm("glm_router_bn4", "glm_router_m5_bn4")?;
        run("glm_router_m5_bn4", &|| {
            ops::glm_router_bn4(gpu, k, x, router.gate, out, s)
        })?;
    }
    let k = gemm("gemm", "dense_gemm_bf16_router_bn32")?;
    run("dense_gemm_bf16_router_bn32", &|| {
        KernelLaunch::new(gpu, k)
            .grid([9, rows.div_ceil(16), 1])
            .block([8, 16, 1])
            .arg_ptr(x)
            .arg_ptr(router.gate)
            .arg_ptr(out)
            .arg_u32(rows)
            .arg_u32(E)
            .arg_u32(H)
            .launch(s)
    })?;
    run("cuBLASLt f32out", &|| {
        spark_runtime::cublaslt::bf16_gemm_act_weight_t_f32out(
            x.0,
            router.gate.0,
            out.0,
            rows,
            E,
            H,
            s,
        )
    })?;
    run("CUTLASS f32out", &|| {
        spark_runtime::cutlass::bf16_gemm_tuned_f32out(
            x.0,
            router.gate.0,
            out.0,
            rows,
            E,
            H,
            H,
            E,
            9,
            s,
        )
    })?;
    gpu.free(out)?;
    Ok(())
}
