// SPDX-License-Identifier: AGPL-3.0-only
//! Full dense-FFN geometry oracle: resident cache, independent CPU and old CUDA.
use super::*;
use half::bf16;
const FP4: [f32; 16] = [
    0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
];
fn scale(row: usize, group: usize) -> u8 {
    0x28 + ((row % 19 * 5 + group * 3) % 24) as u8
}
fn fp8(code: u8) -> f32 {
    (1. + (code & 7) as f32 / 8.) * 2f32.powi((code >> 3) as i32 - 7)
}
fn coefficient(row: usize, k: usize, global: f32) -> f32 {
    // Match the established W4A16 kernel's multiplication order explicitly.
    bf16::from_f32((FP4[(row % 19 * 3 + k * 7) % 16] * fp8(scale(row, k / 16))) * global).to_f32()
}
fn activation(row: usize, k: usize) -> f32 {
    bf16::from_f32(((row % 17 * 5 + k * 3) % 31) as f32 / 37. - 0.4).to_f32()
}
fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn source(gpu: &dyn GpuBackend, n: usize, k: usize, global: f32) -> Result<QuantizedWeight> {
    let packed: Vec<_> = (0..n * k / 2)
        .map(|i| {
            let row = i / (k / 2);
            let x = i % (k / 2) * 2;
            ((row % 19 * 3 + x * 7) % 16) as u8 | (((row % 19 * 3 + (x + 1) * 7) % 16) as u8) << 4
        })
        .collect();
    let scales: Vec<_> = (0..n * k / 16)
        .map(|i| scale(i / (k / 16), i % (k / 16)))
        .collect();
    Ok(QuantizedWeight {
        weight: upload(gpu, &packed)?,
        weight_scale: upload(gpu, &scales)?,
        weight_scale_2: global,
        ..QuantizedWeight::null()
    })
}
fn read(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<f32>> {
    let mut bytes = vec![0; n * 2];
    gpu.copy_d2h_on_stream(ptr, &mut bytes, stream)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect())
}
fn check(
    gpu: &dyn GpuBackend,
    old: &QuantizedWeight,
    cached: &DenseWeight,
    n: usize,
    k: usize,
) -> Result<()> {
    let stream = gpu.default_stream();
    // 19 rows cover the entire deterministic coefficient pattern. Check exact
    // BF16 coefficients before testing reduction-order differences in GEMM.
    let values = read(gpu, cached.weight, 19 * k, stream)?;
    for (i, &v) in values.iter().enumerate() {
        let expected = coefficient(i / k, i % k, old.weight_scale_2);
        ensure!(
            v.to_bits() == expected.to_bits(),
            "cache coefficient row={} k={}: {v} vs{expected}",
            i / k,
            i % k
        );
    }
    let mut cpu = vec![0f32; 17 * 19];
    for t in 0..17 {
        for y in 0..19 {
            for x in 0..k {
                cpu[t * 19 + y] += activation(t, x) * coefficient(y, x, old.weight_scale_2);
            }
        }
    }
    for m in [9usize, 2048] {
        let act: Vec<_> = (0..m * k)
            .flat_map(|i| {
                bf16::from_f32(activation(i / k, i % k))
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let input = upload(gpu, &act)?;
        let out = gpu.alloc(m * n * 2)?;
        let baseline = gpu.alloc(m * n * 2)?;
        ops::cublas_bf16_proj_dense(
            input,
            cached.weight,
            out,
            m as u32,
            n as u32,
            k as u32,
            stream,
        )?;
        ops::w4a16_gemm(
            gpu,
            gpu.kernel("w4a16", "w4a16_gemm")?,
            input,
            old,
            baseline,
            m as u32,
            n as u32,
            k as u32,
            stream,
        )?;
        let got = read(gpu, out, m * n, stream)?;
        let reference = read(gpu, baseline, m * n, stream)?;
        let mut max_error = 0f32;
        for i in 0..m * n {
            let expected = bf16::from_f32(cpu[(i / n % 17) * 19 + i % n % 19]).to_f32();
            let ulp = if expected == 0. {
                f32::MIN_POSITIVE
            } else {
                2f32.powi(expected.abs().log2().floor() as i32 - 7)
            };
            let tolerance = 2. * ulp + 1e-5;
            ensure!(
                got[i].is_finite() && (got[i] - expected).abs() <= tolerance,
                "CPU GEMM M={m} N={n} K={k} index={i}: {} vs{expected}",
                got[i]
            );
            ensure!(
                reference[i].is_finite() && (got[i] - reference[i]).abs() <= tolerance,
                "W4A16 GEMM M={m} N={n} K={k} index={i}: {} vs{}",
                got[i],
                reference[i]
            );
            max_error = max_error.max((got[i] - reference[i]).abs());
        }
        println!("PASS dense BF16 M={m} N={n} K={k} max_w4a16_error={max_error}");
        for ptr in [input, out, baseline] {
            gpu.free(ptr)?;
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires idle GB10, no checkpoint; parent coordinates GPU execution"]
fn dense_prefill_bf16_native_full_projection_geometry() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.hidden_size = 4096;
    config.intermediate_size = 12288;
    config.num_hidden_layers = 45;
    config.mlp_only_layers = vec![0, 1, 2];
    let mut layer = DenseFfnLayer::new(
        DenseFfnWeights {
            gate_proj: source(&gpu, 12288, 4096, 0.0031415926)?,
            up_proj: source(&gpu, 12288, 4096, 0.001271)?,
            down_proj: source(&gpu, 4096, 12288, 0.0085731)?,
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        &gpu,
    )?;
    layer.cache_glm_prefill_bf16(&config, &gpu)?;
    ensure!(
        layer.bf16_weights.is_none(),
        "cache changed generic BF16 decode policy"
    );
    let cached = layer.prefill_bf16_weights.as_ref().unwrap();
    check(
        &gpu,
        &layer.weights.gate_proj,
        &cached.gate_proj,
        12288,
        4096,
    )?;
    check(&gpu, &layer.weights.up_proj, &cached.up_proj, 12288, 4096)?;
    check(
        &gpu,
        &layer.weights.down_proj,
        &cached.down_proj,
        4096,
        12288,
    )?;
    Ok(())
}
