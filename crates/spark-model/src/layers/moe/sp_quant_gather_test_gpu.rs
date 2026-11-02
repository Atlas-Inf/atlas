// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): the row-local steps `route_sp_quantized` runs on one
//! rank's rows -- the router GEMM on `ops::bf16_gemm_f32out`'s CUTLASS tiles
//! (FP32 logits), the sigmoid top-k and the NVFP4 quantization -- leave, on
//! each quarter of a chunk, that quarter's bytes of the whole chunk's run.
//!
//!   cargo test --release -p spark-model --lib sp_quant_gather -- --ignored --nocapture

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

const H: u32 = 4096;
const E: u32 = 288;
const TOPK: u32 = 8;

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// Pseudo-random BF16 values in [-scale, scale).
fn values(seed: u64, n: usize, scale: f32) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let unit = (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
            bf16(unit * scale).to_le_bytes()
        })
        .collect()
}

struct Outs {
    logits: DevicePtr,
    ids: DevicePtr,
    weights: DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
}

impl Outs {
    fn new(gpu: &dyn GpuBackend, rows: u32) -> Result<Self> {
        let n = rows as usize;
        let poison = |bytes: usize| upload(gpu, &vec![0xa5; bytes]);
        Ok(Self {
            logits: poison(n * E as usize * 4)?,
            ids: poison(n * TOPK as usize * 4)?,
            weights: poison(n * TOPK as usize * 4)?,
            packed: poison(n * H as usize / 2)?,
            scales: poison(n * H as usize / 16)?,
        })
    }

    fn bytes(&self, gpu: &dyn GpuBackend, rows: u32) -> Result<[Vec<u8>; 5]> {
        let n = rows as usize;
        Ok([
            read(gpu, self.logits, n * E as usize * 4)?,
            read(gpu, self.ids, n * TOPK as usize * 4)?,
            read(gpu, self.weights, n * TOPK as usize * 4)?,
            read(gpu, self.packed, n * H as usize / 2)?,
            read(gpu, self.scales, n * H as usize / 16)?,
        ])
    }
}

#[allow(clippy::too_many_arguments)]
fn route(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    gate: DevicePtr,
    bias: DevicePtr,
    out: &Outs,
    row0: u32,
    rows: u32,
    stream: u64,
) -> Result<()> {
    let at = |p: DevicePtr, row: u32| p.offset((row0 * row) as usize);
    let logits = at(out.logits, E * 4);
    ops::bf16_gemm_f32out(at(input, H * 2), gate.0, logits, rows, E, H, stream)?;
    let topk = gpu.kernel("moe_topk_sig", "moe_topk_sigmoid_batched_f32")?;
    ops::moe_topk_sigmoid_batched(
        gpu,
        topk,
        logits,
        bias,
        at(out.ids, TOPK * 4),
        at(out.weights, TOPK * 4),
        E,
        TOPK,
        true,
        2.5,
        rows,
        stream,
    )?;
    let quant = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    ops::quantize_bf16_to_nvfp4(
        gpu,
        quant,
        at(input, H * 2),
        at(out.packed, H / 2),
        at(out.scales, H / 16),
        rows,
        H,
        stream,
    )
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn quarter_routes_are_the_whole_chunks_bytes() -> Result<()> {
    // SAFETY: set before any reader of the switch, in this test only.
    unsafe { std::env::set_var("ATLAS_BF16_GEMM_CUTLASS", "1") };
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let gate = upload(&gpu, &values(0x9a7e, (E * H) as usize, 0.03))?;
    let bias = upload(
        &gpu,
        &(0..E)
            .flat_map(|e| ((e % 7) as f32 * 1e-3).to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    for (total, seed) in [(8192u32, 1u64), (8196, 2), (16384, 3)] {
        let input = upload(&gpu, &values(seed, (total * H) as usize, 2.0))?;
        let (whole, quarters) = (Outs::new(&gpu, total)?, Outs::new(&gpu, total)?);
        route(&gpu, input, gate, bias, &whole, 0, total, stream)?;
        let rows = total / 4;
        for part in 0..4 {
            route(
                &gpu,
                input,
                gate,
                bias,
                &quarters,
                part * rows,
                rows,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        let covered = rows * 4;
        let (want, got) = (whole.bytes(&gpu, covered)?, quarters.bytes(&gpu, covered)?);
        for (name, (w, g)) in ["logits", "ids", "weights", "packed", "scales"]
            .iter()
            .zip(want.iter().zip(&got))
        {
            ensure!(w == g, "{total} rows: {name} of the quarters differ");
        }
    }
    println!("router, top-k and NVFP4 rows: quarters == whole at 8192, 8196, 16384");
    Ok(())
}
