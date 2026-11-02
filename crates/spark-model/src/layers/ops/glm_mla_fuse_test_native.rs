// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

/// Native E4M3 bytes and independently varied non-power-of-two FP32 block
/// scales. Applying a BF16-rounded scale before the MMA must fail parity.
pub(super) fn weight(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    n: u32,
) -> Result<(DevicePtr, DevicePtr)> {
    let values: Vec<u8> = (0..n * H).map(|_| rng.next() as u8 & 0xbf).collect();
    let scales: Vec<u8> = (0..n.div_ceil(128) * (H / 128))
        .flat_map(|_| (0.003 + 0.009 * (rng.unit() + 1.0)).to_le_bytes())
        .collect();
    Ok((upload(gpu, &values)?, upload(gpu, &scales)?))
}

#[test]
#[ignore = "needs a GB10 and the glm-5.3-flash kernels"]
fn native_fp8_fused_front_is_byte_exact() -> Result<()> {
    compare_front(true)
}

#[test]
#[ignore = "needs a GB10 and the glm-5.3-flash kernels"]
fn native_fp8_raw_projections_are_byte_exact_with_and_without_touch() -> Result<()> {
    let gpu = backend()?;
    let f = MlaFuse::load(gpu).context("glm_mla_fuse kernels")?;
    let s = gpu.default_stream();
    let mut rng = Rng(0x6d6c_6166_7038_0001);
    let w = Weights::new(gpu, &mut rng, true)?;
    for rows in 1..=32 {
        let (a, b) = (Run::new(gpu, rows)?, Run::new(gpu, rows)?);
        let input = upload(gpu, &bf16_bytes((rows * H) as usize, &mut rng, 1.0, 0.0))?;
        let tier = if rows <= 8 {
            8
        } else if rows <= 16 {
            16
        } else {
            32
        };
        let k = kernel(gpu, "mxfp8_gemv", &format!("fp8b_gemv_tc{tier}"))?;
        gemv(gpu, k, input, w.q_a.0, w.q_a.1, a.q, rows, QL, H, QL, s)?;
        gemv(
            gpu, k, input, w.kv_a.0, w.kv_a.1, a.kv, rows, KVL, H, KVL, s,
        )?;
        let proj = |(data, scales), out, n| MxProj {
            data,
            scales,
            out,
            n,
        };
        for touch in [None, Some((12 << 20, 32))] {
            f.launch_down(
                gpu,
                &DownArgs {
                    fp8_block: true,
                    input,
                    rows,
                    k: H,
                    wk: w.wk,
                    gate: w.gate,
                    keys: b.keys,
                    gates: b.gates,
                    ni: ID,
                    q_a: proj(w.q_a, b.q, QL),
                    kv_a: proj(w.kv_a, b.kv, KVL),
                    touch,
                },
                s,
            )?;
            gpu.synchronize(s)?;
            for (name, want, got, n) in [("q", a.q, b.q, QL), ("kv", a.kv, b.kv, KVL)] {
                let bytes = (rows * n * 2) as usize;
                let (want, got) = (raw(gpu, want, bytes)?, raw(gpu, got, bytes)?);
                let first = want.iter().zip(&got).position(|(a, b)| a != b);
                anyhow::ensure!(
                    first.is_none(),
                    "native raw {name}: rows {rows}, touch {touch:?}, byte {first:?}"
                );
            }
        }
        a.free(gpu)?;
        b.free(gpu)?;
        gpu.free(input)?;
    }
    Ok(())
}

// Match fp8b_gemv_tc*'s ten-argument ABI, including disabled touch geometry.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(stride)
        .arg_u32(0)
        .arg_u32(1)
        .launch(stream)
}
