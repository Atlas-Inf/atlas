// SPDX-License-Identifier: AGPL-3.0-only

//! Producer/consumer coverage for the serving launch attributes. The scalar
//! projection tests synchronize each readback; here only the reference does.
//! Repeated residual FFNs reuse their scratch between dependent launches.

use super::*;

#[test]
#[ignore = "requires GB10 GLM kernels, ATLAS_PDL=1 and ATLAS_GLM_DECODE_GEMV_BATCH=1"]
fn canonical_prefill_chain_needs_no_host_barriers() -> Result<()> {
    ensure!(enabled(), "requires ATLAS_W4A16_TC=1 and canonical verify");
    let gpu = backend()?;
    ensure!(
        spark_runtime::cuda_backend::pdl_enabled(),
        "requires ATLAS_PDL=1"
    );
    ensure!(
        ops::gemv_touch(ops::Twin::W4Tc8).is_some(),
        "requires touch twins"
    );
    W4a16BatchmTiers::resolve(gpu, true);
    let gpu: &dyn GpuBackend = gpu;
    let (m, h, inter) = (26u32, 4096u32, 1024u32);
    let mut rng = Rng(0xcafe_26ab_4455_9911);
    let gate_w = w4(gpu, &mut rng, inter, h)?;
    let up_w = w4(gpu, &mut rng, inter, h)?;
    let down_w = w4(gpu, &mut rng, h, inter)?;
    let input = rows_bytes(&(0..m).map(|_| random_row(&mut rng, h)).collect::<Vec<_>>());
    let norm_w = DenseWeight {
        weight: upload(gpu, &rows_bytes(&[vec![bf16(1.0); h as usize]]))?,
    };
    let hs = (m * h) as usize * 2;
    let is = (m * inter) as usize * 2;
    let hidden = gpu.alloc(hs)?;
    let normed = gpu.alloc(hs)?;
    let gate = gpu.alloc(is)?;
    let up = gpu.alloc(is)?;
    let output = gpu.alloc(hs)?;
    // Resolve once, before either chain: module loading cannot act as an
    // accidental synchronization inside the asynchronous variant.
    let norm_k = gpu.kernel("norm", "rms_norm")?;
    let silu_k = gpu.kernel("moe_silu_mul", "moe_silu_mul")?;
    let add_k = gpu.kernel("residual_add", "bf16_residual_add")?;
    let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(m);
    let stream = gpu.default_stream();
    let run = |barriers: bool, touch: bool| -> Result<Vec<u16>> {
        for (p, bytes) in [(normed, hs), (gate, is), (up, is), (output, hs)] {
            gpu.memset(p, 0xff, bytes)?;
        }
        gpu.copy_h2d(&input, hidden)?;
        let barrier = || {
            if barriers {
                gpu.synchronize(stream)
            } else {
                Ok(())
            }
        };
        let project = |a, w: &W4, c| {
            if touch {
                w4a16(gpu, a, &w.q, c, m, w.n, w.k, stream)
            } else {
                // Dense FFN prefill uses the plain tc8 sweep; KDA verify
                // uses the canonical dispatch's pre-wait weight touch.
                ops::w4a16_gemv_batchm(gpu, tc, a, &w.q, c, m, w.n, w.k, stream)
            }
        };
        for _ in 0..48 {
            ops::rms_norm(gpu, norm_k, hidden, &norm_w, normed, m, h, 1e-5, stream)?;
            barrier()?;
            project(normed, &gate_w, gate)?;
            barrier()?;
            project(normed, &up_w, up)?;
            barrier()?;
            ops::silu_mul(gpu, silu_k, gate, up, gate, m * inter, stream)?;
            barrier()?;
            project(gate, &down_w, output)?;
            barrier()?;
            ops::residual_add(gpu, add_k, hidden, output, m * h, stream)?;
            barrier()?;
        }
        gpu.synchronize(stream)?;
        download(gpu, hidden, (m * h) as usize)
    };
    for touch in [false, true] {
        let reference = run(true, touch)?;
        for trial in 0..8 {
            let actual = run(false, touch)?;
            ensure!(
                actual == reference,
                "26-row FFN chain differs without host barriers: touch={touch} trial={trial}"
            );
        }
    }
    for p in [hidden, normed, gate, up, output, norm_w.weight] {
        gpu.free(p)?;
    }
    for w in [gate_w, up_w, down_w] {
        gpu.free(w.q.weight)?;
        gpu.free(w.q.weight_scale)?;
    }
    Ok(())
}
