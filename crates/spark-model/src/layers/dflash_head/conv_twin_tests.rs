// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_CONV_TWIN`: the switch's parser, and (GPU, ignored) the
//! delta GEMM `conv_delta_gemm` runs from each twin at the GLM-5.3 conv shape
//! (1024 x 4096) against an FP64 reference, next to the BF16 weight it
//! replaces. The twins change drafts, never outputs; this checks that each
//! twin is served (not a silent BF16 fallback) and how far it moves the
//! deltas. Acceptance itself needs the pair.
//!
//!   ATLAS_W4A16_TC=1 cargo test --release -p spark-model --lib conv_twin -- --ignored --nocapture

use anyhow::{Context, Result};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::conv::{ConvTwin, Dflash2Conv};
use super::super::free_state_tests::zero_head;
use super::ConvTwinKind;
use crate::weight_map::DenseWeight;

#[test]
fn conv_twin_switch_parses() {
    assert_eq!(ConvTwinKind::parse(None).unwrap(), None);
    assert_eq!(ConvTwinKind::parse(Some("")).unwrap(), None);
    assert_eq!(ConvTwinKind::parse(Some("0")).unwrap(), None);
    assert_eq!(
        ConvTwinKind::parse(Some("nvfp4")).unwrap(),
        Some(ConvTwinKind::Nvfp4)
    );
    assert_eq!(
        ConvTwinKind::parse(Some(" mxfp8 ")).unwrap(),
        Some(ConvTwinKind::Mxfp8)
    );
    assert!(ConvTwinKind::parse(Some("1")).is_err());
    assert!(ConvTwinKind::parse(Some("fp4")).is_err());
    let words: Vec<u64> = [None, Some(ConvTwinKind::Nvfp4), Some(ConvTwinKind::Mxfp8)]
        .map(ConvTwinKind::word)
        .to_vec();
    assert_eq!(words, [0, 1, 2]);
}

struct Rng(u64);
impl Rng {
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

fn from_bf16(h: u16) -> f64 {
    half::bf16::from_bits(h).to_f64()
}

fn upload(gpu: &dyn GpuBackend, h: &[u16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

fn download(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// RMS error of `got` against `want`, relative to `want`'s RMS.
fn rel_rms(got: &[u16], want: &[f64]) -> f64 {
    let (mut e2, mut r2) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        e2 += (from_bf16(g) - w).powi(2);
        r2 += w * w;
    }
    (e2 / r2).sqrt()
}

#[test]
#[ignore = "requires a GB10, a glm-5.3-flash kernel build and ATLAS_W4A16_TC=1"]
fn conv_twin_deltas_at_the_glm_shape() -> Result<()> {
    let set = atlas_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash/nvfp4 not in this build")?;
    let gpu = AtlasCudaBackend::new(0, &set.modules).context("CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu, true);
    anyhow::ensure!(
        crate::layers::w4a16_gemv_tiers::tc_kernel(8).0 != 0,
        "run with ATLAS_W4A16_TC=1"
    );

    let mut head = zero_head();
    head.hidden_size = 4096;
    head.kernels.small_m_gemv = true;
    head.kernels.dense_gemv_batchm =
        gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    [head.kernels.dense_gemv_tc16, head.kernels.dense_gemv_tc32] =
        super::super::from_weights_kernels::dense_gemv_tc_kernels(gpu, true);
    head.kernels.mxfp8_gemv = super::super::from_weights_kernels::mxfp8_gemv_kernels(gpu, true);
    head.kernels.mxfp8_quantize = gpu.kernel("mxfp8_gemv", "mxfp8_quantize_bf16")?;

    // GLM-5.3's DFlash2 conv: 16-wide groups, two taps, input and output
    // coefficients: [2 * 2 * 256, 4096].
    let (h, group, taps) = (4096usize, 16usize, 2usize);
    let mut rng = Rng(0x5eed_c0de_2026_1011);
    let mut conv = Dflash2Conv::new(
        DenseWeight {
            weight: DevicePtr::NULL,
        },
        DenseWeight {
            weight: DevicePtr::NULL,
        },
        h,
        group,
        taps,
    );
    let n = 2 * conv.kernel_size * conv.num_groups;
    assert_eq!(n, 1024);
    let w_host: Vec<u16> = (0..n * h).map(|_| bf16(rng.unit() * 0.03)).collect();
    conv.kernel_projection.weight = upload(gpu, &w_host)?;
    let w = conv.kernel_projection;
    let stream = gpu.default_stream();

    // A single proposal's gamma = 8 rows, and a four-sequence batch's 32.
    for m in [8usize, 32] {
        let rows: Vec<u16> = (0..m * h).map(|_| bf16(rng.unit() * 2.0)).collect();
        let want: Vec<f64> = (0..m * n)
            .map(|i| {
                let (r, j) = (i / n, i % n);
                (0..h)
                    .map(|c| from_bf16(rows[r * h + c]) * from_bf16(w_host[j * h + c]))
                    .sum()
            })
            .collect();
        let input = upload(gpu, &rows)?;
        let out = gpu.alloc(m * n * 2)?;
        let run = |conv: &mut Dflash2Conv, twin: Option<ConvTwin>| -> Result<Vec<u16>> {
            conv.twin = twin;
            gpu.memset(out, 0xff, m * n * 2)?;
            head.conv_delta_gemm(
                gpu, conv, input, &w, out, m as u32, n as u32, h as u32, stream,
            )?;
            gpu.synchronize(stream)?;
            download(gpu, out, m * n)
        };
        let bf16_out = run(&mut conv, None)?;
        let base = rel_rms(&bf16_out, &want);
        println!("CONV TWIN m={m} bf16  rel-RMS-err {base:.3e}");
        assert!(base < 1e-2, "BF16 delta GEMM off: {base}");
        for (kind, limit) in [(ConvTwinKind::Nvfp4, 0.15), (ConvTwinKind::Mxfp8, 0.05)] {
            let twin = head.conv_twin(gpu, &conv, kind)?;
            gpu.synchronize(stream)?;
            let got = run(&mut conv, Some(twin))?;
            let err = rel_rms(&got, &want);
            let moved = got.iter().zip(&bf16_out).filter(|(a, b)| a != b).count();
            println!(
                "CONV TWIN m={m} {kind:?} rel-RMS-err {err:.3e} ({moved} of {} deltas differ from BF16)",
                m * n
            );
            assert!(err < limit, "{kind:?} twin off at m={m}: {err}");
            assert!(moved > 0, "{kind:?} twin was not served at m={m}");
        }
        gpu.free(input)?;
        gpu.free(out)?;
    }
    Ok(())
}
