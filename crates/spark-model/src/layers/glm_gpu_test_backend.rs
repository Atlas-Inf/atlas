// SPDX-License-Identifier: AGPL-3.0-only

//! Exact GLM target and launch policy for the native RC2 GPU tests.
//! Run each filter in a fresh process. `ATLAS_PDL=1` exercises the same
//! dependent-launch attributes as serving; add `ATLAS_GLM_DECODE_GEMV_BATCH=1`
//! to exercise the canonical projection dispatch through its touch twins.

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::{AtlasCudaBackend, configure_pdl, pdl_enabled};
use spark_runtime::gpu::GpuBackend;
use std::sync::OnceLock;

pub(crate) fn backend() -> Result<&'static AtlasCudaBackend> {
    static GPU: OnceLock<Result<AtlasCudaBackend, String>> = OnceLock::new();
    let gpu = GPU.get_or_init(|| {
        // This must precede every kernel resolution: resolving a function
        // registers its PDL attribute only after the target has opted in.
        configure_pdl("glm-5.3-flash");
        let set = atlas_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
            .ok_or_else(|| "glm-5.3-flash/nvfp4 absent from this build".to_owned())?;
        AtlasCudaBackend::new(0, &set.modules).map_err(|e| format!("CUDA backend: {e:#}"))
    });
    let gpu = gpu.as_ref().map_err(|e| anyhow::anyhow!(e.clone()))?;
    gpu.bind_to_thread()?;
    let requested = std::env::var("ATLAS_PDL").as_deref() == Ok("1");
    ensure!(
        pdl_enabled() == requested,
        "GLM GPU test PDL policy was latched before target configuration; use a fresh process"
    );
    let touch = std::env::var("ATLAS_GLM_DECODE_GEMV_BATCH").as_deref() == Ok("1");
    if touch {
        ensure!(requested, "GLM touch-twin tests require ATLAS_PDL=1");
        super::ops::gemv_touch_resolve(gpu);
        for twin in [
            super::ops::Twin::W4Tc8,
            super::ops::Twin::W4Tc16,
            super::ops::Twin::W4Tc32,
            super::ops::Twin::W4Tc8Pair,
            super::ops::Twin::W4Tc16Pair,
            super::ops::Twin::W4Tc32Pair,
        ] {
            ensure!(
                super::ops::gemv_touch(twin).is_some(),
                "GLM touch twin {twin:?} unresolved"
            );
        }
    }
    eprintln!(
        "GLM GPU launch mode: PDL={} canonical_touch={touch}",
        pdl_enabled()
    );
    Ok(gpu)
}

/// Numeric comparisons alone are not a finiteness gate: identical NaN bits
/// compare equal, and `f64::max` ignores a NaN error. Check the complete GPU
/// output before bitwise or reference comparisons, including non-probe rows.
pub(super) fn finite_bf16(label: &str, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len().is_multiple_of(2),
        "{label}: incomplete BF16 word"
    );
    finite(
        label,
        bytes
            .chunks_exact(2)
            .map(|word| f32::from_bits((u16::from_le_bytes([word[0], word[1]]) as u32) << 16)),
    )
}

pub(super) fn finite_f32(label: &str, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len().is_multiple_of(4),
        "{label}: incomplete FP32 word"
    );
    finite(
        label,
        bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]])),
    )
}

fn finite(label: &str, values: impl Iterator<Item = f32>) -> Result<()> {
    for (i, value) in values.enumerate() {
        ensure!(
            value.is_finite(),
            "{label}: nonfinite output at element {i}: {value}"
        );
    }
    Ok(())
}

#[test]
fn numeric_gates_reject_matching_nan_and_infinity() {
    for bits in [0x7fc0u16, 0xffc0, 0x7f80, 0xff80] {
        let bytes = [0u8, 0, bits as u8, (bits >> 8) as u8];
        assert!(finite_bf16("BF16 fixture", &bytes).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bytes: Vec<_> = [0.0f32, value]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(finite_f32("FP32 fixture", &bytes).is_err());
    }
    assert!(finite_bf16("finite BF16", &[0, 0, 0x80, 0x3f, 0x80, 0xbf]).is_ok());
    assert!(finite_f32("finite FP32", &1.0f32.to_le_bytes()).is_ok());
    assert!(finite_bf16("truncated BF16", &[0]).is_err());
    assert!(finite_f32("truncated FP32", &[0]).is_err());
}
