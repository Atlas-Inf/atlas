// SPDX-License-Identifier: AGPL-3.0-only
// Portions of this file are derived from ExLlamaV3 (MIT, see NOTICE.md).

//! The typed handle for one EXL3 linear: the four store tensors reduced to
//! device pointers + a validated `Exl3Shape`.
//!
//! A loader (M3) builds an `Exl3Weight` from `<p>.trellis` / `<p>.suh` /
//! `<p>.svh` / `<p>.mul1` and hands it to the reconstruct + GEMM path. Like
//! `PackedQ2Weight`, the buffers are owned by the `WeightStore`, so this struct
//! only borrows the pointers (no free on drop).

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::Exl3Shape;

/// The mul1 codebook tag stored in `<p>.mul1` (exllamav3 codebook.cuh `0x83DCD12D`).
pub const MUL1_TAG: u32 = 0x83DC_D12D;

/// EXL3 trellis weight (mul1 codebook): packed 16x16 tiles + fp16 suh/svh.
#[derive(Debug, Clone, Copy)]
pub struct Exl3Weight {
    /// int16 `[in/16, out/16, 16*bits]` trellis words.
    pub trellis: DevicePtr,
    /// fp16 `[in]` sign-and-scale vector over the input features.
    pub suh: DevicePtr,
    /// fp16 `[out]` sign-and-scale vector over the output features.
    pub svh: DevicePtr,
    /// Feature counts and codebook K, derived from the trellis dims.
    pub shape: Exl3Shape,
}

impl Exl3Weight {
    /// Null weight (all pointers NULL). Used for unset placeholders.
    pub fn null() -> Self {
        Self {
            trellis: DevicePtr::NULL,
            suh: DevicePtr::NULL,
            svh: DevicePtr::NULL,
            shape: Exl3Shape {
                in_features: 16,
                out_features: 16,
                bits: 1,
            },
        }
    }

    /// True if the backing trellis buffer is NULL (unset placeholder).
    pub fn is_null(&self) -> bool {
        self.trellis == DevicePtr::NULL
    }
}

/// Build the `Exl3Weight` for linear `prefix` from the store's four tensors.
///
/// Validates dtypes, the trellis dims and the scale lengths, and reads the
/// `mul1` tag back to the host: a value other than `MUL1_TAG` selects a
/// codebook Atlas does not implement, so it is an error, not a guess.
pub fn exl3_from_store(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<Exl3Weight> {
    let name = |suffix: &str| format!("{prefix}.{suffix}");
    let fetch = |suffix: &str| -> Result<(String, &spark_runtime::weights::WeightTensor)> {
        let full = name(suffix);
        let t = store.get(&full).with_context(|| format!("EXL3: {full}"))?;
        Ok((full, t))
    };

    let (trellis_name, trellis) = fetch("trellis")?;
    let (suh_name, suh) = fetch("suh")?;
    let (svh_name, svh) = fetch("svh")?;
    let (mul1_name, mul1) = fetch("mul1")?;

    let expect_dtype =
        |t: &spark_runtime::weights::WeightTensor, full: &str, want: WeightDtype| -> Result<()> {
            anyhow::ensure!(
                t.dtype == want,
                "EXL3: {full} is {:?}, expected {want:?}",
                t.dtype
            );
            Ok(())
        };
    expect_dtype(trellis, &trellis_name, WeightDtype::Int16)?;
    expect_dtype(suh, &suh_name, WeightDtype::FP16)?;
    expect_dtype(svh, &svh_name, WeightDtype::FP16)?;
    expect_dtype(mul1, &mul1_name, WeightDtype::Int32)?;

    let shape = Exl3Shape::from_trellis_dims(&trellis.shape)
        .with_context(|| format!("EXL3: {trellis_name}"))?;
    shape.validate()?;
    shape.check_scales(suh.num_elements(), svh.num_elements())?;

    anyhow::ensure!(
        mul1.num_elements() == 1,
        "EXL3: {mul1_name} has {} elements, expected 1 (the codebook tag)",
        mul1.num_elements()
    );
    let mut buf = [0u8; 4];
    gpu.copy_d2h(mul1.ptr, &mut buf)
        .with_context(|| format!("EXL3: reading {mul1_name} back to host"))?;
    let tag = u32::from_le_bytes(buf);
    anyhow::ensure!(
        tag == MUL1_TAG,
        "EXL3: {mul1_name} is 0x{tag:08X}, expected 0x{MUL1_TAG:08X} (the mul1 codebook) — \
         Atlas implements only the mul1 codebook"
    );

    Ok(Exl3Weight {
        trellis: trellis.ptr,
        suh: suh.ptr,
        svh: svh.ptr,
        shape,
    })
}
