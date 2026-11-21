// SPDX-License-Identifier: AGPL-3.0-only

//! Debug captures for comparing visits with the transformers reference
//! (off unless their environment variables are set).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::batch::Batch;
use super::layer::Inkling2Layer;

impl Inkling2Layer {
    pub(super) fn debug_taps(
        &self,
        gpu: &dyn GpuBackend,
        batch: &Batch<'_>,
        stream: u64,
    ) -> Result<()> {
        let (s, d, p) = (&self.shared, &self.dims, &self.p);
        let h = d.hidden;
        let slot = batch.rows() * h * 4; // between the q, k, v (, mlp) slots of res3 / msg4
        let taps: [(u32, DevicePtr, usize); 20] = [
            (1, s.res3, h * 4),
            (2, s.normed3, h * 2),
            (3, s.qn, d.q_dim * 2),
            (4, s.kn, d.kv_dim * 2),
            (5, s.vn, d.kv_dim * 2),
            (6, s.rn, d.rel_dim * 2),
            (7, s.sink, p.heads * 2),
            (8, s.attn, d.q_dim * 2),
            (9, s.proj, h * 2),
            (10, s.conv_out, h * 2),
            (11, s.resid, h * 4),
            (12, s.mlp_in, h * 4),
            (13, s.mlp_normed, h * 2),
            (14, s.ids, d.top_k * 4),
            (16, s.moe_out, h * 2),
            (17, s.mlp_conv, h * 2),
            // loop passes: the q and k messages and the gated q and k inputs
            (18, s.msg4, h * 4),
            (19, s.msg4.offset(slot), h * 4),
            (20, s.res3, h * 4),
            (21, s.res3.offset(slot), h * 4),
        ];
        for r in 0..batch.rows() {
            for (tag, ptr, row_bytes) in taps {
                tap(
                    gpu,
                    ptr.offset(r * row_bytes),
                    row_bytes,
                    tag,
                    batch.position(r),
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

/// Debug: with `ATLAS_INKLING2_DUMP=<file>`, append `(visit u32, position u32,
/// BF16 hidden row)` records (visit `u32::MAX` = the normed embedding), for
/// comparison with the transformers reference's per-visit hidden states.
pub(super) fn dump(
    gpu: &dyn GpuBackend,
    hidden: DevicePtr,
    batch: &Batch<'_>,
    h: usize,
    visit: u32,
    stream: u64,
) -> Result<()> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var("ATLAS_INKLING2_DUMP").ok()) else {
        return Ok(());
    };
    use std::io::Write;
    gpu.synchronize(stream)?;
    let mut all = vec![0u8; batch.rows() * h * 2];
    gpu.copy_d2h(hidden, &mut all)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for (r, row) in all.chunks_exact(h * 2).enumerate() {
        f.write_all(&visit.to_le_bytes())?;
        f.write_all(&(batch.position(r) as u32).to_le_bytes())?;
        f.write_all(row)?;
    }
    Ok(())
}

/// Debug: the visit whose intermediates `ATLAS_INKLING2_TAP=<file>` records.
/// `dump`'s visit number for the model output (after the final block residual).
pub(super) const FINAL: u32 = 1000;

/// Whether a debug capture is on (each one synchronizes, so graphs stay off).
pub(super) fn enabled() -> bool {
    std::env::var_os("ATLAS_INKLING2_DUMP").is_some() || tap_visit().is_some()
}

pub(super) fn tap_visit() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_INKLING2_TAP")
            .ok()
            .and(std::env::var("ATLAS_INKLING2_TAP_VISIT").ok())
            .and_then(|v| v.parse().ok())
    })
}

/// Append `(tag u32, position u32, len u32, raw bytes)` to `ATLAS_INKLING2_TAP`.
fn tap(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    bytes: usize,
    tag: u32,
    pos: usize,
    stream: u64,
) -> Result<()> {
    use std::io::Write;
    let path = std::env::var("ATLAS_INKLING2_TAP")?;
    gpu.synchronize(stream)?;
    let mut buf = vec![0u8; bytes];
    gpu.copy_d2h(ptr, &mut buf)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(&tag.to_le_bytes())?;
    f.write_all(&(pos as u32).to_le_bytes())?;
    f.write_all(&(bytes as u32).to_le_bytes())?;
    f.write_all(&buf)?;
    Ok(())
}
