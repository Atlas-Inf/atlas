// SPDX-License-Identifier: AGPL-3.0-only

//! Debug captures for comparing visits with the transformers reference
//! (off unless their environment variables are set).

use anyhow::{Context, Result};
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

/// Debug captures, one switch: `ATLAS_INKLING2_DUMP=<file>` appends `(visit
/// u32, position u32, BF16 hidden row)` records (visit `u32::MAX` = the normed
/// embedding) for comparison with the transformers reference's per-visit
/// hidden states; `<file>@<visit>` also records that visit's intermediates to
/// `<file>.tap` (see [`tap`]).
struct Switch {
    dump: String,
    tap_visit: Option<usize>,
}

fn switch() -> Option<&'static Switch> {
    static S: std::sync::OnceLock<Option<Switch>> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let v = std::env::var("ATLAS_INKLING2_DUMP").ok()?;
        Some(match v.rsplit_once('@') {
            Some((path, visit)) => Switch {
                dump: path.to_string(),
                tap_visit: visit.parse().ok(),
            },
            None => Switch {
                dump: v,
                tap_visit: None,
            },
        })
    })
    .as_ref()
}

pub(super) fn dump(
    gpu: &dyn GpuBackend,
    hidden: DevicePtr,
    batch: &Batch<'_>,
    h: usize,
    visit: u32,
    stream: u64,
) -> Result<()> {
    let Some(path) = switch().map(|s| &s.dump) else {
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

/// `dump`'s visit number for the model output (after the final block residual).
pub(super) const FINAL: u32 = 1000;

/// Whether a debug capture is on (each one synchronizes, so graphs stay off).
pub(super) fn enabled() -> bool {
    switch().is_some()
}

/// The visit whose intermediates the switch records, if any.
pub(super) fn tap_visit() -> Option<usize> {
    switch().and_then(|s| s.tap_visit)
}

/// Append `(tag u32, position u32, len u32, raw bytes)` to `<dump file>.tap`.
fn tap(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    bytes: usize,
    tag: u32,
    pos: usize,
    stream: u64,
) -> Result<()> {
    use std::io::Write;
    let path = format!(
        "{}.tap",
        switch()
            .context("inkling2 tap without the dump switch")?
            .dump
    );
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
