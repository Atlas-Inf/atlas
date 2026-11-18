// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_SCORE_PACK_ROUNDTRIP`: a just-built pack's GDN state and/or K/V,
//! rounded in place through a narrower format (`roundtrip_rows`), so the
//! requests on it read what a compressed pack cache would hand them. A
//! measurement hook: the pack's own rows were computed at full precision,
//! exactly as they would be before the cache stored them.
//!
//! `state:fp8|bf16|fp16` rounds every GDN layer's h state (FP32 `[nv, vd,
//! kd]`, a scale a `kd` row for FP8); `kv:fp8|bf16|fp16` rounds the pack's
//! BF16 K and V blocks (a scale a token and head). Comma-separated, e.g.
//! `state:fp8,kv:fp8`.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::KvCacheDtype;

use super::super::super::types::TransformerModel;
use super::score_pack::ssm_ptrs_of;
use crate::traits::SequenceState;

/// The requested formats: (state, kv), each `roundtrip_rows`' `fmt`.
fn parse(spec: &str) -> Result<(Option<u32>, Option<u32>)> {
    let fmt = |v: &str| match v {
        "fp8" => Ok(0),
        "bf16" => Ok(1),
        "fp16" => Ok(2),
        other => bail!("ATLAS_SCORE_PACK_ROUNDTRIP: unknown format {other:?}"),
    };
    let (mut state, mut kv) = (None, None);
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once(':') {
            Some(("state", v)) => state = Some(fmt(v)?),
            Some(("kv", v)) => kv = Some(fmt(v)?),
            _ => bail!("ATLAS_SCORE_PACK_ROUNDTRIP: expected state:FMT or kv:FMT, got {part:?}"),
        }
    }
    Ok((state, kv))
}

impl TransformerModel {
    /// Apply `ATLAS_SCORE_PACK_ROUNDTRIP` (if set) to a just-built pack.
    pub(super) fn score_pack_roundtrip(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        let Ok(spec) = std::env::var("ATLAS_SCORE_PACK_ROUNDTRIP") else {
            return Ok(());
        };
        let (state, kv) = parse(&spec)?;
        let kernel = crate::layers::try_kernel(self.gpu.as_ref(), "residual_add", "roundtrip_rows");
        ensure!(
            kernel.0 != 0,
            "ATLAS_SCORE_PACK_ROUNDTRIP: roundtrip_rows is not in this build"
        );
        let launch = |ptr: DevicePtr, rows: usize, len: usize, dtype: u32, fmt: u32| {
            KernelLaunch::new(self.gpu.as_ref(), kernel)
                .grid([rows.div_ceil(8) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(ptr)
                .arg_u32(rows as u32)
                .arg_u32(len as u32)
                .arg_u32(dtype)
                .arg_u32(fmt)
                .launch(stream)
        };
        if let Some(fmt) = state {
            let kd = self.config.linear_key_head_dim;
            let rows = self.config.ssm_h_state_bytes() / 4 / kd;
            ensure!(
                self.ssm_pool.h_stored_bytes == self.config.ssm_h_state_bytes(),
                "ATLAS_SCORE_PACK_ROUNDTRIP: the SSM pool must hold FP32 state"
            );
            for (i, layer) in self.layers.iter().enumerate() {
                if layer.is_ssm_layer() {
                    launch(ssm_ptrs_of(seq, i)?.0, rows, kd, 0, fmt)?;
                }
            }
        }
        if let Some(fmt) = kv {
            let kv = self.kv_cache.lock();
            let d = self.config.head_dim;
            for l in 0..kv.num_layers() {
                if kv.dtype_for_layer(l) != KvCacheDtype::Bf16 {
                    continue;
                }
                let rows = kv.k_block_stride_bytes_for_layer(l) / 2 / d;
                for &b in &seq.block_table {
                    launch(kv.k_cache_ptr(l, b), rows, d, 1, fmt)?;
                    launch(kv.v_cache_ptr(l, b), rows, d, 1, fmt)?;
                }
            }
        }
        self.gpu.synchronize(stream)?;
        tracing::info!("score pack: rounded through {spec}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn parses_state_and_kv_formats() {
        assert_eq!(parse("state:fp8,kv:fp8").unwrap(), (Some(0), Some(0)));
        assert_eq!(parse("kv:bf16").unwrap(), (None, Some(1)));
        assert_eq!(parse("state:fp16").unwrap(), (Some(2), None));
        assert!(parse("state:int4").is_err());
        assert!(parse("x:fp8").is_err());
    }
}
