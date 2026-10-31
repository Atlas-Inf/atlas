// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in: FP4-Local's KDA q/k/v/o computed on the checkpoint's own MXFP8
//! bytes instead of production's runtime NVFP4 twins (default off).
//!
//! `ATLAS_GLM_FP4L_KDA_MXFP8_DECODE=1`: every decode, multi-sequence decode
//! and verify row (1..=32 a launch, any count chunked) reads the rank's MXFP8
//! twin through `ops::mxfp8_gemv_rows`. Exact weights, and one row-invariant
//! kernel family for every row count, so a verify row and the same row
//! decoded agree bit for bit, as on the canonical W4A16 tier. It reads ~1.03
//! B/param where NVFP4 reads ~0.56 (34 layers x 4 projections: ~1 GB more
//! per rank per step) and adds the twin (~2.4 GB per rank) beside the NVFP4
//! copies prefill still reads. Tiled under `ATLAS_GLM_MXFP8_TILED`.
//!
//! `ATLAS_GLM_FP4L_KDA_W8A8_PREFILL=1`: every prefill projection is the
//! cuBLASLt MXFP8 W8A8 GEMM on the twin's E4M3 bytes (`ops::MxW8a8`,
//! NVIDIA's recipe: exact weights, activations rounded to E4M3 per 32 values,
//! ~2-3e-2 relative from BF16 activations), row-invariant end to end. The
//! transposed NVFP4 prefill twins are not built and the LT FP8 copies are
//! never taken. GLM's TP2 shape (4096 x 4096) is in `cublaslt::MX_SWEEP`
//! but not yet swept, so it runs one in-order kernel for every row count.
//!
//! Both on: nothing reads NVFP4, so the layer keeps only the MXFP8 twin and
//! the W8A8 scale image (~1.06 B/param, below production's NVFP4 decode +
//! transposed prefill twins at ~1.1) and the load skips the BF16 expansion.
//!
//! On a checkpoint that does not ship the projection MXFP8 both are ignored
//! (logged once).

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::ops::MxW8a8;
use crate::tp_shard::TpShardKind;
use crate::weight_map::{Mxfp8Weight, mxfp8_cols_twin, mxfp8_rows_twin, raw_mxfp8};

/// Rows the pinned W8A8 kernel is timed at (a prefill pass's height:
/// `ATLAS_GLM_PREFILL_4096`).
pub(super) const PIN_REF_ROWS: usize = 4096;

/// The switches, read like the loader's other flags (`components::env_flag`:
/// `1` or `true` is on, anything else off).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct KdaMxPolicy {
    pub decode: bool,
    pub w8a8_prefill: bool,
}

impl KdaMxPolicy {
    pub(super) const DECODE: &'static str = crate::layers::glm_native::NativePolicy::KDA_DECODE;
    pub(super) const W8A8: &'static str = crate::layers::glm_native::NativePolicy::KDA_W8A8;

    /// The two switches, each also on under `ATLAS_GLM_FP4L_NATIVE`
    /// (`layers::glm_native`).
    pub(super) fn from_env() -> Self {
        let native = crate::layers::glm_native::NativePolicy::from_env();
        Self {
            decode: native.kda_decode,
            w8a8_prefill: native.kda_w8a8,
        }
    }

    fn any(self) -> bool {
        self.decode || self.w8a8_prefill
    }
}

/// What a KDA projection reads from its MXFP8 twin.
#[derive(Debug, Default)]
pub(super) struct KdaMx {
    /// The decode / verify twin (`decode`).
    pub decode: Option<Mxfp8Weight>,
    /// The prefill W8A8 operand over the same E4M3 bytes (`w8a8_prefill`).
    pub w8a8: Option<MxW8a8>,
}

impl KdaMx {
    /// Whether these twins serve decode, verify AND prefill: nothing reads
    /// the projection's NVFP4 copies, so none are built.
    pub(super) fn serves_all(&self) -> bool {
        self.decode.is_some() && self.w8a8.is_some()
    }

    /// Free what [`kda_twin`] allocated (the error path of a projection
    /// load that fails after it). The W8A8 operand owns only its scale
    /// image; its E4M3 bytes are the twin's, or (W8A8 alone) its own to free.
    pub(super) fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(w) = self.w8a8 {
            // Its E4M3 bytes are the decode twin's unless decode has its own
            // (tiled) copy or there is no decode.
            if self.decode.is_none_or(|t| t.data != w.data) {
                gpu.free(w.data)?;
            }
            w.free(gpu)?;
        }
        if let Some(t) = self.decode {
            gpu.free(t.data)?;
            gpu.free(t.scales)?;
        }
        Ok(())
    }
}

/// This rank's MXFP8 twin of KDA projection `stem`, whose full shape the
/// config says is `[n, k]`, sharded by `kind`, under `policy`, cut from the
/// checkpoint's bytes; `None` when the policy is off or the checkpoint ships
/// `stem` in another format (logged once: the switch is ignored there). The
/// checkpoint tensor must be exactly `[n, k]` and the twin this rank's
/// `[n/tp, k]` (q/k/v) or `[n, k/tp]` (o), or the GEMV would read past it.
/// The twin's slice is the matching slice of the checkpoint tensor, as
/// `mxfp8_load` cuts it, and the same rows the BF16 TP shard keeps. The W8A8
/// operand runs the pinned kernel (`MxW8a8::pinned`), so nothing is tuned at
/// load. Synchronizes the default stream; frees what it allocated on any
/// error.
#[allow(clippy::too_many_arguments)]
pub(super) fn kda_twin(
    policy: KdaMxPolicy,
    store: &WeightStore,
    stem: &str,
    [n, k]: [usize; 2],
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<KdaMx>> {
    if !policy.any() {
        return Ok(None);
    }
    let Some(raw) = raw_mxfp8(store, stem)? else {
        super::mode::require_format(false, &format!("{stem} MXFP8 weights"))?;
        static IGNORED: std::sync::Once = std::sync::Once::new();
        IGNORED.call_once(|| {
            tracing::warn!(
                "{} / {} set, but {stem} is not MXFP8 in this checkpoint: \
                 KDA projections keep their NVFP4 twins",
                KdaMxPolicy::DECODE,
                KdaMxPolicy::W8A8
            )
        });
        return Ok(None);
    };
    ensure!(
        [raw.n, raw.k] == [n, k] && tp_size > 0 && tp_rank < tp_size,
        "{stem}: checkpoint MXFP8 [{}, {}] is not the config's [{n}, {k}] (tp {tp_rank}/{tp_size})",
        raw.n,
        raw.k
    );
    let stream = gpu.default_stream();
    let (mut twin, tn, tk) = match kind {
        TpShardKind::ColumnParallel => {
            mxfp8_rows_twin(gpu, &[(raw, &[raw.n][..])], tp_rank, tp_size, stream)?
        }
        TpShardKind::RowParallel => mxfp8_cols_twin(gpu, raw, tp_rank, tp_size, stream)?,
        TpShardKind::Replicated => bail!("{stem}: KDA projections are TP-sharded"),
    };
    let free_twin = |twin: Mxfp8Weight| -> Result<()> {
        gpu.free(twin.data)?;
        gpu.free(twin.scales)
    };
    let local = match kind {
        TpShardKind::ColumnParallel => [n / tp_size, k],
        _ => [n, k / tp_size],
    };
    if [tn, tk] != local {
        free_twin(twin)?;
        bail!("{stem}: MXFP8 twin [{tn}, {tk}] is not this rank's {local:?}");
    }
    let w8a8 = if policy.w8a8_prefill {
        // Reads the twin's E4M3 bytes in place, untiled, on the pinned
        // kernel (one per shape, never retuned: chunk- and history-invariant
        // prefill), so no row bucket is tuned at load.
        match MxW8a8::from_twin(gpu, &twin, local, 0, stream) {
            Ok(w) => Some(w.pinned_timed(PIN_REF_ROWS, stream)),
            Err(e) => {
                free_twin(twin)?;
                return Err(e.context(format!("{stem}: MXFP8 W8A8 operand")));
            }
        }
    } else {
        if let Err(e) = crate::layers::ops::mxfp8_tile_if_requested(gpu, &mut twin, tn, tk, stream)
        {
            free_twin(twin)?;
            return Err(e.context(format!("{stem}: MXFP8 twin tiling")));
        }
        None
    };
    // ATLAS_GLM_FP4L_KDA_TILED_TWIN: decode reads its own chunk-tiled copy
    // while W8A8 keeps the row-major bytes (+1 B/param resident).
    let tiled_copy = policy.decode && w8a8.is_some() && tiled_twin_requested()?;
    if !policy.decode || tiled_copy {
        // W8A8 reads the E4M3 bytes (its operand) and its own tiled scale
        // image: the row-major scales have no other reader once decode has
        // its own copy (or there is no decode).
        let decode = if tiled_copy {
            match tiled_decode_copy(gpu, &twin, tn, tk, stream) {
                Ok(copy) => Some(copy),
                Err(e) => {
                    if let Some(w) = w8a8 {
                        w.free(gpu)?;
                    }
                    free_twin(twin)?;
                    return Err(e.context(format!("{stem}: tiled MXFP8 decode copy")));
                }
            }
        } else {
            None
        };
        gpu.free(twin.scales)?;
        return Ok(Some(KdaMx { decode, w8a8 }));
    }
    Ok(Some(KdaMx {
        decode: Some(twin),
        w8a8,
    }))
}

/// `ATLAS_GLM_FP4L_KDA_TILED_TWIN=1` with `ATLAS_GLM_MXFP8_TILED=1`: a KDA
/// layer serving W8A8 prefill gives decode / verify a chunk-tiled copy of the
/// twin (the `_tiled` tiers stream ~10% faster than row-major at these
/// shapes, `mxfp8_tiled_bench`) instead of sharing the row-major bytes. The
/// copy costs ~1 B/param (~2.3 GB a rank over 34 layers). Not part of
/// `ATLAS_GLM_FP4L_NATIVE`. Without `ATLAS_GLM_MXFP8_TILED` it is ignored.
fn tiled_twin_requested() -> Result<bool> {
    Ok(
        std::env::var("ATLAS_GLM_FP4L_KDA_TILED_TWIN").as_deref() == Ok("1")
            && crate::layers::ops::mxfp8_tiled_requested()?,
    )
}

/// A copy of untiled `twin` (`[n, k]`) rewritten into the chunk tiling.
fn tiled_decode_copy(
    gpu: &dyn GpuBackend,
    twin: &Mxfp8Weight,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<Mxfp8Weight> {
    let mut copy = Mxfp8Weight::alloc(gpu, n, k)?;
    let built = (|| -> Result<()> {
        gpu.copy_d2d_async(twin.data, copy.data, n * k, stream)?;
        gpu.copy_d2d_async(twin.scales, copy.scales, n * k / 32, stream)?;
        gpu.synchronize(stream)?;
        crate::layers::ops::mxfp8_tile_if_requested(gpu, &mut copy, n, k, stream)
    })();
    if let Err(e) = built {
        gpu.free(copy.data)?;
        gpu.free(copy.scales)?;
        return Err(e);
    }
    Ok(copy)
}

#[cfg(test)]
#[path = "kda_mx_tests.rs"]
mod tests;
