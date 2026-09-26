// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-only native EXL3 overlay for the full-attention q/k/v/o projections.
//!
//! Under `ATLAS_EXL3_NATIVE_DECODE=1` the packed four of these stems survive
//! materialize (see `weight_map::exl3::keeps_packed_with`); this attaches the
//! int8 sq GEMV overlays that consume them for single-token decode and
//! narrow (n <= 2) verify. Prefill and wider batches keep the BF16 copies.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::Qwen3AttentionLayer;
use crate::layers::ops::Exl3LinearDecode;
use crate::weight_map::exl3::{exl3_from_store, exl3_native_decode};

/// The four attention projections as packed EXL3 decode linears.
pub struct Exl3AttnDecode {
    pub q: Exl3LinearDecode,
    pub k: Exl3LinearDecode,
    pub v: Exl3LinearDecode,
    pub o: Exl3LinearDecode,
}

impl Qwen3AttentionLayer {
    /// Build the overlay from `store` for the `...self_attn` stem prefix `p`.
    /// Returns false (and changes nothing) unless the gate is on and
    /// `{p}.q_proj.trellis` is in the store.
    pub fn attach_exl3_decode_from_store(
        &mut self,
        store: &spark_runtime::weights::WeightStore,
        p: &str,
        gpu: &dyn GpuBackend,
    ) -> Result<bool> {
        if !exl3_native_decode() || !store.contains(&format!("{p}.q_proj.trellis")) {
            return Ok(false);
        }
        let build = |name: &str| -> Result<Exl3LinearDecode> {
            let w = exl3_from_store(store, &format!("{p}.{name}"), gpu)?;
            Exl3LinearDecode::new(gpu, w)
        };
        self.exl3_attn = Some(Box::new(Exl3AttnDecode {
            q: build("q_proj")?,
            k: build("k_proj")?,
            v: build("v_proj")?,
            o: build("o_proj")?,
        }));
        Ok(true)
    }
}
