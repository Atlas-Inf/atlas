// SPDX-License-Identifier: AGPL-3.0-only

//! The activation scales of qwen4_exp's W4A4 routed prefill
//! (`forward_prefill_w4a4`), as the engine NVIDIA's FP4-Local card names
//! sets them (vLLM b12x, `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13`,
//! `B12X_W4A16_A4_PREFILL=1`: `fused_moe/oracle/nvfp4.py` and `b12x.py`):
//!
//! - gate/up input: one global scale for the layer, `g1 = 1 / max_e
//!   max(gate, up input_scale)` over EVERY expert (the reference runs TP
//!   with all experts on a rank), so a token is quantized once;
//! - down input: each expert's own `g2[e] = 1 / down input_scale[e]`;
//! - each GEMM's alpha takes the activation scale back out: `alpha =
//!   weight_scale_2 / g`, the reference's epilogue product.
//!
//! The checkpoint ships `{gate,up,down}_proj.input_scale` per expert (the
//! FP4-Local and NVFP4 releases alike; on FP4-Local the gate/up value is one
//! per layer). The fast loader defers them (`skip_activation_scales`), every
//! expert's on every rank; a layer reads its 1536 scalars in one span here.
//! As in the reference, a layer whose scales are not all positive and finite
//! (the MTP layer ships none) keeps BF16 activations.

use std::collections::BTreeMap;

use super::*;
use spark_runtime::weights::{DeferredTensor, WeightDtype, WeightStore};

/// One layer's W4A4 scales and the GEMM tables that undo them.
pub(crate) struct W4a4Scales {
    /// The gate/up input's global scale `g1`.
    pub(super) g1: f32,
    /// `gate_ptrs` / `up_ptrs` / `down_ptrs` with `alpha` in `scale2_vals`.
    pub(super) gate: ExpertPtrTable,
    pub(super) up: ExpertPtrTable,
    pub(super) down: ExpertPtrTable,
    /// `[E]` f32 `g2[e]`, the down input's global scales.
    pub(super) a2_gscale: DevicePtr,
}

/// The host tables of a layer: `g1` and `[alpha_gate | alpha_up | alpha_down
/// | g2]` (`E` f32 each) from every expert's `[gate, up, down]` input scale
/// and the local experts' `[gate, up, down]` weight_scale_2 (`None`:
/// remote, alpha 0, never read). `None` when a scale is not positive and
/// finite.
pub(crate) fn w4a4_host_tables(
    input_scales: &[[f32; 3]],
    weight_scale_2: &[Option<[f32; 3]>],
) -> Option<(f32, Vec<f32>)> {
    let ok = |s: f32| s.is_finite() && s > 0.0;
    if input_scales.is_empty() || !input_scales.iter().flatten().all(|&s| ok(s)) {
        return None;
    }
    let a1 = input_scales
        .iter()
        .fold(0.0f32, |m, s| m.max(s[0]).max(s[1]));
    let g1 = 1.0 / a1;
    let n = input_scales.len();
    let mut t = vec![0.0f32; 4 * n];
    for (e, (is, ws2)) in input_scales.iter().zip(weight_scale_2).enumerate() {
        let g2 = 1.0 / is[2];
        t[3 * n + e] = g2;
        if let Some([g, u, d]) = *ws2 {
            t[e] = g / g1;
            t[n + e] = u / g1;
            t[2 * n + e] = d / g2;
        }
    }
    Some((g1, t))
}

/// The F32 scalars `names`, deferred by the loader: one read per shard span
/// (a layer's input scales sit in ~6 KB of one shard). `None` when any is
/// missing from the store (another loader, or a checkpoint without them).
fn read_deferred_scalars(store: &WeightStore, names: &[String]) -> Result<Option<Vec<f32>>> {
    let mut by_file: BTreeMap<&std::path::Path, Vec<(usize, &DeferredTensor)>> = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        let Some(d) = store.deferred(name) else {
            return Ok(None);
        };
        anyhow::ensure!(
            d.dtype == WeightDtype::FP32 && d.shape.iter().product::<usize>() == 1,
            "`{name}` is {:?} {:?}, expected one F32",
            d.dtype,
            d.shape
        );
        by_file.entry(&d.path).or_default().push((i, d));
    }
    let mut out = vec![0.0f32; names.len()];
    for recs in by_file.values() {
        let lo = recs.iter().map(|r| r.1.offset).min().unwrap_or(0);
        let hi = recs.iter().map(|r| r.1.offset).max().unwrap_or(0) + 4;
        anyhow::ensure!(
            hi - lo <= 64 << 20,
            "input scales spread over {} bytes of one shard",
            hi - lo
        );
        let span = DeferredTensor {
            offset: lo,
            ..recs[0].1.clone()
        };
        spark_runtime::fast_weights::with_deferred_bytes(&span, 0, (hi - lo) as usize, |b| {
            for (i, d) in recs {
                let at = (d.offset - lo) as usize;
                out[*i] = f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
            }
            Ok(())
        })?;
    }
    Ok(Some(out))
}

fn upload_f32s(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let p = gpu.alloc(v.len() * 4)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

impl MoeLayer {
    /// The layer's [`W4a4Scales`] from the checkpoint's input scales, when
    /// `ATLAS_QWEN4EXP_PREFILL_W4A4` runs this model (`mlp` = the layer's
    /// `.mlp` prefix). Leaves none -- BF16 activations -- when the store has
    /// no deferred scales or they are not all usable.
    pub(crate) fn load_w4a4_scales(
        &mut self,
        store: &WeightStore,
        mlp: &str,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if !super::w4a4_requested()
            || config.model_type != "qwen4_exp"
            || self.experts_scale_kind != crate::weight_map::WeightQuantFormat::Nvfp4
        {
            return Ok(());
        }
        let experts = &self.weights.experts;
        let names: Vec<String> = (0..experts.len())
            .flat_map(|e| {
                ["gate", "up", "down"].map(|p| format!("{mlp}.experts.{e}.{p}_proj.input_scale"))
            })
            .collect();
        let scales = read_deferred_scalars(store, &names)?;
        let Some(scales) = scales else {
            if gpu.op_cache().once("moe:w4a4_none") {
                tracing::warn!(
                    "qwen4_exp W4A4 prefill off: the routed experts' input scales were not deferred \
                     by the loader (BF16 activations)"
                );
            }
            return Ok(());
        };
        let is: Vec<[f32; 3]> = scales.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let ws2: Vec<Option<[f32; 3]>> = experts
            .iter()
            .map(|x| {
                (!x.gate_proj.is_null()).then_some([
                    x.gate_proj.weight_scale_2,
                    x.up_proj.weight_scale_2,
                    x.down_proj.weight_scale_2,
                ])
            })
            .collect();
        let Some((g1, t)) = w4a4_host_tables(&is, &ws2) else {
            tracing::info!(
                "{mlp}: input scales not all positive; W4A4 prefill off (BF16 activations)"
            );
            return Ok(());
        };
        let table = upload_f32s(gpu, &t)?;
        let n = experts.len();
        self.w4a4 = Some(W4a4Scales {
            g1,
            gate: self.gate_ptrs.with_scale2(table),
            up: self.up_ptrs.with_scale2(table.offset(4 * n)),
            down: self.down_ptrs.with_scale2(table.offset(8 * n)),
            a2_gscale: table.offset(12 * n),
        });
        if gpu.op_cache().once("moe:w4a4") {
            tracing::info!(
                "qwen4_exp routed prefill: W4A4 at the checkpoint's input scales (b12x A4 prefill; \
                 first layer g1 = {g1:.4e})"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::w4a4_host_tables;

    #[test]
    fn g1_is_the_max_over_every_expert_and_alpha_divides_it_out() {
        // Expert 1 is remote (no weight_scale_2) but its gate scale is the max.
        let is = [[0.5, 0.25, 0.125], [2.0, 1.0, 4.0], [1.0, 1.5, 0.5]];
        let ws2 = [Some([3.0, 6.0, 1.0]), None, Some([1.0, 2.0, 8.0])];
        let (g1, t) = w4a4_host_tables(&is, &ws2).unwrap();
        assert_eq!(g1, 0.5);
        assert_eq!(&t[0..3], &[6.0, 0.0, 2.0]); // gate: ws2 / g1
        assert_eq!(&t[3..6], &[12.0, 0.0, 4.0]); // up
        assert_eq!(&t[9..12], &[8.0, 0.25, 2.0]); // g2 = 1 / down input scale
        assert_eq!(&t[6..9], &[0.125, 0.0, 4.0]); // down: ws2 / g2
    }

    #[test]
    fn a_layer_with_an_unusable_scale_keeps_bf16_activations() {
        let ws2 = [Some([1.0; 3]); 2];
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(w4a4_host_tables(&[[1.0; 3], [1.0, 1.0, bad]], &ws2).is_none());
        }
        assert!(w4a4_host_tables(&[], &[]).is_none());
    }
}
