// SPDX-License-Identifier: AGPL-3.0-only

//! Setters + transposes + transpose_for_prefill_unified_inner.

use super::inplace_transpose::TransposeScratch;
use super::*;
#[path = "helpers_btile_phase.rs"]
mod btile_phase;

#[path = "helpers_checkpoint_down.rs"]
mod checkpoint_down;
#[path = "helpers_unified_phases.rs"]
mod unified_phases;
pub(super) use unified_phases::SharedGateUpReceipt;
#[cfg(test)]
pub(super) use unified_phases::btile_shared_fixture;
#[cfg(test)]
#[path = "helpers_unified_tests.rs"]
mod unified_tests;

impl MoeLayer {
    /// Transpose MoE weights for coalesced prefill GEMM reads.
    ///
    /// Transposes per-expert routed weights [N, K/2] → [K/2, N] to enable
    /// the cp.async pipelined FP8-MMA K64 kernels. This doubles expert
    /// memory (~17 GB for 35B, ~30 GB for 122B) but eliminates the
    /// catastrophic uncoalesced B reads in the fallback grouped GEMM,
    /// cutting MoE prefill time by ~2x.
    /// Set pre-expert norm (Gemma-4 26B: pre_feedforward_layernorm_2).
    /// Applied to input AFTER routing but BEFORE expert dispatch.
    pub fn set_pre_expert_norm(&mut self, norm: crate::weight_map::DenseWeight) {
        self.pre_expert_norm = Some(norm);
    }

    /// Set GeGLU activation for MoE experts (Gemma-4 26B).
    /// Replaces SiLU with GELU in the sorted/unfused path and forces decode
    /// to use the sorted path (avoiding fused SiLU kernels).
    pub fn set_gelu_activation(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.btile_storage.require_legacy()?;
        self.moe_act_mul = gpu.kernel("gelu", "gelu_mul")?;
        self.gelu_activation = true;
        Ok(())
    }

    pub fn transpose_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, true)
    }

    /// Transpose only the gate+up routed weights, leaving the down projection
    /// in its original layout. Cuts the transpose memory cost from ~3×
    /// (gate+up+down) to ~2× per expert. Used by MiniMax M2.7-NVFP4 EP=2
    /// when the full transpose doesn't fit but gate+up does — the fused
    /// `moe_w4a16_fused_gate_up_k64_n128` kernel still runs (capturing the
    /// dominant gate+up bandwidth savings), while down stays on the
    /// uncoalesced grouped-GEMM path.
    pub fn transpose_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, false)
    }

    pub(super) fn transpose_for_prefill_impl(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        include_down: bool,
    ) -> Result<()> {
        self.btile_storage.require_legacy()?;
        let h = config.hidden_size;
        let inter = config.routed_inter_local();
        let shared_inter = config.shared_expert_intermediate_size;

        // Transpose per-expert routed weights for coalesced prefill GEMM reads.
        let num_experts = self.weights.experts.len();
        let mut gate_t = Vec::with_capacity(num_experts);
        let mut up_t = Vec::with_capacity(num_experts);
        let mut down_t = Vec::with_capacity(num_experts);

        // ARM-2 Phase-K Family C: native-MXFP4 routed experts have per-32 E8M0
        // scales ([N, K/32]); NVFP4 is per-16. The scale transpose must use the
        // matching block size or the E8M0 kernels read a mis-shaped scale table.
        let routed_gs =
            if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                32
            } else {
                16
            };
        // ATLAS_QWEN4EXP_PREFILL_MOE_NODUP: the routed experts stay in their
        // checkpoint layout (`forward_prefill_q38_nodup`); the shared expert
        // below is still transposed.
        let routed = !super::skips_routed_transpose(config);
        if !routed {
            self.check_q38_nodup_planes()?;
        }
        for expert in self.weights.experts.iter().filter(|_| routed) {
            if expert.gate_proj.is_null() {
                gate_t.push(QuantizedWeight::null());
                up_t.push(QuantizedWeight::null());
                if include_down {
                    down_t.push(QuantizedWeight::null());
                }
            } else {
                gate_t.push(
                    expert
                        .gate_proj
                        .transpose_for_gemm_gs(gpu, inter, h, routed_gs)?,
                );
                up_t.push(
                    expert
                        .up_proj
                        .transpose_for_gemm_gs(gpu, inter, h, routed_gs)?,
                );
                if include_down {
                    down_t.push(
                        expert
                            .down_proj
                            .transpose_for_gemm_gs(gpu, h, inter, routed_gs)?,
                    );
                }
            }
        }

        if routed {
            self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
            self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
            if include_down {
                self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
            }
        }

        // Transpose shared expert weights (tiny: ~5 MB per layer).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            if include_down {
                self.shared_down_t =
                    Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                        gpu,
                        h,
                        shared_inter,
                    )?);
            }
        }

        Ok(())
    }

    /// Phase 8a unified-layout transpose pass: build persistent transposed
    /// gate/up/down for all experts, freeing the untransposed copies between
    /// phases so the entire pass fits in tight memory budgets that the
    /// non-unified `transpose_for_prefill_impl(true)` would reject.
    ///
    /// Phased flow (memory math for MiniMax M2.7-NVFP4 EP=2 ≈ 47 GB free):
    ///   A. Transpose gate+up               (allocs +39 GB; free ≈ 8 GB)
    ///   B. Free gate+up untransposed       (frees 39 GB; free ≈ 47 GB)
    ///   C. Transpose down                  (allocs +20 GB; free ≈ 27 GB)
    ///   D. Free down untransposed          (frees 20 GB; free ≈ 47 GB)
    ///
    /// Net memory: all originals are freed and the layout is transposed-only.
    /// Models that need a decode-native shared expert use the explicitly named
    /// `transpose_for_prefill_unified_keep_shared` variant below.
    ///
    /// Caller responsibilities:
    ///   1. Set `ATLAS_UNIFIED_MOE_LAYOUT=1` so `MoeLayer::use_t_layout_for_decode()`
    ///      returns true at dispatch time.
    ///   2. Call this method INSTEAD of `transpose_for_prefill` /
    ///      `transpose_gate_up_for_prefill`.
    ///
    /// Returns the original pointers it freed. They are the weight store's
    /// tensors, so the caller marks them in the store or teardown frees them
    /// a second time (#122).
    pub fn transpose_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Vec<DevicePtr>> {
        self.transpose_for_prefill_unified_inner(gpu, config, false, false)
    }

    /// Unified routed-expert layout while retaining only the small shared
    /// expert's decode-native weights. GLM K=4 verification uses this to run
    /// an exact-M=4 shared projection without paying hybrid layout's cost for
    /// every routed expert. Returns the original pointers it freed — same
    /// contract as `transpose_for_prefill_unified`.
    pub fn transpose_for_prefill_unified_keep_shared(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Vec<DevicePtr>> {
        self.transpose_for_prefill_unified_inner(gpu, config, false, true)
    }

    /// Hybrid-layout transpose pass — analogue of `transpose_for_prefill_unified`
    /// that **keeps** the untransposed originals so decode + MTP verify dispatch
    /// can continue using the warp-reduction kernels. Allocates ~58 GB
    /// transposed alongside the existing ~58 GB originals on MiniMax M2.7-NVFP4
    /// EP=2; fits in 122 GB GB10 with KV-cache headroom up to ~32K context.
    /// Caller is responsible for memory-fit gating (factory checks free memory
    /// before invoking this).
    pub fn transpose_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        // Hybrid keeps the originals, so there is nothing freed to report.
        self.transpose_for_prefill_unified_inner(gpu, config, true, true)
            .map(|_| ())
    }

    /// Phased build of the transposed weight set. When `keep_originals` is true
    /// (hybrid-layout mode), Phase B and Phase D frees are skipped so decode
    /// paths still find the untransposed weights. When false (unified-layout
    /// mode), the originals are freed between phases — current Phase 8a
    /// behavior — and returned, so their owner can stop tracking them. The
    /// independent `keep_shared_originals` bit exempts only the shared expert.
    pub(super) fn transpose_for_prefill_unified_inner(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        keep_originals: bool,
        keep_shared_originals: bool,
    ) -> Result<Vec<DevicePtr>> {
        let mut freed = Vec::new();
        self.btile_storage.require_legacy()?;
        let h = config.hidden_size;
        let inter = config.routed_inter_local();

        // ── Layout state is DERIVED, never declared ──────────────────────
        // These two flags used to be read independently from env at
        // construction, so operator intent and the tier that actually ran
        // could disagree. That was not cosmetic: with both
        // ATLAS_UNIFIED_MOE_LAYOUT=1 and ATLAS_HYBRID_MOE_LAYOUT=1 on a box
        // where hybrid does not fit, the orchestrator ran UNIFIED (freeing the
        // [N,K/2] originals) while `use_t_layout_for_decode()` still returned
        // false -- its `!self.hybrid_layout` term read the env flag -- so
        // decode fell through to the originals arm and dereferenced freed
        // device memory.
        //
        // Setting them here, from the branch that is about to execute, makes
        // that disagreement structurally impossible and removes the need for
        // the operator to know either flag exists.
        self.unified_layout = !keep_originals;
        self.hybrid_layout = keep_originals;

        // ── Phase A: transpose gate+up routed experts ──
        // ARM-2 Phase-K Family C: native-MXFP4 routed experts are per-32 E8M0.
        let routed_gs =
            if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
                32
            } else {
                16
            };
        let gate_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.gate_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.gate_proj
                }
            })
            .collect();
        let up_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.gate_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.up_proj
                }
            })
            .collect();
        let mut scratch = TransposeScratch::new();
        let (gate_t, up_t) = if keep_originals {
            (
                self.transpose_experts_gpu(gpu, &gate_src, inter, h, routed_gs)?,
                self.transpose_experts_gpu(gpu, &up_src, inter, h, routed_gs)?,
            )
        } else {
            // Unified: transpose into scratch, copy back over the originals —
            // no per-layer slab allocs and no ~3K small frees per call.
            (
                self.transpose_experts_inplace(gpu, &gate_src, inter, h, routed_gs, &mut scratch)?,
                self.transpose_experts_inplace(gpu, &up_src, inter, h, routed_gs, &mut scratch)?,
            )
        };
        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        self.transpose_unified_shared_gate_up(gpu, config)?;

        if !keep_originals {
            // ── Phase B: free gate+up untransposed ──
            // The previous gate_ptrs / up_ptrs device-side pointer tables now
            // contain stale addresses, but the unified dispatch never reads
            // them (gated by `use_t_layout_for_decode()`).
            for expert in &mut self.weights.experts {
                // The original allocations now hold the transposed data;
                // NULL the fields so no untransposed path can read them.
                if !expert.gate_proj.weight.is_null() {
                    expert.gate_proj.weight = DevicePtr::NULL;
                    expert.gate_proj.weight_scale = DevicePtr::NULL;
                }
                if !expert.up_proj.weight.is_null() {
                    expert.up_proj.weight = DevicePtr::NULL;
                    expert.up_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !keep_shared_originals {
                self.release_unified_shared_gate_up(gpu, config, &mut freed)?;
            }
        }
        scratch.release(gpu)?;
        self.transpose_unified_down_phase(
            gpu,
            config,
            routed_gs,
            keep_originals,
            keep_shared_originals,
            &mut freed,
        )?;
        Ok(freed)
    }

    /// Transpose one projection across ALL routed experts on the GPU, into a
    /// single slab allocation per buffer.
    /// Replaces a per-expert `QuantizedWeight::transpose_for_gemm_gs`, which
    /// round-trips each expert through the host and creates two allocations.
    /// The batched kernel avoids those transfers and allocation pressure.
    /// `src` supplies the per-expert untransposed `[n, k/2]` packed bytes and
    /// `[n, k/group_size]` scales; the returned `QuantizedWeight`s point into
    /// the two slabs and carry the source's scale metadata unchanged.
    #[allow(clippy::too_many_arguments)]
    fn transpose_experts_gpu(
        &self,
        gpu: &dyn GpuBackend,
        src: &[QuantizedWeight],
        n: usize,
        k: usize,
        group_size: usize,
    ) -> Result<Vec<QuantizedWeight>> {
        let num_experts = src.len();
        let (local_experts, compact_slots) =
            super::compact_layout::compact_slot_map(src.iter().map(|w| !w.is_null()));
        let packed_each = n * (k / 2);
        let scale_each = n * (k / group_size);
        anyhow::ensure!(
            packed_each > 0 && scale_each > 0,
            "transpose_experts_gpu: zero-sized projection (n={n} k={k} gs={group_size})"
        );

        if local_experts == 0 {
            return Ok(vec![QuantizedWeight::null(); num_experts]);
        }
        let packed_slab = gpu.alloc(local_experts * packed_each)?;
        let scale_slab = gpu.alloc(local_experts * scale_each)?;

        // Destinations carve the slabs; a NULL source keeps a NULL slot so the
        // kernel's own NULL guard skips that expert (EP-remote convention).
        let mut out = Vec::with_capacity(num_experts);
        for (w, compact_slot) in src.iter().zip(compact_slots) {
            if let Some(slot) = compact_slot {
                out.push(QuantizedWeight {
                    weight: packed_slab.offset(slot * packed_each),
                    weight_scale: scale_slab.offset(slot * scale_each),
                    weight_scale_2: w.weight_scale_2,
                    input_scale: w.input_scale,
                    weight_scale_2_vec: w.weight_scale_2_vec,
                });
            } else {
                out.push(QuantizedWeight::null());
            }
        }

        let src_tbl = build_ptr_table_from_qw(src, gpu)?;
        let dst_tbl = build_ptr_table_from_qw(&out, gpu)?;
        let stream = gpu.default_stream();
        // Packed [n, k/2] -> [k/2, n].
        crate::layers::ops::moe_transpose_u8_batched(
            gpu,
            self.moe_transpose_u8_batched_k,
            src_tbl.packed_ptrs,
            dst_tbl.packed_ptrs,
            n as u32,
            (k / 2) as u32,
            num_experts as u32,
            stream,
        )?;
        // Scales [n, k/group_size] -> [k/group_size, n].
        crate::layers::ops::moe_transpose_u8_batched(
            gpu,
            self.moe_transpose_u8_batched_k,
            src_tbl.scale_ptrs,
            dst_tbl.scale_ptrs,
            n as u32,
            (k / group_size) as u32,
            num_experts as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        // The pointer tables were scratch for the launch only.
        gpu.free(src_tbl.packed_ptrs)?;
        gpu.free(src_tbl.scale_ptrs)?;
        gpu.free(src_tbl.scale2_vals)?;
        gpu.free(dst_tbl.packed_ptrs)?;
        gpu.free(dst_tbl.scale_ptrs)?;
        gpu.free(dst_tbl.scale2_vals)?;
        Ok(out)
    }

    /// Build per-expert swizzled SFB weight-scale tables for the CUTLASS grouped
    /// NVFP4 path (`ATLAS_HOLO_MOE_GROUPED_CUTLASS`). For each expert, swizzle the
    /// `[K/16,N]` `gate_ptrs_t`/`up_ptrs_t` scale into the CUTLASS SFB atom via
    /// `pack_weight_sfb`, then upload the per-expert pointer arrays. The grouped
    /// kernel pairs these with `gate_ptrs.packed` (`[N,K/2]`) + the real per-expert
    /// `scale2`. Requires FAST_MOE=full (gate_ptrs_t/up_ptrs_t present); no-op else.
    /// Shared-expert transposed twins only. CUTLASS grouped MoE keeps routed
    /// experts checkpoint-native, but the shared FP8 cache and prefill shared
    /// GEMMs still consume the transposed shared projections.
    pub fn transpose_shared_only(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_unified_shared_gate_up(gpu, config)?;
        let shared_inter = config.shared_expert_intermediate_size;
        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                config.hidden_size,
                shared_inter,
            )?);
        }
        Ok(())
    }

    /// Free the checkpoint routed-expert scales once CUTLASS owns swizzled
    /// copies (vLLM keeps only the swizzled scales too). Scale pointer tables
    /// are zeroed so any non-CUTLASS routed kernel faults instead of reading
    /// freed memory.
    pub(super) fn release_routed_scales(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut freed = 0usize;
        for expert in self.weights.experts.iter_mut() {
            for proj in [
                &mut expert.gate_proj,
                &mut expert.up_proj,
                &mut expert.down_proj,
            ] {
                if !proj.weight_scale.is_null() {
                    gpu.free(proj.weight_scale)?;
                    proj.weight_scale = DevicePtr::NULL;
                    freed += 1;
                }
            }
        }
        let zeros = vec![0u8; self.weights.experts.len() * 8];
        for table in [&self.gate_ptrs, &self.up_ptrs, &self.down_ptrs] {
            if !table.scale_ptrs.is_null() {
                gpu.copy_h2d(&zeros, table.scale_ptrs)?;
            }
        }
        self.routed_scales_released = true;
        tracing::info!(freed, "CUTLASS grouped: released checkpoint routed scales");
        Ok(())
    }
}

/// Free one original weight and remember it for the store (see
/// `transpose_for_prefill_unified`).
fn free_original(gpu: &dyn GpuBackend, freed: &mut Vec<DevicePtr>, ptr: DevicePtr) -> Result<()> {
    gpu.free(ptr)?;
    freed.push(ptr);
    Ok(())
}
