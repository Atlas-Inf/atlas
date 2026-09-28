// SPDX-License-Identifier: AGPL-3.0-only

//! Pool-layout WY-chunkwise GDN verify arm, shared by the K=17 DFlash path
//! (`gated_delta_rule_wy17`) and the chain-verify K∈{5..16} path
//! (`gated_delta_rule_wy5..wy16`, one K-templated source). Extracted from
//! `trait_decode_batched_conv_gdn.rs` so both arms dispatch the identical
//! body (fused conv_kn epilogue + one wyN launch) and to keep that file
//! under the 500 LoC cap.

use anyhow::Result;
use spark_runtime::gpu::KernelHandle;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

// The `OnceLock<bool>` static that lived here is now a field on
// `layers::ops::ModelLevers` — resolved when the model is built and carried
// on `ForwardContext`, because a static outlives the model whose flags it
// encodes.

impl Qwen3SsmLayer {
    /// The wyN kernel for chain-verify `num_tokens` ∈ {5..16}, or `None`
    /// when out of range, the module is absent (non-gb10 target), or the
    /// `ATLAS_GDN_WYN=0` kill-switch is set — all of which keep the caller
    /// on the sequential per-token fallback.
    pub(super) fn wyn_kernel(&self, num_tokens: usize, wyn_enabled: bool) -> Option<KernelHandle> {
        if !(5..=16).contains(&num_tokens) || !wyn_enabled {
            return None;
        }
        let k = self.gdn_wyn_k[num_tokens - 5];
        (k.0 != 0).then_some(k)
    }

    /// Deferred-commit eligibility: lever on, wyN band (K=5..16), and both
    /// `_defer`/`commit` kernels resolved — else the storing path runs.
    /// Pure + `pub(super)` so the host-side decision is unit-tested.
    pub(super) fn gdn_defer_active(
        lever_on: bool,
        num_tokens: usize,
        defer_k: KernelHandle,
        commit_k: KernelHandle,
    ) -> bool {
        lever_on && (5..=16).contains(&num_tokens) && defer_k.0 != 0 && commit_k.0 != 0
    }

    /// Fused pool-layout WY verify arm for K = `args.num_tokens`:
    /// conv1d+L2norm epilogue (single fused launch writing every rollback
    /// snapshot inline when `gdn_verify_fused_conv_kn` is present and the
    /// conv intermediates are pool-contiguous; per-token loop otherwise),
    /// then ONE `wy_kernel` launch producing all K outputs, Hi_0..Hi_{K-2}
    /// intermediate H snapshots (pool layout, stride `h_bytes`) and the
    /// final H in place. `wy_kernel`'s compile-time K must equal
    /// `args.num_tokens` (wy17 for 17, wy5..wy8 for 5..8).
    ///
    /// Kill-switch `ATLAS_GDN_FUSED_CONV17=0` restores the per-token conv
    /// loop for A/B (applies to every width dispatched through this arm).
    pub(super) fn decode_batched_conv_gdn_wyn(
        &self,
        ssm_state: &mut SsmLayerState,
        ctx: &ForwardContext,
        args: &ConvGdnArgs,
        wy_kernel: KernelHandle,
    ) -> Result<()> {
        let ConvGdnArgs {
            num_tokens,
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            h_bytes,
            conv_bytes,
            qkvz_size,
            conv_dim,
            key_dim,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
            ..
        } = *args;

        let conv_inter_base = ssm_state.conv_state_intermediates[0];
        let inter_contiguous = ssm_state
            .conv_state_intermediates
            .iter()
            .take(num_tokens)
            .enumerate()
            .all(|(t, p)| p.0 == conv_inter_base.0 + (t * conv_bytes) as u64);
        let fused_conv = self.gdn_verify_fused_conv_kn_k.0 != 0
            && inter_contiguous
            && !matches!(
                std::env::var("ATLAS_GDN_FUSED_CONV17").ok().as_deref(),
                Some("0")
            );
        if fused_conv {
            ops::gdn_verify_fused_conv_kn(
                ctx.gpu,
                self.gdn_verify_fused_conv_kn_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                conv_out_buf,
                conv_inter_base,
                num_tokens as u32,
                conv_dim as u32,
                d_conv as u32,
                qk_ch,
                kd as u32,
                qkvz_size as u32, // input stride (BF16 elems between positions)
                conv_dim as u32,  // output stride (BF16 elems between positions)
                (conv_bytes / 4) as u32, // snapshot stride (FP32 elems)
                1e-6,
                stream,
            )?;
        } else {
            for t in 0..(num_tokens as u32) {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                // Skip t == K-1: dead write, no reader (enumeration in
                // trait_decode_batched_conv_gdn.rs). The FUSED conv arm
                // above still writes all K snapshots (on-device kernel),
                // which is why the conv pools keep K per slot.
                if (t as usize) + 1 < num_tokens {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }
        }

        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
        let gate_ptr = gates_buf;
        let beta_ptr = gates_buf.offset(nv * fp32);
        let inter_stride_floats = (h_bytes / 4) as u32;

        // ATLAS_GDN_DEFERRED_COMMIT (wyN K=5..16 only): launch the `_defer`
        // twin — identical outputs, no Hi_t/final-H stores, `h_state` stays
        // H0 — and stash the verify rows' conv-out + gate/beta into
        // per-slot staging so `gated_delta_rule_commit` can replay the
        // accepted prefix after accept. `h_state_intermediates` are unused
        // under defer (arg kept for the signature; the MODE-1 kernel never
        // reads it).
        // `!fp16` mirrors the host-side `gdn_deferred_wyn` gate — wyN has
        // no FP16-h-state twin, and the flag must never disagree with what
        // actually launched.
        let defer = Self::gdn_defer_active(
            ctx.levers.gdn_deferred_commit && !super::ssm_h_fp16_enabled(),
            num_tokens,
            if (5..=16).contains(&num_tokens) {
                self.gdn_wyn_defer_k[num_tokens - 5]
            } else {
                KernelHandle(0)
            },
            self.gdn_commit_k,
        );
        if defer {
            // Staging copies into pool-stable per-slot buffers — they MAY
            // run inside a captured verify graph: pool pointers are
            // replay-stable and a captured memcpy re-copies on replay, so
            // replays re-stage correctly. `gdn_commit_pending` is NOT set
            // here: host code does not run on replay, so the flag is
            // assigned at the verify dispatch entry instead
            // (`mark_gdn_deferred_commit`).
            anyhow::ensure!(
                !ssm_state.gdn_commit_qkv.is_null() && !ssm_state.gdn_commit_gb.is_null(),
                "deferred commit staging missing — the pool was built without                  ATLAS_GDN_DEFERRED_COMMIT"
            );
            ctx.gpu.copy_d2d_async(
                conv_out_buf,
                ssm_state.gdn_commit_qkv,
                num_tokens * conv_dim * bf16,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                gates_buf,
                ssm_state.gdn_commit_gb,
                num_tokens * nv * 2 * fp32,
                stream,
            )?;
        }
        let wy_kernel = if defer {
            self.gdn_wyn_defer_k[num_tokens - 5]
        } else {
            wy_kernel
        };
        let inter_base = if defer {
            spark_runtime::gpu::DevicePtr(0)
        } else {
            ssm_state.h_state_intermediates[0]
        };
        let inter_stride = if defer { 0 } else { inter_stride_floats };
        ops::gdn_decode_wyn(
            ctx.gpu,
            wy_kernel,
            ssm_state.h_state,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_out_buf,
            inter_base,
            inter_stride,
            1, // batch_size
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32, // qk_stride
            conv_dim as u32, // v_stride
            (nv * 2) as u32, // gb_stride
            false,           // contiguous state — this site is batch_size=1
            stream,
        )
    }
}
