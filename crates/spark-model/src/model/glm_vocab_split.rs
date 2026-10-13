// SPDX-License-Identifier: AGPL-3.0-only
//! Exact TP2 vocabulary split for the greedy GLM verify head.
//!
//! Each rank projects only its contiguous half of the BF16 head with the same
//! batched GEMV (per-column arithmetic unchanged), reduces it with
//! `argmax_bf16_value_ban` (the best pair and the best pair outside the
//! min_tokens end tokens), and swaps both `(f32, u32)` pairs per row with its
//! peer; a device merge applies the full-vocabulary tie rule, taking the
//! unbanned pair on rows under a min_tokens floor. The other half of the
//! logits buffer is left stale, so this serves only argmax-consuming verify.
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

use super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::EosBan;
use crate::weight_map::DenseWeight;

/// Local pairs, then peer pairs, above the verify argmax words in scratch.
const PAIRS_OFFSET: usize = 16384;
const PAIRS_BYTES: usize = 1024;

/// `ATLAS_GLM_VERIFY_VOCAB_SPLIT=1`. Read once. Both ranks must run the same
/// value (`startup_parity`): each scores half the vocabulary and swaps pairs.
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_VERIFY_VOCAB_SPLIT").as_deref() == Ok("1"))
}

/// This rank's shard as MXFP8 (first vocab row, weight), when prepared.
static SHARD_MX: OnceLock<(usize, crate::weight_map::Mxfp8Weight)> = OnceLock::new();

/// Quantize this rank's half of a BF16 GLM head to MXFP8 before KV sizing
/// (`ATLAS_GLM_LM_HEAD_MXFP8=1`, ~0.33 GB); the split verify head then
/// streams half the bytes for up to 32 rows at a time.
pub(crate) fn prepare_shard_mxfp8(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lm_head: DevicePtr,
    bf16_head: bool,
) -> Result<()> {
    if config.model_type != "glm5_next"
        || !bf16_head
        || config.ep_world_size != 2
        || !enabled()
        || std::env::var("ATLAS_GLM_LM_HEAD_MXFP8").as_deref() != Ok("1")
    {
        return Ok(());
    }
    let (vocab, h) = (config.vocab_size, config.hidden_size);
    ensure!(
        vocab % 2 == 0,
        "GLM vocab split needs an even vocabulary ({vocab})"
    );
    let shard = vocab / 2;
    let start = config.ep_rank * shard;
    let quantize = gpu.kernel("mxfp8_gemv", "mxfp8_quantize_bf16")?;
    let data = gpu.alloc(shard * h)?;
    let scales = gpu.alloc(shard * h / ops::MXFP8_BLOCK)?;
    let stream = gpu.default_stream();
    ops::mxfp8_quantize(
        gpu,
        quantize,
        lm_head.offset(start * h * 2),
        data,
        scales,
        shard,
        h,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let _ = SHARD_MX.set((start, crate::weight_map::Mxfp8Weight { data, scales }));
    tracing::info!(
        rank = config.ep_rank,
        shard,
        "GLM split verify head: MXFP8 shard ready"
    );
    Ok(())
}

impl TransformerModel {
    /// Why the split head would not serve a verify of `rows` rows, if it
    /// would not. Every term is identical on both ranks, so both decline
    /// together.
    fn glm_split_head_decline(&self, rows: usize) -> Option<&'static str> {
        let Some(comm) = self.comm.as_ref() else {
            return Some("no_comm");
        };
        [
            (!enabled(), "disabled"),
            (self.config.model_type != "glm5_next", "model"),
            (comm.world_size() != 2, "world"),
            (!comm.supports_peer_exchange_async(), "peer_exchange"),
            (self.lm_head_fp8.is_some(), "lm_head_fp8"),
            (self.lm_head_nvfp4.is_some(), "lm_head_nvfp4"),
            (self.overlays.is_some(), "overlays"),
            (self.logit_softcap_kernel.0 != 0, "softcap"),
            (self.dense_gemv_batchm_kernel.0 == 0, "batchm_kernel"),
            (rows == 0 || rows * 16 > PAIRS_BYTES, "rows"),
        ]
        .into_iter()
        .find_map(|(hit, why)| hit.then_some(why))
    }

    /// The split head serves GLM verifies, which then leave half of the
    /// logits buffer stale: verify picks must stay on the raw argmax.
    pub(crate) fn glm_verify_logits_argmax_only(&self) -> bool {
        self.glm_split_head_decline(1).is_none()
    }

    /// Project `normed` [rows, H] and write global argmax IDs to `out` [rows].
    /// Rows whose bit is set in `ban_rows` never pick one of `ban.ids`.
    /// With `allow` (row-major full-vocabulary bitmasks, one per row, from
    /// `glm_verify_masks`), row r picks the best token its mask allows.
    /// Returns `Ok(false)` without launching when the split does not apply.
    pub(super) fn glm_split_head_argmax(
        &self,
        normed: DevicePtr,
        rows: usize,
        out: DevicePtr,
        ban: (u64, &EosBan),
        allow: Option<DevicePtr>,
        stream: u64,
    ) -> Result<bool> {
        if let Some(why) = self.glm_split_head_decline(rows) {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(
                    "GLM vocab-split verify head declined ({why}, rows={rows}); \
                     min_tokens end-token ban inactive"
                )
            });
            return Ok(false);
        }
        let Some(comm) = self.comm.as_ref() else {
            return Ok(false);
        };
        static KERNELS: OnceLock<(KernelHandle, KernelHandle)> = OnceLock::new();
        let (value_k, merge_k) = match KERNELS.get() {
            Some(k) => *k,
            None => *KERNELS.get_or_init(|| {
                (
                    self.gpu
                        .kernel("argmax", "argmax_bf16_value_ban")
                        .unwrap_or(KernelHandle(0)),
                    self.gpu
                        .kernel("argmax", "argmax_pair_merge_ban")
                        .unwrap_or(KernelHandle(0)),
                )
            }),
        };
        if value_k.0 == 0 || merge_k.0 == 0 {
            return Ok(false);
        }
        let vocab = self.config.vocab_size;
        let h = self.config.hidden_size;
        ensure!(
            vocab.is_multiple_of(2),
            "GLM vocab split needs an even vocabulary ({vocab})"
        );
        let shard = vocab / 2;
        let start = comm.rank() * shard;
        let logits = self.buffers.logits();
        let shard_weight = DenseWeight {
            weight: self.lm_head_weight.weight.offset(start * h * 2),
        };
        // Up to 8 rows per batch-M pass; wider owner-batched verifies read the
        // shard once per 32 rows on the tensor cores.
        static TC: OnceLock<(KernelHandle, KernelHandle)> = OnceLock::new();
        let (tc16, tc32) = *TC.get_or_init(|| {
            let k = |name| {
                self.gpu
                    .kernel("dense_gemv_bf16_batchm", name)
                    .unwrap_or(KernelHandle(0))
            };
            (k("dense_gemv_bf16_tc16"), k("dense_gemv_bf16_tc32"))
        });
        let max_m = if rows > ops::DENSE_GEMV_BATCHM_MAX_M as usize && tc32.0 != 0 {
            ops::DENSE_GEMV_TC_MAX_M as usize
        } else {
            ops::DENSE_GEMV_BATCHM_MAX_M as usize
        };
        static MX: OnceLock<[KernelHandle; 3]> = OnceLock::new();
        let mx = SHARD_MX.get().filter(|(s, _)| *s == start).map(|(_, w)| {
            let k = *MX.get_or_init(|| {
                ["mxfp8_gemv_tc8", "mxfp8_gemv_tc16", "mxfp8_gemv_tc32"].map(|name| {
                    self.gpu
                        .kernel("mxfp8_gemv", name)
                        .unwrap_or(KernelHandle(0))
                })
            });
            (*w, k)
        });
        if let Some((w, k)) = mx.filter(|(_, k)| k.iter().all(|k| k.0 != 0)) {
            for first in (0..rows).step_by(32) {
                let m = (rows - first).min(32);
                ops::mxfp8_gemv(
                    self.gpu.as_ref(),
                    k[match m {
                        0..=8 => 0,
                        9..=16 => 1,
                        _ => 2,
                    }],
                    normed.offset(first * h * 2),
                    w.data,
                    w.scales,
                    logits.offset((first * vocab + start) * 2),
                    m as u32,
                    shard as u32,
                    h as u32,
                    vocab as u32,
                    stream,
                )?;
            }
        }
        for first in (0..rows).step_by(max_m).filter(|_| mx.is_none()) {
            let m = (rows - first).min(max_m);
            let (input, output) = (
                normed.offset(first * h * 2),
                logits.offset((first * vocab + start) * 2),
            );
            if m > ops::DENSE_GEMV_BATCHM_MAX_M as usize {
                let kernel = if m <= 16 { tc16 } else { tc32 };
                ops::dense_gemv_bf16_tc(
                    self.gpu.as_ref(),
                    kernel,
                    input,
                    &shard_weight,
                    output,
                    m as u32,
                    shard as u32,
                    h as u32,
                    vocab as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv_batchm(
                    self.gpu.as_ref(),
                    self.dense_gemv_batchm_kernel,
                    input,
                    &shard_weight,
                    output,
                    m as u32,
                    shard as u32,
                    h as u32,
                    vocab as u32,
                    stream,
                )?;
            }
        }
        let local = self.buffers.scratch().offset(PAIRS_OFFSET);
        let peer = local.offset(PAIRS_BYTES);
        let (ban_rows, ban) = ban;
        ensure!(
            ban_rows == 0 || !self.gpu.stream_is_capturing(stream),
            "min_tokens verify ban cannot be graph-captured"
        );
        let local_id = |id: u32| {
            (id as usize)
                .checked_sub(start)
                .filter(|&i| i < shard)
                .map_or(u32::MAX, |i| i as u32)
        };
        let allow = match allow {
            Some(masks) => {
                static ALLOW_K: OnceLock<KernelHandle> = OnceLock::new();
                let k = *ALLOW_K.get_or_init(|| {
                    self.gpu
                        .kernel("argmax", "argmax_bf16_value_ban_allow")
                        .unwrap_or(KernelHandle(0))
                });
                ensure!(
                    k.0 != 0,
                    "masked verify: argmax_bf16_value_ban_allow missing"
                );
                ensure!(
                    !self.gpu.stream_is_capturing(stream),
                    "masked verify cannot be graph-captured"
                );
                Some((k, masks, self.config.vocab_size.div_ceil(32)))
            }
            None => None,
        };
        let ban_ids = ban.head_ids().map(local_id);
        launch_shard_argmax(
            self.gpu.as_ref(),
            value_k,
            allow,
            logits.offset(start * 2),
            local,
            rows,
            (start, shard, vocab),
            ban_ids,
            stream,
        )?;
        comm.peer_exchange_async(local.0, peer.0, rows * 16, stream)?;
        KernelLaunch::new(self.gpu.as_ref(), merge_k)
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(local)
            .arg_ptr(peer)
            .arg_ptr(out)
            .arg_u32(rows as u32)
            .arg_u32(shard as u32)
            .arg_u32(comm.rank() as u32)
            .arg_u32(ban_rows as u32)
            .arg_u32((ban_rows >> 32) as u32)
            .launch(stream)?;
        Ok(true)
    }
}

/// Launch the per-row shard argmax: `argmax_bf16_value_ban`, or under row
/// masks `argmax_bf16_value_ban_allow` (`(kernel, masks, words per row)`):
/// each row's full-vocabulary mask, read from bit `start` on, so the rank's
/// first vocabulary row may fall anywhere in a mask word (GLM-5.3 splits
/// at 77,428).
#[allow(clippy::too_many_arguments)]
fn launch_shard_argmax(
    gpu: &dyn GpuBackend,
    value_k: KernelHandle,
    allow: Option<(KernelHandle, DevicePtr, usize)>,
    shard_logits: DevicePtr,
    local: DevicePtr,
    rows: usize,
    (start, shard, vocab): (usize, usize, usize),
    ban_ids: impl IntoIterator<Item = u32>,
    stream: u64,
) -> Result<()> {
    let kernel = allow.map_or(value_k, |(k, _, _)| k);
    let mut launch = KernelLaunch::new(gpu, kernel)
        .grid([rows as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(shard_logits)
        .arg_ptr(local)
        .arg_u32(shard as u32)
        .arg_u32(vocab as u32);
    if let Some((_, masks, words)) = allow {
        launch = launch
            .arg_ptr(masks)
            .arg_u32(words as u32)
            .arg_u32(start as u32);
    }
    for id in ban_ids {
        launch = launch.arg_u32(id);
    }
    launch.launch(stream)
}

#[cfg(test)]
#[path = "glm_vocab_split_tests.rs"]
mod tests;
