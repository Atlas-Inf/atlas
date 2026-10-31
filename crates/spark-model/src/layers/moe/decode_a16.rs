// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_FP4L_DECODE_A16=1`: every routed FFN row that is not a prompt
//! row (decode, DFlash verify and owner-batched verify rows) runs W4A16, BF16
//! activations against the NVFP4 weights, as the reference does (vLLM b12x
//! with `B12X_W4A16_A4_PREFILL=1`: the scheduler's prompt ranges A4, every
//! other row A16). Prompt rows keep NVFP4 activations (W4A4).
//!
//! The GLM layers' decode entry points hold a [`DecodeRows`] scope; a prompt
//! chunk that carries verify rows (`prefill_with_glm_passengers`) holds a
//! [`PromptRows`] scope, and its routed FFN runs the two halves apart
//! ([`MoeLayer::forward_prefill_mixed`]). Each routed FFN then picks its lane
//! ([`A16Scope`]): the persistent stream twins' `_a16` kernels when they take
//! the batch (`glm_moe_decode_a16.cuh`), else the grouped W4A16 path.

use super::*;
use anyhow::ensure;

/// `ATLAS_GLM_FP4L_DECODE_A16` is `1`. Read once.
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_FP4L_DECODE_A16").as_deref() == Ok("1"))
}

/// The routed lane of the batch in progress on this thread ([`A16Scope`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Activations {
    /// NVFP4 activations (prompt rows, or the switch off).
    A4,
    /// W4A16 on the stream decode twins' `_a16` kernels: the prequant path's
    /// routing, tiles and schedule with BF16 activations
    /// (glm_moe_decode_a16.cuh).
    A16Stream,
    /// W4A16 on the grouped BF16-activation kernels (`prequant_moe` false):
    /// non-prompt batches the stream twins do not take.
    A16Grouped,
}

/// The lane of a routed FFN: A4 unless the switch is on and a [`DecodeRows`]
/// scope holds; then the stream twins when they take the batch (`stream`),
/// else the grouped W4A16 path.
pub(super) fn lane(requested: bool, decode_rows: bool, stream: bool) -> Activations {
    match (requested && decode_rows, stream) {
        (false, _) => Activations::A4,
        (true, true) => Activations::A16Stream,
        (true, false) => Activations::A16Grouped,
    }
}

thread_local! {
    static DECODE_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LANE: std::cell::Cell<Activations> = const { std::cell::Cell::new(Activations::A4) };
    static PROMPT_ROWS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// While held, this thread's layers serve decode or verify rows: the GLM
/// layers' decode entry points (`decode`, `decode_batched`,
/// `decode_multi_seq`, `decode_glm_long_owners`) hold one, prompt prefill
/// does not. Every rank runs the same entry for a step, so every rank picks
/// the same lane.
pub(crate) struct DecodeRows(bool);

impl DecodeRows {
    pub(crate) fn enter() -> Self {
        Self(DECODE_ROWS.with(|c| c.replace(true)))
    }
}

impl Drop for DecodeRows {
    fn drop(&mut self) {
        DECODE_ROWS.with(|c| c.set(self.0));
    }
}

/// While held, this thread's layers serve a prompt chunk of `prompt` rows
/// followed by verify rows (`prefill_with_glm_passengers`): under
/// [`requested`] the routed FFN runs the chunk's rows A4 and the rest A16
/// ([`MoeLayer::forward_prefill_mixed`]), as the reference splits a forward
/// at the scheduler's prompt ranges.
pub(crate) struct PromptRows(Option<usize>);

impl PromptRows {
    pub(crate) fn enter(prompt: usize) -> Self {
        Self(PROMPT_ROWS.with(|c| c.replace(Some(prompt))))
    }

    /// The prompt rows of the mixed batch in progress, taken: the routed FFN
    /// it splits runs both halves without it.
    pub(super) fn take() -> Option<usize> {
        PROMPT_ROWS.with(|c| c.take())
    }
}

impl Drop for PromptRows {
    fn drop(&mut self) {
        PROMPT_ROWS.with(|c| c.set(self.0));
    }
}

/// The lane of one routed FFN, held while it runs.
pub(super) struct A16Scope(Activations);

impl A16Scope {
    /// [`lane`] of this thread's scopes; `stream`: the stream twins' W4A16
    /// kernels take the batch.
    pub(super) fn enter(stream: bool) -> Self {
        let lane = lane(requested(), DECODE_ROWS.with(|c| c.get()), stream);
        Self(LANE.with(|c| c.replace(lane)))
    }

    /// The lane of the routed FFN in progress.
    pub(super) fn lane() -> Activations {
        LANE.with(|c| c.get())
    }
}

impl Drop for A16Scope {
    fn drop(&mut self) {
        LANE.with(|c| c.set(self.0));
    }
}

impl MoeLayer {
    /// A mixed batch's routed FFN ([`PromptRows`]): its verify rows under a
    /// [`DecodeRows`] scope first, their output set aside, then the prompt
    /// rows, then the verify rows' output back after the prompt rows' in
    /// `moe_output`. The routed, shared and router kernels are row-invariant,
    /// so each row's output is the one its half alone gives. False (nothing
    /// run) unless [`requested`] and the batch has verify rows.
    pub(super) fn forward_prefill_mixed(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
        defer_shared_hc: bool,
    ) -> Result<bool> {
        let Some(prompt) = PromptRows::take().filter(|&p| p < num_tokens) else {
            return Ok(false);
        };
        if !requested() {
            return Ok(false);
        }
        ensure!(
            !defer_shared_hc,
            "ATLAS_GLM_FP4L_DECODE_A16: a mixed batch cannot defer its shared expert blend"
        );
        let row = ctx.config.hidden_size * 2;
        let rest = num_tokens - prompt;
        let aside = mixed_aside(ctx.gpu, rest * row, row)?;
        {
            let _rows = DecodeRows::enter();
            self.forward_prefill_mode(input.offset(prompt * row), rest, ctx, stream, false)?;
        }
        let out = ctx.buffers.moe_output();
        ctx.gpu.copy_d2d_async(out, aside, rest * row, stream)?;
        self.forward_prefill_mode(input, prompt, ctx, stream, false)?;
        ctx.gpu
            .copy_d2d_async(aside, out.offset(prompt * row), rest * row, stream)?;
        Ok(true)
    }

    /// The prequant routed path serves this batch: `ATLAS_NVFP4_PREQUANT_MOE`,
    /// unless its [`A16Scope`] took the grouped W4A16 path.
    pub(super) fn prequant_moe(&self) -> bool {
        self.nvfp4_prequant_moe && A16Scope::lane() != Activations::A16Grouped
    }

    /// Whether this batch runs the stream twins' W4A16 kernels.
    pub(super) fn a16_stream(&self) -> bool {
        A16Scope::lane() == Activations::A16Stream
    }
}

/// The verify rows' routed output of a mixed batch while its prompt rows
/// run: one buffer for [`crate::layer::glm_long_owner::MAX_ROWS`] rows,
/// allocated on the first mixed batch.
fn mixed_aside(gpu: &dyn GpuBackend, bytes: usize, row: usize) -> Result<DevicePtr> {
    static ASIDE: std::sync::Mutex<Option<(DevicePtr, usize)>> = std::sync::Mutex::new(None);
    let mut aside = ASIDE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((p, cap)) = *aside
        && cap >= bytes
    {
        return Ok(p);
    }
    let cap = bytes.max(crate::layer::glm_long_owner::MAX_ROWS * row);
    let p = gpu.alloc(cap)?;
    *aside = Some((p, cap));
    Ok(p)
}

#[cfg(test)]
#[path = "decode_a16_tests.rs"]
mod tests;
