// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_FIRST_APPEND`: what `propose()` appends to the drafter
//! context on top of the prefill capture and the verify commits.
//!
//! `propose()` appends row 0 of the model-global capture buffer unless a
//! commit already covered it. On the first propose after prefill nothing of
//! this sequence is in that buffer yet: the slot gets the previous request's
//! last verify row (zeros after boot; under concurrent traffic, whichever
//! sequence wrote last), so drafts — and with them accept counts and verify
//! widths — depend on request history. `Legacy` keeps that, byte for byte.
//!
//! The other variants differ in what the first propose appends, and none of
//! them trusts the shared row: a single-sequence decode copies its capture
//! into the sequence's own row at once (`keep_own_capture`) and the propose
//! appends from there. The shared row is read only right after this
//! sequence's own verify, which a commit suppresses under unified ctx.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BlockDiffusionDraftHead, DflashProposerState};
use crate::speculative::ProposerState;

/// Strict startup switch: an unknown value is an error, never a default.
const FIRST_APPEND_ENV: &str = "ATLAS_DFLASH_FIRST_APPEND";

/// The slot the first propose after prefill appends (stamped `position - 1`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FirstAppend {
    /// Row 0 of the model-global capture buffer, whoever wrote it.
    #[default]
    Legacy,
    /// Nothing: the context is the prefill capture alone.
    None,
    /// This request's own last prompt position, captured during its prefill.
    Own,
    /// A zero row (what `Legacy` gives the first request after boot).
    Zero,
}

impl FirstAppend {
    pub fn parse(raw: Option<&str>) -> Result<Self> {
        Ok(match raw {
            None | Some("legacy") => Self::Legacy,
            Some("none") => Self::None,
            Some("own") => Self::Own,
            Some("zero") => Self::Zero,
            Some(other) => {
                anyhow::bail!("{FIRST_APPEND_ENV}={other:?} is not one of legacy|none|own|zero")
            }
        })
    }

    /// Startup only. Every rank checks it before loading any weights, so a
    /// typo costs no model load; the head then freezes it (`for_head`).
    pub fn from_env() -> Result<Self> {
        let raw = match std::env::var(FIRST_APPEND_ENV) {
            Ok(raw) => Some(raw),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => anyhow::bail!("{FIRST_APPEND_ENV}: {error}"),
        };
        Self::parse(raw.as_deref())
    }

    /// Read at head construction and logged once.
    pub fn for_head() -> Result<Self> {
        let variant = Self::from_env()?;
        tracing::info!(
            "DFlash first context append: {variant:?} ({FIRST_APPEND_ENV}=legacy|none|own|zero)"
        );
        Ok(variant)
    }
}

/// Where the propose-side append takes its slot from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AppendSource {
    Row(DevicePtr),
    Zero,
}

/// What this sequence itself ran last before a propose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Preceding {
    /// Its prefill: this is the first propose.
    Prefill,
    /// Its single-sequence decode: the token is in its own row.
    Decode,
    /// Its verify: the caller's row is its own verify capture.
    Verify,
    /// Nothing the append may use (e.g. decode steps that captured nothing).
    Other,
}

/// Pure source decision for the propose-side append; `None` = no append.
///
/// `suppressed`: a commit already appended this capture (or the debug switch
/// is set). `stack`: the caller's model-global capture row. `own_row`: this
/// sequence's own row (non-legacy variants only).
pub(super) fn decode_append_source(
    variant: FirstAppend,
    suppressed: bool,
    stack: Option<DevicePtr>,
    room: bool,
    preceding: Preceding,
    own_row: Option<DevicePtr>,
) -> Option<AppendSource> {
    let stack = stack.filter(|_| !suppressed && room)?;
    match (variant, preceding) {
        (FirstAppend::Legacy, _) | (_, Preceding::Verify) => Some(AppendSource::Row(stack)),
        (_, Preceding::Decode) | (FirstAppend::Own, Preceding::Prefill) => {
            own_row.map(AppendSource::Row)
        }
        (FirstAppend::Zero, Preceding::Prefill) => Some(AppendSource::Zero),
        _ => None,
    }
}

/// The row of a prefill pass holding its last position: `proc_count` rows,
/// of which this rank holds those from `sp_row0` on, compacted at row 0.
pub(crate) fn own_capture_row(proc_count: usize, sp_row0: usize) -> Option<usize> {
    proc_count.checked_sub(sp_row0 + 1)
}

/// The caller's capture row now holds this sequence's own verify rows, so
/// its next propose may append from it. No-op for other proposers.
pub(crate) fn note_own_capture(state: &mut dyn ProposerState) {
    if let Some(dstate) = state.as_any_mut().downcast_mut::<DflashProposerState>() {
        dstate.own_capture = true;
    }
}

/// A single-sequence decode just left its token's hidden stack in the
/// model-global capture row (`stack`). A sequence with a row of its own
/// (non-legacy variants) copies it there before anything else can overwrite
/// it, for the propose at `seq_len`. No-op for other proposers.
pub(crate) fn keep_own_capture(
    state: Option<&mut (dyn ProposerState + 'static)>,
    stack: Option<DevicePtr>,
    seq_len: usize,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let Some(dstate) = state.and_then(|s| s.as_any_mut().downcast_mut::<DflashProposerState>())
    else {
        return Ok(());
    };
    // Whatever verify capture the row held is gone.
    dstate.own_capture = false;
    if let (Some(stack), Some(own_row)) = (stack, dstate.own_row) {
        // The model's default stream: the one the decode captured on.
        gpu.copy_d2d_async(stack, own_row, dstate.ctx_slot_bytes, gpu.default_stream())?;
        dstate.own_row_at = Some(seq_len);
    }
    Ok(())
}

impl DflashProposerState {
    /// Context bookkeeping at the end of a prefill whose last pass ended at
    /// position `end` of a `prompt_len`-token prompt: the accumulator holds
    /// the window's last positions, and the propose at `seq_len` is the first.
    pub(crate) fn seed_prefill_ctx(&mut self, prompt_len: usize, end: usize, seq_len: usize) {
        let window_start = prompt_len.max(end).saturating_sub(self.max_ctx_len);
        let new_len = end.saturating_sub(window_start).min(self.max_ctx_len);
        self.ctx_len = new_len;
        // Phase I (v2): seed per-slot fixed positions for the prompt
        // captures. Slot i holds prompt position window_start + i (the
        // tail window kept by try_dflash_prefill_capture_layer). Keep
        // parallel to ctx_len. Re-seed idempotently across prefill chunks.
        self.ctx_positions = (window_start..window_start + new_len)
            .map(|i| i as i32)
            .collect();
        self.first_append_at = Some(seq_len);
    }
}

impl BlockDiffusionDraftHead {
    /// What the propose-side append of `dstate` at `position` appends from
    /// `target_hidden_stack`, read without consuming the one-shot markers.
    pub(super) fn decode_append_planned(
        &self,
        dstate: &DflashProposerState,
        target_hidden_stack: Option<DevicePtr>,
        position: usize,
    ) -> Option<AppendSource> {
        // EAGLE-fix: a commit that already appended this capture (in EAGLE
        // order) sets the one-shot flag so it is not appended twice.
        let eagle_skip = dstate.skip_next_decode_append;
        let verify = dstate.own_capture;
        let decode = dstate.own_row_at == Some(position);
        let prefill = dstate.first_append_at == Some(position);
        let preceding = match (verify, decode, prefill) {
            (true, ..) => Preceding::Verify,
            (_, true, _) => Preceding::Decode,
            (_, _, true) => Preceding::Prefill,
            _ => Preceding::Other,
        };
        decode_append_source(
            self.startup.diagnostics.first_append,
            self.startup.diagnostics.no_decode_append || eagle_skip,
            target_hidden_stack,
            dstate.ctx_len < dstate.max_ctx_len,
            preceding,
            dstate.own_row,
        )
    }

    /// The propose-side context append: at most one slot at `ctx_len`. The
    /// state's one-shot markers are consumed whether or not it appends.
    pub(super) fn append_decode_ctx(
        &self,
        dstate: &mut DflashProposerState,
        target_hidden_stack: Option<DevicePtr>,
        position: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let source = self.decode_append_planned(dstate, target_hidden_stack, position);
        dstate.skip_next_decode_append = false;
        dstate.own_capture = false;
        dstate.own_row_at = None;
        dstate.first_append_at = None;
        let Some(source) = source else {
            return Ok(());
        };
        let slot = dstate
            .ctx_hidden_acc
            .offset(dstate.ctx_len * dstate.ctx_slot_bytes);
        match source {
            AppendSource::Row(row) => {
                gpu.copy_d2d_async(row, slot, dstate.ctx_slot_bytes, stream)?
            }
            AppendSource::Zero => gpu.memset_async(slot, 0, dstate.ctx_slot_bytes, stream)?,
        }
        // Phase I (v2): stamp this slot's TRUE absolute position, fixed
        // forever. The just-decoded token sits at `position - 1` (the
        // full-rebuild formula assigns slot ctx_len the position
        // (position - (ctx_len+1)) + ctx_len == position - 1). Keeping
        // ctx_positions parallel to ctx_len lets precompute rope each
        // slot by its own fixed position instead of a sliding base.
        debug_assert_eq!(dstate.ctx_positions.len(), dstate.ctx_len);
        dstate.ctx_positions.push(position.saturating_sub(1) as i32);
        dstate.ctx_len += 1;
        Ok(())
    }
}
