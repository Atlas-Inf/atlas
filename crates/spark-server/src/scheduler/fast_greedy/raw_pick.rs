// SPDX-License-Identifier: AGPL-3.0-only

//! When a row's raw GPU argmax IS its pick, and the one-row pick built on it.
//!
//! Plain decode, the MTP bootstrap and the verify fast paths each skip the
//! host pipeline (D2H + dequant + masks + penalties + argmax) when it cannot
//! move the argmax. They used to decide that differently: decode sent
//! never-thinking rows to the GPU with penalties and bias ignored, but applied
//! them on the host whenever another row in the batch was thinking, and the
//! bootstrap took the raw argmax inside `<think>` with no pipeline at all. A
//! pick then depended on the batch's composition and on which path served it.
//! One rule now: the GPU argmax (lowest index on exact ties, as the host
//! argmax) is used only where the pipeline provably keeps it.

use crate::scheduler::ActiveSeq;
use crate::scheduler::logit_processors::LogitsContext;
use spark_model::traits::Model;
use spark_runtime::gpu::DevicePtr;

/// Whether the host pipeline provably leaves a greedy row's pick at its raw
/// argmax, grammar aside (callers check the grammar bitmask themselves):
/// outside `<think>` (where F2, the mid-word defer, the `</think>` deferral
/// and the A4 floor act), penalties exactly neutral, no logit bias, and no
/// one-shot pin to the tool-call opener. The pick must also not be one of the
/// ids the remaining stages mask ([`raw_pick_masked`]).
pub(in crate::scheduler) fn raw_argmax_is_pick(a: &ActiveSeq) -> bool {
    !a.inside_thinking
        && a.repetition_penalty == 1.0
        && a.presence_penalty == 0.0
        && a.frequency_penalty == 0.0
        && a.lz_penalty == 0.0
        && a.dry_multiplier == 0.0
        && a.logit_bias.is_empty()
        && !tool_pin_armed(a)
}

/// `PinToToolCallStart` is armed: the pick right after `</think>` on a turn
/// that must call a tool is forced to the opener.
pub(in crate::scheduler) fn tool_pin_armed(a: &ActiveSeq) -> bool {
    a.think_just_ended && a.require_tool_call && !a.tool_call_opened && !a.inside_thinking
}

/// Ids the pipeline masks or biases outside `<think>`: `</think>` and
/// `<think>` (post-close mask), and the tool-call opener under the tool-loop
/// bias. A raw argmax on one of them must be redone on the host.
pub(in crate::scheduler) fn raw_pick_masked(a: &ActiveSeq, tok: u32) -> bool {
    Some(tok) == a.think_end_token
        || Some(tok) == a.think_start_token
        || (a.suppress_tool_call && Some(tok) == a.tool_call_start_token)
}

/// One row's pick as plain decode makes it: the GPU argmax where
/// [`raw_argmax_is_pick`] holds and the grammar allows it, otherwise the host
/// pipeline on this row (which advances the row's pipeline state, as decode's
/// does). `gpu_argmax` is the row's argmax when the caller already read it.
pub(in crate::scheduler) fn pick_row(
    model: &dyn Model,
    logits: DevicePtr,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    gpu_argmax: Option<u32>,
) -> anyhow::Result<u32> {
    crate::scheduler::emit_step::PickEffects::clear(a);
    let greedy = a.temperature == 0.0 || ctx.sampling.force_temp_zero;
    let fp32 = model.decode_logits_fp32();
    if greedy && raw_argmax_is_pick(a) {
        let mut top1 = [match gpu_argmax {
            Some(t) => t,
            None => model.argmax_on_device(logits, 0)?,
        }];
        // The min_tokens end-token ban: the masked row's argmax.
        let served =
            crate::scheduler::min_tokens_ban::fix_raw_picks(model, a, &mut top1, logits, fp32)?;
        let [top1] = top1;
        // A terminated matcher masks nothing (`GrammarBitmaskApply`).
        let allowed = a
            .grammar_state
            .as_mut()
            .is_none_or(|gs| !gs.fill_bitmask() || gs.is_token_allowed(top1));
        if served && allowed && !raw_pick_masked(a, top1) {
            return Ok(top1);
        }
    }
    let vocab = model.vocab_size();
    let mut row = crate::scheduler::verify_pipeline_helper::HostRows::take(
        &ctx.scratch.host_bytes,
        vocab * if fp32 { 4 } else { 2 },
    );
    model.copy_logits_to_host(logits, &mut row)?;
    Ok(
        crate::scheduler::verify_pipeline_helper::verify_pick_with_pipeline(
            &row, fp32, vocab, a, ctx,
        ),
    )
}
