// SPDX-License-Identifier: AGPL-3.0-only

//! Suppress `<tool_call>` during thinking, bias it down on tool-loop.
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~152-173.
//!
//! - **Inside thinking**: hard-mask (`-inf`) — tool calls inside
//!   `<think>` are unparsable per the canonical qwen3_coder dialect.
//! - **Outside thinking, tool-loop detected (`seq.suppress_tool_call`)**:
//!   strong negative bias (`-12.0`) instead of `-inf` so the model can
//!   still escape if its evidence for a tool call is overwhelming.
//!
//! GLM's configured native opener is itself a reasoning boundary when tools
//! are active. It remains subject to the explicit tool-loop suppression bias.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct ToolCallDuringThinkingMask;

impl LogitsProcessor for ToolCallDuringThinkingMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        let native_boundary = a.tools_present
            && ctx.glm_tool_boundary.is_some()
            && ctx.glm_tool_boundary == ctx.tool_call_start_token;
        if a.inside_thinking && !native_boundary {
            if let Some(tc_start) = ctx.tool_call_start_token {
                let idx = tc_start as usize;
                if idx < logits.len() {
                    logits[idx] = f32::NEG_INFINITY;
                }
            }
        } else if a.suppress_tool_call
            && let Some(tc_start) = ctx.tool_call_start_token
        {
            let idx = tc_start as usize;
            if idx < logits.len() {
                logits[idx] -= 12.0;
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "tool_call_during_thinking_mask"
    }
}
