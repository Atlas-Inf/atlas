// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-native DeepSeek-V4 and V4.1 chat encoding.
//!
//! The official 0731 checkpoint ships `encoding_dsv4.py`, not a Hugging Face
//! chat template. Keeping the preprocessing in Rust avoids approximating its
//! tool-result merge, reasoning-history, and task-transition rules in Jinja.
//! V4.1-Flash-1017 ships `encoding/encoding.py`, the same encoding with the
//! differences [`Dialect::V41`] names.

use anyhow::{Result, bail};

mod preprocess;
mod render;
mod v41;

pub(super) const BOS: &str = "<｜begin▁of▁sentence｜>";
pub(super) const EOS: &str = "<｜end▁of▁sentence｜>";
pub(super) const USER: &str = "<｜User｜>";
pub(super) const ASSISTANT: &str = "<｜Assistant｜>";
pub(super) const LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub(super) const THINK_START: &str = "<think>";
pub(super) const THINK_END: &str = "</think>";
pub(super) const DSML: &str = "｜DSML｜";
/// V4.1 only: opens the leading system block and any later system message.
pub(super) const SYSTEM: &str = "<｜System｜>";

/// Which checkpoint's encoding to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dialect {
    V4,
    /// V4.1 changes, all from the checkpoint's `encoding.py`: DSML tag names
    /// carry a leading space; reasoning effort is a number from 1 to 100;
    /// `<｜System｜>` opens the system block, and a system message may come
    /// mid-conversation, where it starts a turn like a user message; tools may
    /// be namespaced (`namespace::name`); an image is a placeholder token.
    V41,
}

/// DSML tag names: (block of calls, one call, one parameter).
pub(super) const fn dsml_tags(dialect: Dialect) -> (&'static str, &'static str, &'static str) {
    match dialect {
        Dialect::V4 => ("tool_calls", "invoke", "parameter"),
        Dialect::V41 => (" calls", " invoke", " parameter"),
    }
}

impl Dialect {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::V4 => "DeepSeek-V4",
            Self::V41 => "DeepSeek-V4.1",
        }
    }

    /// The text that opens the prompt in thinking mode; empty otherwise.
    fn effort_prefix(self, value: Option<&str>, thinking: bool) -> Result<String> {
        if !thinking {
            return Ok(String::new());
        }
        match self {
            Self::V4 => Ok(ReasoningEffort::parse(value)?.prefix().to_string()),
            Self::V41 => {
                let budget = match value.unwrap_or("high") {
                    // low, high and max are the encoder's; the OpenAI names
                    // map as the checkpoint's chat_template.jinja maps them.
                    "low" | "minimal" => 50,
                    "high" | "medium" | "xhigh" | "none" => 75,
                    "max" => 100,
                    other => match other.parse::<u8>() {
                        Ok(budget @ 1..=100) => budget,
                        _ => bail!(
                            "invalid DeepSeek-V4.1 reasoning effort '{other}': \
                             expected 1-100, low, high or max"
                        ),
                    },
                };
                Ok(format!(
                    "Reasoning Effort: {budget} (range 1-100, the higher the value, \
                     the more thorough the reasoning)\n\n"
                ))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningEffort {
    Low,
    High,
    Max,
}

impl ReasoningEffort {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("high") {
            "low" | "minimal" | "medium" | "none" => Ok(Self::Low),
            "high" => Ok(Self::High),
            "max" | "xhigh" => Ok(Self::Max),
            other => bail!("invalid DeepSeek-V4 reasoning effort '{other}'"),
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Low => "",
            Self::High => {
                "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
                 You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
                 Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
            }
            Self::Max => {
                "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n\
                 You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n\
                 Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"
            }
        }
    }
}

pub(super) fn encode_messages(
    dialect: Dialect,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> Result<String> {
    if messages.is_empty() {
        bail!("{} requires at least one message", dialect.name());
    }

    let effort = dialect.effort_prefix(reasoning_effort, enable_thinking)?;
    let mut messages = messages.to_vec();
    if dialect == Dialect::V41 {
        v41::prepare(&mut messages)?;
    }
    preprocess::attach_top_level_tools(&mut messages, tools)?;
    let mut messages = preprocess::merge_tool_messages(&messages, dialect)?;
    preprocess::sort_tool_results_by_call_order(&mut messages);
    if dialect == Dialect::V41 {
        v41::images_to_placeholders(&mut messages)?;
    }

    let drop_thinking = !messages.iter().any(|m| {
        m.get("tools")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|v| !v.is_empty())
    });
    if enable_thinking && drop_thinking {
        messages = preprocess::drop_thinking_messages(&messages, dialect);
    }

    render::render_messages(&messages, enable_thinking, drop_thinking, dialect, &effort)
}

pub(super) fn python_json(value: &serde_json::Value) -> Result<String> {
    render::python_json(value)
}
