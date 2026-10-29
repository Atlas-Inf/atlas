// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl std::str::FromStr for ToolCallFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hermes" => Ok(Self::Hermes),
            "qwen3_coder" => Ok(Self::Qwen3Coder),
            "qwen3_xml" => Ok(Self::Qwen3Xml),
            "gemma4" => Ok(Self::Gemma4),
            "mistral" => Ok(Self::Mistral),
            "minimax_xml" => Ok(Self::MinimaxXml),
            "deepseek_v4" | "dsml" => Ok(Self::DeepseekV4),
            "bare_json" => Ok(Self::BareJson),
            "poolside_v1" => Ok(Self::PoolsideV1),
            other => Err(format!(
                "Unknown tool call parser '{other}'. Supported: hermes, qwen3_coder, qwen3_xml, gemma4, mistral, minimax_xml, deepseek_v4, bare_json, poolside_v1",
            )),
        }
    }
}

impl ToolCallFormat {
    /// Create a boxed parser implementation for this format.
    pub fn into_parser(self) -> Box<dyn ToolCallParser> {
        match self {
            Self::Hermes => Box::new(HermesParser),
            Self::Qwen3Coder => Box::new(Qwen3CoderParser),
            Self::Qwen3Xml => Box::new(Qwen3XmlParser),
            Self::Gemma4 => Box::new(Gemma4Parser),
            Self::Mistral => Box::new(MistralNativeParser),
            Self::MinimaxXml => Box::new(MinimaxXmlParser),
            Self::DeepseekV4 => Box::new(DeepseekV4DsmlParser),
            Self::BareJson => Box::new(BareJsonParser),
            Self::PoolsideV1 => Box::new(PoolsideV1Parser),
        }
    }

    /// F66/F69 (2026-04-29): does this parser have a registered XGrammar
    /// schema? Parsers without a grammar fall through to unconstrained
    /// decoding, which (per fix39 testing on MiniMax M2.7) can produce
    /// token-doubled output at the tool-call boundary.
    ///
    /// F69 routed this through the [`ToolCallParser::has_tool_grammar`]
    /// trait method so the per-parser answer is the parser's own — the
    /// previous duplicated `match self` on the enum variant could drift
    /// from the parser's actual `compile_tool_grammar` override.
    pub fn has_grammar(self) -> bool {
        self.into_parser().has_tool_grammar()
    }

    /// F66 (2026-04-29): canonical name used in CLI flags and logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Hermes => "hermes",
            Self::Qwen3Coder => "qwen3_coder",
            Self::Qwen3Xml => "qwen3_xml",
            Self::Gemma4 => "gemma4",
            Self::Mistral => "mistral",
            Self::MinimaxXml => "minimax_xml",
            Self::DeepseekV4 => "deepseek_v4",
            Self::BareJson => "bare_json",
            Self::PoolsideV1 => "poolside_v1",
        }
    }
}

// ── MiniMax XML parser (MiniMax M2.7) ──

// ── Sub-modules (split from monolithic file) ──
mod bare_json;
mod deepseek_v4_dsml;
mod fuzzy_match;
mod gemma4;
mod helpers_a;
mod helpers_b;
mod hermes;
mod minimax_xml;
mod mistral;
mod parse_dispatch;
mod parse_single_a;
mod parse_single_b;
mod parse_tools_tag;
mod pipeline;
mod pipeline_helpers;
mod poolside_v1;

mod qwen3_coder;
mod qwen3_xml;
mod streaming;
mod streaming_emit;
mod streaming_flush;
mod streaming_impl;
mod type_coerce;
pub(crate) mod validation;

pub use bare_json::*;
pub use deepseek_v4_dsml::*;
pub use gemma4::*;
use helpers_a::*;
use helpers_b::*;
pub use hermes::*;
pub use minimax_xml::*;
pub use mistral::*;
pub use parse_dispatch::*;
use parse_single_a::*;
use parse_single_b::*;
use parse_tools_tag::*;
pub use pipeline::*;
use pipeline_helpers::*;
pub use poolside_v1::*;
pub use qwen3_coder::*;
pub use qwen3_xml::*;
pub use streaming::*;
pub use type_coerce::coerce_all;
pub use validation::*;

#[cfg(test)]
mod tests;
