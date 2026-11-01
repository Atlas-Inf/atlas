// SPDX-License-Identifier: AGPL-3.0-only

//! Misc grammar compilation: structural-tag, JSON schema, JSON, EBNF, and
//! the tool-call-or-JSON union.

use xgrammar::{CompiledGrammar, Grammar};

use super::engine::{GrammarEngine, GrammarError};
use crate::tool_parser::{ToolCallParser, ToolDefinition};

/// Largest JSON schema text compiled for response_format.
pub(crate) const MAX_SCHEMA_TEXT_BYTES: usize = 1 << 20;
/// Largest compiled response_format grammar structure. Masks are excluded:
/// they fill lazily as the grammar is used, and the compiler cache counts
/// them only when it next measures (on insert), so one live grammar's masks
/// grow with the parser states its requests visit. A strict object of 96
/// required object-valued properties compiles to a few MiB.
pub(crate) const MAX_SCHEMA_GRAMMAR_BYTES: usize = 64 << 20;

impl GrammarEngine {
    /// Build and compile a structural tag grammar from raw JSON components.
    /// This bypasses xgrammar-rs's `compile_structural_tag` wrapper to access
    /// `at_least_one` and `stop_after_first` parameters.
    ///
    /// Inside [`Self::compile_tool_or_json_grammar`] the tags become the
    /// union's tool branch: calls from the first output token
    /// (`at_least_one`), parallel calls after it (no `stop_after_first`).
    pub(super) fn compile_structural_tag_raw(
        &mut self,
        triggers: &[String],
        tags: &[serde_json::Value],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        let (at_least_one, stop_after_first) = if self.json_alternative.is_some() {
            (true, false)
        } else {
            (at_least_one, stop_after_first)
        };
        let structural_tag_json = serde_json::json!({
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": triggers,
                "tags": tags,
                "at_least_one": at_least_one,
                "stop_after_first": stop_after_first,
            }
        })
        .to_string();

        let grammar = Grammar::from_structural_tag(&structural_tag_json)
            .map_err(GrammarError::Compilation)?;
        match &self.json_alternative {
            Some(json) => self
                .compiler
                .compile_schema_grammar(&Grammar::union(&[json.clone(), grammar])),
            None => self.compiler.compile_grammar(&grammar),
        }
        .map_err(GrammarError::Compilation)
    }

    /// Compile `parser`'s tool-call grammar unioned with the response_format
    /// JSON (`GrammarSpec::ToolCallOrJson`; `schema` `None` is json_object):
    /// the output is either tool calls from its first token, or the JSON with
    /// nothing around it. The parser builds its auto-mode tags as for any
    /// request (`use_triggers`); `compile_structural_tag_raw` makes them the
    /// union's tool branch. `None` when the parser has no tool grammar.
    pub fn compile_tool_or_json_grammar(
        &mut self,
        parser: &dyn ToolCallParser,
        tools: &[ToolDefinition],
        schema: Option<&str>,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        let json = match Self::json_grammar(schema) {
            Ok(json) => json,
            Err(e) => return Some(Err(e)),
        };
        self.json_alternative = Some(json);
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            parser.compile_tool_grammar(self, tools, true)
        }));
        // Cleared on every path: a later tool-only grammar must not inherit it.
        self.json_alternative = None;
        compiled.unwrap_or_else(|p| std::panic::resume_unwind(p))
    }

    /// The union's answer branch first-token mask: the response_format JSON
    /// compiled alone (as a response_format request would, so it is cached
    /// alike), at its start state. Picks the union's branch (`branch`).
    pub fn json_start_mask(&mut self, schema: Option<&str>) -> Result<Box<[i32]>, GrammarError> {
        let compiled = match schema {
            Some(schema) => self.compile_json_schema(schema)?,
            None => self.compile_json_grammar()?,
        };
        let mut matcher = xgrammar::GrammarMatcher::new(&compiled, None, false, -1)
            .map_err(GrammarError::Compilation)?;
        let mut mask = xgrammar::allocate_token_bitmask(1, self.vocab_size());
        matcher.fill_next_token_bitmask(&mut mask, 0, false);
        Ok(mask)
    }

    /// The response_format JSON as a grammar, with the options and limits of
    /// [`Self::compile_json_schema`] / [`Self::compile_json_grammar`].
    fn json_grammar(schema: Option<&str>) -> Result<Grammar, GrammarError> {
        match schema {
            None => Grammar::builtin_json_grammar(),
            Some(schema) => {
                Self::check_schema_text(schema)?;
                Grammar::from_json_schema(
                    schema,
                    Self::JSON_ANY_WHITESPACE,
                    None,
                    None::<(&str, &str)>,
                    Self::JSON_STRICT_MODE,
                    Some(Self::MAX_JSON_SCHEMA_WHITESPACE),
                    false,
                )
            }
        }
        .map_err(GrammarError::Compilation)
    }

    fn check_schema_text(schema: &str) -> Result<(), GrammarError> {
        if schema.len() > MAX_SCHEMA_TEXT_BYTES {
            return Err(GrammarError::Compilation(format!(
                "JSON schema is {} bytes, over the {MAX_SCHEMA_TEXT_BYTES}-byte limit",
                schema.len()
            )));
        }
        Ok(())
    }

    // ── JSON schema grammar ──

    /// Cap on consecutive inter-token whitespace characters in
    /// response_format json_schema grammars. Unlimited whitespace
    /// (`None`) gives a degenerating model a grammar-legal runway:
    /// measured on the FP8 flagship (issue #131, N=8 strict json_schema
    /// at temp 0), 5/8 requests padded hundreds of \n/\t between JSON
    /// tokens until max_tokens, leaving unterminated JSON. 8 permits
    /// normal pretty-printing (newline + indent) while making runaway
    /// whitespace grammar-illegal, so the matcher forces a structural
    /// token instead.
    const MAX_JSON_SCHEMA_WHITESPACE: i32 = 8;
    /// Whitespace between JSON tokens (bounded by the cap above).
    const JSON_ANY_WHITESPACE: bool = true;
    /// Unlisted properties are refused unless the schema allows them.
    const JSON_STRICT_MODE: bool = true;

    /// Compile a grammar that enforces a JSON schema.
    ///
    /// Refuses a schema over [`MAX_SCHEMA_TEXT_BYTES`] before compiling; the
    /// compiler refuses (and never caches) one whose grammar is over
    /// [`MAX_SCHEMA_GRAMMAR_BYTES`] or over the FSM encoding limits.
    /// response_format reports either as an invalid request (HTTP 400).
    pub fn compile_json_schema(&mut self, schema: &str) -> Result<CompiledGrammar, GrammarError> {
        Self::check_schema_text(schema)?;
        self.compiler
            .compile_json_schema(
                schema,
                Self::JSON_ANY_WHITESPACE,
                None,                 // indent
                None::<(&str, &str)>, // separators
                Self::JSON_STRICT_MODE,
                Some(Self::MAX_JSON_SCHEMA_WHITESPACE),
            )
            .map_err(GrammarError::Compilation)
    }

    /// Compile the built-in JSON grammar (any valid JSON).
    pub fn compile_json_grammar(&mut self) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_builtin_json_grammar()
            .map_err(GrammarError::Compilation)
    }

    /// Compile a grammar from an EBNF string.
    pub fn compile_ebnf(
        &mut self,
        ebnf: &str,
        root_rule: &str,
    ) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_grammar_from_ebnf(ebnf, root_rule)
            .map_err(GrammarError::Compilation)
    }
}
