// SPDX-License-Identifier: AGPL-3.0-only

//! `emit_step` test modules (the test files sit beside `emit_step.rs`).

use super::*;

#[cfg(test)]
#[path = "../cancel_tests.rs"]
mod cancellation_tests;

#[cfg(test)]
#[path = "../emit_thinking_tests.rs"]
mod thinking_tests;

#[cfg(test)]
#[path = "../glm_tool_boundary_tests.rs"]
mod glm_tool_boundary_tests;

#[cfg(test)]
#[path = "../glm_native_eos_tests.rs"]
mod glm_native_eos_tests;

#[cfg(test)]
#[path = "../strict_grammar_tests.rs"]
mod strict_grammar_tests;
