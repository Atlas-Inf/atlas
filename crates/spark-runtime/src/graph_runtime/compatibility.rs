// SPDX-License-Identifier: AGPL-3.0-only

use super::{GraphFallbackReason, GraphMode, GraphPhase};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatibilityRule {
    pub id: String,
    pub phases: Vec<GraphPhase>,
    pub modes: Vec<GraphMode>,
    pub fallback_reason: GraphFallbackReason,
    pub detail: String,
}

impl CompatibilityRule {
    pub fn applies(&self, phase: GraphPhase, mode: GraphMode) -> bool {
        self.phases.contains(&phase) && self.modes.contains(&mode)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompatibilityDecision {
    Supported,
    EagerFallback {
        rule_id: String,
        reason: GraphFallbackReason,
        detail: String,
    },
}

pub struct CompatibilityRegistry {
    rules: BTreeMap<String, CompatibilityRule>,
}

impl CompatibilityRegistry {
    pub fn new(rules: impl IntoIterator<Item = CompatibilityRule>) -> Result<Self, String> {
        let mut indexed = BTreeMap::new();
        for rule in rules {
            if rule.id.trim().is_empty() {
                return Err("graph compatibility rule id must not be empty".to_string());
            }
            if rule.phases.is_empty() || rule.modes.is_empty() {
                return Err(format!(
                    "graph compatibility rule {:?} must name at least one phase and mode",
                    rule.id
                ));
            }
            let id = rule.id.clone();
            if indexed.insert(id.clone(), rule).is_some() {
                return Err(format!("duplicate graph compatibility rule {id:?}"));
            }
        }
        Ok(Self { rules: indexed })
    }

    pub fn decide<'a>(
        &self,
        active_rule_ids: impl IntoIterator<Item = &'a str>,
        phase: GraphPhase,
        mode: GraphMode,
    ) -> CompatibilityDecision {
        for id in active_rule_ids {
            if let Some(rule) = self.rules.get(id)
                && rule.applies(phase, mode)
            {
                return CompatibilityDecision::EagerFallback {
                    rule_id: rule.id.clone(),
                    reason: rule.fallback_reason,
                    detail: rule.detail.clone(),
                };
            }
        }
        CompatibilityDecision::Supported
    }

    pub fn rules(&self) -> impl Iterator<Item = &CompatibilityRule> {
        self.rules.values()
    }
}

#[cfg(test)]
#[path = "compatibility_tests.rs"]
mod tests;
