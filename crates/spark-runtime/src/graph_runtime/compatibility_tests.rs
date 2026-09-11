// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn registry_returns_a_visible_reason() {
    let registry = CompatibilityRegistry::new([CompatibilityRule {
        id: "host_topk".into(),
        phases: vec![GraphPhase::Decode],
        modes: vec![GraphMode::Full],
        fallback_reason: GraphFallbackReason::HostSynchronization,
        detail: "top-k selection reads device state on the host".into(),
    }])
    .unwrap();
    assert!(matches!(
        registry.decide(["host_topk"], GraphPhase::Decode, GraphMode::Full),
        CompatibilityDecision::EagerFallback {
            reason: GraphFallbackReason::HostSynchronization,
            ..
        }
    ));
}
