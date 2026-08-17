// SPDX-License-Identifier: AGPL-3.0-only

use super::contract::LightningDsparkProfile;
use super::product_policy::{LightningDsparkProductPolicy, LightningDsparkRuntimeToggles};

fn product_toggles() -> LightningDsparkRuntimeToggles {
    LightningDsparkRuntimeToggles {
        option_b_enabled: true,
        proposal_lane_count: 1,
        proposal_graph_eligible: true,
        target_verify_graph_eligible: true,
        batched_verify_enabled: true,
        seam_serial_enabled: false,
        draft_cap_override: None,
        adaptive_enabled: false,
    }
}

#[test]
fn exact_lightning_profile_and_product_toggles_admit() {
    let policy = LightningDsparkProductPolicy::try_new(
        LightningDsparkProfile::lightning(),
        product_toggles(),
    )
    .expect("official Lightning product policy");

    assert_eq!(policy.profile().algorithm, "DSpark");
    assert_eq!(policy.profile().served_gamma, 4);
    assert_eq!(policy.profile().num_drafts, 3);
    assert_eq!(policy.profile().checkpoint.block_size, 8);
    assert_eq!(policy.runtime_toggles().proposal_lane_count, 1);
}

#[test]
fn profile_validate_remains_authoritative() {
    let mut profile = LightningDsparkProfile::lightning();
    profile.num_drafts = 4;

    let error = LightningDsparkProductPolicy::try_new(profile, product_toggles())
        .expect_err("profile K drift must remain rejected");
    assert!(error.to_string().contains("num_drafts"), "{error}");
    assert!(error.to_string().contains("expected 3"), "{error}");
}

#[test]
fn rejects_each_forbidden_product_runtime_toggle() {
    let cases = [
        (
            "option_b_enabled",
            LightningDsparkRuntimeToggles {
                option_b_enabled: false,
                ..product_toggles()
            },
        ),
        (
            "proposal_lane_count",
            LightningDsparkRuntimeToggles {
                proposal_lane_count: 2,
                ..product_toggles()
            },
        ),
        (
            "proposal_graph_eligible",
            LightningDsparkRuntimeToggles {
                proposal_graph_eligible: false,
                ..product_toggles()
            },
        ),
        (
            "target_verify_graph_eligible",
            LightningDsparkRuntimeToggles {
                target_verify_graph_eligible: false,
                ..product_toggles()
            },
        ),
        (
            "batched_verify_enabled",
            LightningDsparkRuntimeToggles {
                batched_verify_enabled: false,
                ..product_toggles()
            },
        ),
        (
            "seam_serial_enabled",
            LightningDsparkRuntimeToggles {
                seam_serial_enabled: true,
                ..product_toggles()
            },
        ),
        (
            "draft_cap_override",
            LightningDsparkRuntimeToggles {
                draft_cap_override: Some(2),
                ..product_toggles()
            },
        ),
        (
            "adaptive_enabled",
            LightningDsparkRuntimeToggles {
                adaptive_enabled: true,
                ..product_toggles()
            },
        ),
    ];

    for (field, toggles) in cases {
        let error =
            LightningDsparkProductPolicy::try_new(LightningDsparkProfile::lightning(), toggles)
                .expect_err("diagnostic/runtime override must fail closed");
        assert!(error.to_string().contains(field), "{field}: {error}");
    }
}

#[test]
fn generic_runtime_shape_is_not_a_lightning_product_policy() {
    let error = LightningDsparkProductPolicy::try_new(
        LightningDsparkProfile::lightning(),
        LightningDsparkRuntimeToggles {
            proposal_lane_count: 3,
            ..product_toggles()
        },
    )
    .expect_err("multi-lane generic DFlash must not become Lightning");
    assert!(error.to_string().contains("proposal_lane_count"));
}
