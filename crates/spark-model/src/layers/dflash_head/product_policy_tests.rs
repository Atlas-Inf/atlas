// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use super::contract::LightningDsparkProfile;
use super::product_policy::{
    LightningDsparkIdentityLatch, LightningDsparkProductPolicy, LightningDsparkRuntimeToggles,
};

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
        batch_parity_enabled: false,
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

fn reader(values: &[(&str, &str)]) -> impl FnMut(&str) -> Option<String> {
    let values = values
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect::<HashMap<_, _>>();
    move |name| values.get(name).cloned()
}

fn read_product(
    values: &[(&str, &str)],
) -> Result<LightningDsparkRuntimeToggles, super::product_policy::LightningDsparkPolicyError> {
    LightningDsparkRuntimeToggles::from_reader(reader(values))
}

#[test]
fn explicit_batch_parity_is_admitted_and_frozen_without_other_diagnostics() {
    let toggles = read_product(&[
        ("ATLAS_DFLASH_OPTION_B", "1"),
        ("ATLAS_DFLASH_BATCH_PARITY", "1"),
    ])
    .unwrap();
    assert!(toggles.batch_parity_enabled);
    toggles.validate().unwrap();
    let execution = super::DsparkStartupExecution::from_lightning(toggles);
    assert!(execution.diagnostics.batch_parity);
    assert!(!execution.diagnostics.verify_trace);
    assert!(!execution.graph_ineligible_diags);
}

#[test]
fn raw_reader_rejects_malformed_or_zero_lane_count() {
    for raw in ["bad", ""] {
        let error = read_product(&[("ATLAS_DFLASH_PROPOSE_LANES", raw)])
            .expect_err("malformed lane count must fail closed");
        assert!(error.to_string().contains("proposal_lane_count"), "{error}");
    }
    let error = read_product(&[("ATLAS_DFLASH_PROPOSE_LANES", "0")])
        .expect_err("zero lane count must fail closed");
    assert!(error.to_string().contains("proposal_lane_count"), "{error}");
}

#[test]
fn raw_reader_represents_valid_non_one_lane_for_validation() {
    let toggles = read_product(&[
        ("ATLAS_DFLASH_OPTION_B", "1"),
        ("ATLAS_DFLASH_PROPOSE_LANES", "2"),
    ])
    .expect("valid lane count parses before product validation");
    assert_eq!(toggles.proposal_lane_count, 2);
    let error = toggles
        .validate()
        .expect_err("product requires exactly one lane");
    assert!(error.to_string().contains("proposal_lane_count"), "{error}");
}

#[test]
fn raw_reader_rejects_malformed_draft_cap_but_valid_cap_reaches_validation() {
    for raw in ["bad", ""] {
        let error = read_product(&[("ATLAS_DFLASH_DRAFT_CAP", raw)])
            .expect_err("malformed draft cap must fail closed");
        assert!(error.to_string().contains("draft_cap_override"), "{error}");
    }
    let toggles = read_product(&[
        ("ATLAS_DFLASH_OPTION_B", "1"),
        ("ATLAS_DFLASH_DRAFT_CAP", "2"),
    ])
    .expect("valid draft cap parses before product validation");
    assert_eq!(toggles.draft_cap_override, Some(2));
    let error = toggles
        .validate()
        .expect_err("product rejects draft cap override");
    assert!(error.to_string().contains("draft_cap_override"), "{error}");
}

#[test]
fn raw_reader_uses_runtime_gemma_true_semantics() {
    let toggles = read_product(&[("ATLAS_DIAG_GEMMA4", "true")])
        .expect("Gemma diagnostic value is a valid boolean opt-in");
    assert!(!toggles.proposal_graph_eligible);
    assert!(!toggles.target_verify_graph_eligible);
    assert!(toggles.validate().is_err());
}

#[test]
fn raw_reader_rejects_debug_dump_and_option_b_no_ctx() {
    let toggles = read_product(&[("ATLAS_DFLASH_DEBUG_DUMP", "1")])
        .expect("debug dump is represented as a diagnostic");
    assert!(!toggles.proposal_graph_eligible);
    assert!(toggles.validate().is_err());

    let error = read_product(&[("ATLAS_DFLASH_OPTION_B_NO_CTX", "0")])
        .expect_err("presence of Option B no-context must fail closed");
    assert!(error.to_string().contains("option_b_no_ctx"), "{error}");
}

#[test]
fn raw_reader_rejects_every_existing_forbidden_proposal_diagnostic() {
    for name in [
        "ATLAS_DFLASH_PROPOSE_NO_GRAPH",
        "ATLAS_DFLASH_DEBUG_NO_GRAPH",
        "ATLAS_DEBUG_NO_GRAPH",
        "ATLAS_DIAG_GEMMA4",
        "ATLAS_DFLASH_DEBUG_DUMP_FULL",
        "ATLAS_DFLASH_OPTION_B_DIAG",
        "ATLAS_DFLASH_PRECOMPUTE_DUMP",
        "ATLAS_DFLASH_VERIFY_TRACE",
        "ATLAS_DFLASH_LOG_DRAFTS",
        "ATLAS_DFLASH_DEBUG_FORCE_PATTERN",
        "ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
        "ATLAS_DFLASH_DEBUG_CTX_OFF",
        "ATLAS_DFLASH_DEBUG_CTX_USED",
        "ATLAS_DFLASH_BLOCK_DUMP",
        "ATLAS_DFLASH_DEBUG_DUMP",
    ] {
        let toggles = read_product(&[(name, "1")]).expect("diagnostic reader should parse");
        assert!(toggles.validate().is_err(), "{name} must be forbidden");
    }
}

fn admitted_policy() -> LightningDsparkProductPolicy {
    LightningDsparkProductPolicy::try_new(LightningDsparkProfile::lightning(), product_toggles())
        .expect("official Lightning policy")
}

#[test]
fn identity_latch_transitions_between_generic_and_lightning() {
    let mut latch = LightningDsparkIdentityLatch::default();
    assert!(
        latch.policy().is_none(),
        "default/generic is not product identity"
    );

    latch.install_lightning(admitted_policy());
    assert_eq!(latch.policy(), Some(&admitted_policy()));

    latch.install_generic();
    assert!(
        latch.policy().is_none(),
        "generic install clears stale identity"
    );
}

#[test]
fn startup_execution_from_lightning_freezes_product_values() {
    let toggles = exact_product_toggles();
    let execution = super::DsparkStartupExecution::from_lightning(toggles);
    assert!(execution.option_b_enabled);
    assert_eq!(execution.proposal_lane_count, 1);
    assert_eq!(execution.draft_cap_override, None);
    assert!(!execution.option_b_no_ctx);
    assert!(!execution.debug_dump);
    assert!(!execution.graph_ineligible_diags);
}

#[test]
fn startup_execution_from_lightning_marks_ineligible_toggles() {
    // Toggles that pass raw parsing but fail product validation (lane 2,
    // draft cap) still freeze deterministically; graph eligibility is
    // derived so an ineligible toggle is visible on the execution struct.
    let mut toggles = exact_product_toggles();
    toggles.proposal_graph_eligible = false;
    let execution = super::DsparkStartupExecution::from_lightning(toggles);
    assert!(execution.graph_ineligible_diags);
}

#[test]
fn startup_execution_from_env_lenient_defaults() {
    // Generic DFlash defaults: Option B off, one lane, no cap, no dumps.
    let execution = super::DsparkStartupExecution::from_env_lenient();
    assert_eq!(
        execution.proposal_lane_count.max(1),
        execution.proposal_lane_count
    );
    assert!(!execution.debug_dump);
}

#[test]
fn startup_execution_is_frozen_not_env_read() {
    // Constructing from Lightning toggles must not consult the process
    // environment at all: identical toggles produce identical execution
    // regardless of any ambient diagnostic variables.
    let toggles = exact_product_toggles();
    let a = super::DsparkStartupExecution::from_lightning(toggles);
    let b = super::DsparkStartupExecution::from_lightning(toggles);
    assert_eq!(a, b);
}

fn exact_product_toggles() -> super::LightningDsparkRuntimeToggles {
    super::LightningDsparkRuntimeToggles {
        option_b_enabled: true,
        proposal_lane_count: 1,
        proposal_graph_eligible: true,
        target_verify_graph_eligible: true,
        batched_verify_enabled: true,
        seam_serial_enabled: false,
        draft_cap_override: None,
        adaptive_enabled: false,
        batch_parity_enabled: false,
    }
}

#[test]
fn raw_reader_treats_present_but_non_one_graph_kills_as_ineligible() {
    // A present-but-`0` kill switch made the pre-freeze runtime eager;
    // admission must agree instead of claiming graph eligibility.
    for name in [
        "ATLAS_DFLASH_DEBUG_NO_GRAPH",
        "ATLAS_DEBUG_NO_GRAPH",
        "ATLAS_DFLASH_PROPOSE_NO_GRAPH",
    ] {
        let toggles = read_product(&[("ATLAS_DFLASH_OPTION_B", "1"), (name, "0")])
            .expect("present-but-non-one kill switch parses");
        assert!(
            !toggles.proposal_graph_eligible,
            "{name}=0 must be graph-ineligible"
        );
        let error = toggles
            .validate()
            .expect_err("product must reject a present graph kill switch");
        assert!(
            error.to_string().contains("proposal_graph_eligible"),
            "{error}"
        );
    }
    let toggles = read_product(&[
        ("ATLAS_DFLASH_OPTION_B", "1"),
        ("ATLAS_DFLASH_VERIFY_COMPUTE_SERIAL", "1"),
    ])
    .expect("serial verify probe parses");
    assert!(!toggles.target_verify_graph_eligible);
}

#[test]
fn structural_graph_state_gates_eligibility() {
    let allowed = super::LightningStructuralGraphState {
        target_suppress_graphs: false,
        lora_installed: false,
        distributed_topology: false,
    };
    assert!(allowed.graphs_allowed());
    let denied_cases: [super::LightningStructuralGraphState; 3] = [
        super::LightningStructuralGraphState {
            target_suppress_graphs: true,
            lora_installed: false,
            distributed_topology: false,
        },
        super::LightningStructuralGraphState {
            target_suppress_graphs: false,
            lora_installed: true,
            distributed_topology: false,
        },
        super::LightningStructuralGraphState {
            target_suppress_graphs: false,
            lora_installed: false,
            distributed_topology: true,
        },
    ];
    for denied in denied_cases {
        assert!(!denied.graphs_allowed());
    }
}

#[test]
fn startup_diagnostics_default_all_off_for_product() {
    let execution = super::DsparkStartupExecution::from_lightning(exact_product_toggles());
    assert_eq!(execution.diagnostics, super::DsparkDiagnostics::default());
    assert_eq!(execution.diagnostics.propose_warmup_n, 2);
}

#[test]
fn shared_structural_gate_rejects_each_eager_cause() {
    // This is the exact function the production setter calls; if the
    // setter stops calling it, setter_rejects_structurally_eager_target
    // in impl_b3_accessors (which routes through this same function)
    // loses its enforcement and this pin makes the contract explicit.
    for state in [
        super::LightningStructuralGraphState {
            target_suppress_graphs: true,
            lora_installed: false,
            distributed_topology: false,
        },
        super::LightningStructuralGraphState {
            target_suppress_graphs: false,
            lora_installed: true,
            distributed_topology: false,
        },
        super::LightningStructuralGraphState {
            target_suppress_graphs: false,
            lora_installed: false,
            distributed_topology: true,
        },
    ] {
        let error = super::enforce_lightning_structural_gate(state)
            .expect_err("structurally eager state must fail the shared gate");
        assert!(error.to_string().contains("structurally eager"), "{error}");
    }
    super::enforce_lightning_structural_gate(super::LightningStructuralGraphState {
        target_suppress_graphs: false,
        lora_installed: false,
        distributed_topology: false,
    })
    .expect("clean structural state passes the shared gate");
}
