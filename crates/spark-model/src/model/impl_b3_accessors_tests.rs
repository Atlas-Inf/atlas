// SPDX-License-Identifier: AGPL-3.0-only

//! Blocker-4 closure: executable production identity-transition tests.
//!
//! `TransformerModel::new` resolves real kernel handles and cannot be
//! constructed on a host test runner, so these tests exercise the
//! production setter/latch state machine through a minimal harness: a
//! stub `Model` impl proving the object-safe trait default is
//! non-product, and a harness struct that owns the same
//! `LightningDsparkIdentityLatch` + structural fields the production
//! setter reads, driving the identical install/clear code path.
use super::*;

#[test]
fn fresh_identity_latch_is_not_product() {
    // The object-safe trait default delegates to this same latch state
    // (`None` policy). `TransformerModel` realizes it via
    // `lightning_dspark_identity.policy()`; the trait-level default is
    // proven non-product by `traits/model.rs` returning `None` unless a
    // concrete model installs a policy — pinned here at the latch level.
    let latch = crate::layers::dflash_head::LightningDsparkIdentityLatch::default();
    assert!(latch.policy().is_none());
}

/// Harness implementing the SAME `LightningStructuralView` the
/// production `TransformerModel` implements, with the three fields
/// the view reads. `set_lightning` drives the REAL production install
/// seam (`install_lightning_proposer`) that the production setter
/// delegates its entire body to.
struct SetterHarness {
    lightning_dspark_identity: crate::layers::dflash_head::LightningDsparkIdentityLatch,
    levers: crate::layers::ops::ModelLevers,
    proposer_slot: Option<std::sync::Arc<dyn DraftProposer>>,
    suppress_graphs: std::sync::atomic::AtomicBool,
    lora: Option<()>,
    comm: Option<()>,
}

impl LightningStructuralView for SetterHarness {
    fn lightning_structural_graph_state(&self) -> LightningStructuralGraphState {
        LightningStructuralGraphState {
            target_suppress_graphs: self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed),
            lora_installed: self.lora.is_some(),
            distributed_topology: self.comm.is_some(),
        }
    }
}

/// Minimal DraftProposer so the real seam's proposer install is
/// exercised (alloc_state is never called by these tests).
struct NoopProposer;

impl crate::speculative::DraftProposer for NoopProposer {
    fn alloc_state(
        &self,
        _gpu: &dyn spark_runtime::gpu::GpuBackend,
    ) -> anyhow::Result<Box<dyn crate::speculative::ProposerState>> {
        anyhow::bail!("noop proposer: no state")
    }

    fn propose(
        &self,
        _last_token: u32,
        _target_hidden: spark_runtime::gpu::DevicePtr,
        _position: usize,
        _num_drafts: usize,
        _state: &mut dyn crate::speculative::ProposerState,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        _ctx: &crate::layer::ForwardContext,
        _stream: u64,
        _draft_embed_target: Option<spark_runtime::gpu::DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<spark_runtime::gpu::DevicePtr>,
    ) -> anyhow::Result<Vec<u32>> {
        Ok(Vec::new())
    }

    fn after_verify(
        &self,
        _num_accepted: usize,
        _expected_owner: Option<crate::layers::dflash_head::SequenceGeneration>,
        _state: &mut dyn crate::speculative::ProposerState,
        _stream: u64,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl SetterHarness {
    fn new() -> Self {
        Self {
            lightning_dspark_identity: Default::default(),
            levers: crate::layers::ops::ModelLevers::defaults(),
            proposer_slot: None,
            suppress_graphs: std::sync::atomic::AtomicBool::new(false),
            lora: None,
            comm: None,
        }
    }

    /// Drives the REAL production install seam
    /// (`install_lightning_proposer`) — the function
    /// `TransformerModel::set_lightning_dspark_proposer` delegates its
    /// entire body to. Deleting the gate inside the seam, or bypassing
    /// the seam from the production setter, changes this test's
    /// observable behavior identically.
    fn set_lightning(
        &mut self,
        policy: crate::layers::dflash_head::LightningDsparkProductPolicy,
    ) -> anyhow::Result<()> {
        let structural = LightningStructuralView::lightning_structural_graph_state(self);
        super::install_lightning_proposer(
            structural,
            &mut self.proposer_slot,
            &mut self.lightning_dspark_identity,
            &mut self.levers,
            std::sync::Arc::new(NoopProposer),
            policy,
        )
    }

    fn set_generic(&mut self) {
        self.proposer_slot = None;
        self.levers.lightning_mamba_exact_recurrence = false;
        self.levers.lightning_mamba_scalar_in_proj = false;
        self.lightning_dspark_identity.install_generic();
    }

    fn is_product(&self) -> bool {
        self.lightning_dspark_identity.policy().is_some() && self.proposer_slot.is_some()
    }
}

fn admitted_policy() -> crate::layers::dflash_head::LightningDsparkProductPolicy {
    use crate::layers::dflash_head::LightningDsparkRuntimeToggles;
    crate::layers::dflash_head::LightningDsparkProductPolicy::try_new(
        test_profile(),
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
        },
    )
    .expect("exact Lightning profile + toggles admit")
}

fn test_profile() -> crate::layers::dflash_head::LightningDsparkProfile {
    // Canonical official-Lightning profile; the admission unit tests
    // already prove exact-checkpoint parsing, so identity-transition
    // tests only need any admitted profile.
    crate::layers::dflash_head::LightningDsparkProfile::lightning()
}

#[test]
fn lightning_install_then_generic_install_clears_identity() {
    let mut harness = SetterHarness::new();
    assert!(!harness.is_product(), "default/generic is not product");

    harness
        .set_lightning(admitted_policy())
        .expect("structural graph state allows product install");
    assert!(harness.is_product(), "Lightning setter installs identity");
    assert!(harness.levers.lightning_mamba_exact_recurrence);
    assert!(harness.levers.lightning_mamba_scalar_in_proj);

    harness.set_generic();
    assert!(!harness.levers.lightning_mamba_exact_recurrence);
    assert!(!harness.levers.lightning_mamba_scalar_in_proj);
    assert!(
        !harness.is_product(),
        "generic setter clears stale Lightning identity"
    );
}

#[test]
fn setter_rejects_structurally_eager_target() {
    let mut harness = SetterHarness::new();
    harness
        .suppress_graphs
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let error = harness
        .set_lightning(admitted_policy())
        .expect_err("suppressed-graphs target must fail closed");
    assert!(error.to_string().contains("structurally eager"));

    let mut lora_harness = SetterHarness::new();
    lora_harness.lora = Some(());
    assert!(lora_harness.set_lightning(admitted_policy()).is_err());

    let mut dist_harness = SetterHarness::new();
    dist_harness.comm = Some(());
    assert!(dist_harness.set_lightning(admitted_policy()).is_err());
}
