// SPDX-License-Identifier: AGPL-3.0-only

//! Post-construction proposer-wiring accessors for [`TransformerModel`].
//! Split out of `impl_b3.rs` (500-LoC cap) — borrow/install hooks only.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use super::types::TransformerModel;
use crate::layers::dflash_head::{LightningDsparkProductPolicy, LightningStructuralGraphState};
use crate::speculative::DraftProposer;

impl TransformerModel {
    /// Borrow the GPU backend for post-construction wiring (e.g. installing
    /// a DFlash proposer that needs to allocate paged KV caches against the
    /// same GPU the target uses).
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// Borrow the model config for post-construction wiring (e.g. building the
    /// DeepSeek-V4 MTP proposer, which needs `hidden_size` / `kv_lora_rank` /
    /// `qk_rope_head_dim` to size its private MLA KV cache).
    pub fn config_ref(&self) -> &ModelConfig {
        &self.config
    }

    /// Install a DFlash drafter as the active proposer, replacing whatever
    /// MTP proposer (if any) `TransformerModel::new` built. The target's
    /// hidden-state capture buffer is already allocated when the config's
    /// `dflash_capture_layers` is non-empty (factory.rs populates it before
    /// construction), so this method only swaps the proposer slot.
    ///
    /// Mutually exclusive with `--speculative` MTP at the CLI level
    /// (clap `conflicts_with`); this method does not enforce that — the
    /// caller is expected to have validated the flag combination already.
    pub fn set_dflash_proposer(&mut self, proposer: std::sync::Arc<dyn DraftProposer>) {
        if self.proposer.is_some() {
            tracing::info!("DFlash: replacing existing MTP proposer with BlockDiffusionDraftHead");
        }
        self.proposer = Some(proposer);
        self.lightning_dspark_identity.install_generic();
    }

    /// Atomically install the admitted official Lightning proposer and its
    /// immutable startup policy. Generic DFlash and MTP must use
    /// [`Self::set_dflash_proposer`] and therefore remain non-product.
    ///
    /// Fails closed when the constructed target is structurally eager
    /// (graph suppression, LoRA, or a distributed topology): an admitted
    /// policy must never claim product graph eligibility the runtime
    /// cannot honor.
    pub fn set_lightning_dspark_proposer(
        &mut self,
        proposer: std::sync::Arc<dyn DraftProposer>,
        policy: LightningDsparkProductPolicy,
    ) -> anyhow::Result<()> {
        let structural = LightningStructuralGraphState {
            target_suppress_graphs: self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed),
            lora_installed: self.lora.is_some(),
            distributed_topology: self.comm.is_some(),
        };
        anyhow::ensure!(
            structural.graphs_allowed(),
            "Lightning DSpark product admission rejected: target is structurally eager              (suppress_graphs={}, lora={}, distributed={})",
            structural.target_suppress_graphs,
            structural.lora_installed,
            structural.distributed_topology
        );
        self.proposer = Some(proposer);
        self.lightning_dspark_identity.install_lightning(policy);
        Ok(())
    }
}

#[cfg(test)]
mod identity_transition_tests {
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

    /// Mirror of the production setter's structural read set. The
    /// production `set_lightning_dspark_proposer` computes exactly this
    /// state from the live model and gates on `graphs_allowed()`.
    struct SetterHarness {
        lightning_dspark_identity: crate::layers::dflash_head::LightningDsparkIdentityLatch,
        suppress_graphs: std::sync::atomic::AtomicBool,
        lora: Option<()>,
        comm: Option<()>,
    }

    impl SetterHarness {
        fn new() -> Self {
            Self {
                lightning_dspark_identity: Default::default(),
                suppress_graphs: std::sync::atomic::AtomicBool::new(false),
                lora: None,
                comm: None,
            }
        }

        /// Structurally identical to the production setter body.
        fn set_lightning(
            &mut self,
            policy: crate::layers::dflash_head::LightningDsparkProductPolicy,
        ) -> anyhow::Result<()> {
            let structural = LightningStructuralGraphState {
                target_suppress_graphs: self
                    .suppress_graphs
                    .load(std::sync::atomic::Ordering::Relaxed),
                lora_installed: self.lora.is_some(),
                distributed_topology: self.comm.is_some(),
            };
            anyhow::ensure!(structural.graphs_allowed(), "structurally eager target");
            self.lightning_dspark_identity.install_lightning(policy);
            Ok(())
        }

        fn set_generic(&mut self) {
            self.lightning_dspark_identity.install_generic();
        }

        fn is_product(&self) -> bool {
            self.lightning_dspark_identity.policy().is_some()
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

        harness.set_generic();
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
}
