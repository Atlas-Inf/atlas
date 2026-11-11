// SPDX-License-Identifier: AGPL-3.0-only

//! Launch admission of strict native GLM MTP (`ATLAS_GLM_STRICT_NATIVE_MTP`).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_model::weight_loader::glm5_mode;

use crate::cli;

/// `--speculative` on the explicit strict native lane.
pub(super) fn requested(args: &cli::ServeArgs) -> bool {
    args.speculative && !args.dflash && glm5_mode::strict_native_mtp()
}

/// The GLM long-context lane: the DFlash lane or explicit strict native MTP.
pub(super) fn long_context_lane(args: &cli::ServeArgs) -> bool {
    (args.dflash && spark_model::speculative::glm_repair_policy::dflash_enabled())
        || requested(args)
}

/// Whether this launch's `--speculative` is the admitted native lane: C1..C4
/// with equal active and admitted limits, on the pair or four ranks.
pub(super) fn admitted(args: &cli::ServeArgs, ep_v2: bool) -> bool {
    requested(args)
        && (1..=4).contains(&args.max_batch_size)
        && args.max_num_seqs == args.max_batch_size
        && (args.max_batch_size == 1 || ep_v2)
        && glm5_mode::supported_parallelism(args.world_size, args.tp_size, args.ep_size)
}

/// Native GLM allocates predictor weights/workspace before the free-memory
/// snapshot and budgets its private KV jointly with target KV. Its eager owner
/// staging is only tens of MiB; the generic 4 GiB speculation reserve charged
/// that memory twice. GDN scratch and the KV growth floor remain. Earlier GLM
/// admission checks constrain this exception to matching TP2/EP2 or TP4/EP4
/// C1..C4.
pub(super) fn preallocated(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    long_context: bool,
) -> Result<bool> {
    let native = config.model_type == "glm5_next"
        && args.speculative
        && long_context
        && glm5_mode::strict_native_mtp()
        && matches!(glm5_mode::selected()?, Some(glm5_mode::ServingMode::Strict));
    if native {
        for flag in [
            "ATLAS_GLM_MTP1_VERIFY_GRAPH",
            "ATLAS_GLM_TP_VERIFY_GRAPH",
            "ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS",
        ] {
            anyhow::ensure!(
                matches!(std::env::var(flag).ok().as_deref(), None | Some("0")),
                "strict native GLM memory accounting requires eager verification: {flag} must be disabled"
            );
        }
    }
    Ok(native)
}
