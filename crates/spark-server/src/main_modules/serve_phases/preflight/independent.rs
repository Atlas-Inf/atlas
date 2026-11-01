// SPDX-License-Identifier: AGPL-3.0-only
//! Preparation seam: the long-context GLM lane resolves its topology before
//! allocation accounting, so every reserve uses the local shapes it allocates.
use super::super::{Topology, resolve_topology};
use super::*;

pub(crate) fn prepare_reserve<B>(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
    init_backend: impl FnOnce() -> Result<(B, usize)>,
) -> Result<(B, usize, Option<Topology>, ReservePreflight)> {
    // Selected profile errors must be reported before CUDA initialization.
    // OFF retains the original backend -> reserve -> late topology order.
    let topology = prepare_topology(args, config)?;
    let (backend, free_mem) = init_backend()?;
    let reserve = preflight_reserve(args, config, free_mem)?;
    Ok((backend, free_mem, topology, reserve))
}

fn prepare_topology(args: &cli::ServeArgs, config: &mut ModelConfig) -> Result<Option<Topology>> {
    if spark_model::weight_loader::qwen4exp_mode::selected()?.is_some() {
        anyhow::ensure!(
            !args.no_fast_load && std::env::var("ATLAS_FAST_LOAD").as_deref() != Ok("0"),
            "Qwen serving modes require fast loading to retain deferred input scales"
        );
        let topology = resolve_topology(args, config)?;
        spark_model::weight_loader::qwen4exp_mode::validate(
            config,
            args.speculative,
            args.dflash || args.self_speculative || args.ngram_speculative,
        )?;
        return Ok(Some(topology));
    }
    if spark_model::speculative::glm_repair_policy::long_context_enabled() {
        anyhow::ensure!(
            args.dflash
                && spark_model::speculative::glm_repair_policy::dflash_enabled()
                && config.model_type == "glm5_next",
            "long-context GLM preparation requires the GLM DFlash lane"
        );
        let topology = resolve_topology(args, config)?;
        anyhow::ensure!(
            topology.world_size == 2
                && topology.tp_size == 2
                && topology.ep_size == 2
                && config.tp_rank == config.ep_rank
                && config.ep_rank == args.rank
                && args.rank < topology.world_size,
            "resolved long-context GLM topology mismatch before GPU initialization"
        );
        // The serve handoff consumes Some(topology), so global heads are divided
        // exactly once and every reserve uses the same local shapes as allocation.
        return Ok(Some(topology));
    }
    Ok(None)
}

#[cfg(test)]
mod tests;
