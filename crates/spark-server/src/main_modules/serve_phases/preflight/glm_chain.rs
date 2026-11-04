// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 beyond two ranks: TP4/EP4 over the RDMA chain (`spark_comm`'s
//! `rdma_pair::chain`), and what of the two-rank lanes it cannot take.
//!
//! Many GLM-5 lanes split work between exactly two ranks or are built for
//! TP2's 32 local attention heads (TP4 has 16). At more ranks each either
//! falls back to the generic path by its own guard, which is said here once
//! at startup, or would fail at load or on first use, or serve wrong
//! numbers: those are refused here, all at once, at the GLM-5 gate. The
//! table is the one place that knows which is which; a lane is requested
//! when its variable is set to anything but `0` (every one is opt-in). An
//! option whose own validation requires a refused lane (the pipelined sparse
//! prefill of the sparse prefill TC, say) is refused with it, so a profile
//! fails here rather than at model build.

use anyhow::{Result, bail, ensure};

/// What a requested two-rank lane does with more ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Beyond {
    /// Its guard keeps it off; the generic path serves.
    Off(&'static str),
    /// It would fail or mis-run: the operator must turn it off.
    Refused(&'static str),
}

use Beyond::{Off, Refused};

const C4: &str = "C4 is the pair's non-speculative four-sequence decode (TP2 MLA/KDA geometry); \
                  speculative serving takes four sequences through the owner-batched verify";

const LANES: &[(&str, Beyond)] = &[
    (
        "ATLAS_GLM_K3_MLA_QUERY_COMPARE",
        Refused("the scalar query comparison is TP2's"),
    ),
    (
        "ATLAS_GLM_KV_SHARD_OVERLAP",
        Refused("the KV shard's overlap lane swaps with one peer; the chain all-gathers"),
    ),
    ("ATLAS_GLM_C4_DECODE", Refused(C4)),
    ("ATLAS_GLM_C4_SPARSE", Refused(C4)),
    ("ATLAS_GLM_C4_GROUPED_MOE", Refused(C4)),
    (
        "ATLAS_GLM_INDEPENDENT_DECODE",
        Refused("independent decode is a TP2 lane"),
    ),
    ("ATLAS_GLM_MTP_REPAIR", Refused("MTP repair is a TP2 lane")),
    (
        "ATLAS_GLM_MTP_DISTRIBUTED",
        Refused("MTP repair is a TP2 lane"),
    ),
    (
        "ATLAS_GLM_K5_FUSED_TP_HC",
        Off("the fused TP HC exchange is the pair's"),
    ),
    (
        "ATLAS_GLM_TP_VERIFY_GRAPH",
        Off("TP verify graphs are TP2's; verify runs eager"),
    ),
    (
        "ATLAS_GLM_MTP1_VERIFY_GRAPH",
        Off("TP verify graphs are TP2's; verify runs eager"),
    ),
];

/// At `world` > 2: log each requested lane that stays off, and refuse the
/// requested lanes that cannot run, naming every one. `var` reads the
/// environment.
pub(super) fn check(world: usize, var: impl Fn(&str) -> Option<String>) -> Result<()> {
    if world <= 2 {
        return Ok(());
    }
    let mut refused = Vec::new();
    for &(name, beyond) in LANES {
        let Some(value) = var(name).filter(|v| !v.is_empty() && v != "0") else {
            continue;
        };
        match beyond {
            Off(why) => tracing::warn!("GLM-5 at {world} ranks: {name}={value} stays off: {why}"),
            Refused(why) => refused.push((format!("{name}={value}"), why)),
        }
    }
    if refused.is_empty() {
        return Ok(());
    }
    let list: Vec<String> = refused
        .iter()
        .map(|(what, why)| format!("{what} ({why})"))
        .collect();
    bail!(
        "GLM-5 at {world} ranks cannot run these two-rank settings; turn them off: {}",
        list.join("; ")
    )
}

/// Whether `(world, tp, ep)` is a GLM-5 parallelism, given whether the
/// RDMA chain is requested: the dual-Spark pair (EP2 with TP1 or TP2), or
/// TP4/EP4 over four Sparks in a chain.
pub(super) fn parallelism_supported(world: usize, tp: usize, ep: usize, chain: bool) -> Result<()> {
    if world == 2 && ep == 2 && matches!(tp, 1 | 2) {
        return Ok(());
    }
    ensure!(
        world == 4 && tp == 4 && ep == 4,
        "GLM-5 supports --world-size 2 --ep-size 2 with --tp-size 1 (EP fallback) or 2 \
         (overlapping TP+EP), or --world-size 4 --tp-size 4 --ep-size 4 over the RDMA chain"
    );
    ensure!(
        chain,
        "GLM-5 on four ranks runs over the RDMA chain: set ATLAS_RDMA_ALLREDUCE=1 \
         ATLAS_RDMA_ONESHOT=1 ATLAS_RDMA_PAIR_CHAIN=1 and each rank's ATLAS_RDMA_LEFT_RAILS / \
         ATLAS_RDMA_RIGHT_RAILS"
    );
    Ok(())
}

#[cfg(test)]
#[path = "glm_chain_tests.rs"]
mod tests;
