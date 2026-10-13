// SPDX-License-Identifier: AGPL-3.0-only

//! Construction of the prefix cache's NVMe spill tier (default OFF).
//!
//! * `ATLAS_KV_NVME_DIR=<dir>` — enable: evicted prefix-cache blocks are
//!   written to an O_DIRECT record file in `<dir>` instead of being dropped.
//! * `ATLAS_KV_NVME_GB=<GiB>` — REQUIRED with the dir (no implicit default):
//!   the per-rank disk budget; the coldest on-disk blocks are dropped past it.
//! * `ATLAS_GLM_NVME_FAST=1` — move the same records through the fast path
//!   (`kv_cache/nvme_fast.rs`: pitched copies, run-sized I/O, write-behind).
//! * `ATLAS_GLM_NVME_KEEP=1` — a restored block keeps its record, so evicting
//!   it again writes nothing (`NvmePrefixTier::set_keep_restored`).
//!
//! The record file is unlinked right after creation (unix), so its space is
//! returned when the process exits for ANY reason — nothing to clean up after
//! a crash, and no stale file can ever be read by a later process. A file a
//! process left behind by dying before that unlink is swept at the next start.
//!
//! Without `ATLAS_KV_NVME_DIR` none of the other three is read: a rank starts
//! exactly as it does with them unset ([`attach`] warns about each).
//!
//! The tier refuses to start when `ATLAS_SSM_TIER` keeps spilled snapshots in
//! host RAM ([`snapshots_off_host`]), and on a multi-rank world with
//! `ATLAS_EP_PEER_LIFELINE=0` ([`word_for`]); every rank stops.

use std::path::PathBuf;

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixCache;

use crate::model::ssm_tier::SpillHome;

pub(super) const DIR_VAR: &str = "ATLAS_KV_NVME_DIR";
pub(super) const GB_VAR: &str = "ATLAS_KV_NVME_GB";
pub(super) const FAST_VAR: &str = "ATLAS_GLM_NVME_FAST";
pub(super) const KEEP_VAR: &str = "ATLAS_GLM_NVME_KEEP";
/// Record files are `<prefix><pid>.r<rank>.swap`.
const FILE_PREFIX: &str = "atlas-kv-prefix.";

#[derive(Debug, PartialEq)]
pub(super) struct NvmeKvConfig {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    pub fast: bool,
    pub keep: bool,
}

/// The tier's configuration from the environment.
pub(super) fn config_from_env() -> Result<Option<NvmeKvConfig>> {
    let var = |name| std::env::var(name).ok();
    config_from(
        var(DIR_VAR).as_deref(),
        var(GB_VAR).as_deref(),
        [(FAST_VAR, var(FAST_VAR)), (KEEP_VAR, var(KEEP_VAR))],
    )
}

/// A strict `=1` / `=0` switch (unset or empty = off).
fn switch((name, value): &(&str, Option<String>)) -> Result<bool> {
    match value.as_deref().map(str::trim) {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{name}={other:?}: expected 1 (on) or 0 (off)"),
    }
}

/// Env-free parse (strict, PCND): `None` when the tier is off.
pub(super) fn config_from(
    dir: Option<&str>,
    gb: Option<&str>,
    [fast, keep]: [(&str, Option<String>); 2],
) -> Result<Option<NvmeKvConfig>> {
    let dir = dir.map(str::trim).filter(|s| !s.is_empty());
    let gb = gb.map(str::trim).filter(|s| !s.is_empty());
    let Some(dir) = dir else {
        // The budget and the two switches only shape a tier that is on.
        // Without it they are not read at all — whatever their value, a rank
        // starts exactly as it does with them unset ([`attach`] says so).
        return Ok(None);
    };
    let (fast, keep) = (switch(&fast)?, switch(&keep)?);
    let Some(gb) = gb else {
        bail!("{DIR_VAR} is set: {GB_VAR}=<GiB> (per-rank disk budget) is required");
    };
    let v: f64 = gb
        .parse()
        .map_err(|e| anyhow::anyhow!("{GB_VAR}={gb:?} is not a number: {e}"))?;
    ensure!(v.is_finite() && v > 0.0, "{GB_VAR}={gb:?} must be > 0");
    Ok(Some(NvmeKvConfig {
        dir: PathBuf::from(dir),
        budget_bytes: (v * (1u64 << 30) as f64) as u64,
        fast,
        keep,
    }))
}

/// Host memory the spill tiers take AFTER the KV pool is sized, which
/// `build_model` therefore leaves out of the pool (0 with the KV tier off):
///
/// * the pinned staging (`PagedKvCache::nvme_staging_bytes`);
/// * the tree's index for a FULL disk budget — every on-disk block keeps its
///   radix node (`NVME_HOST_BYTES_PER_BLOCK`), so the budget in GiB is also a
///   budget in host RAM;
/// * `ssm_lazy_bytes`: what the SSM snapshot tier commits on first use.
///
/// Never fails: a bad configuration is reported by the rank exchange
/// ([`verify_ranks`]; an early error here would leave the peers in it).
pub(super) fn host_reserve_bytes(record_bytes: usize, ssm_lazy_bytes: usize) -> usize {
    let cfg = config_from_env().ok().flatten();
    let reserve = reserve_for(cfg.as_ref(), record_bytes, ssm_lazy_bytes);
    if reserve > 0 {
        tracing::info!(
            "NVMe spill tiers: reserving {:.1} MiB of host memory out of the KV budget \
             (staging + on-disk index at a full {GB_VAR} + {:.1} MiB SSM tier arena/staging)",
            reserve as f64 / (1u64 << 20) as f64,
            ssm_lazy_bytes as f64 / (1u64 << 20) as f64,
        );
    }
    reserve
}

fn reserve_for(cfg: Option<&NvmeKvConfig>, record_bytes: usize, ssm_lazy_bytes: usize) -> usize {
    let Some(cfg) = cfg else {
        return 0;
    };
    let Ok(slots) = max_slots(cfg.budget_bytes, record_bytes) else {
        return 0;
    };
    PagedKvCache::nvme_staging_bytes(record_bytes, cfg.fast)
        + slots as usize * spark_runtime::prefix_cache::NVME_HOST_BYTES_PER_BLOCK
        + ssm_lazy_bytes
}

/// Records that fit the budget (refuses a budget below one record).
pub(super) fn max_slots(budget_bytes: u64, record_bytes: usize) -> Result<u32> {
    let n = budget_bytes / record_bytes.max(1) as u64;
    ensure!(
        n >= 1,
        "{GB_VAR}: {budget_bytes} B cannot hold one {record_bytes} B KV block record"
    );
    Ok(n.min(u32::MAX as u64 - 1) as u32)
}

/// What every rank must agree on before serving: the KV tier's geometry and
/// budget, where the SSM tier keeps its spills (with a spill tier on, the
/// restore-depth agreement in `prefill_b/pc_policy.rs` covers a snapshot
/// fault-in that succeeds on one rank only, and the home sets the host
/// reserve), whether restored records are kept (it changes which blocks the
/// budget drops, so the ranks' trees would drift apart), and the I/O path (it
/// sets the staging reserve, and a write failure reaches the tree at a
/// different time on each path — ranks on different paths are not one
/// configuration). Never [`TIER_OFF`] or [`FAILED_RANK`].
fn rank_fingerprint(
    slots: u32,
    record_bytes: usize,
    ssm_home: Option<SpillHome>,
    (fast, keep): (bool, bool),
) -> u32 {
    let ssm = match ssm_home {
        None => 0,
        Some(SpillHome::HostRam) => 1,
        Some(SpillHome::Disk { .. }) => 2,
        Some(SpillHome::Peer) => 3,
    };
    let h = atlas_tier::hash::mix64(
        atlas_tier::hash::mix64(slots as u64, record_bytes as u64),
        ssm + 1 + 4 * keep as u64 + 8 * fast as u64,
    );
    (h % (FAILED_RANK as u64 - 1)) as u32 + 1
}

/// The word of a rank with no spill tier.
const TIER_OFF: u32 = 0;
/// The word a rank sends when its OWN configuration is unusable: never equal
/// to a real one, so every peer fails the check too instead of serving on.
pub(super) const FAILED_RANK: u32 = u32::MAX;

/// This rank's spill-tier word for the startup agreement
/// (`glm::agree_kv_blocks` carries it in the upper half of the word every rank
/// already gathers, so the tiers add no collective): [`TIER_OFF`] with neither
/// `ATLAS_KV_NVME_DIR` nor an SSM tier (`ssm_home`) — whatever else the
/// environment holds — and otherwise a fingerprint of everything in
/// [`rank_fingerprint`]. `Err` when the KV tier's environment does not parse,
/// its budget cannot hold one record, or [`word_for`] refuses the combination.
pub(super) fn rank_word(record_bytes: usize, ssm_home: Option<SpillHome>) -> Result<u32> {
    let lifeline = std::env::var("ATLAS_EP_PEER_LIFELINE").as_deref() != Ok("0");
    word_for(
        config_from_env()?.as_ref(),
        record_bytes,
        ssm_home,
        lifeline,
    )
}

fn word_for(
    cfg: Option<&NvmeKvConfig>,
    record_bytes: usize,
    ssm_home: Option<SpillHome>,
    lifeline: bool,
) -> Result<u32> {
    let Some(cfg) = cfg else {
        // The SSM tier alone: no KV record, but the ranks run the restore
        // agreement together (`pc_agree_restore`) or not at all.
        return Ok(ssm_home.map_or(TIER_OFF, |home| {
            rank_fingerprint(0, 0, Some(home), (false, false))
        }));
    };
    snapshots_off_host(ssm_home)?;
    // What can still fail after this exchange is rank-local ([`attach`]); the
    // lifeline is what stops the peer of a rank that fails there.
    ensure!(
        lifeline,
        "{DIR_VAR} requires the EP peer lifeline: unset ATLAS_EP_PEER_LIFELINE=0"
    );
    let slots = max_slots(cfg.budget_bytes, record_bytes)?;
    let switches = (cfg.fast, cfg.keep);
    Ok(rank_fingerprint(slots, record_bytes, ssm_home, switches))
}

/// Hosts hang on unified-memory exhaustion: the KV pool is sized around what
/// the tiers take from the host ([`host_reserve_bytes`]), and a snapshot store
/// that keeps its spills in RAM (78 MB each on GLM-5.3) has no such bound —
/// the legacy store, a unified store without a usable
/// `ATLAS_SSM_TIER_SWAP_DIR`, or a blob that is not an O_DIRECT record. The
/// KV tier does not start beside one: checked before the rank exchange
/// ([`word_for`], so every rank stops) and at [`attach`] (a single rank has
/// no exchange).
fn snapshots_off_host(ssm_home: Option<SpillHome>) -> Result<()> {
    ensure!(
        ssm_home != Some(SpillHome::HostRam),
        "{DIR_VAR} with ATLAS_SSM_TIER: this rank's snapshot tier keeps its spills in host RAM, \
         which the KV pool is not sized around. Set ATLAS_SSM_TIER_UNIFIED=1 and \
         ATLAS_SSM_TIER_SWAP_DIR=<a directory of the node's NVMe> (see this rank's \
         'unified SSM tier' log line for why the swap file was not used), or unset ATLAS_SSM_TIER"
    );
    Ok(())
}

/// After the exchange: this rank's own error first, then any peer that
/// failed, then any config mismatch. `all` is every rank's word, in rank
/// order (a rank whose `local` is an error sent [`FAILED_RANK`]).
pub(super) fn verify_ranks(local: Result<u32>, all: &[u32]) -> Result<()> {
    let word = local?;
    if let Some(bad) = all.iter().position(|&v| v == FAILED_RANK) {
        bail!("spill-tier setup failed on rank {bad} (see its log); every rank stops");
    }
    ensure!(
        all.iter().all(|&v| v == word),
        "spill-tier config differs across ranks (words {all:x?}, {TIER_OFF} = off); set \
         identical {DIR_VAR}/{GB_VAR}/{FAST_VAR}/{KEEP_VAR} and ATLAS_SSM_TIER* on every rank"
    );
    Ok(())
}

/// Attach the tier to a freshly built KV cache (after the sparse index) and
/// enable spill-on-evict in the prefix cache; nothing happens with the tier
/// off. The ranks have already agreed on the configuration
/// ([`rank_word`]); what can still fail here is rank-local (directory, disk
/// reservation, staging), and a rank that stops takes its peers with it
/// through the EP peer lifeline ([`word_for`] requires it).
pub(super) fn attach(
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
    rank: usize,
    ssm_home: Option<SpillHome>,
) -> Result<()> {
    let cfg = config_from_env()?;
    if cfg.is_some() {
        snapshots_off_host(ssm_home)?;
    } else {
        for var in [GB_VAR, FAST_VAR, KEEP_VAR] {
            if std::env::var(var).is_ok_and(|v| !matches!(v.trim(), "" | "0")) {
                tracing::warn!(
                    "{var} is set but {DIR_VAR} is not: the NVMe prefix tier is off and {var} \
                     has no effect"
                );
            }
        }
    }
    let record = kv_cache.nvme_record_bytes();
    setup_local(cfg, rank, record, kv_cache, prefix_cache, gpu).map(|_| ())
}

/// The snapshot tier's swap directory, when it is configured to use one.
fn ssm_swap_dir(dir: Option<String>) -> Option<PathBuf> {
    let on =
        std::env::var_os("ATLAS_SSM_TIER").is_some() && crate::model::ssm_tier::ssm_tier_unified();
    dir.filter(|d| on && !d.is_empty()).map(PathBuf::from)
}

/// `dir` (created if missing) is on a real disk and this process can create
/// files in it — checked with a throwaway file, so a read-only or foreign
/// mount fails here, by name, instead of inside a tier.
fn usable_disk_dir(var: &str, dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|e| anyhow::anyhow!("{var}={}: cannot create: {e}", dir.display()))?;
    if let Some(kind) = atlas_tier::unsuitable_swap_fs(dir) {
        bail!(
            "{var}={} is on {kind}; bind-mount a directory of the node's NVMe instead",
            dir.display()
        );
    }
    let probe = dir.join(format!("atlas-dir-probe.{}.swap", std::process::id()));
    let _ = std::fs::remove_file(&probe);
    let made = atlas_tier::SharedRecordFile::create(&probe, 4096);
    let _ = std::fs::remove_file(&probe);
    made.map(|_| ())
        .map_err(|e| e.context(format!("{var}={} is not usable", dir.display())))
}

/// Env-free body of [`attach`]: validate, create the record file, attach it
/// and enable the tree side. Returns the slot budget (0 = off).
fn setup_local(
    cfg: Option<NvmeKvConfig>,
    rank: usize,
    record: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
) -> Result<u32> {
    let Some(cfg) = cfg else {
        return Ok(0);
    };
    let slots = max_slots(cfg.budget_bytes, record)?;
    ensure!(
        prefix_cache.is_active(),
        "{DIR_VAR} requires --enable-prefix-caching (it spills evicted prefix-cache blocks)"
    );
    let tier = prefix_cache
        .nvme()
        .ok_or_else(|| anyhow::anyhow!("this prefix cache has no NVMe spill tier"))?;
    usable_disk_dir(DIR_VAR, &cfg.dir)?;
    if let Some(dir) = ssm_swap_dir(std::env::var("ATLAS_SSM_TIER_SWAP_DIR").ok()) {
        // The snapshot tier falls back to HOST RAM when its directory is
        // unusable; with the KV tier on, that is a misconfigured install.
        usable_disk_dir("ATLAS_SSM_TIER_SWAP_DIR", &dir)?;
    }
    let stale = atlas_tier::remove_stale_swap_files(&cfg.dir, FILE_PREFIX);
    if stale > 0 {
        tracing::info!(
            "prefix cache NVMe spill tier: removed {stale} stale record file(s) from {}",
            cfg.dir.display()
        );
    }
    let path = cfg
        .dir
        .join(format!("{FILE_PREFIX}{}.r{rank}.swap", std::process::id()));
    // Records hold prompt-derived KV: owner-only, and never through a
    // pre-planted file or symlink (remove_file drops a link, not its target;
    // an exclusive create refuses anything that reappears). ONE open per
    // path: nothing re-opens the name after another process could have
    // swept it.
    let _ = std::fs::remove_file(&path);
    if cfg.fast {
        let store = atlas_tier::SharedRecordFile::create(&path, record)?;
        // The whole budget, now: a disk that cannot hold it fails the start
        // (every rank stops) instead of the spills later, and writers never
        // serialise on block allocation.
        let reserved = store
            .reserve(slots as u64 * record as u64)
            .map_err(|e| e.context(format!("{GB_VAR}: the disk budget does not fit {DIR_VAR}")));
        #[cfg(unix)]
        let _ = std::fs::remove_file(&path);
        if !reserved? {
            tracing::warn!(
                "prefix cache NVMe spill tier: this filesystem cannot reserve space; the record \
                 file grows on demand (slower concurrent writes, and it can run out of disk)"
            );
        }
        kv_cache.attach_nvme_fast(std::sync::Arc::new(store), gpu)?;
    } else {
        let store = atlas_tier::DirectSwapFile::create_new(&path, record)?;
        kv_cache.attach_nvme_spill(Box::new(store), gpu)?;
    }
    // Anonymous from here on: freed at exit. (Another process's stale sweep
    // may have unlinked it first — the descriptor is what matters.)
    #[cfg(unix)]
    let _ = std::fs::remove_file(&path);
    ensure!(tier.enable(slots), "prefix cache NVMe tier already enabled");
    tier.set_keep_restored(cfg.keep);
    let per_token = record as f64 / kv_cache.block_size() as f64;
    let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
    tracing::info!(
        "prefix cache NVMe spill tier ON (rank {rank}): {} ({DIR_VAR}, O_DIRECT, unlinked), \
         {record} B/block record ({per_token:.0} B/token: latent + pooled index, no raw \
         tails), budget {:.1} GiB = {slots} blocks = {} tokens; {} I/O ({FAST_VAR}), \
         restored records {} ({KEEP_VAR}), {:.1} MiB pinned staging, up to {:.1} MiB host index",
        cfg.dir.display(),
        cfg.budget_bytes as f64 / (1u64 << 30) as f64,
        slots as u64 * kv_cache.block_size() as u64,
        if cfg.fast { "fast" } else { "synchronous" },
        if cfg.keep { "kept" } else { "released" },
        mib(PagedKvCache::nvme_staging_bytes(record, cfg.fast)),
        mib(slots as usize * spark_runtime::prefix_cache::NVME_HOST_BYTES_PER_BLOCK),
    );
    Ok(slots)
}

#[cfg(test)]
#[path = "kv_nvme_tests.rs"]
mod tests;
