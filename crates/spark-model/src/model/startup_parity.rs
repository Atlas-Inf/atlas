// SPDX-License-Identifier: AGPL-3.0-only

//! Startup agreement on the settings every rank must run with the same value.
//!
//! Each rank reads its switches from its own environment and command line,
//! and the worker mirrors the head command by command. A rank whose value of
//! one of these differs (a stale profile on one node) issues other
//! collectives or command words than its peer, takes another share of work
//! the pair splits, or fills a cache differently. The pair then deadlocks or
//! pairs unrelated collectives at some prompt-dependent step (one rank's
//! restore-depth reduction read as the other's KV-admission vote, which
//! refuses a cold prompt), or serves wrong numerics silently.
//!
//! So every rank gathers every rank's values once, right after the
//! communicator comes up, and fails on any difference, naming the setting and
//! both values. The gather is itself a collective: it runs whatever the
//! settings are, and its size is the length of `SETTINGS` plus the
//! caller's, which no setting changes. Only a build changes it, so a gather
//! of one word, the table's id, goes first and fails ranks on different
//! builds before their settings gathers could mispair. A rank whose parser
//! refuses a value has no settings to gather: it takes part in that first
//! gather with `REFUSED`, which ends the agreement there on every rank.
//!
//! `ATLAS_STARTUP_PARITY=warn` logs a disagreement or a refusal and boots
//! anyway; the gathers are the same, and ranks on different builds still
//! fail.
//!
//! # What belongs in the table
//!
//! A setting whose mismatch changes the command words or collectives of a
//! step (their count, order, size or transport), the share of a split
//! computation a rank takes, the lane both ranks must take through a step,
//! or a cache both ranks must fill alike. Its value comes from the parser the
//! feature itself reads, never from a second reading of the variable. A
//! setting that only matters under a switch reads 0 while that switch is off
//! (`while_on`), so stale leftovers do not fail a boot.
//!
//! Not here: what the ranks already reconcile (`ATLAS_KV_MAX_BLOCKS` takes
//! the pair's minimum; the RDMA pair compares its capacity and one-shot
//! settings at bootstrap), what is local to a rank (the drafter, the rails,
//! logging, and the scheduler, which only the head runs: the worker follows
//! its commands), and kernel choices that only reorder one rank's own
//! arithmetic. Not here either, and not reconciled: the switches that pick
//! the routed-MoE arm of a row count off the expert-TP lane
//! (`ATLAS_MOE_DECODE_ARM`, `ATLAS_GLM_C3_GROUPED_MOE`, ...). On that lane
//! every arm but the K=5 one is the grouped prefill MoE, so only the K=5
//! switches are carried.

use anyhow::{Result, anyhow, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;

#[path = "startup_parity_table.rs"]
mod table;
use table::SETTINGS;

/// One agreed setting: its name and a rank's value.
pub type Setting = (&'static str, u64);

/// A setting that only matters while its switch is `on`.
fn while_on(on: bool, value: fn() -> usize) -> Result<u64> {
    Ok(if on { value() as u64 } else { 0 })
}

/// `ATLAS_STARTUP_PARITY=warn`: log what [`agree`] would fail on and boot.
fn warn_only() -> bool {
    std::env::var("ATLAS_STARTUP_PARITY").as_deref() == Ok("warn")
}

/// This process's settings: [`SETTINGS`], then the `caller`'s.
fn settings(caller: &[Setting]) -> Result<Vec<Setting>> {
    SETTINGS
        .iter()
        .map(|&(name, read)| Ok((name, read()?)))
        .chain(caller.iter().copied().map(Ok))
        .collect()
}

/// Call on every rank right after a multi-rank communicator comes up, before
/// any other collective. Fails, on every rank, when a setting differs across
/// the ranks or a rank's parser refuses one, and logs why before returning:
/// a rank that fails is gone before its peer's teardown reaches its own
/// report. `caller` carries the settings resolved outside this crate: the
/// same names in the same order on every rank.
pub fn agree(comm: &dyn CommBackend, gpu: &dyn GpuBackend, caller: &[Setting]) -> Result<()> {
    agree_on(settings(caller), warn_only(), comm, gpu)
        .inspect_err(|why| tracing::error!("Startup settings agreement: {why:#}"))
}

/// What a rank whose parser refused a setting gathers in place of its
/// table's id.
const REFUSED: u64 = 0;

/// Names a table: FNV-1a over its setting names, in order. Never [`REFUSED`].
fn table_id(settings: &[Setting]) -> u64 {
    settings
        .iter()
        .flat_map(|s| s.0.bytes().chain([0]))
        .fold(0xcbf2_9ce4_8422_2325, |id, b| {
            (id ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
        | 1
}

/// Fails with `why`; `warn` (`ATLAS_STARTUP_PARITY=warn`) logs it instead.
fn fail(warn: bool, why: anyhow::Error) -> Result<()> {
    if !warn {
        return Err(why);
    }
    tracing::warn!("Startup settings agreement ignored (ATLAS_STARTUP_PARITY=warn): {why:#}");
    Ok(())
}

fn agree_on(
    ours: Result<Vec<Setting>>,
    warn: bool,
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let me = comm.rank();
    // The table is part of the build and the second gather is as long as the
    // table, so first compare the tables, in a gather of one word. Every rank
    // takes part, a rank without settings too, and every rank reads the same
    // words, so they all stop here or all go on.
    let table = ours.as_ref().map_or(REFUSED, |ours| table_id(ours));
    let tables = gather_words(comm, gpu, &[table])?;
    let ours = match (ours, tables.iter().position(|&t| t == REFUSED)) {
        (Err(why), _) => {
            return fail(
                warn,
                why.context(format!("rank {me} refuses one of its settings")),
            );
        }
        (Ok(_), Some(rank)) => {
            return fail(
                warn,
                anyhow!("rank {rank} refuses one of its settings: see its log"),
            );
        }
        (Ok(ours), None) => ours,
    };
    ensure!(
        tables.iter().all(|&t| t == table),
        "rank {me} compares other settings at startup than its peers: \
         the ranks run different builds"
    );
    let values: Vec<u64> = ours.iter().map(|s| s.1).collect();
    let all = gather_words(comm, gpu, &values)?;
    let differ: Vec<String> = all
        .chunks(ours.len())
        .enumerate()
        .flat_map(|(rank, theirs)| {
            ours.iter()
                .zip(theirs)
                .filter(|(ours, theirs)| ours.1 != **theirs)
                .map(move |((name, ours), theirs)| {
                    format!("{name}: rank {me} has {ours}, rank {rank} has {theirs}")
                })
        })
        .collect();
    if differ.is_empty() {
        return Ok(());
    }
    fail(
        warn,
        anyhow!(
            "every rank must run the same settings ({})",
            differ.join("; ")
        ),
    )
}

/// Every rank's `words`, in rank order: one all-gather of `8 * words.len()`
/// bytes a rank.
pub(crate) fn gather_words(
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
    words: &[u64],
) -> Result<Vec<u64>> {
    let ours: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut all = vec![0u8; ours.len() * comm.world_size()];
    let buf = gpu.alloc(ours.len() + all.len())?;
    let recv = buf.offset(ours.len());
    let gathered = gpu
        .copy_h2d(&ours, buf)
        .and_then(|()| comm.all_gather(buf.0, recv.0, ours.len()))
        .and_then(|()| gpu.copy_d2h(recv, &mut all));
    gpu.free(buf)?;
    gathered?;
    Ok(all
        .chunks_exact(8)
        .map(|w| u64::from_le_bytes(w.try_into().expect("8-byte words")))
        .collect())
}

#[cfg(test)]
#[path = "startup_parity_tests.rs"]
mod tests;
