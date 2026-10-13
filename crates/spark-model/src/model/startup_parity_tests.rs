// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the startup agreement, against a communicator whose
//! all-gathers land the other ranks' words around this rank's own.

use std::sync::Mutex;

use anyhow::bail;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

/// Rank `rank` of a world. `rounds[i]` is what every rank holds at the i-th
/// gather (this rank's own entry is ignored). Records the per-rank byte count
/// of every gather; `down` fails them.
struct World<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    rounds: Vec<Vec<Vec<u64>>>,
    gathers: Mutex<Vec<usize>>,
    down: bool,
}

impl<'a> World<'a> {
    fn new(gpu: &'a MockGpuBackend, rank: usize, rounds: Vec<Vec<Vec<u64>>>) -> Self {
        Self {
            gpu,
            rank,
            rounds,
            gathers: Mutex::default(),
            down: false,
        }
    }

    /// The agreement's two gathers when every rank runs the table of `ours`
    /// and the ranks hold `values`.
    fn agreeing(
        gpu: &'a MockGpuBackend,
        rank: usize,
        ours: &[Setting],
        values: Vec<Vec<u64>>,
    ) -> Self {
        let tables = vec![vec![table_id(ours)]; values.len()];
        Self::new(gpu, rank, vec![tables, values])
    }
}

impl CommBackend for World<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.rounds[0].len()
    }
    fn all_gather(&self, send: u64, recv: u64, bytes: usize) -> Result<()> {
        anyhow::ensure!(!self.down, "the peer is gone");
        let round = {
            let mut gathers = self.gathers.lock().unwrap();
            gathers.push(bytes);
            &self.rounds[gathers.len() - 1]
        };
        let mut own = vec![0u8; bytes];
        self.gpu.copy_d2h(DevicePtr(send), &mut own)?;
        for (rank, words) in round.iter().enumerate() {
            let theirs: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let chunk = if rank == self.rank { &own } else { &theirs };
            assert_eq!(chunk.len(), bytes, "rank {rank} gathers another size");
            self.gpu
                .copy_h2d(chunk, DevicePtr(recv).offset(rank * bytes))?;
        }
        Ok(())
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected reduce-scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected receive")
    }
}

const CALLER: [Setting; 2] = [("--block-size", 16), ("--ssm-cache-slots", 16)];

fn values(settings: &[Setting]) -> Vec<u64> {
    settings.iter().map(|s| s.1).collect()
}

/// `agree_on(ours)` as rank `rank` of a pair whose peer holds `peer`.
fn pair(rank: usize, ours: &[Setting], peer: Vec<u64>) -> Result<()> {
    let gpu = MockGpuBackend::new();
    let world = World::agreeing(&gpu, rank, ours, vec![peer.clone(), peer]);
    let agreed = agree_on(Ok(ours.to_vec()), false, &world, &gpu);
    assert_eq!(gpu.alloc_count(), 0, "the gather buffer is freed");
    agreed
}

/// The gathers of `agree_on` as rank `rank` of a pair: the peer gathers
/// `peer_table`, then (if the agreement gets that far) `peer`.
fn gathers(
    rank: usize,
    ours: Result<Vec<Setting>>,
    warn: bool,
    peer_table: u64,
    peer: Vec<u64>,
) -> (Result<()>, Vec<usize>) {
    let gpu = MockGpuBackend::new();
    let rounds = vec![vec![vec![peer_table]; 2], vec![peer; 2]];
    let world = World::new(&gpu, rank, rounds);
    let agreed = agree_on(ours, warn, &world, &gpu);
    assert_eq!(gpu.alloc_count(), 0, "the gather buffer is freed");
    (agreed, world.gathers.into_inner().unwrap())
}

#[test]
fn matching_ranks_agree() {
    let ours = settings(&CALLER).unwrap();
    for rank in 0..2 {
        pair(rank, &ours, values(&ours)).unwrap();
    }
}

#[test]
fn each_setting_that_differs_is_named_with_both_values_on_both_ranks() {
    let ours = settings(&CALLER).unwrap();
    for (i, &(name, value)) in ours.iter().enumerate() {
        let mut peer = values(&ours);
        peer[i] = value + 7;
        for (rank, other) in [(0, 1), (1, 0)] {
            let why = pair(rank, &ours, peer.clone()).unwrap_err().to_string();
            let want = format!(
                "{name}: rank {rank} has {value}, rank {other} has {}",
                value + 7
            );
            assert!(why.contains(&want), "{why}");
            // Nothing else is blamed.
            assert_eq!(why.matches(" has ").count(), 2, "{why}");
        }
    }
}

#[test]
fn every_difference_is_listed() {
    let ours = [("A", 1), ("B", 2), ("C", 3)];
    let why = pair(1, &ours, vec![1, 5, 0]).unwrap_err().to_string();
    assert!(why.contains("B: rank 1 has 2, rank 0 has 5"), "{why}");
    assert!(why.contains("C: rank 1 has 3, rank 0 has 0"), "{why}");
    assert!(!why.contains("A: "), "{why}");
}

#[test]
fn every_rank_of_a_wider_world_is_compared() {
    let gpu = MockGpuBackend::new();
    let ours = [("A", 1), ("B", 2)];
    let values = vec![vec![1, 2], vec![], vec![1, 2], vec![9, 2]];
    let world = World::agreeing(&gpu, 1, &ours, values);
    let why = agree_on(Ok(ours.to_vec()), false, &world, &gpu)
        .unwrap_err()
        .to_string();
    assert!(why.contains("A: rank 1 has 1, rank 3 has 9"), "{why}");
    assert_eq!(why.matches(" has ").count(), 2, "{why}");
}

#[test]
fn the_gather_has_one_size_whatever_the_settings_are() {
    let names = settings(&CALLER).unwrap();
    let words = SETTINGS.len() + CALLER.len();
    assert_eq!(names.len(), words);
    // Every flag off, every flag on, and junk: the same collective.
    for value in [0, 1, u64::MAX] {
        let ours: Vec<Setting> = names.iter().map(|s| (s.0, value)).collect();
        let gpu = MockGpuBackend::new();
        let world = World::agreeing(&gpu, 0, &ours, vec![vec![value; words]; 2]);
        agree_on(Ok(ours), false, &world, &gpu).unwrap();
        assert_eq!(*world.gathers.lock().unwrap(), [8, 8 * words]);
    }
    // The table itself: this process's own flags through `agree`.
    let gpu = MockGpuBackend::new();
    let world = World::agreeing(&gpu, 1, &names, vec![values(&names); 2]);
    agree(&world, &gpu, &CALLER).unwrap();
    assert_eq!(*world.gathers.lock().unwrap(), [8, 8 * words]);
}

#[test]
fn a_rank_with_another_table_fails_before_the_settings_are_gathered() {
    // Another build: a setting more, less, renamed or moved.
    let ours = [("A", 1), ("B", 2)];
    for theirs in [
        &[("A", 1), ("B", 2), ("C", 3)][..],
        &[("A", 1)],
        &[("A", 1), ("b", 2)],
        &[("B", 2), ("A", 1)],
        &[("AB", 1), ("", 2)],
    ] {
        assert_ne!(table_id(&ours), table_id(theirs));
        for rank in 0..2 {
            let gpu = MockGpuBackend::new();
            let tables = vec![vec![table_id(theirs)]; 2];
            let world = World::new(&gpu, rank, vec![tables]);
            // `ATLAS_STARTUP_PARITY=warn` does not let these ranks go on:
            // their settings gathers would mispair.
            let why = agree_on(Ok(ours.to_vec()), true, &world, &gpu)
                .unwrap_err()
                .to_string();
            assert!(why.contains("different builds"), "{why}");
            assert_eq!(*world.gathers.lock().unwrap(), [8]);
        }
    }
    // The values are not part of it.
    assert_eq!(table_id(&ours), table_id(&[("A", 7), ("B", 0)]));
    assert_ne!(table_id(&[]), REFUSED);
}

#[test]
fn a_rank_whose_parser_refuses_a_setting_still_gathers_and_both_ranks_fail() {
    let ours = [("A", 1), ("B", 2)];
    let table = table_id(&ours);
    for (rank, peer) in [(0, 1), (1, 0)] {
        // The rank with the refused value: its one-word gather, then its
        // parser's message.
        let refusal = || Err(anyhow::anyhow!("ATLAS_X must be 0 or 1"));
        let (agreed, sizes) = gathers(rank, refusal(), false, table, vec![1, 2]);
        let why = format!("{:#}", agreed.unwrap_err());
        assert!(
            why.contains(&format!("rank {rank} refuses one of its settings"))
                && why.contains("ATLAS_X must be 0 or 1"),
            "{why}"
        );
        assert_eq!(sizes, [8]);
        // Its peer: the same one gather, and the refusing rank named.
        let (agreed, sizes) = gathers(peer, Ok(ours.to_vec()), false, REFUSED, vec![]);
        let why = agreed.unwrap_err().to_string();
        assert!(
            why.contains(&format!("rank {rank} refuses one of its settings")),
            "{why}"
        );
        assert_eq!(sizes, [8]);
        // Both refusing: each reports its own.
        let (agreed, sizes) = gathers(rank, refusal(), false, REFUSED, vec![]);
        assert!(format!("{:#}", agreed.unwrap_err()).contains("ATLAS_X must be 0 or 1"));
        assert_eq!(sizes, [8]);
    }
}

#[test]
fn warn_only_logs_and_boots_with_the_same_gathers() {
    let ours = [("A", 1), ("B", 2)];
    let table = table_id(&ours);
    // A difference: both gathers, as when failing.
    let differs = |warn| gathers(1, Ok(ours.to_vec()), warn, table, vec![1, 5]);
    let (strict, strict_sizes) = differs(false);
    assert!(strict.is_err());
    let (warned, sizes) = differs(true);
    warned.unwrap();
    assert_eq!(sizes, [8, 16]);
    assert_eq!(sizes, strict_sizes);
    // A refusal, this rank's or its peer's: the one gather.
    let refused = Err(anyhow::anyhow!("junk"));
    let (warned, sizes) = gathers(0, refused, true, table, vec![]);
    warned.unwrap();
    assert_eq!(sizes, [8]);
    let (warned, sizes) = gathers(0, Ok(ours.to_vec()), true, REFUSED, vec![]);
    warned.unwrap();
    assert_eq!(sizes, [8]);
}

#[test]
fn the_table_carries_each_setting_once() {
    let names: Vec<&str> = SETTINGS.iter().map(|s| s.0).collect();
    for required in [
        "ATLAS_EP_PROTOCOL=v2",
        "ATLAS_NO_TAIL_SPLIT",
        "ATLAS_GLM_PREFILL_SP",
        "ATLAS_GLM_INDEX_SPLIT",
        "ATLAS_GLM_INDEX_SPLIT_MIN_CTX",
        "ATLAS_GLM_INDEX_SPLIT_CHECK",
        "ATLAS_GLM_PC_EVICT",
        "ATLAS_GLM_PC_BRANCH",
        "ATLAS_GLM_PC_BRANCH_MIN",
        "ATLAS_GLM_PC_FINISH_LEAF",
        "ATLAS_GLM_PC_FINISH_LEAF_BLOCKS",
        "ATLAS_GLM_PC_WRITE_FLOOR",
        "ATLAS_GLM_KV_WRITE_FLOOR_LEGACY",
        "ATLAS_GLM_KDA_MULTI_SEQ",
        "ATLAS_GLM_MLA_MULTI_SEQ",
        "ATLAS_GLM_KDA_BATCHED_FFN",
        "ATLAS_GLM_C4_DECODE",
        "ATLAS_GLM_LONG_BATCH_FFN",
        "ATLAS_GLM_LONG_BATCH_SERIAL",
        "ATLAS_GLM_MTP_DISTRIBUTED",
        "ATLAS_STARTUP_PARITY=warn",
    ] {
        assert_eq!(
            names.iter().filter(|&&n| n == required).count(),
            1,
            "{required}"
        );
    }
    let mut unique = names.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), names.len(), "a setting is listed twice");
}

#[test]
fn a_setting_under_a_switch_that_is_off_reads_zero() {
    assert_eq!(while_on(false, || 2048).unwrap(), 0);
    assert_eq!(while_on(true, || 2048).unwrap(), 2048);
}

#[test]
fn gather_words_returns_every_rank_in_order_and_frees_on_failure() {
    let gpu = MockGpuBackend::new();
    let world = World::new(&gpu, 1, vec![vec![vec![3, 4], vec![], vec![7, 8]]]);
    assert_eq!(
        gather_words(&world, &gpu, &[5, 6]).unwrap(),
        [3, 4, 5, 6, 7, 8]
    );
    assert_eq!(gpu.alloc_count(), 0);

    let down = World {
        down: true,
        ..World::new(&gpu, 0, vec![vec![vec![]; 2]])
    };
    assert!(gather_words(&down, &gpu, &[1]).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}
