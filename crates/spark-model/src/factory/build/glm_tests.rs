// SPDX-License-Identifier: AGPL-3.0-only

//! Replicated-cache capacity and SSM-only restore agreement. The communicator
//! fixture is retained from Jason McCartney's original shard agreement tests.

use super::*;
use crate::model::ssm_tier::SpillHome;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::Mutex;

/// One rank of a pair whose peer gathers the word `peer`; the byte count of
/// every gather.
struct Ranks<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    peer: u64,
    gathers: Mutex<Vec<usize>>,
}

impl spark_comm::CommBackend for Ranks<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_gather(&self, send: u64, recv: u64, bytes: usize) -> Result<()> {
        self.gathers.lock().unwrap().push(bytes);
        let mut ours = [0u8; 8];
        self.gpu.copy_d2h(DevicePtr(send), &mut ours)?;
        let (mine, theirs) = (8 * self.rank, 8 * (1 - self.rank));
        self.gpu.copy_h2d(&ours, DevicePtr(recv).offset(mine))?;
        self.gpu
            .copy_h2d(&self.peer.to_le_bytes(), DevicePtr(recv).offset(theirs))
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected all-reduce")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected receive")
    }
}

fn agree(
    rank: usize,
    blocks: usize,
    home: Option<SpillHome>,
    peer: u64,
) -> (Result<usize>, Vec<usize>) {
    let gpu = MockGpuBackend::new();
    let ranks = Ranks {
        gpu: &gpu,
        rank,
        peer,
        gathers: Mutex::default(),
    };
    let tier = super::super::ssm_spill::rank_word(home);
    let agreed = agree_kv_blocks(Some(&ranks), &gpu, blocks, None, tier);
    assert_eq!(gpu.alloc_count(), 0, "the gather buffer is freed");
    (agreed, ranks.gathers.into_inner().unwrap())
}

#[test]
fn unsharded_ranks_gather_their_block_counts_and_take_the_smaller() {
    for rank in 0..2 {
        let (agreed, gathers) = agree(rank, 177_000, None, 150_000);
        assert_eq!((agreed.unwrap(), gathers), (150_000, vec![8]));
    }
}

#[test]
fn ssm_only_tiers_keep_rank_agreement_without_a_kv_nvme_tier() {
    let homes = [
        None,
        Some(SpillHome::HostRam),
        Some(SpillHome::Disk { hot_slots: 2 }),
        Some(SpillHome::Peer),
    ];
    for rank in 0..2 {
        for home in homes {
            for peer_home in homes {
                let peer_word = u64::from(super::super::ssm_spill::rank_word(peer_home)) << 32;
                let (result, gathers) = agree(rank, 177_000, home, peer_word | 150_000);
                assert_eq!(gathers, vec![8]);
                if home == peer_home {
                    assert_eq!(result.unwrap(), 150_000);
                } else {
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("SSM spill-tier config differs")
                    );
                }
            }
        }
    }
}
