// SPDX-License-Identifier: AGPL-3.0-only

//! SSM-only rank agreement retained while the KV NVMe tier is deferred.
//! Extracted from Jason McCartney's spill-tier agreement in `361e54ec`.

use crate::model::ssm_tier::SpillHome;
use anyhow::{Result, ensure};

/// Preserve the SSM-only fingerprints used by the combined spill-tier path.
/// Zero means no tier; active backing modes must agree before serving.
pub(super) fn rank_word(home: Option<SpillHome>) -> u32 {
    let Some(home) = home else { return 0 };
    let backing = match home {
        SpillHome::HostRam => 1,
        SpillHome::Disk { .. } => 2,
        SpillHome::Peer => 3,
    };
    let hash = atlas_tier::hash::mix64(atlas_tier::hash::mix64(0, 0), backing + 1);
    (hash % (u32::MAX as u64 - 1)) as u32 + 1
}

pub(super) fn verify_ranks(local: u32, all: &[u32]) -> Result<()> {
    ensure!(
        all.iter().all(|&word| word == local),
        "SSM spill-tier config differs across ranks (words {all:x?}, 0 = off); \
         set identical ATLAS_SSM_TIER settings and backing on every rank"
    );
    Ok(())
}
