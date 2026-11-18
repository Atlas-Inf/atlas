// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};

/// GLM-style coincident TP/EP groups, not a Cartesian four-rank mesh.
/// Experts are whole and contiguous; router, latent transforms and shared
/// MLP are replicated. Reduce routed H outputs, then add shared once/rank.
#[derive(Debug, Clone, Copy)]
pub struct EpPlan {
    pub rank: usize,
    pub world: usize,
    pub experts: usize,
}

impl EpPlan {
    pub fn new(rank: usize, tp: usize, ep: usize, experts: usize) -> Result<Self> {
        ensure!(
            (tp == 1 || tp == 2)
                && tp == ep
                && rank < ep
                && experts > 0
                && experts.is_multiple_of(ep),
            "Hy4 requires coincident TP/EP=1 or 2"
        );
        Ok(Self {
            rank,
            world: ep,
            experts,
        })
    }
    pub fn owns(self, expert: usize) -> bool {
        let width = self.experts / self.world;
        (self.rank * width..(self.rank + 1) * width).contains(&expert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tp2_ep2_owns_each_whole_expert_once() {
        let plans = [
            EpPlan::new(0, 2, 2, 512).unwrap(),
            EpPlan::new(1, 2, 2, 512).unwrap(),
        ];
        for e in 0..512 {
            assert_eq!(plans.iter().filter(|p| p.owns(e)).count(), 1);
        }
        assert!(plans[0].owns(255));
        assert!(!plans[0].owns(256));
        assert!(plans[1].owns(256));
    }
    #[test]
    fn rejects_empty_invalid_or_cartesian_mesh() {
        for (rank, tp, ep, n) in [
            (0, 2, 1, 512),
            (0, 1, 2, 512),
            (2, 2, 2, 512),
            (0, 2, 2, 511),
            (0, 1, 1, 0),
        ] {
            assert!(EpPlan::new(rank, tp, ep, n).is_err());
        }
    }
}
