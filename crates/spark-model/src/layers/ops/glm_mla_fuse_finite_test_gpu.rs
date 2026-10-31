// SPDX-License-Identifier: AGPL-3.0-only

//! Reject matching nonfinite outputs while preserving untouched cache bytes.

use super::*;
use crate::layers::glm_gpu_test_backend::{finite_bf16, finite_f32};

impl Out {
    pub(super) fn check_finite(&self, start: u32, rows: u32, owner: ShardRank) -> Result<()> {
        for (name, bytes) in [
            ("q latent", &self.q),
            ("kv latent", &self.kv),
            ("index keys", &self.keys),
            ("index gates", &self.gates),
            ("raw tails", &self.tail),
            ("pooled keys", &self.pool),
        ] {
            finite_bf16(name, bytes)?;
        }
        // The fixture poisons untouched latent storage. Only slots written
        // by this rank are valid FP8/scales, in the production block layout.
        for token in start..start + rows {
            let block = physical(token / BS);
            if block % owner.world != owner.rank {
                continue;
            }
            let base = (block / owner.world) as usize * LATENT_BLOCK;
            let offset = (token % BS) as usize;
            let values = &self.latent[base + offset * 512..][..512];
            anyhow::ensure!(
                values.iter().all(|q| q & 0x7f != 0x7f),
                "latent FP8 contains NaN at token {token}"
            );
            let scales = &self.latent[base + BS as usize * 512 + offset * 16..][..16];
            finite_f32("latent FP8 scales", scales)?;
            anyhow::ensure!(
                scales
                    .chunks_exact(4)
                    .all(|s| f32::from_le_bytes(s.try_into().unwrap()) > 0.0),
                "latent FP8 scale is not positive at token {token}"
            );
        }
        Ok(())
    }
}

#[test]
fn latent_finite_gate_checks_written_slots_only() {
    let mut output = Out {
        q: vec![],
        kv: vec![],
        keys: vec![],
        gates: vec![],
        latent: vec![0xff; LATENT_BLOCK],
        tail: vec![],
        pool: vec![],
        ids: vec![],
        counts: vec![],
    };
    let own = ShardRank { rank: 0, world: 2 };
    let peer = ShardRank { rank: 1, world: 2 };
    // Rank 1 owns no slot here; poison is deliberately not numeric data.
    assert!(output.check_finite(0, 1, peer).is_ok());
    assert!(output.check_finite(0, 1, own).is_err());
    output.latent[..512].fill(0);
    let scales = BS as usize * 512;
    for word in output.latent[scales..scales + 16].chunks_exact_mut(4) {
        word.copy_from_slice(&1.0f32.to_le_bytes());
    }
    assert!(output.check_finite(0, 1, own).is_ok());
    output.latent[511] = 0x7f;
    assert!(output.check_finite(0, 1, own).is_err());
    output.latent[511] = 0;
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        output.latent[scales..scales + 4].copy_from_slice(&bad.to_le_bytes());
        assert!(output.check_finite(0, 1, own).is_err());
    }
}
