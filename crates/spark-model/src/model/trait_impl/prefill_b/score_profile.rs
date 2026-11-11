// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in phase timing for score passes (`ATLAS_SCORE_TRACE=2`).

use anyhow::Result;

use super::super::super::types::TransformerModel;

/// Phases timed by `ATLAS_SCORE_TRACE=2` (each mark synchronizes).
#[derive(Clone, Copy)]
pub(super) enum Phase {
    Setup,
    Attn,
    SsmProj,
    SsmConv,
    SsmGdn,
    SsmOut,
}

pub(super) struct ScoreProfile {
    on: bool,
    last: std::time::Instant,
    us: [u128; 6],
}

impl ScoreProfile {
    pub(super) fn new() -> Self {
        Self {
            on: std::env::var("ATLAS_SCORE_TRACE").is_ok_and(|v| v == "2"),
            last: std::time::Instant::now(),
            us: [0; 6],
        }
    }

    /// Charge the time since the previous mark to `phase`.
    pub(super) fn mark(&mut self, m: &TransformerModel, stream: u64, phase: Phase) -> Result<()> {
        if self.on {
            m.gpu.synchronize(stream)?;
            let now = std::time::Instant::now();
            self.us[phase as usize] += now.duration_since(self.last).as_micros();
            self.last = now;
        }
        Ok(())
    }

    pub(super) fn report(&self, rows: usize, streams: usize) {
        if self.on {
            let [setup, attn, proj, conv, gdn, out] = self.us;
            tracing::info!(
                "score profile: rows={rows} streams={streams} setup={setup}us attn={attn}us \
                 ssm_proj={proj}us ssm_conv={conv}us ssm_gdn={gdn}us ssm_out_moe={out}us"
            );
        }
    }
}
