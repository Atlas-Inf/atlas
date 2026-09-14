// SPDX-License-Identifier: AGPL-3.0-only

//! Low-rank hyper-connection reference: the collapse and its injection gains.

use super::{PleDims, grouped_rms_norm, linear, sigmoid};

/// One hyper-connection block's weights. Widths are `hc_count * hidden` — the
/// residual is that many streams concatenated, not `hidden` with a gate.
pub struct HyperConnectionWeights<'a> {
    /// `[hc_count * hidden]`, an offset from 1
    pub hc_norm: &'a [f32],
    /// `[hc_lowrank, hc_count * hidden]`
    pub mix_down: &'a [f32],
    /// `[hc_count * hidden, hc_lowrank]`
    pub mix_up: &'a [f32],
    /// `[hc_count, hc_count * hidden]`. `None` on the trunk and MTP mixers,
    /// which mix without injecting.
    pub block_inject: Option<&'a [f32]>,
}

/// What a hyper-connection block hands the layer that follows it.
pub struct HyperConnectionOut {
    /// `[hidden]` — the mixed input the block actually computes on.
    pub mixed: Vec<f32>,
    /// `[hc_count]` — per-stream injection gains, centred on 1. Empty when the
    /// block has no `block_inject_weight`.
    pub injection: Vec<f32>,
}

/// One position through a hyper-connection block.
pub fn hyper_connection_forward(
    dims: &PleDims,
    w: &HyperConnectionWeights<'_>,
    lowrank: usize,
    hyper_input: &[f32],
) -> HyperConnectionOut {
    let (wide, hidden, hc) = (dims.wide(), dims.hidden, dims.hc_count);
    assert_eq!(
        hyper_input.len(),
        wide,
        "hyper input must be hc_count * hidden"
    );

    let normed = grouped_rms_norm(hyper_input, hidden, w.hc_norm, dims.eps);
    // Divided by hc_count BEFORE the activation, not after.
    let down: Vec<f32> = linear(&normed, w.mix_down, lowrank, wide)
        .into_iter()
        .map(|v| {
            let scaled = v / hc as f32;
            scaled * sigmoid(scaled)
        })
        .collect();
    let gate: Vec<f32> = linear(&down, w.mix_up, wide, lowrank)
        .into_iter()
        .map(sigmoid)
        .collect();

    // Mean across streams: `hc_count * hidden` in, `hidden` out.
    let mut mixed = vec![0f32; hidden];
    for stream in 0..hc {
        let span = stream * hidden..(stream + 1) * hidden;
        for ((slot, g), n) in mixed.iter_mut().zip(&gate[span.clone()]).zip(&normed[span]) {
            *slot += g * n;
        }
    }
    for value in &mut mixed {
        *value /= hc as f32;
    }

    let injection = match w.block_inject {
        None => Vec::new(),
        // Centred on 1, not 0.5.
        Some(inject) => linear(&normed, inject, hc, wide)
            .into_iter()
            .map(|v| 2.0 * sigmoid(v / hc as f32))
            .collect(),
    };
    HyperConnectionOut { mixed, injection }
}
