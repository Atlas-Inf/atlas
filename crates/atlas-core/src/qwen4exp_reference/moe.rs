// SPDX-License-Identifier: AGPL-3.0-only

//! 512-expert MoE reference: routing, SwiGLU, and the weighted sum.

use super::{linear, sigmoid};

/// One MoE block's weights. Expert bodies are borrowed per expert so a caller
/// can dequantize lazily — the published trunk experts are 120.8 B parameters
/// and no reference wants them all in f32 at once.
pub struct MoeWeights<'a> {
    /// `[num_experts, hidden]` router.
    pub router: &'a [f32],
    /// `[1, hidden]` — gates the shared expert through a sigmoid.
    pub shared_gate: &'a [f32],
    /// gate, up `[inter, hidden]`; down `[hidden, inter]`.
    pub shared_expert: [&'a [f32]; 3],
}

pub struct MoeDims {
    pub hidden: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
    pub shared_intermediate: usize,
    /// Renormalise the top-K routing weights to sum to 1.
    pub norm_topk_prob: bool,
}

fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| g * sigmoid(*g) * u)
        .collect()
}

/// Route one token and return `(expert, weight)` pairs, most-probable first.
///
/// Softmax over ALL experts first, then top-K — not top-K then softmax. The two
/// differ, and the second one silently renormalises away the router's
/// confidence.
pub fn moe_route(dims: &MoeDims, router: &[f32], x: &[f32]) -> Vec<(usize, f32)> {
    let logits = linear(x, router, dims.num_experts, dims.hidden);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let total: f32 = exp.iter().sum();

    let mut probs: Vec<(usize, f32)> = exp.iter().map(|e| e / total).enumerate().collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs.truncate(dims.top_k);
    if dims.norm_topk_prob {
        let sum: f32 = probs.iter().map(|(_, w)| w).sum();
        for (_, w) in &mut probs {
            *w /= sum;
        }
    }
    probs
}

/// One token through the MoE block.
///
/// `expert` yields `(gate_up, down)` for a routed expert: `gate_up` is
/// `[2*intermediate, hidden]` with gate first and up second — HF's native
/// stacked layout, which a quantized checkpoint splits per `nn.Linear`.
pub fn moe_forward<'a>(
    dims: &MoeDims,
    w: &MoeWeights<'_>,
    x: &[f32],
    mut expert: impl FnMut(usize) -> Option<(&'a [f32], &'a [f32])>,
) -> Vec<f32> {
    assert_eq!(x.len(), dims.hidden);
    let mut out = vec![0f32; dims.hidden];

    for (index, weight) in moe_route(dims, w.router, x) {
        let Some((gate_up, down)) = expert(index) else {
            continue;
        };
        let fused = linear(x, gate_up, dims.intermediate * 2, dims.hidden);
        let (gate, up) = fused.split_at(dims.intermediate);
        let activated = swiglu(gate, up);
        for (slot, value) in
            out.iter_mut()
                .zip(linear(&activated, down, dims.hidden, dims.intermediate))
        {
            *slot += value * weight;
        }
    }

    // The shared expert runs for every token, gated by its own sigmoid.
    let gate = linear(x, w.shared_expert[0], dims.shared_intermediate, dims.hidden);
    let up = linear(x, w.shared_expert[1], dims.shared_intermediate, dims.hidden);
    let shared = linear(
        &swiglu(&gate, &up),
        w.shared_expert[2],
        dims.hidden,
        dims.shared_intermediate,
    );
    let shared_scale = sigmoid(linear(x, w.shared_gate, 1, dims.hidden)[0]);
    for (slot, value) in out.iter_mut().zip(shared) {
        *slot += shared_scale * value;
    }
    out
}
