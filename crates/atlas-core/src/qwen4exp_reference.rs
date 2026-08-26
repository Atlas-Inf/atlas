// SPDX-License-Identifier: AGPL-3.0-only

//! CPU references for the `qwen4_exp` layers that Atlas does not have yet.
//!
//! Not the serving path — these are what a GPU implementation gets checked
//! against. Each is checked here against HuggingFace's own module at real
//! weights, because each has steps that are easy to get plausibly wrong, and
//! "plausibly wrong" in a language model means fluent output that is subtly
//! not the model you loaded.
//!
//! In the PLE tower:
//!
//! * its RMSNorm is **grouped** (each of `hc_count` streams normalises over its
//!   own `hidden_size` slice) and its weight is an **offset from 1**
//!   (`x * (1 + w)`), not a plain scale;
//! * the gate takes a **signed square root** — `sign(g) * sqrt(max(|g|, 1e-6))`;
//! * the depthwise conv is **dilated by `ngram_size`**, so its state is
//!   `(kernel - 1) * ngram_size` wide rather than `kernel - 1`.
//!
//! In the hyper-connection block:
//!
//! * the mixing gate divides by `hc_count` BEFORE its activation, twice — once
//!   into the low-rank SiLU and again into the injection sigmoid;
//! * the block output is a **mean** across streams, not a sum or a concat, so
//!   it comes back `hidden` wide from a `hc_count * hidden` input;
//! * the injection weights are `2 * sigmoid(...)`, centred on 1 rather than
//!   0.5.

/// Grouped RMS norm. `weight` is an offset from 1, applied across the full row.
pub fn grouped_rms_norm(x: &[f32], group: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len() % group, 0, "row must divide into groups");
    assert_eq!(weight.len(), x.len(), "norm weight must match the row");
    let mut out = vec![0f32; x.len()];
    for (chunk_index, chunk) in x.chunks_exact(group).enumerate() {
        let mean_square = chunk.iter().map(|v| v * v).sum::<f32>() / group as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        let base = chunk_index * group;
        for (offset, value) in chunk.iter().enumerate() {
            out[base + offset] = value * scale * (1.0 + weight[base + offset]);
        }
    }
    out
}

/// `y = W x` for a row-major `[out_dim, in_dim]` weight.
pub fn linear(x: &[f32], weight: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_dim);
    assert_eq!(weight.len(), out_dim * in_dim);
    (0..out_dim)
        .map(|row| {
            weight[row * in_dim..(row + 1) * in_dim]
                .iter()
                .zip(x)
                .map(|(w, v)| w * v)
                .sum()
        })
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Everything the tower needs that is not an activation.
pub struct PleWeights<'a> {
    /// `[hc_count * hidden, ple_embed_dim]`
    pub key_proj: &'a [f32],
    /// `[hidden, ple_embed_dim]`
    pub value_proj: &'a [f32],
    /// `[hc_count * hidden]`, each an offset from 1
    pub norm_key: &'a [f32],
    pub norm_query: &'a [f32],
    pub norm_conv: &'a [f32],
    /// `[hc_count * hidden, kernel]`, depthwise
    pub conv1d: &'a [f32],
}

pub struct PleDims {
    pub hidden: usize,
    pub hc_count: usize,
    pub ple_embed_dim: usize,
    pub kernel: usize,
    /// Conv dilation, which the reference sets to `ngram_size`.
    pub dilation: usize,
    pub eps: f32,
}

impl PleDims {
    pub fn wide(&self) -> usize {
        self.hidden * self.hc_count
    }
    /// Left context the dilated conv needs: `(kernel - 1) * dilation`.
    pub fn conv_state_len(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }
}

/// One PLE forward over a whole sequence, starting from a fresh conv state.
///
/// `embeddings` is `[seq, ple_embed_dim]` — the gathered, dequantized n-gram
/// rows. `hidden_states` is `[seq, hc_count * hidden]`. Returns the same shape
/// as `hidden_states`.
pub fn ple_forward(
    dims: &PleDims,
    w: &PleWeights<'_>,
    embeddings: &[f32],
    hidden_states: &[f32],
) -> Vec<f32> {
    let (wide, hidden, e) = (dims.wide(), dims.hidden, dims.ple_embed_dim);
    let seq = embeddings.len() / e;
    assert_eq!(hidden_states.len(), seq * wide, "hidden_states shape");

    // Per position: project, norm, gate, scale the shared value.
    let mut gated = vec![0f32; seq * wide];
    for t in 0..seq {
        let emb = &embeddings[t * e..(t + 1) * e];
        let key = linear(emb, w.key_proj, wide, e);
        let key_normed = grouped_rms_norm(&key, hidden, w.norm_key, dims.eps);
        let value = linear(emb, w.value_proj, hidden, e);
        let query_normed = grouped_rms_norm(
            &hidden_states[t * wide..(t + 1) * wide],
            hidden,
            w.norm_query,
            dims.eps,
        );

        for stream in 0..dims.hc_count {
            let span = stream * hidden..(stream + 1) * hidden;
            let dot: f32 = key_normed[span.clone()]
                .iter()
                .zip(&query_normed[span.clone()])
                .map(|(k, q)| k * q)
                .sum::<f32>()
                / (hidden as f32).sqrt();
            // Signed square root, floored so the gradient at zero is finite.
            let gate = dot.abs().max(1e-6).sqrt() * if dot < 0.0 { -1.0 } else { 1.0 };
            let gate = sigmoid(gate);
            for h in 0..hidden {
                gated[t * wide + stream * hidden + h] = gate * value[h];
            }
        }
    }

    // The conv runs on a normalised copy; the residual adds the un-normalised one.
    let mut normed = Vec::with_capacity(seq * wide);
    for t in 0..seq {
        normed.extend(grouped_rms_norm(
            &gated[t * wide..(t + 1) * wide],
            hidden,
            w.norm_conv,
            dims.eps,
        ));
    }

    // Depthwise, dilated, causal: zero-padded on the left by (kernel-1)*dilation.
    let mut out = gated;
    for t in 0..seq {
        for channel in 0..wide {
            let mut acc = 0f32;
            for tap in 0..dims.kernel {
                let back = (dims.kernel - 1 - tap) * dims.dilation;
                if let Some(source) = t.checked_sub(back) {
                    acc += w.conv1d[channel * dims.kernel + tap] * normed[source * wide + channel];
                }
            }
            out[t * wide + channel] += silu(acc);
        }
    }
    out
}

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

/// Full-attention weights for one layer. The indexer is absent on purpose:
/// it is a no-op below `indexer_budget + compress_ratio - 1` tokens of
/// context, which is where a first bring-up lives.
pub struct AttnWeights<'a> {
    /// `[2 * num_heads * head_dim, hidden]` — query and its gate, interleaved
    /// per head.
    pub q_proj: &'a [f32],
    /// `[num_kv_heads * head_dim, hidden]`
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    /// `[hidden, num_heads * head_dim]` — consumes only the query half.
    pub o_proj: &'a [f32],
    /// `[head_dim]`, offsets from 1, applied per head.
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}

pub struct AttnDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// `head_dim * partial_rotary_factor`. Only this prefix of each head is
    /// rotated; the rest carries through untouched.
    pub rotary_dim: usize,
    pub eps: f32,
}

/// Rotate the first `rotary_dim` of one head in place, rotate-half style.
fn apply_rope(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let rotary = cos.len();
    let half = rotary / 2;
    let original: Vec<f32> = head[..rotary].to_vec();
    for i in 0..rotary {
        // rotate_half: [-x2, x1]
        let rotated = if i < half {
            -original[i + half]
        } else {
            original[i - half]
        };
        head[i] = original[i] * cos[i] + rotated * sin[i];
    }
}

/// Causal multi-head attention over a whole sequence.
///
/// `cos` and `sin` are `[seq, rotary_dim]` — supplied rather than derived, so
/// this stays independent of how MRoPE builds them. `hidden` is
/// `[seq, hidden]`; the result is the same shape.
pub fn attention_forward(
    dims: &AttnDims,
    w: &AttnWeights<'_>,
    hidden: &[f32],
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    let (h, hd, rot) = (dims.hidden, dims.head_dim, dims.rotary_dim);
    let seq = hidden.len() / h;
    let q_dim = dims.num_heads * hd;
    let kv_dim = dims.num_kv_heads * hd;
    let group = dims.num_heads / dims.num_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut queries = vec![0f32; seq * q_dim];
    let mut gates = vec![0f32; seq * q_dim];
    let mut keys = vec![0f32; seq * kv_dim];
    let mut values = vec![0f32; seq * kv_dim];

    for t in 0..seq {
        let x = &hidden[t * h..(t + 1) * h];
        let (c, s) = (&cos[t * rot..(t + 1) * rot], &sin[t * rot..(t + 1) * rot]);

        // q_proj emits [query | gate] per head, hence the 2x head stride.
        let fused = linear(x, w.q_proj, q_dim * 2, h);
        for head in 0..dims.num_heads {
            let src = head * hd * 2;
            let dst = t * q_dim + head * hd;
            let mut q = grouped_rms_norm(&fused[src..src + hd], hd, w.q_norm, dims.eps);
            apply_rope(&mut q, c, s);
            queries[dst..dst + hd].copy_from_slice(&q);
            gates[dst..dst + hd].copy_from_slice(&fused[src + hd..src + hd * 2]);
        }

        let k = linear(x, w.k_proj, kv_dim, h);
        for head in 0..dims.num_kv_heads {
            let src = head * hd;
            let dst = t * kv_dim + head * hd;
            let mut kh = grouped_rms_norm(&k[src..src + hd], hd, w.k_norm, dims.eps);
            apply_rope(&mut kh, c, s);
            keys[dst..dst + hd].copy_from_slice(&kh);
        }
        values[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&linear(x, w.v_proj, kv_dim, h));
    }

    let mut out = vec![0f32; seq * h];
    for t in 0..seq {
        // Attention output, then the gate, then o_proj.
        let mut context = vec![0f32; q_dim];
        for head in 0..dims.num_heads {
            let kv_head = head / group;
            let q = &queries[t * q_dim + head * hd..t * q_dim + (head + 1) * hd];

            let mut scores = Vec::with_capacity(t + 1);
            for past in 0..=t {
                let k = &keys[past * kv_dim + kv_head * hd..past * kv_dim + (kv_head + 1) * hd];
                scores.push(q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                total += *score;
            }
            for (past, weight) in scores.iter().enumerate() {
                let v = &values[past * kv_dim + kv_head * hd..past * kv_dim + (kv_head + 1) * hd];
                let slot = &mut context[head * hd..(head + 1) * hd];
                for (c, value) in slot.iter_mut().zip(v) {
                    *c += weight / total * value;
                }
            }
        }
        // Elementwise sigmoid gate, applied before the output projection.
        for (c, g) in context.iter_mut().zip(&gates[t * q_dim..(t + 1) * q_dim]) {
            *c *= sigmoid(*g);
        }
        out[t * h..(t + 1) * h].copy_from_slice(&linear(&context, w.o_proj, h, q_dim));
    }
    out
}

/// Gated-delta-net weights for one linear-attention layer.
pub struct GdnWeights<'a> {
    /// `[2*key_dim + value_dim, hidden]` — q, k and v fused.
    pub in_proj_qkv: &'a [f32],
    /// `[value_dim, hidden]` — the output gate.
    pub in_proj_z: &'a [f32],
    /// `[num_v_heads, hidden]`
    pub in_proj_a: &'a [f32],
    pub in_proj_b: &'a [f32],
    /// `[2*key_dim + value_dim, kernel]`, depthwise causal.
    pub conv1d: &'a [f32],
    /// `[num_v_heads]`
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    /// `[value_head_dim]` — a PLAIN multiplier here, not an offset from 1.
    /// `Qwen4ExpTextRMSNormGated` differs from `Qwen4ExpTextRMSNorm` in exactly
    /// this, and they sit a few lines apart in the reference.
    pub norm: &'a [f32],
    /// `[hidden, value_dim]`
    pub out_proj: &'a [f32],
}

pub struct GdnDims {
    pub hidden: usize,
    pub num_k_heads: usize,
    pub key_head_dim: usize,
    pub num_v_heads: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
    /// `true` for `output_gate_type = "sigmoid"`, `false` for SiLU.
    pub sigmoid_gate: bool,
}

fn softplus(x: f32) -> f32 {
    // Numerically stable: log1p(exp(-|x|)) + max(x, 0).
    (-x.abs()).exp().ln_1p() + x.max(0.0)
}

fn l2norm(x: &mut [f32], eps: f32) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    for v in x {
        *v *= inv;
    }
}

/// One gated-delta-net layer over a whole sequence, from a zero state.
///
/// `hidden` is `[seq, hidden]`; the result is the same shape.
pub fn gdn_forward(dims: &GdnDims, w: &GdnWeights<'_>, hidden: &[f32]) -> Vec<f32> {
    let h = dims.hidden;
    let seq = hidden.len() / h;
    let key_dim = dims.num_k_heads * dims.key_head_dim;
    let value_dim = dims.num_v_heads * dims.value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let group = dims.num_v_heads / dims.num_k_heads;
    let (kd, vd) = (dims.key_head_dim, dims.value_head_dim);

    // Project, then a depthwise causal conv with SiLU over the fused q/k/v.
    let mut projected = vec![0f32; seq * conv_dim];
    for t in 0..seq {
        projected[t * conv_dim..(t + 1) * conv_dim].copy_from_slice(&linear(
            &hidden[t * h..(t + 1) * h],
            w.in_proj_qkv,
            conv_dim,
            h,
        ));
    }
    let mut mixed = vec![0f32; seq * conv_dim];
    for t in 0..seq {
        for c in 0..conv_dim {
            let mut acc = 0f32;
            for tap in 0..dims.conv_kernel {
                // Left-padded: tap `kernel-1` is the current position.
                if let Some(src) = t.checked_sub(dims.conv_kernel - 1 - tap) {
                    acc += w.conv1d[c * dims.conv_kernel + tap] * projected[src * conv_dim + c];
                }
            }
            mixed[t * conv_dim + c] = acc * sigmoid(acc); // SiLU
        }
    }

    // Per-head gates. `g` is negative: it becomes a decay through exp().
    let mut recurrent = vec![0f32; dims.num_v_heads * kd * vd];
    let mut context = vec![0f32; seq * value_dim];
    for t in 0..seq {
        let x = &hidden[t * h..(t + 1) * h];
        let a = linear(x, w.in_proj_a, dims.num_v_heads, h);
        let b = linear(x, w.in_proj_b, dims.num_v_heads, h);
        let row = &mixed[t * conv_dim..(t + 1) * conv_dim];
        let (q_all, rest) = row.split_at(key_dim);
        let (k_all, v_all) = rest.split_at(key_dim);

        for head in 0..dims.num_v_heads {
            // q and k are shared across `group` value heads.
            let k_head = head / group;
            let mut q: Vec<f32> = q_all[k_head * kd..(k_head + 1) * kd].to_vec();
            let mut k: Vec<f32> = k_all[k_head * kd..(k_head + 1) * kd].to_vec();
            l2norm(&mut q, 1e-6);
            l2norm(&mut k, 1e-6);
            let scale = 1.0 / (kd as f32).sqrt();
            for value in &mut q {
                *value *= scale;
            }
            let v = &v_all[head * vd..(head + 1) * vd];

            let decay = (-w.a_log[head].exp() * softplus(a[head] + w.dt_bias[head])).exp();
            let beta = sigmoid(b[head]);
            let state = &mut recurrent[head * kd * vd..(head + 1) * kd * vd];

            for value in state.iter_mut() {
                *value *= decay;
            }
            // delta rule: correct the state by how wrong its recall of v is.
            let mut recall = vec![0f32; vd];
            for (ki, kv) in k.iter().enumerate() {
                for (vi, slot) in recall.iter_mut().enumerate() {
                    *slot += state[ki * vd + vi] * kv;
                }
            }
            let delta: Vec<f32> = v
                .iter()
                .zip(&recall)
                .map(|(target, got)| (target - got) * beta)
                .collect();
            for (ki, kv) in k.iter().enumerate() {
                for (vi, d) in delta.iter().enumerate() {
                    state[ki * vd + vi] += kv * d;
                }
            }
            let out = &mut context[t * value_dim + head * vd..t * value_dim + (head + 1) * vd];
            for (ki, qv) in q.iter().enumerate() {
                for (vi, slot) in out.iter_mut().enumerate() {
                    *slot += state[ki * vd + vi] * qv;
                }
            }
        }
    }

    // Gated RMS norm per value head, then the output projection.
    let mut out = vec![0f32; seq * h];
    for t in 0..seq {
        let z = linear(&hidden[t * h..(t + 1) * h], w.in_proj_z, value_dim, h);
        let mut gated = vec![0f32; value_dim];
        for head in 0..dims.num_v_heads {
            let span = head * vd..(head + 1) * vd;
            let x = &context[t * value_dim + span.start..t * value_dim + span.end];
            let mean_square = x.iter().map(|v| v * v).sum::<f32>() / vd as f32;
            let inv = 1.0 / (mean_square + dims.eps).sqrt();
            for (i, slot) in gated[span.clone()].iter_mut().enumerate() {
                let g = z[span.start + i];
                let activated = if dims.sigmoid_gate {
                    sigmoid(g)
                } else {
                    g * sigmoid(g)
                };
                // Plain multiplier, not (1 + w).
                *slot = x[i] * inv * w.norm[i] * activated;
            }
        }
        out[t * h..(t + 1) * h].copy_from_slice(&linear(&gated, w.out_proj, h, value_dim));
    }
    out
}

/// Standard RoPE tables, `[seq, rotary_dim]` each.
///
/// MRoPE collapses to this for text: the three grids carry identical position
/// ids, so `apply_interleaved_mrope` copies each section onto itself. Only a
/// multimodal prompt makes the grids differ.
pub fn rope_tables(seq: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0f32; seq * rotary_dim];
    let mut sin = vec![0f32; seq * rotary_dim];
    for t in 0..seq {
        for i in 0..half {
            let freq = t as f32 / theta.powf(2.0 * i as f32 / rotary_dim as f32);
            // emb = cat(freqs, freqs), so each frequency appears twice.
            for slot in [i, i + half] {
                cos[t * rotary_dim + slot] = freq.cos();
                sin[t * rotary_dim + slot] = freq.sin();
            }
        }
    }
    (cos, sin)
}

/// Scatter a block's `hidden`-wide output back across the `hc_count` residual
/// streams, scaled per stream.
pub fn broadcast_inject(mixed: &[f32], injection: &[f32], hidden: usize) -> Vec<f32> {
    let mut out = vec![0f32; injection.len() * hidden];
    for (stream, gain) in injection.iter().enumerate() {
        for (slot, value) in out[stream * hidden..(stream + 1) * hidden]
            .iter_mut()
            .zip(mixed)
        {
            *slot = value * gain;
        }
    }
    out
}
