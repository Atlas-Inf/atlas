// SPDX-License-Identifier: AGPL-3.0-only

//! QSA indexer reference — which tokens a query is allowed to attend.
//!
//! Transcribed from `Qwen4ExpTextQSAIndexer.forward` in the vendored reference
//! (`bench/qwen4_exp/ref/modeling_qwen4_exp.py`), NOT from the CUDA kernel. That
//! direction matters: an oracle read off the kernel it is meant to check proves
//! only that the transcription was faithful.
//!
//! The mechanism, in the order the reference does it:
//!
//! 1. `index_qk_proj(hidden)` -> `[n_heads * hd | kv_heads * hd]`; with
//!    `indexer_kv_heads = 1` the key half is ONE `hd`-wide row per token.
//! 2. the query half takes `q_layernorm` (offset-from-1) then rope at the
//!    query's own position.
//! 3. the visible prefix is cut into `ratio`-token blocks. Only COMPLETE
//!    blocks are scored; the remainder is a tail that is always visible.
//! 4. a block's key is the MEAN of its tokens' raw keys, computed in f32, then
//!    `k_layernorm`, then rope **at the block's FIRST token position** — not at
//!    the block's centre and not at the query.
//! 5. `score(block) = sum_heads relu(q_h . k_block) / sqrt(hd)`.
//! 6. the top `min(block_topk, complete_blocks)` blocks contribute all their
//!    tokens; `block_topk = indexer_budget / ratio`.
//!
//! THE INERTNESS PROPERTY, which is what makes short contexts exact rather
//! than approximate: `min()` means that at or below `indexer_budget` visible
//! tokens there are at most `block_topk` complete blocks, so EVERY block is
//! selected and the indexer cannot mask anything. Both independent ports
//! measured the same threshold — 2048 masks nothing, 2052 masks 4 — and
//! `selects_everything_below_the_budget` pins it here without a GPU.

use super::{grouped_rms_norm, linear, rope_tables};

/// Indexer geometry. The published checkpoint: 4 query heads, 1 key head, 128
/// dims, budget 2048, ratio 4 — so `block_topk` is 512.
#[derive(Clone, Debug)]
pub struct QsaDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Rotated prefix of each indexer head. `head_dim * partial_rotary_factor`
    /// of the ATTENTION head_dim — 64 of the indexer's 128 on this checkpoint,
    /// so the upper half carries through untouched.
    pub rotary_dim: usize,
    pub budget: usize,
    pub ratio: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

impl QsaDims {
    /// `indexer_budget / compress_ratio` — how many blocks survive selection.
    pub fn block_topk(&self) -> usize {
        self.budget / self.ratio
    }

    /// Width of `index_qk_proj`'s output: queries and the single key row.
    pub fn qk_width(&self) -> usize {
        (self.n_heads + self.kv_heads) * self.head_dim
    }
}

pub struct QsaWeights<'a> {
    /// `[qk_width, hidden]`
    pub index_qk_proj: &'a [f32],
    /// `[head_dim]`, an offset from 1
    pub q_layernorm: &'a [f32],
    /// `[head_dim]`, an offset from 1
    pub k_layernorm: &'a [f32],
}

/// Every intermediate the GPU pipeline also produces, so a parity test can say
/// WHICH stage diverged instead of only that the selection differs.
pub struct QsaStages {
    /// `[seq, head_dim]` — the raw per-token key rows, pre-pooling.
    pub raw_keys: Vec<f32>,
    /// `[seq, n_heads, head_dim]` — normed and roped queries.
    pub q_post: Vec<f32>,
    /// `[blocks, head_dim]` — pooled, normed, roped block keys.
    pub block_keys: Vec<f32>,
    /// `[seq, blocks]` — relu-summed scores per query.
    pub scores: Vec<f32>,
    /// Per query, the token positions it may attend, ASCENDING. Selected
    /// blocks' tokens plus the incomplete tail.
    pub selected: Vec<Vec<u32>>,
}

/// Rope over the rotary PREFIX of one head, in place. `rotate_half` — the same
/// convention the attention oracle uses, which is checked against HF at 8.0e-7.
fn rope_head(head: &mut [f32], pos: usize, rotary: usize, theta: f32) {
    if rotary == 0 {
        return;
    }
    let (cos, sin) = rope_tables(pos + 1, rotary, theta);
    let c = &cos[pos * rotary..(pos + 1) * rotary];
    let s = &sin[pos * rotary..(pos + 1) * rotary];
    let half = rotary / 2;
    let original: Vec<f32> = head[..rotary].to_vec();
    for i in 0..rotary {
        let rotated = if i < half {
            -original[i + half]
        } else {
            original[i - half]
        };
        head[i] = original[i] * c[i] + rotated * s[i];
    }
}

/// One block's key: MEAN of its tokens' raw keys, then `k_layernorm`, then rope
/// **at the block's FIRST token**. Exposed because the GPU pipeline computes
/// exactly this in `qsa_block_pool`, and a parity test must compare against the
/// code [`qsa_select`] runs rather than a second copy of the formula.
///
/// `block_raw` is `[ratio, head_dim]` — the block's tokens, in order.
pub fn qsa_block_key(
    dims: &QsaDims,
    k_layernorm: &[f32],
    block_raw: &[f32],
    first_pos: usize,
) -> Vec<f32> {
    let hd = dims.head_dim;
    assert_eq!(block_raw.len(), dims.ratio * hd, "block_raw is [ratio, hd]");
    let mut pooled = vec![0f32; hd];
    for token in 0..dims.ratio {
        for (slot, v) in pooled
            .iter_mut()
            .zip(&block_raw[token * hd..(token + 1) * hd])
        {
            *slot += *v;
        }
    }
    for slot in &mut pooled {
        *slot /= dims.ratio as f32;
    }
    let mut key = grouped_rms_norm(&pooled, hd, k_layernorm, dims.eps);
    rope_head(&mut key, first_pos, dims.rotary_dim, dims.rope_theta);
    key
}

/// One query head: `q_layernorm` then rope at the query's own position.
/// `raw` is the head's slice of the projection output, `[head_dim]`.
pub fn qsa_query_head(dims: &QsaDims, q_layernorm: &[f32], raw: &[f32], pos: usize) -> Vec<f32> {
    let mut q = grouped_rms_norm(raw, dims.head_dim, q_layernorm, dims.eps);
    rope_head(&mut q, pos, dims.rotary_dim, dims.rope_theta);
    q
}

/// `sum_heads relu(q_h . k) / sqrt(head_dim)`. The relu is PER HEAD; `relu` of
/// the summed dot is a different function and the natural misreading.
pub fn qsa_block_score(dims: &QsaDims, q_post: &[f32], block_key: &[f32]) -> f32 {
    let hd = dims.head_dim;
    let mut acc = 0f32;
    for head in 0..dims.n_heads {
        let dot: f32 = q_post[head * hd..(head + 1) * hd]
            .iter()
            .zip(block_key)
            .map(|(a, b)| a * b)
            .sum();
        acc += dot.max(0.0);
    }
    acc / (hd as f32).sqrt()
}

/// Run the indexer over a whole causal sequence: query `t` sees `0..=t`.
///
/// `hidden` is `[seq, hidden]` — the ATTENTION BLOCK INPUT, which is what the
/// reference projects, not the post-attention output.
pub fn qsa_select(dims: &QsaDims, w: &QsaWeights<'_>, hidden: &[f32]) -> QsaStages {
    let (hd, nh) = (dims.head_dim, dims.n_heads);
    let seq = hidden.len() / dims.hidden;
    assert_eq!(hidden.len(), seq * dims.hidden, "hidden shape");
    assert_eq!(dims.kv_heads, 1, "the reference squeezes a single key head");

    // ── stage 1: project, then split queries from the single key row.
    let mut raw_keys = Vec::with_capacity(seq * hd);
    let mut q_post = Vec::with_capacity(seq * nh * hd);
    for t in 0..seq {
        let qk = linear(
            &hidden[t * dims.hidden..(t + 1) * dims.hidden],
            w.index_qk_proj,
            dims.qk_width(),
            dims.hidden,
        );
        for head in 0..nh {
            // `q_layernorm` is per HEAD, so the group size is head_dim.
            q_post.extend_from_slice(&qsa_query_head(
                dims,
                w.q_layernorm,
                &qk[head * hd..(head + 1) * hd],
                t,
            ));
        }
        raw_keys.extend_from_slice(&qk[nh * hd..(nh + 1) * hd]);
    }

    // ── stage 2: pool, norm and rope one key per COMPLETE block. Causality
    //    makes the block set a prefix, so it is computed once for the whole
    //    sequence and each query uses the prefix it can see.
    let total_blocks = seq / dims.ratio;
    let mut block_keys = Vec::with_capacity(total_blocks * hd);
    for b in 0..total_blocks {
        let first = b * dims.ratio;
        // Roped at the block's FIRST token, which is `group_starts` in the
        // reference. The centre or the query position would both be plausible
        // and both wrong.
        block_keys.extend_from_slice(&qsa_block_key(
            dims,
            w.k_layernorm,
            &raw_keys[first * hd..(first + dims.ratio) * hd],
            first,
        ));
    }

    // ── stage 3: score and select, per query.
    let mut scores = vec![0f32; seq * total_blocks];
    let mut selected = Vec::with_capacity(seq);
    for t in 0..seq {
        let visible = t + 1; // causal
        let complete = visible / dims.ratio;
        let mut ranked: Vec<(usize, f32)> = Vec::with_capacity(complete);
        for b in 0..complete {
            // relu PER HEAD, then summed — see `qsa_block_score`.
            let s = qsa_block_score(
                dims,
                &q_post[t * nh * hd..(t + 1) * nh * hd],
                &block_keys[b * hd..(b + 1) * hd],
            );
            scores[t * total_blocks + b] = s;
            ranked.push((b, s));
        }
        // Descending by score; ties by lower block index, so the result is
        // deterministic where torch's topk is not specified to be.
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        ranked.truncate(dims.block_topk().min(complete));

        let mut tokens: Vec<u32> = Vec::new();
        for (b, _) in &ranked {
            for token in b * dims.ratio..(b + 1) * dims.ratio {
                tokens.push(token as u32);
            }
        }
        // The incomplete tail is ALWAYS visible — including the current token,
        // which is therefore not force-included by the selection itself.
        for token in complete * dims.ratio..visible {
            tokens.push(token as u32);
        }
        tokens.sort_unstable();
        selected.push(tokens);
    }

    QsaStages {
        raw_keys,
        q_post,
        block_keys,
        scores,
        selected,
    }
}

#[cfg(test)]
#[path = "qsa_tests.rs"]
mod qsa_tests;
