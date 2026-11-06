// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side geometry of one score pass over one or more requests.
//!
//! Stream order is every request's head first (requests in order), then every
//! request's branches (requests in order, branches in order). A request's
//! head is `prefix[0..F)` with `F = floor(L / block) * block`, omitted when
//! F == 0; each of its branches is `prefix ++ branch`, processed from F and
//! sharing the head's `F / block` whole blocks.
//!
//! A `prefix_bidir` request marks every one of its streams with the state end
//! S = L (its prefix length): in the full-attention layers state rows read
//! keys k < S and branch rows read every key of their stream. Its head reads
//! the state's partial last block `[F, S)` from its first branch (every
//! branch writes the same tail), so the head's KV extent is S, not F.

/// One request: a shared prefix, its branches, and whether the prefix is a
/// bidirectional state block (`prefix_bidir`).
pub(in crate::model) type ScoreRequest<'a> = (&'a [u32], &'a [Vec<u32>], bool);

pub(super) struct ScorePlan {
    /// Full token list per stream.
    pub tokens: Vec<Vec<u32>>,
    /// First processed position per stream (0 for heads, F for branches).
    pub starts: Vec<usize>,
    /// Streams `0..heads` are heads; the rest are branches.
    pub heads: usize,
    /// Per stream: the head stream a branch forks from (None for heads and for
    /// branches of a request without a head).
    pub head_of: Vec<Option<usize>>,
    /// Per stream: whole blocks shared from its head (0 for heads).
    pub shared_blocks: Vec<usize>,
    /// Branches per request, in request order.
    pub branch_counts: Vec<usize>,
    /// Per stream: the state end S of a `prefix_bidir` request, -1 if causal.
    pub state_end: Vec<i32>,
    /// Per stream: the KV extent its attention reads (S for a bidirectional
    /// head with a partial tail; else its token count).
    pub kv_len: Vec<usize>,
    /// Per head stream: the branch stream whose tail block it reads, if any.
    pub tail_from: Vec<Option<usize>>,
}

impl ScorePlan {
    pub(super) fn new(reqs: &[ScoreRequest<'_>], bs: usize) -> Self {
        let mut p = Self {
            tokens: Vec::new(),
            starts: Vec::new(),
            heads: 0,
            head_of: Vec::new(),
            shared_blocks: Vec::new(),
            branch_counts: Vec::with_capacity(reqs.len()),
            state_end: Vec::new(),
            kv_len: Vec::new(),
            tail_from: Vec::new(),
        };
        let s_end = |prefix: &[u32], bidir: bool| if bidir { prefix.len() as i32 } else { -1 };
        let mut head_idx = Vec::with_capacity(reqs.len());
        for &(prefix, _, bidir) in reqs {
            let f = (prefix.len() / bs) * bs;
            head_idx.push((f > 0).then_some(p.tokens.len()));
            if f > 0 {
                p.push(prefix[..f].to_vec(), 0, None, 0, s_end(prefix, bidir));
            }
        }
        p.heads = p.tokens.len();
        p.tail_from = vec![None; p.heads];
        for (&(prefix, branches, bidir), head) in reqs.iter().zip(head_idx) {
            let f = (prefix.len() / bs) * bs;
            if let Some(h) = head
                && bidir
                && prefix.len() > f
            {
                p.tail_from[h] = Some(p.tokens.len());
                p.kv_len[h] = prefix.len();
            }
            for b in branches.iter() {
                let mut t = Vec::with_capacity(prefix.len() + b.len());
                t.extend_from_slice(prefix);
                t.extend_from_slice(b);
                p.push(t, f, head, f / bs, s_end(prefix, bidir));
            }
            p.branch_counts.push(branches.len());
        }
        p
    }

    fn push(&mut self, tokens: Vec<u32>, start: usize, head: Option<usize>, shared: usize, s: i32) {
        self.kv_len.push(tokens.len());
        self.tokens.push(tokens);
        self.starts.push(start);
        self.head_of.push(head);
        self.shared_blocks.push(shared);
        self.state_end.push(s);
    }

    /// Whether any stream uses the prefix-bidirectional mask.
    pub(super) fn any_bidir(&self) -> bool {
        self.state_end.iter().any(|&s| s >= 0)
    }

    pub(super) fn streams(&self) -> usize {
        self.tokens.len()
    }

    pub(super) fn rows(&self, i: usize) -> usize {
        self.tokens[i].len() - self.starts[i]
    }

    pub(super) fn total_rows(&self) -> usize {
        (0..self.streams()).map(|i| self.rows(i)).sum()
    }

    /// KV blocks this pass allocates (shared head blocks counted once).
    pub(super) fn own_blocks(&self, bs: usize) -> usize {
        (0..self.streams())
            .map(|i| self.tokens[i].len().div_ceil(bs) - self.shared_blocks[i])
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_request_shares_whole_blocks_and_replays_the_tail() {
        let prefix: Vec<u32> = (0..37).collect();
        let branches = [vec![100, 101], vec![200]];
        let plan = ScorePlan::new(&[(&prefix, &branches, false)], 16);
        assert_eq!(plan.heads, 1);
        assert_eq!(plan.starts, vec![0, 32, 32]);
        assert_eq!(plan.head_of, vec![None, Some(0), Some(0)]);
        assert_eq!(plan.shared_blocks, vec![0, 2, 2]);
        assert_eq!(plan.tokens[0], (0..32).collect::<Vec<u32>>());
        assert_eq!((plan.rows(1), plan.rows(2)), (5 + 2, 5 + 1));
        assert_eq!(plan.total_rows(), 32 + 7 + 6);
        assert_eq!(plan.own_blocks(16), 2 + 1 + 1);
    }

    #[test]
    fn heads_come_first_and_branches_keep_request_order() {
        let (p1, p2): (Vec<u32>, Vec<u32>) = ((0..20).collect(), vec![7, 8, 9]);
        let (b1, b2) = ([vec![1], vec![2]], [vec![3]]);
        let plan = ScorePlan::new(&[(&p1, &b1, false), (&p2, &b2, false)], 16);
        // Request 2's prefix is under a block: no head, its branch starts at 0.
        assert_eq!(plan.heads, 1);
        assert_eq!(plan.head_of, vec![None, Some(0), Some(0), None]);
        assert_eq!(plan.starts, vec![0, 16, 16, 0]);
        assert_eq!(plan.branch_counts, vec![2, 1]);
        assert_eq!(plan.tokens[3], vec![7, 8, 9, 3]);
    }

    #[test]
    fn bidir_head_reads_its_first_branchs_tail() {
        let prefix: Vec<u32> = (0..37).collect();
        let (b1, b2) = ([vec![100, 101], vec![200]], [vec![300]]);
        let short = vec![5, 6];
        let plan = ScorePlan::new(&[(&prefix, &b1, true), (&short, &b2, false)], 16);
        assert_eq!(plan.heads, 1);
        assert_eq!(plan.state_end, vec![37, 37, 37, -1]);
        assert_eq!(plan.tail_from, vec![Some(1)]);
        assert_eq!(plan.kv_len, vec![37, 39, 38, 3]);
        assert!(plan.any_bidir());
    }
}
