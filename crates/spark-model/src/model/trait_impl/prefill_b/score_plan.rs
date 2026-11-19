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
//!
//! A request on a knowledge pack of P tokens (`score_pack.rs`) is the same
//! geometry shifted by P: the pack's K/V already fill positions `[0, P)`, so
//! every position above (F, S, starts, KV extents) is absolute and every
//! stream's tokens begin with the pack's. Its head is `[P, F)` with
//! `F = floor((P + L) / block) * block`, omitted when F <= P; a stream that
//! reads the pack directly (its head, or a branch without one) shares the
//! pack's `P / block` whole blocks and gets its own copy of the pack's partial
//! last block, and a branch with a head shares the head's F / block blocks
//! (the pack's among them) as before.

/// One request: a shared prefix, its branches, whether the prefix is a
/// bidirectional state block (`prefix_bidir`), and the knowledge pack it
/// continues, if any (index into the pass's packs, and the pack's tokens).
pub(in crate::model) type ScoreRequest<'a> =
    (&'a [u32], &'a [Vec<u32>], bool, Option<(usize, &'a [u32])>);

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
    /// Per stream: whole blocks shared from its head, or from its pack when
    /// it has no head (0 for a head or headless branch without a pack).
    pub shared_blocks: Vec<usize>,
    /// Per stream: the pass's pack its request continues, if any.
    pub pack_of: Vec<Option<usize>>,
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
    pub(super) fn new<'a>(reqs: &[ScoreRequest<'a>], bs: usize) -> Self {
        let mut p = Self {
            tokens: Vec::new(),
            starts: Vec::new(),
            heads: 0,
            head_of: Vec::new(),
            shared_blocks: Vec::new(),
            pack_of: Vec::new(),
            branch_counts: Vec::with_capacity(reqs.len()),
            state_end: Vec::new(),
            kv_len: Vec::new(),
            tail_from: Vec::new(),
        };
        // Per request: (pack tokens, its index, state end S, head end F).
        fn geometry<'a>(
            r: &ScoreRequest<'a>,
            bs: usize,
        ) -> (&'a [u32], Option<usize>, usize, usize) {
            let (k, base) = r.3.map_or((None, &[][..]), |(k, t)| (Some(k), t));
            let end = base.len() + r.0.len();
            (base, k, end, (end / bs) * bs)
        }
        let geo = |r: &ScoreRequest<'a>| geometry(r, bs);
        let joined = |base: &[u32], a: &[u32], b: &[u32]| {
            let mut t = Vec::with_capacity(base.len() + a.len() + b.len());
            t.extend_from_slice(base);
            t.extend_from_slice(a);
            t.extend_from_slice(b);
            t
        };
        let s_end = |end: usize, bidir: bool| if bidir { end as i32 } else { -1 };
        let mut head_idx = Vec::with_capacity(reqs.len());
        for r in reqs {
            let (base, k, end, f) = geo(r);
            let (prefix, bidir) = (r.0, r.2);
            let p0 = base.len();
            head_idx.push((f > p0).then_some(p.tokens.len()));
            if f > p0 {
                let t = joined(base, &prefix[..f - p0], &[]);
                p.push(t, p0, None, p0 / bs, k, s_end(end, bidir));
            }
        }
        p.heads = p.tokens.len();
        p.tail_from = vec![None; p.heads];
        for (r, head) in reqs.iter().zip(head_idx) {
            let (base, k, end, f) = geo(r);
            let (prefix, branches, bidir) = (r.0, r.1, r.2);
            if let Some(h) = head
                && bidir
                && end > f
            {
                p.tail_from[h] = Some(p.tokens.len());
                p.kv_len[h] = end;
            }
            let (start, shared) = match head {
                Some(_) => (f, f / bs),
                None => (base.len(), base.len() / bs),
            };
            for b in branches.iter() {
                p.push(
                    joined(base, prefix, b),
                    start,
                    head,
                    shared,
                    k,
                    s_end(end, bidir),
                );
            }
            p.branch_counts.push(branches.len());
        }
        p
    }

    /// A pack build (`score_pack.rs`): one stream over all of `tokens` from
    /// position 0, no branches; under `bidir` every row reads every key.
    pub(super) fn pack(tokens: &[u32], bidir: bool) -> Self {
        let p = tokens.len();
        Self {
            tokens: vec![tokens.to_vec()],
            starts: vec![0],
            heads: 1,
            head_of: vec![None],
            shared_blocks: vec![0],
            pack_of: vec![None],
            branch_counts: Vec::new(),
            state_end: vec![if bidir { p as i32 } else { -1 }],
            kv_len: vec![p],
            tail_from: vec![None],
        }
    }

    fn push(
        &mut self,
        tokens: Vec<u32>,
        start: usize,
        head: Option<usize>,
        shared: usize,
        pack: Option<usize>,
        s: i32,
    ) {
        self.kv_len.push(tokens.len());
        self.tokens.push(tokens);
        self.starts.push(start);
        self.head_of.push(head);
        self.shared_blocks.push(shared);
        self.pack_of.push(pack);
        self.state_end.push(s);
    }

    /// Whether stream `i` reads its pack directly (its K/V blocks, entry SSM
    /// state and conv window), rather than through its head.
    pub(super) fn reads_pack(&self, i: usize) -> Option<usize> {
        self.pack_of[i].filter(|_| self.head_of[i].is_none())
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
        let plan = ScorePlan::new(&[(&prefix, &branches, false, None)], 16);
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
        let plan = ScorePlan::new(&[(&p1, &b1, false, None), (&p2, &b2, false, None)], 16);
        // Request 2's prefix is under a block: no head, its branch starts at 0.
        assert_eq!(plan.heads, 1);
        assert_eq!(plan.head_of, vec![None, Some(0), Some(0), None]);
        assert_eq!(plan.starts, vec![0, 16, 16, 0]);
        assert_eq!(plan.branch_counts, vec![2, 1]);
        assert_eq!(plan.tokens[3], vec![7, 8, 9, 3]);
    }

    #[test]
    fn bidir_head_reads_its_first_branch_tail() {
        let prefix: Vec<u32> = (0..37).collect();
        let (b1, b2) = ([vec![100, 101], vec![200]], [vec![300]]);
        let short = vec![5, 6];
        let plan = ScorePlan::new(
            &[(&prefix, &b1, true, None), (&short, &b2, false, None)],
            16,
        );
        assert_eq!(plan.heads, 1);
        assert_eq!(plan.state_end, vec![37, 37, 37, -1]);
        assert_eq!(plan.tail_from, vec![Some(1)]);
        assert_eq!(plan.kv_len, vec![37, 39, 38, 3]);
        assert!(plan.any_bidir());
    }

    #[test]
    fn a_pack_shifts_every_position_and_its_streams_share_its_blocks() {
        // Pack of 40 tokens (2 whole blocks + 8), state of 30 (S = 70), so
        // F = 64: the head is [40, 64) and the branches replay [64, 70).
        let pack: Vec<u32> = (1000..1040).collect();
        let prefix: Vec<u32> = (0..30).collect();
        let branches = [vec![100, 101], vec![200]];
        let plan = ScorePlan::new(&[(&prefix, &branches, true, Some((0, &pack[..])))], 16);
        assert_eq!(plan.heads, 1);
        assert_eq!(plan.starts, vec![40, 64, 64]);
        assert_eq!(plan.shared_blocks, vec![2, 4, 4]);
        assert_eq!(plan.pack_of, vec![Some(0); 3]);
        assert_eq!((plan.reads_pack(0), plan.reads_pack(1)), (Some(0), None));
        assert_eq!(plan.state_end, vec![70, 70, 70]);
        assert_eq!(plan.tokens[0][..40], pack[..]);
        assert_eq!(plan.tokens[0][40..], prefix[..24]);
        assert_eq!(plan.tokens[1].len(), 72);
        assert_eq!((plan.rows(0), plan.rows(1), plan.rows(2)), (24, 8, 7));
        // The head's tail block [64, 70) is read from its first branch.
        assert_eq!(
            (plan.tail_from.clone(), plan.kv_len[0]),
            (vec![Some(1)], 70)
        );
        // Own blocks: head 4 - 2 (its copy of the pack's tail block counted),
        // each branch 5 - 4.
        assert_eq!(plan.own_blocks(16), 2 + 1 + 1);
    }

    #[test]
    fn a_short_state_on_a_pack_has_no_head_and_its_branches_read_the_pack() {
        // P = 40, L = 6: S = 46 < 48, no whole block past the pack.
        let pack: Vec<u32> = (1000..1040).collect();
        let prefix: Vec<u32> = (0..6).collect();
        let branches = [vec![7], vec![8]];
        let plain: Vec<u32> = (0..20).collect();
        let reqs = [
            (&prefix[..], &branches[..], true, Some((0, &pack[..]))),
            (&plain[..], &branches[..], false, None),
        ];
        let plan = ScorePlan::new(&reqs, 16);
        // Only the plain request has a head.
        assert_eq!(plan.heads, 1);
        assert_eq!(plan.head_of, vec![None, None, None, Some(0), Some(0)]);
        assert_eq!(plan.starts[1..3], [40, 40]);
        assert_eq!(plan.shared_blocks[1..3], [2, 2]);
        assert_eq!(plan.reads_pack(1), Some(0));
        assert_eq!(plan.reads_pack(3), None);
        assert_eq!(plan.state_end, vec![-1, 46, 46, -1, -1]);
        assert_eq!(plan.rows(1), 7);
    }

    #[test]
    fn a_pack_build_is_one_whole_stream() {
        let tokens: Vec<u32> = (0..40).collect();
        let plan = ScorePlan::pack(&tokens, true);
        assert_eq!((plan.heads, plan.streams(), plan.total_rows()), (1, 1, 40));
        assert_eq!(
            (plan.state_end.clone(), plan.kv_len.clone()),
            (vec![40], vec![40])
        );
        assert_eq!(plan.own_blocks(16), 3);
        assert!(plan.branch_counts.is_empty());
        assert_eq!(ScorePlan::pack(&tokens, false).state_end, vec![-1]);
    }

    #[test]
    fn a_block_aligned_pack_needs_no_tail_copy() {
        let pack: Vec<u32> = (1000..1032).collect();
        let prefix: Vec<u32> = (0..20).collect();
        let branches = [vec![7]];
        let plan = ScorePlan::new(&[(&prefix, &branches, true, Some((0, &pack[..])))], 16);
        assert_eq!(plan.starts, vec![32, 48]);
        assert_eq!(plan.shared_blocks, vec![2, 3]);
        assert_eq!(plan.own_blocks(16), 1 + 1);
    }
}
