// SPDX-License-Identifier: AGPL-3.0-only

//! Exact DFlash tree verify (`ATLAS_DFLASH_TREE`): the draft tree, its
//! best-first construction from the DFlash2 selector's lattice, and the
//! device descriptor a tree verify's layers read.
//!
//! The DFlash2 selector scores candidate `j` of draft position `d` after
//! predecessor token `t` as `unary[d][j] + <pred[t] * h[d], succ[cand[d][j]]>`
//! and walks the argmax chain. A tree expands each node's children from the
//! same scores with the node's own token as predecessor, so the one drafter
//! forward that proposes the chain serves any tree. Nodes are admitted
//! best-first by path probability (each step's softmax over its parent's
//! candidates), which the offline replay of 15k anchor positions measured
//! at +11% tokens a round over the 7-draft chain at 12 rows (code), +12%
//! (math), +22% (prose) — `best:11` in `scripts/dev/tree_eval.py`'s terms.
//!
//! Rows are laid out in depth-first pre-order with row 0 the root (the last
//! committed token, as in the chain): every parent precedes its children and
//! a node's first child is its most probable, so a pure chain is rows
//! `0..k` with parents `t - 1`. The verify gives row `r` position
//! `prefix + depth(r)` and the state of its parent, so its logits are serial
//! decode's for the path ending at `r`; the accepted path is the walk that
//! follows each row's pick into the child carrying it.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use spark_runtime::gpu::DevicePtr;

/// Widest tree verify (rows, root included): the exact single-sequence
/// verify's cap (`model::exact_verify::DFLASH_MAX_ROWS`).
pub const MAX_ROWS: usize = 16;

/// `ATLAS_DFLASH_TREE=<rows>`: the tree verify's row budget (2..=16), 0 =
/// off. Also sizes the DFlash verify pools (`dspark_pool::dflash_verify_rows`)
/// past the chain's γ rows.
pub fn rows_lever() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let rows = parse_rows(std::env::var("ATLAS_DFLASH_TREE").ok().as_deref());
        if rows > 0 {
            tracing::info!("ATLAS_DFLASH_TREE: exact tree verify at C1, up to {rows} rows");
        }
        rows
    })
}

/// [`rows_lever`]'s parse: a width in 2..=MAX_ROWS, anything else off.
pub fn parse_rows(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        None | Some("") | Some("0") => 0,
        Some(s) => match s.parse::<usize>() {
            Ok(n) if (2..=MAX_ROWS).contains(&n) => n,
            _ => {
                tracing::warn!("ATLAS_DFLASH_TREE={s:?} ignored: expects 2..={MAX_ROWS} rows");
                0
            }
        },
    }
}

/// One draft tree in depth-first pre-order. Row 0 is the root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftTree {
    pub tokens: Vec<u32>,
    /// Parent row, -1 for the root.
    pub parent: Vec<i32>,
    pub depth: Vec<u8>,
}

impl DraftTree {
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    pub fn max_depth(&self) -> usize {
        self.depth.iter().copied().max().unwrap_or(0) as usize
    }

    /// A chain: rows `0..=drafts.len()`, each the previous row's child.
    pub fn chain(root: u32, drafts: &[u32]) -> Self {
        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(root);
        tokens.extend_from_slice(drafts);
        Self {
            parent: (0..tokens.len() as i32).map(|t| t - 1).collect(),
            depth: (0..tokens.len()).map(|t| t as u8).collect(),
            tokens,
        }
    }

    /// Whether every row's parent is the row before it.
    pub fn is_chain(&self) -> bool {
        self.parent
            .iter()
            .enumerate()
            .all(|(t, &p)| p == t as i32 - 1)
    }

    /// The rows on the path from the root to `row`, root first.
    pub fn path_to(&self, row: usize) -> Vec<usize> {
        let mut path = vec![row];
        let mut r = row;
        while self.parent[r] >= 0 {
            r = self.parent[r] as usize;
            path.push(r);
        }
        path.reverse();
        path
    }

    /// The path the picks accept: from the root, step into the child whose
    /// token is the current row's pick while one exists. `pick[r]` is row
    /// `r`'s verified token. Root first; the last row's pick is the bonus.
    pub fn walk(&self, pick: &[u32]) -> Vec<usize> {
        let mut path = vec![0usize];
        let mut cur = 0usize;
        loop {
            let want = pick[cur];
            let next = (cur + 1..self.rows())
                .find(|&c| self.parent[c] == cur as i32 && self.tokens[c] == want);
            match next {
                Some(c) => {
                    path.push(c);
                    cur = c;
                }
                None => return path,
            }
        }
    }

    /// Structural invariants: a root at row 0, every other row's parent an
    /// earlier row one level up, subtrees contiguous (pre-order), sibling
    /// tokens distinct.
    pub fn validate(&self) -> anyhow::Result<()> {
        let n = self.rows();
        anyhow::ensure!(
            n >= 1 && self.parent.len() == n && self.depth.len() == n,
            "draft tree: ragged rows"
        );
        anyhow::ensure!(
            self.parent[0] == -1 && self.depth[0] == 0,
            "draft tree: row 0 is not the root"
        );
        for r in 1..n {
            let p = self.parent[r];
            anyhow::ensure!(
                p >= 0 && (p as usize) < r,
                "draft tree: row {r} parent {p} is not an earlier row"
            );
            anyhow::ensure!(
                self.depth[r] == self.depth[p as usize] + 1,
                "draft tree: row {r} depth"
            );
            // Pre-order: the row before r is r's parent or a descendant of
            // r's parent (so p's subtree is contiguous).
            anyhow::ensure!(
                self.path_to(r - 1).contains(&(p as usize)),
                "draft tree: row {r} breaks pre-order"
            );
            for s in 1..r {
                anyhow::ensure!(
                    !(self.parent[s] == p && self.tokens[s] == self.tokens[r]),
                    "draft tree: rows {s} and {r} are siblings with one token"
                );
            }
        }
        Ok(())
    }
}

/// The selector's candidates for each draft position of one propose: row
/// `d` is draft position `d + 1` (the selector's mask row `d + 1`).
#[derive(Debug, Clone)]
pub struct Lattice {
    /// The token the block was drafted after (the root).
    pub anchor: u32,
    /// Draft positions `1..=ban_depth` may not end the turn (min_tokens).
    pub ban_depth: u32,
    pub end_ids: [u32; 4],
    pub ids: Vec<Vec<u32>>,
    pub unary: Vec<Vec<f32>>,
}

impl Lattice {
    pub fn depth(&self) -> usize {
        self.ids.len()
    }
}

#[derive(Debug)]
struct Cand {
    prob: f64,
    parent: usize,
    depth: usize,
    token: u32,
}

impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    /// Most probable first; ties to the shallower, then the lower token.
    fn cmp(&self, o: &Self) -> Ordering {
        self.prob
            .total_cmp(&o.prob)
            .then_with(|| o.depth.cmp(&self.depth))
            .then_with(|| o.token.cmp(&self.token))
    }
}

/// Best-first draft tree of at most `rows` rows (root included, `<=
/// MAX_ROWS`) over `lat`. `score(d, pred)` returns the selector's score of
/// every candidate of draft position `d` after token `pred` (unary
/// included, `lat.ids[d]` order). A node's probability is its parent's
/// times the softmax of its score among its siblings; banned and invalid
/// candidates (`id >= vocab` or a non-finite unary) never enter.
pub fn best_first(
    lat: &Lattice,
    rows: usize,
    vocab: u32,
    score: &mut dyn FnMut(usize, u32) -> Vec<f32>,
) -> DraftTree {
    let rows = rows.clamp(1, MAX_ROWS);
    // Admitted nodes: (parent node, depth, token, prob); node 0 the root.
    let mut nodes: Vec<(usize, usize, u32, f64)> = vec![(usize::MAX, 0, lat.anchor, 1.0)];
    let mut heap = BinaryHeap::new();
    let push_children = |heap: &mut BinaryHeap<Cand>,
                         score: &mut dyn FnMut(usize, u32) -> Vec<f32>,
                         node: usize,
                         nodes: &[(usize, usize, u32, f64)]| {
        let (_, depth, token, prob) = nodes[node];
        if depth >= lat.depth() {
            return;
        }
        let d = depth;
        let s = score(d, token);
        let banned = |id: u32| (d as u32) < lat.ban_depth && lat.end_ids.contains(&id);
        let ok: Vec<bool> = lat.ids[d]
            .iter()
            .zip(&lat.unary[d])
            .zip(&s)
            .map(|((&id, &u), &v)| id < vocab && u.is_finite() && v.is_finite() && !banned(id))
            .collect();
        let m = s
            .iter()
            .zip(&ok)
            .filter(|(_, ok)| **ok)
            .map(|(v, _)| *v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            return;
        }
        let z: f64 = s
            .iter()
            .zip(&ok)
            .filter(|(_, ok)| **ok)
            .map(|(v, _)| (*v as f64 - m).exp())
            .sum();
        // A token the selector lists more than once is one child carrying
        // the summed mass: siblings never share a token (`validate`).
        let mut merged: Vec<(u32, f64)> = Vec::new();
        for (j, &id) in lat.ids[d].iter().enumerate() {
            if ok[j] {
                let p = (s[j] as f64 - m).exp() / z;
                match merged.iter_mut().find(|(t, _)| *t == id) {
                    Some((_, q)) => *q += p,
                    None => merged.push((id, p)),
                }
            }
        }
        for (id, p) in merged {
            heap.push(Cand {
                prob: prob * p,
                parent: node,
                depth: depth + 1,
                token: id,
            });
        }
    };
    push_children(&mut heap, score, 0, &nodes);
    while nodes.len() < rows {
        let Some(c) = heap.pop() else { break };
        nodes.push((c.parent, c.depth, c.token, c.prob));
        let node = nodes.len() - 1;
        push_children(&mut heap, score, node, &nodes);
    }
    // Depth-first pre-order, children most probable first.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (n, &(p, ..)) in nodes.iter().enumerate().skip(1) {
        children[p].push(n);
    }
    for c in &mut children {
        c.sort_by(|&a, &b| nodes[b].3.total_cmp(&nodes[a].3));
    }
    let mut tree = DraftTree {
        tokens: Vec::with_capacity(nodes.len()),
        parent: Vec::with_capacity(nodes.len()),
        depth: Vec::with_capacity(nodes.len()),
    };
    let mut stack = vec![(0usize, -1i32)];
    while let Some((n, prow)) = stack.pop() {
        let row = tree.tokens.len() as i32;
        tree.tokens.push(nodes[n].2);
        tree.parent.push(prow);
        tree.depth.push(nodes[n].1 as u8);
        for &c in children[n].iter().rev() {
            stack.push((c, row));
        }
    }
    tree
}

/// What a tree verify's layers read, in device memory at fixed addresses
/// (uploaded before a captured graph replays): the rows' parents (`i32`) for
/// the GDN layers, and each attention layer's KV slot copies
/// (`kv_slot_copy_bf16`: `kv_max` int64 pairs, `kv_count` live).
#[derive(Clone, Copy, Debug)]
pub struct TreeDev {
    pub parents: DevicePtr,
    pub kv_entries: DevicePtr,
    pub kv_count: DevicePtr,
    pub kv_max: u32,
}

thread_local! {
    static CURRENT: Cell<Option<TreeDev>> = const { Cell::new(None) };
}

/// The tree verify forward in flight on this thread (`enter`), if any.
pub fn current() -> Option<TreeDev> {
    CURRENT.with(|c| c.get())
}

/// Marks a tree verify forward in flight until dropped.
pub struct TreeScope(());

impl Drop for TreeScope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(None));
    }
}

/// Enter a tree verify forward: layers run their tree arms against `dev`
/// until the returned scope drops.
pub fn enter(dev: TreeDev) -> TreeScope {
    CURRENT.with(|c| c.set(Some(dev)));
    TreeScope(())
}

#[cfg(test)]
#[path = "draft_tree_tests.rs"]
mod tests;
