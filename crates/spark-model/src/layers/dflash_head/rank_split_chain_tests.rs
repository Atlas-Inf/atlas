// SPDX-License-Identifier: AGPL-3.0-only

//! The rank split over the four-rank chain.

use super::*;

/// One rank of a four-rank chain on the mock backends: a share swap gathers
/// every rank's share from the others' devices (each rank allocates the same
/// buffers in the same order, so a share lies at one address on all), a
/// broadcast lands the head's rows its call left in `head_rows`.
struct Chain<'a> {
    gpus: [&'a MockGpuBackend; 4],
    rank: usize,
    head_rows: &'a RefCell<Vec<u8>>,
    sizes: RefCell<Vec<usize>>,
}

// SAFETY: test-only, used on one thread.
unsafe impl Send for Chain<'_> {}
unsafe impl Sync for Chain<'_> {}

impl CommBackend for Chain<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        4
    }
    fn gather_async(&self, base: u64, stride: i64, bytes: usize, _: u64) -> Result<bool> {
        self.sizes.borrow_mut().push(bytes);
        for r in (0..4).filter(|&r| r != self.rank) {
            let at = DevicePtr(base + r as u64 * stride as u64);
            let mut part = vec![0u8; bytes];
            self.gpus[r].copy_d2h(at, &mut part)?;
            self.gpus[self.rank].copy_h2d(&part, at)?;
        }
        Ok(true)
    }
    fn supports_gather_async(&self, _: usize) -> bool {
        true
    }
    fn broadcast_async(&self, ptr: u64, bytes: usize, root: usize, _: u64) -> Result<bool> {
        assert_eq!(root, 0, "the head's rows go out");
        self.sizes.borrow_mut().push(bytes);
        if self.rank == 0 {
            let mut rows = vec![0u8; bytes];
            self.gpus[0].copy_d2h(DevicePtr(ptr), &mut rows)?;
            *self.head_rows.borrow_mut() = rows;
        } else {
            self.gpus[self.rank].copy_h2d(&self.head_rows.borrow(), DevicePtr(ptr))?;
        }
        Ok(true)
    }
    fn exchange_async(&self, _: u64, _: u64, _: usize, _: bool, _: u64) -> Result<bool> {
        bail!("the chain gathers and broadcasts, it does not exchange")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected reduce-scatter")
    }
    fn barrier(&self) -> Result<()> {
        bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected receive")
    }
}

/// `[gamma, n]` rows from every rank's share of `ranks`, each `[gamma, its
/// rows]` at the start of its part.
fn joined_shares(gamma: usize, n: usize, parts: &[Vec<u8>]) -> Vec<u8> {
    let ranks = parts.len();
    (0..gamma)
        .flat_map(|r| {
            parts.iter().enumerate().flat_map(move |(k, p)| {
                let row = share(n, k, ranks).1 * 2;
                p[r * row..(r + 1) * row].to_vec()
            })
        })
        .collect()
}

#[test]
fn the_chains_four_shares_cover_every_row_once_and_start_on_a_cta() {
    for n in [160, 64, 98, 154_856, 12_288, 4096] {
        let shares: Vec<_> = (0..4).map(|r| share(n, r, 4)).collect();
        assert_eq!(shares[0].0, 0);
        for w in shares.windows(2) {
            assert_eq!(w[0].0 + w[0].1, w[1].0);
            assert!(w[1].0 % 16 == 0 && w[0].1 > 0, "{n}");
        }
        assert_eq!(shares[3].0 + shares[3].1, n);
    }
    // The GLM-5.3 drafter's head at four ranks: CTA cuts under each quarter.
    assert_eq!(share(154_856, 1, 4), (38_704, 38_720));
    assert_eq!(share(154_856, 3, 4), (116_128, 38_728));
    assert_eq!(widest(154_856, 4), 38_728);
}

#[test]
fn the_chain_gathers_shares_where_the_unsplit_launch_writes_whole_rows() {
    let g = Geometry {
        ranks: 4,
        ..geometry(ALL)
    };
    assert!(g.validate().is_ok());
    let ranks: Vec<Rank> = (0..4).map(|_| Rank::new(g)).collect();
    let gpus = [&ranks[0].gpu, &ranks[1].gpu, &ranks[2].gpu, &ranks[3].gpu];
    let head_rows = RefCell::new(Vec::new());
    let comms: Vec<Chain> = (0..4)
        .map(|rank| Chain {
            gpus,
            rank,
            head_rows: &head_rows,
            sizes: RefCell::default(),
        })
        .collect();
    let run = |swap: Swap| {
        for (rank, (r, c)) in ranks.iter().zip(&comms).enumerate() {
            r.split
                .swap(swap, rank, &r.gpu, c, &Frame::serial(&r.scratch), 0)
                .unwrap();
        }
    };
    // Every rank's share of `swap`, distinct bytes a rank.
    let fill = |swap: Swap, tag: u8| -> Vec<Vec<u8>> {
        let bytes = g.bytes(swap);
        (0..4)
            .map(|k| {
                let r = &ranks[k];
                r.fill(r.split.own(swap, k), bytes, tag + k as u8);
                r.read(r.split.own(swap, k), bytes)
            })
            .collect()
    };

    // The head's MLP input reaches every worker's `norm_buf`.
    let hidden = g.gamma * g.hidden * 2;
    ranks[0].fill(ranks[0].scratch.norm_buf, hidden, 0x11);
    let x = ranks[0].read(ranks[0].scratch.norm_buf, hidden);
    run(Swap::Input(0));
    for r in &ranks {
        assert_eq!(r.read(r.scratch.norm_buf, hidden), x);
    }

    // Every rank holds the whole gated activation, the shares in rank order.
    let parts = fill(Swap::Activation(0), 0x20);
    run(Swap::Activation(0));
    let whole = joined_shares(g.gamma, g.inter, &parts);
    for r in &ranks {
        assert_eq!(r.read(r.scratch.mlp_intermediate, whole.len()), whole);
    }

    // The head holds the whole down projection and the whole logits.
    let parts = fill(Swap::Output(0), 0x30);
    run(Swap::Output(0));
    assert_eq!(
        ranks[0].read(ranks[0].scratch.stream_acc, hidden),
        joined_shares(g.gamma, g.hidden, &parts)
    );
    let parts = fill(Swap::Logits, 0x50);
    run(Swap::Logits);
    let vocab = g.gamma * g.vocab * 2;
    assert_eq!(
        ranks[0].read(ranks[0].scratch.logits, vocab),
        joined_shares(g.gamma, g.vocab, &parts)
    );

    // Same sizes, same order, on every rank; a stopped propose drains the
    // rest at the planned sizes.
    for c in &comms[1..] {
        assert_eq!(*c.sizes.borrow(), *comms[0].sizes.borrow());
    }
    let rank = &ranks[1];
    let chain = &comms[1];
    chain.sizes.borrow_mut().clear();
    rank.split.begin(g.gamma, CtxRows::default()).unwrap();
    rank.split.finish(chain, 0).unwrap();
    let planned: Vec<usize> = g.swaps().iter().map(|&s| g.bytes(s)).collect();
    assert_eq!(*chain.sizes.borrow(), planned);
}

#[test]
fn the_attention_split_names_its_part_and_plans_its_swaps() {
    use Piece::*;
    assert_eq!(Parts::parse(Some("mlp,head,attn")).unwrap(), Some(ATTN));
    // `1` keeps its meaning: the MLP and the head.
    assert_eq!(Parts::parse(Some("1")).unwrap(), Some(ALL));
    assert_eq!(Parts::word(Some(ATTN)), 3 | 1 << 5);
    for ranks in [2, 4] {
        let g = Geometry {
            ranks,
            ..geometry(ATTN)
        };
        assert!(g.validate().is_ok());
        let (head, worker) = (g.steps(0), g.steps(ranks - 1));
        assert_eq!(swaps_of(&head), swaps_of(&worker));
        assert_eq!(g.swaps().len(), 7 * g.layers + 2);
        let pieces = |steps: Vec<Step>| -> Vec<Piece> {
            steps
                .into_iter()
                .filter_map(|s| match s {
                    Step::Run(p) => Some(p),
                    Step::Swap(_) => None,
                })
                .collect()
        };
        assert_eq!(
            pieces(worker),
            [
                Qkv(0),
                OProj(0),
                GateUp(0),
                Down(0),
                Qkv(1),
                OProj(1),
                GateUp(1),
                Down(1),
                Vocab
            ]
        );
        assert_eq!(
            pieces(head)[..8],
            [
                AttnPre(0),
                Qkv(0),
                AttnCore(0),
                OProj(0),
                ProjectRest(0),
                GateUp(0),
                Down(0),
                Residual(0)
            ]
        );
    }
    let g = geometry(ATTN);
    // q's 64-row share plus k's and v's 32 a rank, at gamma rows.
    assert_eq!(g.bytes(Swap::Qkv(0)), 3 * (64 + 32 + 32) * 2);
    assert_eq!(g.bytes(Swap::AttnIn(0)), 3 * 64 * 2);
    assert_eq!(g.bytes(Swap::AttnOut(0)), 3 * 128 * 2);
    assert_eq!(g.bytes(Swap::OProj(0)), 3 * 32 * 2);
}

#[test]
fn the_chain_joins_the_attention_shares_where_the_unsplit_projections_write() {
    let g = Geometry {
        ranks: 4,
        ..geometry(ATTN)
    };
    let ranks: Vec<Rank> = (0..4).map(|_| Rank::new(g)).collect();
    let gpus = [&ranks[0].gpu, &ranks[1].gpu, &ranks[2].gpu, &ranks[3].gpu];
    let head_rows = RefCell::new(Vec::new());
    let comms: Vec<Chain> = (0..4)
        .map(|rank| Chain {
            gpus,
            rank,
            head_rows: &head_rows,
            sizes: RefCell::default(),
        })
        .collect();
    let run = |swap: Swap| {
        for (rank, (r, c)) in ranks.iter().zip(&comms).enumerate() {
            r.split
                .swap(swap, rank, &r.gpu, c, &Frame::serial(&r.scratch), 0)
                .unwrap();
        }
    };
    for r in &ranks {
        r.split.begin(g.gamma, CtxRows::default()).unwrap();
    }
    // The normed input and the attention rows reach every worker.
    let hidden = g.gamma * g.hidden * 2;
    ranks[0].fill(ranks[0].scratch.norm_buf, hidden, 0x11);
    run(Swap::AttnIn(0));
    let x = ranks[0].read(ranks[0].scratch.norm_buf, hidden);
    for r in &ranks[1..] {
        assert_eq!(r.read(r.scratch.norm_buf, hidden), x);
    }
    let attn = g.gamma * g.q_dim * 2;
    ranks[0].fill(ranks[0].scratch.attn_out, attn, 0x12);
    run(Swap::AttnOut(0));
    let y = ranks[0].read(ranks[0].scratch.attn_out, attn);
    for r in &ranks[1..] {
        assert_eq!(r.read(r.scratch.attn_out, attn), y);
    }
    // Every rank's q, k and v shares, joined on the head into whole rows.
    let bytes = g.bytes(Swap::Qkv(0));
    let parts: Vec<Vec<u8>> = (0..4)
        .map(|k| {
            let r = &ranks[k];
            r.fill(r.split.own(Swap::Qkv(0), k), bytes, 0x40 + k as u8);
            r.read(r.split.own(Swap::Qkv(0), k), bytes)
        })
        .collect();
    run(Swap::Qkv(0));
    let (wq, wk) = (widest(g.q_dim, 4), widest(g.kv_dim, 4));
    for (at, dst, n) in [
        (0, ranks[0].scratch.q_buf, g.q_dim),
        (wq, ranks[0].scratch.k_buf, g.kv_dim),
        (wq + wk, ranks[0].scratch.v_buf, g.kv_dim),
    ] {
        let subs: Vec<Vec<u8>> = parts
            .iter()
            .map(|p| p[at * g.gamma * 2..].to_vec())
            .collect();
        assert_eq!(
            ranks[0].read(dst, g.gamma * n * 2),
            joined_shares(g.gamma, n, &subs),
            "part at {at}"
        );
    }
    // And o_proj's shares, into the rows o_proj writes.
    let out = g.bytes(Swap::OProj(0));
    let parts: Vec<Vec<u8>> = (0..4)
        .map(|k| {
            let r = &ranks[k];
            r.fill(r.split.own(Swap::OProj(0), k), out, 0x60 + k as u8);
            r.read(r.split.own(Swap::OProj(0), k), out)
        })
        .collect();
    run(Swap::OProj(0));
    assert_eq!(
        ranks[0].read(ranks[0].scratch.stream_acc, hidden),
        joined_shares(g.gamma, g.hidden, &parts)
    );
    for c in &comms[1..] {
        assert_eq!(*c.sizes.borrow(), *comms[0].sizes.borrow());
    }
}
