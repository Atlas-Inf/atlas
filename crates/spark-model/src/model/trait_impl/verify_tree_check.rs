// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DFLASH_TREE_CHECK=1` (diagnostic, default off): hold every tree
//! verify (`verify_tree.rs`) to serial decode, byte for byte.
//!
//! Before the verify, every root-to-leaf path of the tree is decoded
//! serially on the live state (which is then put back), keeping each node's
//! logits and final-hidden row; the accepted path those rows' argmax walk
//! predicts is decoded once more and the state it leaves is kept: every GDN
//! layer's h and conv state, and every attention layer's K/V at the path's
//! positions (which are then cleared in the cache, so only the commit can
//! put them back). After the forward each row's logits and final hidden are
//! compared with its node's; after the commit, when the scheduler accepted
//! the predicted path, the live GDN state and the path's K/V are compared
//! with the kept ones. `TREE_CHECK` lines report every difference, and a
//! summary every 16 checked steps.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use super::exact_verify_check::{argmax, bf16s};
use crate::speculative::draft_tree::DraftTree;
use crate::traits::SequenceState;

const BF16: usize = 2;

static STEPS: AtomicU64 = AtomicU64::new(0);
static ROW_DIFFS: AtomicU64 = AtomicU64::new(0);
static COMMITS: AtomicU64 = AtomicU64::new(0);
static STATE_DIFFS: AtomicU64 = AtomicU64::new(0);
static PATH_MISSES: AtomicU64 = AtomicU64::new(0);

pub(super) fn active() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_DFLASH_TREE_CHECK").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "ATLAS_DFLASH_TREE_CHECK armed: every tree verify is preceded by serial \
                 decode of its paths (diagnostic, slow)"
            );
        }
        on
    })
}

/// What the serial pass kept for the comparisons.
pub(super) struct TreeSerial {
    logits: Vec<Vec<u8>>,
    hidden: Vec<Vec<u8>>,
    /// The path the serial rows' argmax walk accepts.
    path: Vec<usize>,
    /// After that path: (h, conv) per GDN layer, then K and V rows per
    /// attention layer and path position.
    gdn: Vec<(Vec<u8>, Vec<u8>)>,
    kv: Vec<(Vec<u8>, Vec<u8>)>,
}

impl TransformerModel {
    fn read(&self, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; bytes];
        self.gpu
            .copy_d2h_on_stream(p, &mut v, self.gpu.default_stream())?;
        Ok(v)
    }

    fn gdn_bytes(&self) -> (usize, usize) {
        (
            self.ssm_pool.h_stored_bytes,
            self.config.ssm_conv_state_bytes(),
        )
    }

    fn gdn_snapshot(&self, seq: &SequenceState) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let (hb, cb) = self.gdn_bytes();
        self.serial_ssm_states(seq)
            .into_iter()
            .map(|(h, c)| Ok((self.read(h, hb)?, self.read(c, cb)?)))
            .collect()
    }

    fn gdn_restore(&self, seq: &SequenceState, saved: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        for ((h, c), (hs, cs)) in self.serial_ssm_states(seq).into_iter().zip(saved) {
            self.gpu.copy_h2d(hs, h)?;
            self.gpu.copy_h2d(cs, c)?;
        }
        Ok(())
    }

    /// K/V rows of positions `pos` in every attention layer (read, and with
    /// `clear` zeroed afterwards).
    fn kv_rows(
        &self,
        seq: &SequenceState,
        pos: std::ops::Range<usize>,
        clear: bool,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let kv = self.kv_cache.lock();
        let bs = kv.block_size();
        let row = self.config.num_key_value_heads * self.config.head_dim * BF16;
        let mut out = Vec::new();
        for layer in 0..kv.num_layers() {
            for p in pos.clone() {
                let slot = seq.block_table[p / bs] as usize * bs + p % bs;
                let (k, v) = (
                    kv.k_pool_ptr(layer).offset(slot * row),
                    kv.v_pool_ptr(layer).offset(slot * row),
                );
                out.push((self.read(k, row)?, self.read(v, row)?));
                if clear {
                    let s = self.gpu.default_stream();
                    self.gpu.memset_zero_async(k, row, s)?;
                    self.gpu.memset_zero_async(v, row, s)?;
                }
            }
        }
        Ok(out)
    }

    /// Decode `tokens` serially on `seq`, returning each step's logits and
    /// final-hidden rows; the GDN state and tokens are put back, the K/V of
    /// the decoded positions is left.
    fn serial_path(
        &self,
        seq: &mut SequenceState,
        tokens: &[u32],
        keep: bool,
    ) -> Result<(Vec<Vec<u8>>, Vec<Vec<u8>>, Option<Vec<(Vec<u8>, Vec<u8>)>>)> {
        let stream = self.gpu.default_stream();
        let (vocab, h) = (self.config.vocab_size, self.config.hidden_size);
        let saved = self.gdn_snapshot(seq)?;
        let (tokens_len, seq_len) = (seq.tokens.len(), seq.seq_len);
        let (mut logits, mut hidden) = (Vec::new(), Vec::new());
        for &t in tokens {
            let out = self.decode_dispatch(t, seq, stream)?;
            logits.push(self.read(out, vocab * BF16)?);
            hidden.push(self.read(self.buffers.norm_output(), h * BF16)?);
        }
        let after = if keep {
            Some(self.gdn_snapshot(seq)?)
        } else {
            None
        };
        seq.tokens.truncate(tokens_len);
        seq.seq_len = seq_len;
        self.gdn_restore(seq, &saved)?;
        for (i, l) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::FullAttention {
                l.rollback_aux_verify(
                    seq.layer_states[i].as_mut(),
                    0,
                    tokens.len(),
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }
        self.gpu.synchronize(stream)?;
        Ok((logits, hidden, after))
    }

    /// Before a tree verify: every node's serial rows, and the state after
    /// the path their argmax walk accepts.
    pub(super) fn tree_check_serial(
        &self,
        tree: &DraftTree,
        seq: &mut SequenceState,
    ) -> Result<TreeSerial> {
        let k = tree.rows();
        let (mut logits, mut hidden) = (vec![Vec::new(); k], vec![Vec::new(); k]);
        let leaves = (0..k).filter(|&r| !tree.parent.contains(&(r as i32)));
        for leaf in leaves {
            let path = tree.path_to(leaf);
            let toks: Vec<u32> = path.iter().map(|&r| tree.tokens[r]).collect();
            let (l, h, _) = self.serial_path(seq, &toks, false)?;
            for (i, &r) in path.iter().enumerate() {
                logits[r] = l[i].clone();
                hidden[r] = h[i].clone();
            }
        }
        let picks: Vec<u32> = logits.iter().map(|l| argmax(&bf16s(l)) as u32).collect();
        let path = tree.walk(&picks);
        let toks: Vec<u32> = path.iter().map(|&r| tree.tokens[r]).collect();
        let (_, _, gdn) = self.serial_path(seq, &toks, true)?;
        let kv = self.kv_rows(seq, seq.seq_len..seq.seq_len + path.len(), true)?;
        Ok(TreeSerial {
            logits,
            hidden,
            path,
            gdn: gdn.unwrap_or_default(),
            kv,
        })
    }

    /// After the forward: each row's logits and final hidden vs its node's.
    pub(super) fn tree_check_rows(&self, serial: &TreeSerial, pos: usize) -> Result<()> {
        let (vocab, h) = (self.config.vocab_size, self.config.hidden_size);
        let k = serial.logits.len();
        let vl = self.read(self.buffers.logits(), k * vocab * BF16)?;
        let vh = self.read(self.buffers.norm_output(), k * h * BF16)?;
        let mut bad = 0u64;
        for r in 0..k {
            let lrow = &vl[r * vocab * BF16..(r + 1) * vocab * BF16];
            let hrow = &vh[r * h * BF16..(r + 1) * h * BF16];
            if lrow != &serial.logits[r][..] || hrow != &serial.hidden[r][..] {
                bad += 1;
                let (s, v) = (bf16s(&serial.logits[r]), bf16s(lrow));
                let diff = s
                    .iter()
                    .zip(&v)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                tracing::warn!(
                    "TREE_CHECK pos {pos} row {r}/{k}: {diff}/{vocab} logits differ \
                     (hidden equal: {}); argmax serial {} tree {}",
                    hrow == &serial.hidden[r][..],
                    argmax(&s),
                    argmax(&v)
                );
            }
        }
        ROW_DIFFS.fetch_add(bad, Ordering::Relaxed);
        STEPS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// After the commit of `path` on `seq` (whose prefix ended at `pre_len`).
    pub(super) fn tree_check_commit(
        &self,
        serial: &TreeSerial,
        seq: &SequenceState,
        path: &[usize],
        pre_len: usize,
    ) -> Result<()> {
        if path != serial.path {
            PATH_MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                "TREE_CHECK pos {pre_len}: committed path {path:?}, the serial picks' {:?} \
                 (pipeline picks differ: state not compared)",
                serial.path
            );
        } else {
            self.gpu.synchronize(self.secondary_stream)?;
            let gdn = self.gdn_snapshot(seq)?;
            let kv = self.kv_rows(seq, pre_len..pre_len + path.len(), false)?;
            let gdn_bad = gdn.iter().zip(&serial.gdn).filter(|(a, b)| a != b).count();
            let kv_bad = kv.iter().zip(&serial.kv).filter(|(a, b)| a != b).count();
            if gdn_bad + kv_bad > 0 {
                STATE_DIFFS.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    "TREE_CHECK pos {pre_len}: after committing {path:?}, {gdn_bad}/{} GDN \
                     layer states and {kv_bad}/{} KV rows differ from serial decode",
                    gdn.len(),
                    kv.len()
                );
            }
            COMMITS.fetch_add(1, Ordering::Relaxed);
        }
        let steps = STEPS.load(Ordering::Relaxed);
        if steps.is_multiple_of(16) {
            tracing::info!(
                "TREE_CHECK summary: {steps} tree verifies, {} rows differ; {} commits \
                 checked, {} with state differences; {} paths not checked (picks differ)",
                ROW_DIFFS.load(Ordering::Relaxed),
                COMMITS.load(Ordering::Relaxed),
                STATE_DIFFS.load(Ordering::Relaxed),
                PATH_MISSES.load(Ordering::Relaxed),
            );
        }
        Ok(())
    }
}
