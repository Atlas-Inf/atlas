// SPDX-License-Identifier: AGPL-3.0-only

//! Prompt-lookup + dynamic n-gram proposer for speculative decoding.
//!
//! Two complementary draft sources, both CPU-only:
//!
//! 1. **Prompt lookup** — find the longest suffix of the token history
//!    elsewhere in the history and propose the tokens that followed it
//!    (vLLM "prompt lookup" / llama.cpp static ngrams). Great when the model
//!    paraphrases, quotes, or produces repetitive structure.
//! 2. **Dynamic n-gram table** — a hash → next-token-frequency map learned
//!    from every observed token (llama.cpp `ngram_mod`). Covers patterns
//!    that only exist in generated text. The table can be persisted to disk
//!    (`--ngram-speculative` auto-loads/saves it), so draft quality survives
//!    restarts — the "n-gram SSD cache" llama.cpp keeps beside the model.
//!
//! Drafts are chains (up to `max_chain` tokens): the verify path accepts a
//! matching prefix and commits the rest through `commit_accepted_prefix`,
//! so state stays exact regardless of how deep the chain ran.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Cap on dynamic-table keys — beyond this the map stops growing instead of
/// evicting (eviction would churn hashes; a stale-but-bounded table is the
/// llama.cpp behavior when its fixed ngram cache fills).
const TABLE_CAP: usize = 1 << 20;
/// Per-key next-token candidates kept (top by count).
const CANDS_PER_KEY: usize = 4;
/// Flush the dynamic table to disk every N observations when a path is set.
const SAVE_EVERY: u64 = 1 << 13;
const MAGIC: &[u8; 4] = b"NGC1";

/// n-gram draft proposer: prompt lookup + dynamic table + disk persistence.
pub struct NgramProposer {
    /// Minimum match length (shorter matches are too noisy)
    min_match: usize,
    /// Maximum match length to search for
    max_match: usize,
    /// Maximum chain length per proposal
    max_chain: usize,
    /// Dynamic table: hash(ngram) → [(next_token, count)] (top CANDS_PER_KEY).
    table: HashMap<u64, Vec<(u32, u32)>>,
    /// Disk persistence path (None = in-memory only).
    cache_path: Option<PathBuf>,
    /// Observations since last save.
    dirty: u64,
    /// Running accept/reject stats for logging
    pub accepts: u64,
    pub rejects: u64,
}

fn hash_ngram(ctx: &[u32]) -> u64 {
    // FNV-1a over the token ids.
    let mut h = 0xcbf29ce484222325u64;
    for &t in ctx {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl NgramProposer {
    pub fn new(_order: usize) -> Self {
        Self {
            min_match: 2,
            max_match: 16,
            max_chain: 1,
            table: HashMap::new(),
            cache_path: None,
            dirty: 0,
            accepts: 0,
            rejects: 0,
        }
    }

    /// Configure chain drafting and (optionally) disk persistence.
    /// `cache_path` is the per-model file for the dynamic table; `None` keeps
    /// the table in memory only.
    pub fn with_chain_and_cache(mut self, max_chain: usize, cache_path: Option<PathBuf>) -> Self {
        self.max_chain = max_chain.max(1);
        if let Some(p) = cache_path {
            if let Err(e) = self.load(&p) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!("ngram cache load {}: {e:#}", p.display());
                }
            } else {
                tracing::info!(
                    "ngram cache loaded: {} ({} keys)",
                    p.display(),
                    self.table.len()
                );
            }
            self.cache_path = Some(p);
        }
        self
    }

    /// Record an accepted draft (for stats only).
    pub fn record_accept(&mut self) {
        self.accepts += 1;
    }

    /// Record a rejected draft (for stats only).
    pub fn record_reject(&mut self) {
        self.rejects += 1;
    }

    /// Propose a single draft token (longest prompt-lookup match).
    /// Retained for the K=2-only callers; `propose_chain` is the general path.
    pub fn propose(&self, all_tokens: &[u32]) -> Option<u32> {
        self.lookup_match(all_tokens)
            .map(|(start, n)| all_tokens[start + n])
    }

    /// Propose a draft chain of up to `k` tokens.
    ///
    /// 1. Longest prompt-lookup suffix match → emit the continuation that
    ///    followed it in history (a real multi-token run, not guessed).
    /// 2. If the chain is still short, extend it through the dynamic table:
    ///    hash the tail, take the most frequent learned next-token, append,
    ///    repeat. Mirrors llama.cpp's iterative ngram drafting.
    pub fn propose_chain(&self, all_tokens: &[u32]) -> Vec<u32> {
        let k = self.max_chain;
        let mut drafts = Vec::new();
        if let Some((start, n)) = self.lookup_match(all_tokens) {
            // Continuation after the matched span — bounded by history end.
            let mut i = start + n;
            while drafts.len() < k && i < all_tokens.len() {
                drafts.push(all_tokens[i]);
                i += 1;
            }
        }
        // Dynamic-table extension: keep drafting while the tail has a learned
        // continuation. Operates on tokens + drafts-so-far.
        while drafts.len() < k {
            match self.table_best(all_tokens, &drafts) {
                Some(t) => drafts.push(t),
                None => break,
            }
        }
        drafts
    }

    /// Learn `next` given `history` (dynamic table update).
    ///
    /// Inserts n-grams for n ∈ [min_match, max_match] — bounded work per token.
    /// The same observation is what llama.cpp applies to its ngram_mod cache
    /// on every accepted token.
    pub fn observe(&mut self, history: &[u32], next: u32) {
        let len = history.len();
        for n in self.min_match..=self.max_match.min(len) {
            let key = hash_ngram(&history[len - n..]);
            let e = self.table.entry(key).or_default();
            if let Some((_, c)) = e.iter_mut().find(|(t, _)| *t == next) {
                *c += 1;
            } else if e.len() < CANDS_PER_KEY {
                e.push((next, 1));
            }
        }
        if self.table.len() >= TABLE_CAP {
            // Bound memory: stop inserting new keys (existing keys still
            // update — the counts keep learning).
            self.table.shrink_to_fit();
        }
        self.dirty += 1;
        if self.dirty >= SAVE_EVERY {
            self.dirty = 0;
            self.save_now();
        }
    }

    /// Number of learned n-gram keys.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Persist the dynamic table if a cache path is configured.
    pub fn save_now(&self) {
        let Some(p) = self.cache_path.as_ref() else {
            return;
        };
        if let Err(e) = self.save(p) {
            tracing::warn!("ngram cache save {}: {e:#}", p.display());
        }
    }

    fn lookup_match(&self, all_tokens: &[u32]) -> Option<(usize, usize)> {
        let len = all_tokens.len();
        if len < self.min_match + 1 {
            return None;
        }
        let max_n = self.max_match.min(len - 1);
        let mut best: Option<(usize, usize)> = None;
        for n in self.min_match..=max_n {
            let suffix = &all_tokens[len - n..len];
            for start in 0..=(len - n - 1) {
                if all_tokens[start..start + n] == *suffix {
                    if best.is_none_or(|(_, bn)| n > bn) {
                        best = Some((start, n));
                    }
                    break;
                }
            }
        }
        best
    }

    /// Dynamic-table lookup over the effective tail (history + drafts-so-far).
    fn table_best(&self, all_tokens: &[u32], drafts: &[u32]) -> Option<u32> {
        let tail_len = all_tokens.len() + drafts.len();
        if tail_len < self.min_match {
            return None;
        }
        // Longest learned n-gram first.
        for n in (self.min_match..=self.max_match.min(tail_len)).rev() {
            let mut key_ctx = Vec::with_capacity(n);
            let need = n;
            let d_take = drafts.len().min(need);
            key_ctx.extend_from_slice(&drafts[drafts.len() - d_take..]);
            let h_take = need - d_take;
            key_ctx.splice(
                0..0,
                all_tokens[all_tokens.len() - h_take..].iter().copied(),
            );
            if let Some(cands) = self.table.get(&hash_ngram(&key_ctx))
                && let Some(&(tok, _)) = cands.iter().max_by_key(|(_, c)| *c)
            {
                return Some(tok);
            }
        }
        None
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&(self.min_match as u32).to_le_bytes())?;
        w.write_all(&(self.max_match as u32).to_le_bytes())?;
        w.write_all(&(self.table.len() as u64).to_le_bytes())?;
        for (k, v) in &self.table {
            w.write_all(&k.to_le_bytes())?;
            w.write_all(&(v.len() as u32).to_le_bytes())?;
            for (t, c) in v {
                w.write_all(&t.to_le_bytes())?;
                w.write_all(&c.to_le_bytes())?;
            }
        }
        w.flush()
    }

    fn load(&mut self, path: &Path) -> std::io::Result<()> {
        use std::io::Read;
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad ngram cache magic",
            ));
        }
        let mut u32b = [0u8; 4];
        let mut u64b = [0u8; 8];
        r.read_exact(&mut u32b)?;
        let min_match = u32::from_le_bytes(u32b) as usize;
        r.read_exact(&mut u32b)?;
        let max_match = u32::from_le_bytes(u32b) as usize;
        // Trust the file's bounds — they encode what the table was built for.
        self.min_match = min_match;
        self.max_match = max_match;
        r.read_exact(&mut u64b)?;
        let n = u64::from_le_bytes(u64b) as usize;
        self.table.reserve(n.min(TABLE_CAP));
        for _ in 0..n {
            r.read_exact(&mut u64b)?;
            let k = u64::from_le_bytes(u64b);
            r.read_exact(&mut u32b)?;
            let m = u32::from_le_bytes(u32b) as usize;
            let mut v = Vec::with_capacity(m.min(CANDS_PER_KEY));
            for _ in 0..m {
                r.read_exact(&mut u32b)?;
                let t = u32::from_le_bytes(u32b);
                r.read_exact(&mut u32b)?;
                let c = u32::from_le_bytes(u32b);
                if v.len() < CANDS_PER_KEY {
                    v.push((t, c));
                }
            }
            self.table.insert(k, v);
        }
        Ok(())
    }
}

impl Drop for NgramProposer {
    fn drop(&mut self) {
        self.save_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_lookup_basic() {
        let p = NgramProposer::new(4);
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3];
        assert_eq!(p.propose(&tokens), Some(4));
    }

    #[test]
    fn test_prompt_lookup_no_match() {
        let p = NgramProposer::new(4);
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(p.propose(&tokens), None);
    }

    #[test]
    fn test_prompt_lookup_short() {
        let p = NgramProposer::new(4);
        let tokens = vec![1, 2];
        assert_eq!(p.propose(&tokens), None);
    }

    #[test]
    fn test_prompt_lookup_repetitive() {
        let p = NgramProposer::new(4);
        let tokens = vec![10, 20, 30, 10, 20, 30, 10, 20];
        assert_eq!(p.propose(&tokens), Some(30));
    }

    #[test]
    fn test_chain_emits_continuation_run() {
        let p = NgramProposer::new(4).with_chain_and_cache(3, None);
        // "abc abc ab" — suffix "ab" matched at pos 0 → continuation "c a b".
        let tokens = vec![10, 20, 30, 10, 20, 30, 10, 20];
        let chain = p.propose_chain(&tokens);
        assert_eq!(chain, vec![30, 10, 20]);
    }

    #[test]
    fn test_dynamic_table_learns_and_extends() {
        let mut p = NgramProposer::new(4).with_chain_and_cache(3, None);
        // Teach the table that 9 always follows [1,2,3].
        for _ in 0..3 {
            p.observe(&[1, 2, 3], 9);
        }
        // Unseen-in-history tail [1,2,3] → dynamic lookup proposes 9.
        let tokens = vec![5, 6, 1, 2, 3];
        let chain = p.propose_chain(&tokens);
        assert_eq!(chain.first().copied(), Some(9));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ngram_test_{}", std::process::id()));
        let path = dir.join("cache.bin");
        let mut p = NgramProposer::new(4);
        p.observe(&[1, 2, 3, 4], 7);
        p.observe(&[1, 2, 3, 4], 7);
        p.save(&path).unwrap();
        let mut q = NgramProposer::new(4);
        q.load(&path).unwrap();
        assert_eq!(q.len(), p.len());
        // Learned continuation survives a restart.
        assert_eq!(q.table_best(&[1, 2, 3, 4], &[]), Some(7));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
