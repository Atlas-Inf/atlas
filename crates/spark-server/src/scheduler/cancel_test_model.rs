// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only logits provider; every compute entry point is deliberately rejected.
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(super) struct TestModel {
    pub tokens: Vec<u32>,
    pub host_logits: bool,
    pub cancel_after_sampling: Option<Arc<AtomicBool>>,
    pub cancel_after_row_commit: Option<Arc<AtomicBool>>,
    /// A scripted DFlash verify on an argmax-only head (`strict_spec` tests);
    /// `None` keeps every verify entry point rejected as before.
    pub verify: Option<Arc<VerifyScript>>,
}

/// Picks one scripted verify returns, and the wire calls the step made.
#[derive(Default)]
pub(super) struct VerifyScript {
    pub picks: Vec<u32>,
    pub is_ep: bool,
    pub fail_candidates: bool,
    pub fail_full_rows: bool,
    pub log: std::sync::Mutex<Vec<String>>,
}

thread_local! {
    /// A scripted logits stream for this test thread: one vocab row per
    /// position, read from `cursor` on (the device pointer's byte offset
    /// selects further rows). When set, the logits reads and the GPU argmax
    /// entry points serve it as BF16 (argmax lowest index on ties, as the GPU
    /// kernels) instead of the fixed fixture.
    pub(super) static SCRIPT: std::cell::RefCell<Option<Script>> =
        const { std::cell::RefCell::new(None) };
}

pub(super) struct Script {
    pub rows: Vec<Vec<f32>>,
    pub cursor: usize,
}

/// Run `f` on the scripted stream from `ptr` on: the flattened BF16-element
/// offset into it, and the rows (`None` without a script).
fn scripted<T>(ptr: DevicePtr, f: impl FnOnce(usize, &[Vec<f32>]) -> T) -> Option<T> {
    SCRIPT.with(|s| {
        let s = s.borrow();
        let s = s.as_ref()?;
        Some(f(s.cursor * 2048 + ptr.0 as usize / 2, &s.rows))
    })
}

/// Candidates per scripted rank.
pub(super) const SPLIT_CANDIDATES: usize = 16;

/// `row` (BF16-exact) as two ranks' candidates would leave it.
fn split_row(row: &[f32], temperature: f32, specials: &[u32]) -> spark_model::model::SplitRow {
    let half = row.len() / 2;
    let mut present: Vec<(u32, f32)> = Vec::new();
    let mut bound = (f32::NEG_INFINITY, u32::MAX);
    let mut mass = 0.0f64;
    for r in 0..2 {
        let mut ids: Vec<usize> = (r * half..(r + 1) * half).collect();
        ids.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        let (best, rest) = ids.split_at(SPLIT_CANDIDATES);
        present.extend(best.iter().map(|&i| (i as u32, row[i])));
        let last = best[SPLIT_CANDIDATES - 1];
        if row[last] > bound.0 || (row[last] == bound.0 && (last as u32) < bound.1) {
            bound = (row[last], last as u32);
        }
        for &i in rest {
            if specials.contains(&(i as u32)) {
                present.push((i as u32, row[i]));
            } else {
                mass += ((row[i] / temperature) as f64).exp();
            }
        }
    }
    spark_model::model::SplitRow {
        present,
        absent: Some(spark_runtime::sampler::Absent {
            bound: bound.0,
            bound_id: bound.1,
            log_mass: mass.ln() as f32,
        }),
    }
}

impl TestModel {
    fn log(&self, entry: String) {
        if let Some(v) = &self.verify {
            v.log.lock().unwrap().push(entry);
        }
    }
}

impl Model for TestModel {
    fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn prefill_chunk(
        &self,
        _: &[u32],
        _: &mut SequenceState,
        _: usize,
        _: usize,
        _: bool,
        _: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_batch(&self, _: &[u32], _: &mut [&mut SequenceState], _: u64) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn generate_speculative(
        &self,
        _: &[u32],
        _: &spark_runtime::sampler::SamplingParams,
        _: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_verify_graphed(
        &self,
        _: &[u32; 2],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_verify_graphed_k3(
        &self,
        _: &[u32; 3],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn decode_verify_graphed_k4(
        &self,
        _: &[u32; 4],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn run_mtp_propose(
        &self,
        _: u32,
        _: usize,
        _: &mut SequenceState,
        _: u64,
    ) -> Result<Option<u32>> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn run_mtp_propose_multi(
        &self,
        _: u32,
        _: usize,
        _: usize,
        _: &mut SequenceState,
        _: u64,
        _: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn trim_proposer_state(&self, _: &mut SequenceState, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
    fn vocab_size(&self) -> usize {
        2048
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn decode_logits_fp32(&self) -> bool {
        self.host_logits
    }
    fn copy_logits_to_host(&self, ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let bf16 = |at: usize, rows: &[Vec<f32>]| {
            let vals = rows.iter().flatten().skip(at);
            for (bytes, v) in dst.chunks_exact_mut(2).zip(vals) {
                bytes.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
        };
        if scripted(ptr, bf16).is_some() {
            return Ok(());
        }
        assert!(self.host_logits);
        for (i, bytes) in dst.chunks_exact_mut(4).enumerate() {
            let value = if i % 2048 == 101 { 5.0f32 } else { 0.0f32 };
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, ptr: DevicePtr, _: u64) -> Result<u32> {
        let first_wins = |at: usize, r: &[Vec<f32>]| {
            spark_runtime::sampler::argmax_first_wins_f32(&r[at / 2048])
        };
        scripted(ptr, first_wins).ok_or_else(|| anyhow::anyhow!("unexpected scalar argmax"))
    }
    fn argmax_batch(&self, ptr: DevicePtr, n: usize, _: u64) -> Result<Vec<u32>> {
        let first_wins = |at: usize, rows: &[Vec<f32>]| {
            rows[at / 2048..at / 2048 + n]
                .iter()
                .map(|r| spark_runtime::sampler::argmax_first_wins_f32(r))
                .collect()
        };
        if let Some(t) = scripted(ptr, first_wins) {
            return Ok(t);
        }
        assert_eq!(n, self.tokens.len());
        if let Some(flag) = &self.cancel_after_sampling {
            flag.store(true, Ordering::Release);
        }
        Ok(self.tokens.clone())
    }
    fn decode_marconi_checkpoint(&self, _: &mut SequenceState) {
        if let Some(flag) = &self.cancel_after_row_commit {
            flag.store(true, Ordering::Release);
        }
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        anyhow::bail!("unexpected compute in cancellation test")
    }
    fn cache_sequence(&self, _: &SequenceState) {}
    fn free_sequence(&self, _: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _: &mut SequenceState, _: usize) -> Result<bool> {
        Ok(false)
    }
    fn detach_slot_for_reuse(&self, _: &mut SequenceState) {}
    fn save_hidden_for_mtp(&self, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
    fn is_ep(&self) -> bool {
        self.verify.as_ref().is_some_and(|v| v.is_ep)
    }
    fn begin_glm_long_owner_tail(&self, _: u32, owner: usize, _: &[u32]) -> Result<()> {
        anyhow::ensure!(self.verify.is_some(), "no scripted owner tail");
        self.log(format!("owner {owner}"));
        Ok(())
    }
    fn verify_logits_argmax_only(&self) -> bool {
        self.verify.is_some()
    }
    /// A scripted split head (`split_rows` tests): two ranks of the scripted
    /// rows from the cursor on, each sending its best [`SPLIT_CANDIDATES`].
    fn verify_split_sampling(&self) -> bool {
        self.verify.is_some() && SCRIPT.with(|s| s.borrow().is_some())
    }
    fn verify_split_candidates(
        &self,
        rows: usize,
        temperature: f32,
        specials: &[u32],
    ) -> Result<Vec<spark_model::model::SplitRow>> {
        self.log(format!("candidates {rows}"));
        anyhow::ensure!(
            !self.verify.as_ref().is_some_and(|v| v.fail_candidates),
            "injected split-candidate exchange failure"
        );
        scripted(DevicePtr::NULL, |at, all| {
            (0..rows)
                .map(|r| split_row(&all[at / 2048 + r], temperature, specials))
                .collect()
        })
        .ok_or_else(|| anyhow::anyhow!("no scripted rows"))
    }
    fn verify_split_full_rows(&self, rows: usize, out: &mut [u8]) -> Result<()> {
        self.log(format!("full {rows}"));
        anyhow::ensure!(
            !self.verify.as_ref().is_some_and(|v| v.fail_full_rows),
            "injected split-full-row exchange failure"
        );
        self.copy_logits_to_host(DevicePtr::NULL, out)
    }
    fn ep_broadcast_cmd_for_seq(&self, _: u32, cmd: u32) -> Result<()> {
        self.log(format!("seq_cmd {cmd:#x}"));
        Ok(())
    }
    fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> {
        self.log(format!("cmd {cmd:#x}"));
        Ok(())
    }
    fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        self.log(format!("tokens {tokens:?}"));
        Ok(Vec::new())
    }
    fn prepare_verify_row_masks(&self, rows: usize, masks: &[u32]) -> Result<()> {
        anyhow::ensure!(self.verify.is_some(), "no masked verify");
        self.log(format!("upload {rows} rows {} words", masks.len()));
        Ok(())
    }
    fn send_verify_row_masks(&self, rows: usize) -> Result<()> {
        anyhow::ensure!(self.verify.is_some(), "no masked verify");
        self.log(format!("send {rows}"));
        Ok(())
    }
    fn decode_verify_dflash(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        s: u64,
    ) -> Result<Vec<u32>> {
        let Some(v) = &self.verify else {
            return self.decode_verify_graphed_kgamma(tokens, seq, s);
        };
        self.log(format!("verify {tokens:?}"));
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len += tokens.len();
        Ok(v.picks[..tokens.len()].to_vec())
    }
}
