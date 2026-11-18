// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
pub(super) const ALLOC: u32 = 1;
pub(super) const ROW: u32 = 2;
pub(super) const FREE: u32 = 3;
pub(super) const STOP: u32 = u32::MAX;
impl Hy4Model {
    pub(super) fn eager_stream(&self, caller: u64) -> Result<u64> {
        let gpu = self.borrow_owner().gpu.as_ref();
        let stream = gpu.default_stream();
        if caller != stream {
            gpu.synchronize(caller)?;
        }
        Ok(stream)
    }
    pub(super) fn send(&self, words: [u32; 4]) -> Result<()> {
        let s = self.borrow_owner();
        let r = self.borrow_dependent();
        if let Some(c) = s.comm.as_deref().filter(|c| c.world_size() > 1) {
            ensure!(c.rank() == 0, "Hy4 command sender must be rank zero");
            s.gpu.copy_h2d(
                &words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
                r.command,
            )?;
            c.broadcast(r.command.0, 4, 0)?;
            c.broadcast(r.command.offset(4).0, 12, 0)?;
        }
        Ok(())
    }
    pub(super) fn worker(&self) -> Result<bool> {
        let s = self.borrow_owner();
        let r = self.borrow_dependent();
        let c = s
            .comm
            .as_deref()
            .context("Hy4 worker communicator missing")?;
        ensure!(
            c.world_size() > 1 && c.rank() != 0,
            "Hy4 worker rank mismatch"
        );
        c.receive_idle_command_word(r.command.0)?;
        c.broadcast(r.command.offset(4).0, 12, 0)?;
        let mut bytes = [0; 16];
        s.gpu.copy_d2h(r.command, &mut bytes)?;
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let slot = words[1] as usize;
        let stream = s.gpu.default_stream();
        match words[0] {
            ALLOC => {
                let requests = r.requests.lock();
                if let Some(request) = requests.get(&slot) {
                    ensure!(
                        request.borrow_dependent().history.is_empty(),
                        "Hy4 cannot reallocate an active worker slot"
                    );
                } else {
                    drop(requests);
                    r.allocate(s, slot)?;
                }
            }
            ROW => {
                let requests = r.requests.lock();
                let request = requests.get(&slot).context("Hy4 worker slot missing")?;
                ensure!(
                    request.borrow_dependent().history.len() == words[3] as usize,
                    "Hy4 worker position mismatch"
                );
                drop(requests);
                r.row(s, slot, words[2], stream)?;
            }
            FREE => {
                s.gpu.synchronize(stream)?;
                r.requests.lock().remove(&slot);
            }
            STOP => return Ok(false),
            op => anyhow::bail!("Hy4 unknown eager command {op}"),
        }
        Ok(true)
    }
    pub(super) fn forward_token(
        &self,
        token: u32,
        seq: &mut crate::traits::SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        let s = self.borrow_owner();
        let r = self.borrow_dependent();
        let stream = self.eager_stream(stream)?;
        {
            let requests = r.requests.lock();
            let request = requests
                .get(&seq.slot_idx)
                .context("Hy4 sequence not allocated")?;
            ensure!(
                request.borrow_dependent().history.len() == seq.seq_len
                    && seq.seq_len < s.max_seq
                    && (token as usize) < s.config.vocab_size,
                "Hy4 sequence position/token mismatch"
            );
        }
        self.send([
            ROW,
            u32::try_from(seq.slot_idx)?,
            token,
            u32::try_from(seq.seq_len)?,
        ])?;
        let ptr = r.row(s, seq.slot_idx, token, stream)?;
        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(ptr)
    }
}
