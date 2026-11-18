// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::hy4::timing;
use spark_runtime::kernel_args::KernelLaunch;

impl Runtime<'_> {
    pub(super) fn row(
        &self,
        s: &Shared,
        slot: usize,
        token: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let mut requests = self.requests.lock();
        let cell = requests
            .get_mut(&slot)
            .context("Hy4 request slot missing")?;
        cell.with_dependent_mut(|owner, data| {
            ensure!(
                !data.poisoned
                    && data.history.len() < s.max_seq
                    && (token as usize) < s.config.vocab_size,
                "Hy4 request poisoned, capacity exhausted or invalid token"
            );
            let result = (|| {
                let gpu = s.gpu.as_ref();
                let c = &s.config;
                let h = c.hidden_size;
                let timer = timing::Token::begin(gpu, stream, c.tp_rank, data.history.len());
                let row_span = timing::span("other", stream)?;
                KernelLaunch::new(gpu, self.kernels[0])
                    .grid([((4 * h) as u32).div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(self.embed)
                    .arg_ptr(self.highway)
                    .arg_u32(token)
                    .arg_u32(h as u32)
                    .launch(stream)?;
                for (cell, state) in owner.layers.iter().zip(&mut data.layers) {
                    let b = cell.borrow_dependent();
                    if let Some(e) = &b.engram {
                        let _engram = timing::span("engram", stream)?;
                        e.forward(
                            self.highway,
                            &data.history,
                            token,
                            state.engram.as_mut().context("Hy4 Engram state missing")?,
                            s.comm.as_deref(),
                            stream,
                        )?;
                    }
                    let hc = timing::span("ihc_norms", stream)?;
                    b.hc_attn
                        .collapse(gpu, self.highway, self.collapsed, self.post, 1, stream)?;
                    self.norm(s, self.collapsed, b.input_norm, self.normed, stream)?;
                    drop(hc);
                    let p = data.history.len() as i64;
                    let attention = timing::span("attention", stream)?;
                    state.attn.forward(
                        self.normed,
                        self.output,
                        &[[p; 3]],
                        s.comm.as_deref(),
                        stream,
                    )?;
                    drop(attention);
                    let hc = timing::span("ihc_norms", stream)?;
                    b.hc_attn.scatter(
                        gpu,
                        [self.output, self.highway, self.post, self.highway],
                        1,
                        stream,
                    )?;
                    b.hc_mlp
                        .collapse(gpu, self.highway, self.collapsed, self.post, 1, stream)?;
                    self.norm(s, self.collapsed, b.mlp_norm, self.normed, stream)?;
                    drop(hc);
                    b.ffn.forward(
                        self.normed,
                        self.output,
                        1,
                        &mut state.ffn,
                        s.comm.as_deref(),
                        stream,
                    )?;
                    let _hc = timing::span("ihc_norms", stream)?;
                    b.hc_mlp.scatter(
                        gpu,
                        [self.output, self.highway, self.post, self.highway],
                        1,
                        stream,
                    )?;
                }
                let hc = timing::span("ihc_norms", stream)?;
                self.head.collapse(
                    gpu,
                    self.highway,
                    self.collapsed,
                    DevicePtr::NULL,
                    1,
                    stream,
                )?;
                self.norm(s, self.collapsed, self.final_norm, self.normed, stream)?;
                drop(hc);
                let head = timing::span("lm_head", stream)?;
                KernelLaunch::new(gpu, self.kernels[2])
                    .grid([(self.vocab.rows as u32).div_ceil(8), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(self.normed)
                    .arg_ptr(self.vocab.weight)
                    .arg_ptr(self.logits.offset(self.vocab.start * 4))
                    .arg_u32(self.vocab.rows as u32)
                    .arg_u32(h as u32)
                    .arg_u32(0)
                    .launch(stream)?;
                drop(head);
                let gather = timing::span("logits_gather", stream)?;
                self.vocab.gather(s, self.logits, stream)?;
                drop(gather);
                drop(row_span);
                gpu.synchronize(stream)?;
                timer.finish()?;
                Ok(self.logits)
            })();
            if result.is_ok() {
                data.history.push(token);
            } else {
                data.poisoned = true;
            }
            result
        })
    }
    fn norm(
        &self,
        s: &Shared,
        input: DevicePtr,
        weight: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(s.gpu.as_ref(), self.kernels[1])
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight)
            .arg_ptr(out)
            .arg_u32(s.config.hidden_size as u32)
            .arg_f32(s.config.rms_norm_eps as f32)
            .launch(stream)
    }
}
