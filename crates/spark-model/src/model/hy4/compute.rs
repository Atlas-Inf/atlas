// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
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
                        e.forward(
                            self.highway,
                            &data.history,
                            token,
                            state.engram.as_mut().context("Hy4 Engram state missing")?,
                            s.comm.as_deref(),
                            stream,
                        )?;
                    }
                    b.hc_attn
                        .collapse(gpu, self.highway, self.collapsed, self.post, 1, stream)?;
                    self.norm(s, self.collapsed, b.input_norm, self.normed, stream)?;
                    let p = data.history.len() as i64;
                    state.attn.forward(
                        self.normed,
                        self.output,
                        &[[p; 3]],
                        s.comm.as_deref(),
                        stream,
                    )?;
                    b.hc_attn.scatter(
                        gpu,
                        [self.output, self.highway, self.post, self.highway],
                        1,
                        stream,
                    )?;
                    b.hc_mlp
                        .collapse(gpu, self.highway, self.collapsed, self.post, 1, stream)?;
                    self.norm(s, self.collapsed, b.mlp_norm, self.normed, stream)?;
                    b.ffn.forward(
                        self.normed,
                        self.output,
                        1,
                        &mut state.ffn,
                        s.comm.as_deref(),
                        stream,
                    )?;
                    b.hc_mlp.scatter(
                        gpu,
                        [self.output, self.highway, self.post, self.highway],
                        1,
                        stream,
                    )?;
                }
                self.head.collapse(
                    gpu,
                    self.highway,
                    self.collapsed,
                    DevicePtr::NULL,
                    1,
                    stream,
                )?;
                self.norm(s, self.collapsed, self.final_norm, self.normed, stream)?;
                KernelLaunch::new(gpu, self.kernels[2])
                    .grid([(c.vocab_size as u32).div_ceil(128), 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(self.normed)
                    .arg_ptr(self.lm_head)
                    .arg_ptr(self.logits)
                    .arg_u32(c.vocab_size as u32)
                    .arg_u32(h as u32)
                    .arg_u32(u32::from(self.lm_head_f32))
                    .launch(stream)?;
                gpu.synchronize(stream)?;
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
