// SPDX-License-Identifier: AGPL-3.0-only

//! Stream ordering and CUDA launch for one one-shot exchange.

use super::*;

impl OneShot {
    /// Send `src[..bytes]` to the peer and land its payload in `dst` (added
    /// in place when `add`, else copied), all on `stream` and capturable.
    /// `false` (nothing enqueued) when the size is ineligible or the kernel
    /// is missing; both ranks see the same answer.
    pub(in crate::nccl_backend) fn enqueue(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        let kernel = self.kernel.load(Ordering::Relaxed);
        if kernel == 0 || !self.cfg.eligible(bytes) {
            return Ok(false);
        }
        let ctrl = self.dev + ctrl_off(self.cfg.max) as u64;
        let stage = ctrl + STAGE as u64;
        let sm = self.cfg.sm_stage;
        let fence = self.cfg.stage_fence && sm == 0;
        if sm == 0 {
            cu(
                unsafe { cuStreamWaitValue32_v2(stream, stage, 0, CU_STREAM_WAIT_VALUE_EQ) },
                "cuStreamWaitValue32(one-shot stage)",
            )?;
            cu(
                unsafe { cuMemcpyAsync(self.dev, src, bytes, stream) },
                "cuMemcpyAsync(one-shot stage)",
            )?;
        }
        if fence {
            // Default flags fence the copy before the word lands.
            cu(
                unsafe { cuStreamWriteValue32_v2(stream, stage, bytes as u32, 0) },
                "cuStreamWriteValue32(one-shot stage)",
            )?;
        }
        let blocks = (bytes / 16).div_ceil(THREADS).clamp(1, MAX_BLOCKS) as u32;
        let mut args = (
            dst,
            self.dev + recv_off(self.cfg.max, 0) as u64,
            self.cfg.max as u64,
            bytes as u32,
            add as u32,
            ctrl + FLAGS as u64,
            stripes(bytes, self.rails, self.cfg.stripe_min).len() as u32,
            self.state,
            if fence { 0 } else { stage },
            ctrl + POISON as u64,
            self.cfg.timeout_ns,
            src,
            self.dev,
            sm,
        );
        let mut params: [*mut c_void; 14] = [
            (&raw mut args.0).cast(),
            (&raw mut args.1).cast(),
            (&raw mut args.2).cast(),
            (&raw mut args.3).cast(),
            (&raw mut args.4).cast(),
            (&raw mut args.5).cast(),
            (&raw mut args.6).cast(),
            (&raw mut args.7).cast(),
            (&raw mut args.8).cast(),
            (&raw mut args.9).cast(),
            (&raw mut args.10).cast(),
            (&raw mut args.11).cast(),
            (&raw mut args.12).cast(),
            (&raw mut args.13).cast(),
        ];
        let status = if sm != 0 && self.cfg.pdl {
            // The kernel enters with `atlas_pdl_enter()` before it reads
            // anything its predecessor wrote.
            let mut value = [0u8; 64];
            value[..4].copy_from_slice(&1i32.to_ne_bytes());
            let mut attr = CuLaunchAttribute {
                id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                pad: [0; 4],
                value,
            };
            let config = CuLaunchConfig {
                grid: [blocks, 1, 1],
                block: [THREADS as u32, 1, 1],
                shared_mem_bytes: 0,
                stream,
                attrs: &mut attr,
                num_attrs: 1,
            };
            unsafe { cuLaunchKernelEx(&config, kernel, params.as_mut_ptr(), std::ptr::null_mut()) }
        } else {
            unsafe {
                cuLaunchKernel(
                    kernel,
                    blocks,
                    1,
                    1,
                    THREADS as u32,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            }
        };
        if status != 0 && fence {
            // `stage` is already on the stream: the proxy will serve an op no
            // kernel consumes, and its count would stay ahead of the device
            // sequence (stale receive slots). Stop the channel instead.
            poison_host(self.host + ctrl_off(self.cfg.max), POISON_LAUNCH);
        }
        cu(status, "cuLaunchKernel(rdma_oneshot_bf16)")?;
        Ok(true)
    }
}
