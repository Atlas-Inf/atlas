// SPDX-License-Identifier: AGPL-3.0-only
//! Recording GPU and pair for the GLM KV shard tests: kernel launches by
//! symbol, stream/event fences and pair exchanges, in call order.
use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle, mock::MockGpuBackend};
use std::{ffi::c_void, sync::Mutex};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Launch {
        symbol: String,
        grid: [u32; 3],
        shared: u32,
        stream: u64,
        args: Vec<Vec<u8>>,
    },
    Record {
        event: u64,
        stream: u64,
    },
    Wait {
        stream: u64,
        event: u64,
    },
    Exchange {
        send: u64,
        recv: u64,
        bytes: usize,
        stream: u64,
    },
    Copy {
        src: u64,
        dst: u64,
        bytes: usize,
    },
}

#[derive(Default)]
pub(crate) struct ShardGpu {
    inner: MockGpuBackend,
    /// `(module, symbol)` of every kernel lookup; a handle is its index + 1.
    symbols: Mutex<Vec<(String, String)>>,
    pub(crate) ops: Mutex<Vec<Op>>,
}

impl ShardGpu {
    /// The module each looked-up symbol was asked of.
    pub(crate) fn modules(&self) -> Vec<(String, String)> {
        self.symbols.lock().unwrap().clone()
    }

    pub(crate) fn ops(&self) -> Vec<Op> {
        self.ops.lock().unwrap().clone()
    }

    /// Launched symbols and the stream/event/exchange steps between them.
    pub(crate) fn order(&self) -> Vec<String> {
        self.ops()
            .iter()
            .map(|op| match op {
                Op::Launch { symbol, .. } => symbol.clone(),
                Op::Record { event, stream } => format!("record e{event} s{stream}"),
                Op::Wait { stream, event } => format!("wait s{stream} e{event}"),
                Op::Exchange { bytes, stream, .. } => format!("exchange {bytes} s{stream}"),
                Op::Copy { bytes, .. } => format!("copy {bytes}"),
            })
            .collect()
    }
}

/// A pointer argument as the launch recorded it.
pub(crate) fn ptr(p: u64) -> Vec<u8> {
    p.to_ne_bytes().to_vec()
}
/// A `u32` argument as the launch recorded it.
pub(crate) fn word(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

impl GpuBackend for ShardGpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(s, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(s, d, n)
    }
    fn copy_d2d_async(&self, s: DevicePtr, d: DevicePtr, n: usize, _: u64) -> Result<()> {
        self.ops.lock().unwrap().push(Op::Copy {
            src: s.0,
            dst: d.0,
            bytes: n,
        });
        Ok(())
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, symbol: &str) -> Result<KernelHandle> {
        let mut symbols = self.symbols.lock().unwrap();
        symbols.push((module.to_owned(), symbol.to_owned()));
        Ok(KernelHandle(symbols.len() as u64))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut c_void],
    ) -> Result<()> {
        bail!("expected the production typed launch")
    }
    fn launch_typed(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        assert_eq!(block, [256, 1, 1]);
        let symbol = self.symbols.lock().unwrap()[kernel.0 as usize - 1]
            .1
            .clone();
        self.ops.lock().unwrap().push(Op::Launch {
            symbol,
            grid,
            shared,
            stream,
            args: args
                .iter()
                .map(|a| match a {
                    KernelArg::Buffer(p) => ptr(p.0),
                    KernelArg::Bytes(b) => b.to_vec(),
                })
                .collect(),
        });
        Ok(())
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.ops.lock().unwrap().push(Op::Record { event, stream });
        Ok(())
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.ops.lock().unwrap().push(Op::Wait { stream, event });
        Ok(())
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.inner.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, s)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.inner.sm_count()
    }
}

/// One rank of a pair whose exchanges land in the GPU's trace.
pub(crate) struct ShardPair<'a> {
    pub(crate) gpu: &'a ShardGpu,
    pub(crate) rank: usize,
}

impl spark_comm::CommBackend for ShardPair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn exchange_async(&self, send: u64, recv: u64, n: usize, add: bool, s: u64) -> Result<bool> {
        assert!(!add, "shard exchanges copy, never add");
        self.gpu.ops.lock().unwrap().push(Op::Exchange {
            send,
            recv,
            bytes: n,
            stream: s,
        });
        Ok(true)
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
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
