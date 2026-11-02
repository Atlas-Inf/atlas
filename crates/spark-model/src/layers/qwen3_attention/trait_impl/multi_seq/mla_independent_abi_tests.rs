// SPDX-License-Identifier: AGPL-3.0-only

use super::select;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle, mock::MockGpuBackend};
use std::{ffi::c_void, sync::Mutex};

#[derive(Debug, PartialEq)]
enum Arg {
    Ptr(u64),
    U32(u32),
}
#[derive(Debug, PartialEq)]
struct Launch {
    kernel: u64,
    grid: [u32; 3],
    block: [u32; 3],
    stream: u64,
    args: Vec<Arg>,
}
#[derive(Default)]
struct Recorder {
    inner: MockGpuBackend,
    launches: Mutex<Vec<Launch>>,
}
macro_rules! forward {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {$(
        fn $name(&self, $($arg:$ty),*) -> $ret { self.inner.$name($($arg),*) }
    )*};
}
impl GpuBackend for Recorder {
    forward! {
        alloc(n:usize) -> Result<DevicePtr>;
        alloc_managed(n:usize) -> Result<DevicePtr>;
        free(p:DevicePtr) -> Result<()>;
        copy_h2d(s:&[u8],d:DevicePtr) -> Result<()>;
        copy_d2h(s:DevicePtr,d:&mut [u8]) -> Result<()>;
        copy_d2d(s:DevicePtr,d:DevicePtr,n:usize) -> Result<()>;
        synchronize(s:u64) -> Result<()>;
        default_stream() -> u64;
        kernel(m:&str,f:&str) -> Result<KernelHandle>;
        op_cache() -> &spark_runtime::op_cache::OpCache;
        memset(p:DevicePtr,v:u8,n:usize) -> Result<()>;
        memset_async(p:DevicePtr,v:u8,n:usize,s:u64) -> Result<()>;
        total_memory() -> Result<usize>;
        free_memory() -> Result<usize>;
        sm_count() -> Result<u32>;
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
        anyhow::bail!("untyped launch loses ABI evidence")
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
        assert_eq!(shared, 0);
        let args = args
            .iter()
            .map(|arg| match arg {
                KernelArg::Buffer(p) => Arg::Ptr(p.0),
                KernelArg::Bytes(b) => Arg::U32(u32::from_le_bytes((*b).try_into().unwrap())),
            })
            .collect();
        self.launches.lock().unwrap().push(Launch {
            kernel: kernel.0,
            grid,
            block,
            stream,
            args,
        });
        Ok(())
    }
}

#[test]
fn exact_independent_exports_use_existing_nine_argument_mla_abi() {
    let gpu = Recorder::default();
    let handles = std::array::from_fn(|i| KernelHandle((i + 2) as u64));
    for rows in 2..=8 {
        for (n, k) in [(512, 256), (256, 512)] {
            for padding in [0, 8] {
                let ih = k + padding;
                let oh = n + padding;
                let ir = 32 * ih + 4 * padding;
                let or = 32 * oh + 4 * padding;
                let kernel = select(rows, true, handles).unwrap().unwrap();
                crate::layers::ops::mla_batched_gemv_batchm(
                    &gpu,
                    kernel,
                    DevicePtr(0x1000),
                    DevicePtr(0x2000),
                    DevicePtr(0x3000),
                    n,
                    k,
                    32,
                    ih,
                    oh,
                    ir,
                    or,
                    37,
                )
                .unwrap();
                assert_eq!(
                    gpu.launches.lock().unwrap().pop().unwrap(),
                    Launch {
                        kernel: rows as u64,
                        grid: [n.div_ceil(8), 32, 1],
                        block: [256, 1, 1],
                        stream: 37,
                        args: vec![
                            Arg::Ptr(0x1000),
                            Arg::Ptr(0x2000),
                            Arg::Ptr(0x3000),
                            Arg::U32(n),
                            Arg::U32(k),
                            Arg::U32(ih),
                            Arg::U32(oh),
                            Arg::U32(ir),
                            Arg::U32(or)
                        ],
                    }
                );
            }
        }
    }
    assert_eq!(gpu.inner.alloc_count(), 0);
    assert_eq!(gpu.inner.sync_count(), 0);
    assert_eq!(gpu.inner.d2d_count(), 0);
}
