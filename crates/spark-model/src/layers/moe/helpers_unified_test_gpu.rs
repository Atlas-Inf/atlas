// SPDX-License-Identifier: AGPL-3.0-only
//! Byte storage and actual typed-ABI recorder. CUDA launches do NOT modify bytes.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Arg {
    Ptr(DevicePtr),
    Bytes(Vec<u8>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Alloc(DevicePtr, usize),
    H2d(DevicePtr, Vec<u8>),
    D2h(DevicePtr, usize),
    D2d(DevicePtr, DevicePtr, usize),
    Free(DevicePtr),
    Sync(u64),
    Launch(u64, [u32; 3], [u32; 3], u32, u64, Vec<Arg>),
}
struct State {
    next: u64,
    bytes: BTreeMap<u64, Vec<u8>>,
    events: Vec<Event>,
    fail: usize,
    peak: usize,
}
pub(super) struct RecordingGpu {
    cache: spark_runtime::op_cache::OpCache,
    state: Mutex<State>,
    pub host_shared: bool,
}
impl RecordingGpu {
    pub fn new(host_shared: bool) -> Self {
        Self {
            cache: spark_runtime::op_cache::OpCache::new(),
            state: Mutex::new(State {
                next: 0x10000,
                bytes: BTreeMap::new(),
                events: Vec::new(),
                fail: usize::MAX,
                peak: 0,
            }),
            host_shared,
        }
    }
    fn record(s: &mut State, event: Event) -> Result<()> {
        s.events.push(event);
        ensure!(s.events.len() != s.fail, "injected op {}", s.fail);
        Ok(())
    }
    pub fn trace(&self) -> Vec<Event> {
        self.state.lock().unwrap().events.clone()
    }
    pub fn clear(&self) {
        let mut s = self.state.lock().unwrap();
        s.events.clear();
        s.fail = usize::MAX;
        s.peak = s.bytes.values().map(Vec::len).sum();
    }
    pub fn fail(&self, at: usize) {
        self.state.lock().unwrap().fail = at;
    }
    pub fn profile(&self) -> (usize, usize) {
        let s = self.state.lock().unwrap();
        (s.bytes.values().map(Vec::len).sum(), s.peak)
    }
    pub fn live(&self) -> BTreeMap<u64, usize> {
        self.state
            .lock()
            .unwrap()
            .bytes
            .iter()
            .map(|(&p, b)| (p, b.len()))
            .collect()
    }
    pub fn reuse_next(&self, ptr: DevicePtr) {
        let mut s = self.state.lock().unwrap();
        assert!(!s.bytes.contains_key(&ptr.0));
        s.next = ptr.0;
    }
    pub fn read(&self, p: DevicePtr, n: usize) -> Vec<u8> {
        let s = self.state.lock().unwrap();
        let (&base, bytes) = s.bytes.range(..=p.0).next_back().expect("known allocation");
        let offset = (p.0 - base) as usize;
        bytes[offset..offset + n].to_vec()
    }
}
impl GpuBackend for RecordingGpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        let mut s = self.state.lock().unwrap();
        let p = DevicePtr(s.next);
        Self::record(&mut s, Event::Alloc(p, n))?;
        s.next += (n as u64).div_ceil(256) * 256 + 256;
        s.bytes.insert(p.0, vec![0xcd; n]);
        s.peak = s.peak.max(s.bytes.values().map(Vec::len).sum());
        Ok(p)
    }
    fn alloc_managed(&self, _: usize) -> Result<DevicePtr> {
        bail!("unexpected managed alloc")
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        // CudaBackend removes ownership before cuMemFree, even on failure.
        ensure!(
            s.bytes.remove(&p.0).is_some(),
            "non-base or duplicate free {p:?}"
        );
        Self::record(&mut s, Event::Free(p))
    }
    fn copy_h2d(&self, b: &[u8], p: DevicePtr) -> Result<()> {
        let mut s = self.state.lock().unwrap();
        Self::record(&mut s, Event::H2d(p, b.to_vec()))?;
        let dst = s.bytes.get_mut(&p.0).expect("H2D base");
        ensure!(dst.len() == b.len(), "full allocation H2D");
        dst.copy_from_slice(b);
        Ok(())
    }
    fn copy_d2h(&self, p: DevicePtr, b: &mut [u8]) -> Result<()> {
        Self::record(&mut self.state.lock().unwrap(), Event::D2h(p, b.len()))?;
        b.copy_from_slice(&self.read(p, b.len()));
        Ok(())
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, n: usize) -> Result<()> {
        Self::record(&mut self.state.lock().unwrap(), Event::D2d(src, dst, n))?;
        let bytes = self.read(src, n);
        let mut s = self.state.lock().unwrap();
        let (&base, _) = s
            .bytes
            .range(..=dst.0)
            .next_back()
            .expect("D2D destination");
        let offset = (dst.0 - base) as usize;
        let dest = s.bytes.get_mut(&base).unwrap();
        ensure!(offset + n <= dest.len(), "D2D destination bounds");
        dest[offset..offset + n].copy_from_slice(&bytes);
        Ok(())
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        bail!("expected typed ABI")
    }
    fn launch_typed(
        &self,
        k: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        Self::record(
            &mut self.state.lock().unwrap(),
            Event::Launch(
                k.0,
                grid,
                block,
                shared,
                stream,
                args.iter()
                    .map(|a| match a {
                        KernelArg::Buffer(p) => Arg::Ptr(*p),
                        KernelArg::Bytes(b) => Arg::Bytes(b.to_vec()),
                    })
                    .collect(),
            ),
        )
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        Self::record(&mut self.state.lock().unwrap(), Event::Sync(stream))
    }
    fn default_stream(&self) -> u64 {
        77
    }
    fn kernel(&self, _: &str, symbol: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(match symbol {
            "transpose_u8" if self.host_shared => 0,
            "transpose_u8" => 101,
            "moe_transpose_u8_batched" => 102,
            _ => 103,
        }))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        &self.cache
    }
    fn memset(&self, _: DevicePtr, _: u8, _: usize) -> Result<()> {
        bail!("unexpected memset")
    }
    fn memset_async(&self, _: DevicePtr, _: u8, _: usize, _: u64) -> Result<()> {
        bail!("unexpected memset")
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(128usize << 30)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(120usize << 30)
    }
    fn sm_count(&self) -> Result<u32> {
        Ok(48)
    }
}
