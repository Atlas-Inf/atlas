// SPDX-License-Identifier: AGPL-3.0-only
//! Sparse allocation authority and table bytes only; no numerical GPU emulation.
use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
use std::collections::{BTreeMap, HashMap};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Arg {
    Ptr(DevicePtr),
    Bytes(Vec<u8>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Lookup(String, String),
    Alloc(DevicePtr, usize),
    H2d(DevicePtr, usize),
    Read(DevicePtr, usize, u64),
    Copy(DevicePtr, DevicePtr, usize, u64),
    Sync(u64),
    Free(DevicePtr),
    Memset(DevicePtr, usize),
    Launch(u64, [u32; 3], [u32; 3], u32, u64, Vec<Arg>),
}
pub(super) struct Gpu {
    cache: spark_runtime::op_cache::OpCache,
    pub events: Arc<Mutex<Vec<Event>>>,
    memory: Mutex<BTreeMap<u64, (usize, Vec<u8>)>>,
    pub scalars: Mutex<HashMap<u64, u32>>,
    next: Mutex<u64>,
    pub capture: AtomicBool,
    pub fail: AtomicUsize,
    pub lookup_failure: AtomicUsize,
    pub lookup_zero: AtomicBool,
    /// Kernel names the target lacks.
    pub missing: Mutex<Vec<String>>,
    lookups: AtomicUsize,
}
impl Gpu {
    pub fn new() -> Self {
        Self {
            cache: spark_runtime::op_cache::OpCache::new(),
            events: Arc::new(Mutex::new(Vec::new())),
            memory: Mutex::new(BTreeMap::new()),
            scalars: Mutex::new(HashMap::new()),
            next: Mutex::new(0x8000_0000_0000),
            capture: AtomicBool::new(false),
            fail: AtomicUsize::new(usize::MAX),
            lookup_failure: AtomicUsize::new(usize::MAX),
            lookup_zero: AtomicBool::new(false),
            missing: Mutex::new(Vec::new()),
            lookups: AtomicUsize::new(0),
        }
    }
    fn record(&self, event: Event) -> Result<()> {
        let mut e = self.events.lock().unwrap();
        e.push(event);
        ensure!(
            e.len() != self.fail.load(Ordering::Relaxed),
            "injected op {}",
            e.len()
        );
        Ok(())
    }
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
        self.fail.store(usize::MAX, Ordering::Relaxed);
        self.lookups.store(0, Ordering::Relaxed);
    }
    pub fn trace(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }
    pub fn change(&self, ptr: DevicePtr, offset: usize, bytes: &[u8]) {
        self.memory.lock().unwrap().get_mut(&ptr.0).unwrap().1[offset..offset + bytes.len()]
            .copy_from_slice(bytes);
    }
    pub fn allocation_count(&self) -> usize {
        self.memory.lock().unwrap().len()
    }
    pub fn next_allocation(&self, ptr: DevicePtr) {
        *self.next.lock().unwrap() = ptr.0;
    }
}
impl GpuBackend for Gpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        let mut next = self.next.lock().unwrap();
        let p = DevicePtr(*next);
        self.record(Event::Alloc(p, n))?;
        *next += n.div_ceil(256) as u64 * 256 + 256;
        self.memory.lock().unwrap().insert(p.0, (n, Vec::new()));
        Ok(p)
    }
    fn alloc_managed(&self, _: usize) -> Result<DevicePtr> {
        bail!("managed allocation")
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        if p.is_null() {
            return Ok(());
        }
        ensure!(
            self.memory.lock().unwrap().remove(&p.0).is_some(),
            "nonowned free {p:?}"
        );
        self.record(Event::Free(p))
    }
    fn copy_h2d(&self, b: &[u8], p: DevicePtr) -> Result<()> {
        self.record(Event::H2d(p, b.len()))?;
        let mut m = self.memory.lock().unwrap();
        let (capacity, data) = m
            .get_mut(&p.0)
            .ok_or_else(|| anyhow::anyhow!("nonowned H2D"))?;
        ensure!(
            b.len() <= *capacity && b.len() <= 65536,
            "bounded fixture upload"
        );
        *data = b.to_vec();
        Ok(())
    }
    fn copy_d2h(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        bail!("explicit stream required")
    }
    fn copy_d2h_on_stream(&self, p: DevicePtr, b: &mut [u8], s: u64) -> Result<()> {
        self.record(Event::Read(p, b.len(), s))?;
        if let Some(bits) = self.scalars.lock().unwrap().get(&p.0) {
            ensure!(b.len() == 4, "scalar read");
            b.copy_from_slice(&bits.to_le_bytes());
            return Ok(());
        }
        let m = self.memory.lock().unwrap();
        let (cap, data) = m
            .get(&p.0)
            .ok_or_else(|| anyhow::anyhow!("nonowned read"))?;
        ensure!(
            b.len() <= *cap && b.len() <= data.len(),
            "uninitialized/out-of-bounds read"
        );
        b.copy_from_slice(&data[..b.len()]);
        Ok(())
    }
    fn copy_d2d(&self, a: DevicePtr, b: DevicePtr, n: usize) -> Result<()> {
        self.copy_d2d_async(a, b, n, 0)
    }
    fn copy_d2d_async(&self, a: DevicePtr, b: DevicePtr, n: usize, s: u64) -> Result<()> {
        self.record(Event::Copy(a, b, n, s))
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
        bail!("typed ABI required")
    }
    fn launch_typed(
        &self,
        k: KernelHandle,
        g: [u32; 3],
        b: [u32; 3],
        m: u32,
        s: u64,
        a: &[KernelArg<'_>],
    ) -> Result<()> {
        self.record(Event::Launch(
            k.0,
            g,
            b,
            m,
            s,
            a.iter()
                .map(|v| match v {
                    KernelArg::Buffer(p) => Arg::Ptr(*p),
                    KernelArg::Bytes(b) => Arg::Bytes(b.to_vec()),
                })
                .collect(),
        ))
    }
    fn kernel(&self, m: &str, n: &str) -> Result<KernelHandle> {
        self.record(Event::Lookup(m.into(), n.into()))?;
        let count = self.lookups.fetch_add(1, Ordering::Relaxed) + 1;
        if count == self.lookup_failure.load(Ordering::Relaxed) {
            if self.lookup_zero.load(Ordering::Relaxed) {
                return Ok(KernelHandle(0));
            }
            bail!("injected lookup");
        }
        ensure!(
            !self.missing.lock().unwrap().iter().any(|name| name == n),
            "no kernel {n}"
        );
        let id = m
            .bytes()
            .chain(n.bytes())
            .fold(17u64, |v, b| v.wrapping_mul(131).wrapping_add(u64::from(b)));
        Ok(KernelHandle(id))
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.record(Event::Sync(s))
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn stream_is_capturing(&self, _: u64) -> bool {
        self.capture.load(Ordering::Relaxed)
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        &self.cache
    }
    fn memset(&self, p: DevicePtr, _: u8, n: usize) -> Result<()> {
        self.record(Event::Memset(p, n))
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, _: u64) -> Result<()> {
        self.memset(p, v, n)
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
