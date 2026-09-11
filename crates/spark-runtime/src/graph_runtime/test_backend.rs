// SPDX-License-Identifier: AGPL-3.0-only

use crate::gpu::mock::MockGpuBackend;
use crate::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};

pub(super) struct TestBackend {
    inner: MockGpuBackend,
    pub(super) graph: Mutex<TestGraphState>,
    op_cache: crate::op_cache::OpCache,
}

pub(super) struct TestGraphState {
    next_graph: u64,
    next_event: u64,
    captures: HashSet<u64>,
    pub(super) begin_calls: usize,
    pub(super) abort_calls: usize,
    launches: Vec<(u64, u64)>,
    destroyed_graphs: Vec<u64>,
    events: HashMap<u64, bool>,
    fail_begin: bool,
    fail_end: bool,
    fail_launch: bool,
}

impl TestBackend {
    pub(super) fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            graph: Mutex::new(TestGraphState {
                next_graph: 1,
                next_event: 1,
                captures: HashSet::new(),
                begin_calls: 0,
                abort_calls: 0,
                launches: Vec::new(),
                destroyed_graphs: Vec::new(),
                events: HashMap::new(),
                fail_begin: false,
                fail_end: false,
                fail_launch: false,
            }),
            op_cache: crate::op_cache::OpCache::new(),
        }
    }

    pub(super) fn set_fail_begin(&self, fail: bool) {
        self.graph.lock().fail_begin = fail;
    }

    pub(super) fn set_fail_launch(&self, fail: bool) {
        self.graph.lock().fail_launch = fail;
    }

    pub(super) fn complete_events(&self) {
        for complete in self.graph.lock().events.values_mut() {
            *complete = true;
        }
    }

    pub(super) fn destroyed_graphs(&self) -> Vec<u64> {
        self.graph.lock().destroyed_graphs.clone()
    }
}

impl GpuBackend for TestBackend {
    fn alloc(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
        self.inner.alloc(bytes)
    }

    fn alloc_managed(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> anyhow::Result<()> {
        self.inner.free(ptr)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> anyhow::Result<()> {
        self.inner.copy_h2d(src, dst)
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> anyhow::Result<()> {
        self.inner.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> anyhow::Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> anyhow::Result<()> {
        self.inner
            .launch(func, grid, block, shared_mem, stream, params)
    }

    fn synchronize(&self, stream: u64) -> anyhow::Result<()> {
        self.inner.synchronize(stream)?;
        self.complete_events();
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        1
    }

    fn kernel(&self, module: &str, function: &str) -> anyhow::Result<KernelHandle> {
        self.inner.kernel(module, function)
    }

    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn begin_capture(&self, stream: u64) -> anyhow::Result<()> {
        let mut state = self.graph.lock();
        state.begin_calls += 1;
        if state.fail_begin {
            anyhow::bail!("injected begin failure");
        }
        if !state.captures.insert(stream) {
            anyhow::bail!("nested capture on stream {stream}");
        }
        Ok(())
    }

    fn end_capture(&self, stream: u64) -> anyhow::Result<GraphHandle> {
        let mut state = self.graph.lock();
        if !state.captures.remove(&stream) {
            anyhow::bail!("stream {stream} is not capturing");
        }
        if state.fail_end {
            anyhow::bail!("injected end failure");
        }
        let handle = GraphHandle(state.next_graph);
        state.next_graph += 1;
        Ok(handle)
    }

    fn abort_capture_if_active(&self, stream: u64) {
        let mut state = self.graph.lock();
        if state.captures.remove(&stream) {
            state.abort_calls += 1;
        }
    }

    fn launch_graph(&self, graph: GraphHandle, stream: u64) -> anyhow::Result<()> {
        let mut state = self.graph.lock();
        if state.fail_launch {
            anyhow::bail!("injected launch failure");
        }
        state.launches.push((graph.0, stream));
        Ok(())
    }

    fn destroy_graph(&self, graph: GraphHandle) -> anyhow::Result<()> {
        self.graph.lock().destroyed_graphs.push(graph.0);
        Ok(())
    }

    fn create_event(&self) -> anyhow::Result<u64> {
        let mut state = self.graph.lock();
        let event = state.next_event;
        state.next_event += 1;
        state.events.insert(event, false);
        Ok(event)
    }

    fn record_event(&self, event: u64, _stream: u64) -> anyhow::Result<()> {
        if !self.graph.lock().events.contains_key(&event) {
            anyhow::bail!("unknown event {event}");
        }
        Ok(())
    }

    fn event_query(&self, event: u64) -> anyhow::Result<bool> {
        self.graph
            .lock()
            .events
            .get(&event)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("unknown event {event}"))
    }

    fn destroy_event(&self, event: u64) -> anyhow::Result<()> {
        self.graph.lock().events.remove(&event);
        Ok(())
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> anyhow::Result<()> {
        self.inner.memset(ptr, value, bytes)
    }

    fn memset_async(
        &self,
        ptr: DevicePtr,
        value: u8,
        bytes: usize,
        stream: u64,
    ) -> anyhow::Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }

    fn total_memory(&self) -> anyhow::Result<usize> {
        self.inner.total_memory()
    }

    fn free_memory(&self) -> anyhow::Result<usize> {
        self.inner.free_memory()
    }

    fn sm_count(&self) -> anyhow::Result<u32> {
        self.inner.sm_count()
    }
}
