// SPDX-License-Identifier: AGPL-3.0-only

//! `impl GpuBackend for AtlasCudaBackend` — production CUDA backend trait body.
//!
//! ## Safety contract for the `unsafe { cu*(...) }` calls below
//!
//! Every unsafe block in this file wraps a single CUDA Driver API call.
//! The invariants the driver requires are uniform:
//!
//! - **Context bound**: a CUDA primary context for the device is current
//!   on the calling thread. `AtlasCudaBackend::new` binds it once via
//!   `cuCtxSetCurrent`, and we never run on a thread that hasn't been
//!   bound.
//! - **Pointer provenance**: every `DevicePtr` came from a prior
//!   successful `cuMemAlloc_v2` / `cuMemAllocHost_v2` /
//!   `cuMemAllocManaged` and has not yet been freed. `DevicePtr(0)` is
//!   treated as "not allocated" by callers.
//! - **Sizes in bytes**: every `bytes: usize` argument is the exact
//!   byte count of the allocation (callers compute it from typed
//!   sizes); the driver does no bounds-checking.
//! - **Stream / event lifetimes**: handles are owned by `Self` and
//!   freed in `Drop` after `cuStreamSynchronize`, so they outlive every
//!   in-flight launch that captured them.
//! - **`extern "C"` ABI**: matches the cudarc-generated bindings used
//!   in `super::*` imports; see `cudarc` for the full ABI surface.
//!
//! Per-site `// SAFETY:` comments are omitted because the contract is
//! identical for every call. Anything that *deviates* from this
//! contract gets a per-site `// SAFETY:` comment explaining the
//! exception.

use std::ffi::c_void;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use atlas_core::registry::{RawCudaFunc, cuda_error_text};
use cudarc::driver::LaunchConfig;

use super::{
    AtlasCudaBackend, cuMemAlloc_v2, cuMemAllocManaged, cuMemFree_v2, cuMemGetInfo_v2,
    cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2, cuStreamSynchronize,
};
use crate::gpu::{
    ConditionalGraphTemplate, ConditionalNodeKind, DevicePtr, GpuBackend, GraphHandle, KernelHandle,
};

/// D2H call counter + one-shot caller identification
/// (`ATLAS_D2H_TRACE=<N>`: log a backtrace on the Nth call, and the running
/// count on every 10000th).
///
/// Every `copy_d2h*` below pairs its async copy with a `cuStreamSynchronize`,
/// so each call BLOCKS the host until the GPU drains. An nsys trace of a 1K
/// Laguna prefill counted 32,343 D2H + 32,533 syncs inside the prefill span,
/// accounting for 212.8 ms of 306 ms of GPU starvation (58% idle). This exists
/// to name whoever is issuing them.
static D2H_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn d2h_trace_tick() {
    use std::sync::atomic::Ordering;
    let n = D2H_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let Ok(target) = std::env::var("ATLAS_D2H_TRACE") else {
        return;
    };
    let target: u64 = target.parse().unwrap_or(0);
    if target != 0 && n == target {
        tracing::warn!(
            "ATLAS_D2H_TRACE: call #{n} backtrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
    if n.is_multiple_of(10_000) {
        tracing::warn!("ATLAS_D2H_TRACE: {n} D2H copies so far (each forces a stream sync)");
    }
}

/// Enqueue an H2D copy on `stream` and return without waiting. Shared by both
/// async H2D entry points so the two differ ONLY in the ordering they add
/// afterwards, never in the copy itself.
fn h2d_enqueue(src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
    let status =
        unsafe { cuMemcpyHtoDAsync_v2(dst.0, src.as_ptr() as *const c_void, src.len(), stream) };
    if status != 0 {
        bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
    }
    Ok(())
}

/// Say once, loudly, that a page-locked buffer reached the transient H2D path.
///
/// This is the tripwire the whole `pinned_hosts` registry exists to arm. It is a
/// warning and not a `bail!` because the copy is still CORRECT — the sync above
/// restores the guarantee — but it is a real, silent latency regression, and the
/// call site almost certainly wants `copy_h2d_async_retained` instead.
fn warn_pinned_transient_source() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "copy_h2d_async was handed a PAGE-LOCKED source. That copy is genuinely \
             asynchronous, so the promise that the caller may drop the buffer on return \
             is now being paid for with a cuStreamSynchronize on every such call. If the \
             source outlives the next sync, switch the call site to \
             copy_h2d_async_retained; if it does not, this sync is what keeps it from \
             being a use-after-free."
        );
    });
}

impl GpuBackend for AtlasCudaBackend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.ensure_capture_safe("device allocation")?;
        let mut dptr: u64 = 0;
        let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
        if status != 0 {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
            bail!(
                "cuMemAlloc_v2 failed: status {status}, requested {bytes} bytes \
                 (device reports {:.1} MB free / {:.1} GB total)",
                free as f64 / (1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        self.record_alloc(DevicePtr(dptr));
        Ok(DevicePtr(dptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.ensure_capture_safe("managed device allocation")?;
        let mut dptr: u64 = 0;
        const CU_MEM_ATTACH_GLOBAL: u32 = 0x1;
        let status = unsafe { cuMemAllocManaged(&mut dptr, bytes, CU_MEM_ATTACH_GLOBAL) };
        if status != 0 {
            bail!(
                "cuMemAllocManaged failed: status {status}, requested {bytes} bytes. \
                 Check system swap space: swapon --show"
            );
        }
        self.record_alloc(DevicePtr(dptr));
        Ok(DevicePtr(dptr))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        self.ensure_capture_safe("device free")?;
        // Off the ledger BEFORE the free: an entry that survives a successful
        // free would be double-freed at teardown. On a real failure, restore
        // ownership to the backend ledger so `sweep_unreleased` is the final
        // cleanup backstop even if the sequence-local state is dropped.
        self.forget_alloc(ptr);
        let status = unsafe { cuMemFree_v2(ptr.0) };
        // A context that is already being destroyed reports every free as
        // failing, and at process exit that is the normal case, not an error:
        // the driver has reclaimed the allocation by definition. Two other
        // free paths in this crate already consult `is_teardown_noop`; this one
        // did not, so wiring `Model::teardown` into shutdown turned a benign
        // status 4 into `ERROR model teardown reported a failure` on every
        // clean exit — the exact species of false alarm this work set out to
        // remove.
        if status != 0 && !atlas_core::registry::is_teardown_noop(status) {
            self.record_alloc(ptr);
            bail!(
                "cuMemFree_v2 failed: status {status}, ptr {ptr}; \
                 pointer restored to backend cleanup ledger"
            );
        }
        Ok(())
    }

    fn sweep_unreleased(&self) -> usize {
        AtlasCudaBackend::sweep_unreleased(self)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.ensure_capture_safe("synchronous host-to-device copy")?;
        AtlasCudaBackend::copy_h2d_impl(self, src, dst)
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.ensure_capture_safe("synchronous device-to-host copy")?;
        d2h_trace_tick();
        AtlasCudaBackend::copy_d2h_impl(self, src, dst)
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        self.ensure_capture_safe("blocking device-to-host copy")?;
        d2h_trace_tick();
        AtlasCudaBackend::copy_d2h_on_stream_impl(self, src, dst, stream)
    }

    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // Deliberately NO cuStreamSynchronize — that is the entire point.
        // `copy_d2h`/`copy_d2h_on_stream` drain the stream inside every call,
        // so a multi-chunk gather pays one full drain per chunk (the SSM spill's
        // 60 chunks × 66 MB measured ~400 ms = ~165 MB/s, vs ~28 ms for the
        // async H2D scatter of the same bytes). The caller MUST issue exactly
        // one `synchronize(stream)` before touching `dst`.
        d2h_trace_tick();
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 (async) failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        AtlasCudaBackend::copy_d2d_impl(self, src, dst, bytes)
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let raw_func = RawCudaFunc(func.0 as *mut c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared_mem,
        };
        let registry = self.registry();
        unsafe { registry.launch_on_stream(raw_func, cfg, stream, params) }.map_err(|e| {
            // A launch failure may have destroyed the CUDA context. Probe and
            // latch before the error is flattened into a string and bubbled —
            // see `fault_probe`. This does not change control flow: the caller
            // still receives its error either way.
            super::fault_probe::note_failure("kernel launch", &e.to_string());
            anyhow::anyhow!("Kernel launch failed: {e}")
        })
    }

    fn stream_is_capturing(&self, stream: u64) -> bool {
        // SCALE's libcuda does not export cuStreamIsCapturing; report
        // not-capturing there (gfx1151 telemetry taps then sample eagerly —
        // acceptable for a default-off measurement knob).
        #[cfg(atlas_scale)]
        {
            let _ = stream;
            false
        }
        #[cfg(not(atlas_scale))]
        {
            let mut status: u32 = 0;
            // CU_STREAM_CAPTURE_STATUS_NONE = 0; treat query failure as
            // capturing (conservative: the tap skips its sample).
            let rc = unsafe { super::cuStreamIsCapturing(stream, &mut status) };
            rc != 0 || status != 0
        }
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.ensure_capture_safe("host stream synchronization")?;
        let status = unsafe { cuStreamSynchronize(stream) };
        if status != 0 {
            bail!("cuStreamSynchronize failed: {}", cuda_error_text(status));
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        self.default_stream
    }

    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn debug_sync_kernels(&self) -> bool {
        AtlasCudaBackend::debug_sync_kernels(self)
    }

    fn kernel_registry(&self) -> Option<std::sync::Arc<atlas_core::registry::AtlasRegistry>> {
        Some(self.registry().clone())
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        self.ensure_capture_safe("dynamic kernel lookup")?;
        // The DISPATCH SITE, not this line: `#[track_caller]` here and on the
        // trait declaration carries the `.kernel(…)` / `try_kernel(…)` caller's
        // `file:line` through, which is the only part of an unresolved-lookup
        // report an operator can act on.
        let site = std::panic::Location::caller();
        // Ephemeral OnceLock — no cross-call caching, but kernel() is only
        // called at model init time. Layers store the returned KernelHandle.
        let cache: OnceLock<RawCudaFunc> = OnceLock::new();
        let registry = self.registry();
        match registry.raw_function_cached(&cache, module, func_name) {
            Ok(raw) => {
                crate::kernel_audit::record(module, func_name, true, site);
                Ok(KernelHandle(raw.0 as u64))
            }
            Err(e) => {
                // Optional kernels (try_kernel) land here and fall back silently;
                // the audit makes that visible in the startup kernel table.
                crate::kernel_audit::record(module, func_name, false, site);
                Err(anyhow::anyhow!("Kernel lookup {module}::{func_name}: {e}"))
            }
        }
    }

    fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        self.ensure_capture_safe("transient host-to-device copy")?;
        h2d_enqueue(src, dst, stream)?;
        // The trait promises the caller may drop `src` right now. From PAGEABLE
        // memory the driver already made that true by staging the bytes before
        // returning. From PAGE-LOCKED memory it did not — the DMA engine reads
        // these pages after the enqueue — so buy the same guarantee with an
        // explicit wait rather than let ~90 call sites that drop their source
        // immediately turn into use-after-frees the day a buffer gets pinned.
        //
        // Costs nothing on the path everything takes today: no Atlas call site
        // reaches here with a pinned source (the ones that own pinned staging
        // use `copy_h2d_async_retained`), so `is_pinned` is a lock-free-ish read
        // of a three-entry table that says "no".
        if crate::pinned_hosts::is_pinned(src) {
            warn_pinned_transient_source();
            let sync = unsafe { cuStreamSynchronize(stream) };
            if sync != 0 {
                bail!(
                    "cuStreamSynchronize after pinned-source H2D failed: {}",
                    cuda_error_text(sync)
                );
            }
        }
        Ok(())
    }

    fn copy_h2d_async_retained(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        // The caller has promised `src` outlives the next sync on `stream`, so
        // no implicit ordering is added — that is the whole reason this variant
        // exists (a 60-chunk pinned scatter must not pay 60 stream drains).
        h2d_enqueue(src, dst, stream)
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, stream) };
        if status != 0 {
            // See copy_d2d_impl: on 901 the backtrace names the reporter,
            // which bounds where the capture-poisoning op ran.
            tracing::error!(
                "copy_d2d_async failed (status {status}) at:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
            bail!("cuMemcpyDtoDAsync_v2 (copy_d2d_async) failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2d_2d_async(
        &self,
        src: DevicePtr,
        src_pitch: usize,
        dst: DevicePtr,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
        stream: u64,
    ) -> Result<()> {
        // One pitched copy (cudaMemcpyDeviceToDevice = 3) on the caller's stream,
        // replacing a per-row copy_d2d_async loop. cudart is linked (cutlass/
        // flashinfer use the runtime API); a CUstream handle is a valid
        // cudaStream_t.
        unsafe extern "C" {
            fn cudaMemcpy2DAsync(
                dst: *mut c_void,
                dpitch: usize,
                src: *const c_void,
                spitch: usize,
                width: usize,
                height: usize,
                kind: i32,
                stream: u64,
            ) -> i32;
        }
        let status = unsafe {
            cudaMemcpy2DAsync(
                dst.0 as *mut c_void,
                dst_pitch,
                src.0 as *const c_void,
                src_pitch,
                width_bytes,
                height,
                3,
                stream,
            )
        };
        if status != 0 {
            bail!("cudaMemcpy2DAsync failed: status {status}");
        }
        Ok(())
    }

    fn graph_capabilities(&self) -> crate::graph_runtime::GraphCapabilities {
        let conditional_nodes = self.conditional_nodes_supported_cu();
        crate::graph_runtime::GraphCapabilities {
            basic_graphs: true,
            debug_dot: cfg!(not(atlas_scale)),
            graph_upload: false,
            conditional_nodes,
            while_nodes: conditional_nodes,
            native_serialization: false,
        }
    }

    fn graph_environment(&self) -> Option<crate::graph_runtime::GraphEnvironment> {
        match self.graph_environment_cu() {
            Ok(environment) => Some(environment),
            Err(error) => {
                tracing::warn!("CUDA graph environment query failed: {error:#}");
                None
            }
        }
    }

    fn create_conditional_graph(
        &self,
        kind: ConditionalNodeKind,
        predicate: DevicePtr,
        setter: KernelHandle,
        body_count: usize,
    ) -> Result<ConditionalGraphTemplate> {
        self.create_conditional_graph_cu(kind, predicate, setter, body_count)
    }
    fn begin_conditional_branch(
        &self,
        template: &ConditionalGraphTemplate,
        branch: usize,
        stream: u64,
    ) -> Result<()> {
        if self.graph_fault(super::graph_faults::GraphFaultPoint::Capture) {
            bail!("injected CUDA conditional branch capture failure");
        }
        self.begin_capture_guard()?;
        let result = self.begin_conditional_branch_cu(template, branch, stream);
        if result.is_err() {
            self.finish_capture_guard();
        }
        result
    }
    fn end_conditional_branch(
        &self,
        template: &ConditionalGraphTemplate,
        branch: usize,
        stream: u64,
    ) -> Result<()> {
        if self.graph_fault(super::graph_faults::GraphFaultPoint::Instantiate) {
            self.abort_capture_if_active_cu(stream);
            self.finish_capture_guard();
            bail!("injected CUDA conditional branch capture failure");
        }
        let result = self.end_conditional_branch_cu(template, branch, stream);
        self.finish_capture_guard();
        result
    }
    fn instantiate_conditional_graph(
        &self,
        template: &ConditionalGraphTemplate,
        dot_path: Option<&std::path::Path>,
    ) -> Result<GraphHandle> {
        self.instantiate_conditional_graph_cu(template, dot_path)
    }
    fn destroy_conditional_graph_template(&self, template: &ConditionalGraphTemplate) {
        self.destroy_conditional_graph_template_cu(template)
    }

    fn begin_capture(&self, stream: u64) -> Result<()> {
        if self.graph_fault(super::graph_faults::GraphFaultPoint::Capture) {
            bail!("injected CUDA graph capture failure");
        }
        self.begin_capture_guard()?;
        let result = self.begin_capture_cu(stream);
        if result.is_err() {
            self.finish_capture_guard();
        }
        result
    }
    fn end_capture(&self, stream: u64) -> Result<GraphHandle> {
        if self.graph_fault(super::graph_faults::GraphFaultPoint::Instantiate) {
            self.abort_capture_if_active_cu(stream);
            self.finish_capture_guard();
            bail!("injected CUDA graph instantiation failure");
        }
        let result = self.end_capture_cu(stream);
        self.finish_capture_guard();
        result
    }
    fn end_capture_with_dot(
        &self,
        stream: u64,
        dot_path: Option<&std::path::Path>,
    ) -> Result<GraphHandle> {
        if self.graph_fault(super::graph_faults::GraphFaultPoint::Instantiate) {
            self.abort_capture_if_active_cu(stream);
            self.finish_capture_guard();
            bail!("injected CUDA graph instantiation failure");
        }
        let result = self.end_capture_with_dot_cu(stream, dot_path);
        self.finish_capture_guard();
        result
    }

    fn abort_capture_if_active(&self, stream: u64) {
        self.abort_capture_if_active_cu(stream);
        self.finish_capture_guard();
    }

    fn launch_graph(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        self.ensure_capture_safe("graph launch")?;
        if self.graph_fault(super::graph_faults::GraphFaultPoint::Replay) {
            bail!("injected CUDA graph replay failure before submission");
        }
        self.launch_graph_cu(graph, stream)
    }
    fn destroy_graph(&self, graph: GraphHandle) -> Result<()> {
        self.ensure_capture_safe("graph destruction")?;
        self.destroy_graph_cu(graph)
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.ensure_capture_safe("synchronous memset")?;
        self.memset_cu(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.memset_async_cu(ptr, value, bytes, stream)
    }
    fn total_memory(&self) -> Result<usize> {
        self.total_memory_cu()
    }
    fn free_memory(&self) -> Result<usize> {
        self.free_memory_cu()
    }
    fn live_alloc_count(&self) -> usize {
        self.live_alloc_len()
    }
    fn sm_count(&self) -> Result<u32> {
        self.sm_count_cu()
    }
    fn create_stream(&self) -> Result<u64> {
        self.ensure_capture_safe("stream creation")?;
        self.create_stream_cu()
    }
    fn bind_to_thread(&self) -> Result<()> {
        self.bind_to_thread_cu()
    }
    fn create_event(&self) -> Result<u64> {
        self.ensure_capture_safe("event creation")?;
        if self.graph_fault(super::graph_faults::GraphFaultPoint::Event) {
            bail!("injected CUDA graph retirement-event failure");
        }
        self.create_event_cu()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_cu(event, stream)
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_cu(stream, event)
    }
    fn event_synchronize(&self, event: u64) -> Result<()> {
        self.ensure_capture_safe("host event synchronization")?;
        self.event_synchronize_cu(event)
    }
    fn event_query(&self, event: u64) -> Result<bool> {
        self.ensure_capture_safe("host event query")?;
        self.event_query_cu(event)
    }
    fn destroy_event(&self, event: u64) -> Result<()> {
        self.ensure_capture_safe("event destruction")?;
        self.destroy_event_cu(event)
    }
    fn host_ptr_to_device(&self, host: *mut u8) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        let status =
            unsafe { super::cuMemHostGetDevicePointer_v2(&mut dptr, host as *mut c_void, 0) };
        if status != 0 {
            bail!("cuMemHostGetDevicePointer_v2 failed: status {status}");
        }
        Ok(DevicePtr(dptr))
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        self.ensure_capture_safe("pinned host allocation")?;
        self.alloc_host_pinned_cu(bytes)
    }
    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        self.ensure_capture_safe("pinned host free")?;
        self.free_host_pinned_cu(ptr, _bytes)
    }
}
