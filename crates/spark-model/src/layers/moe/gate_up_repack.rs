// SPDX-License-Identifier: AGPL-3.0-only
//! Unpublished construction transaction. No loader caller or forward accessor.
//! Staged until the complete reader family exists; never enable via a flag here.
#![allow(dead_code)]
use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;
#[path = "gate_up_btile_arena.rs"]
mod arena;
#[cfg(test)]
#[path = "gate_up_btile_arena_tests.rs"]
mod arena_tests;
#[path = "gate_up_btile_binding.rs"]
mod binding;
#[cfg(test)]
#[path = "gate_up_btile_binding_tests.rs"]
mod binding_tests;
#[cfg(test)]
#[path = "forward_c2_tests.rs"]
mod c2_tests;
#[path = "gate_up_btile_call.rs"]
mod call;
#[path = "gate_up_btile_decode.rs"]
mod decode;
#[path = "gate_up_btile_grouped.rs"]
mod grouped;
#[path = "gate_up_btile_kernels.rs"]
mod kernels;
#[path = "gate_up_btile_load.rs"]
mod load;
#[path = "gate_up_btile_preflight.rs"]
mod preflight;
pub(in crate::layers::moe) use load::ConstructionToken;
#[path = "gate_up_btile_resident.rs"]
mod resident;
pub(in crate::layers::moe) use resident::Storage;
#[cfg(test)]
#[path = "gate_up_btile_compact_reader_tests.rs"]
mod compact_reader_tests;
#[cfg(test)]
#[path = "gate_up_btile_model_fixture.rs"]
mod model_fixture;
#[cfg(test)]
#[path = "gate_up_btile_prefill_tests.rs"]
mod prefill_tests;
#[cfg(test)]
#[path = "gate_up_btile_reader_tests.rs"]
mod reader_tests;
#[cfg(test)]
#[path = "gate_up_btile_resident_tests.rs"]
mod resident_tests;
#[cfg(test)]
#[path = "gate_up_btile_unsupported_tests.rs"]
mod unsupported_tests;
#[cfg(test)]
#[path = "gate_up_btile_validity_tests.rs"]
mod validity_tests;
#[cfg(test)]
pub(crate) use model_fixture::model_fixture;
#[cfg(test)]
#[path = "gate_up_btile_launch_tests.rs"]
mod launch_tests;
#[path = "gate_up_native_source.rs"]
mod native_source;
#[cfg(test)]
#[path = "qwen4exp_fast_tests.rs"]
mod qwen4exp_fast_tests;
#[cfg(test)]
#[path = "gate_up_btile_test_gpu.rs"]
mod recording;
#[cfg(test)]
#[path = "gate_up_btile_shared_tests.rs"]
mod shared_tests;
use native_source::{NativeGateUpLayer, PACKED_BYTES, Span};

pub(super) struct RepackWorkspace<'a> {
    gpu: &'a dyn GpuBackend,
    scratch: Option<Span>,
    stream: u64,
    poisoned: bool,
}
pub(super) struct UnpublishedBTileLayer<'s, 'g> {
    source: NativeGateUpLayer<'s, 'g>,
}
impl<'a> RepackWorkspace<'a> {
    pub(super) fn new(gpu: &'a dyn GpuBackend, stream: u64) -> Result<Self> {
        ensure!(
            !gpu.stream_is_capturing(stream),
            "repack workspace during capture"
        );
        let ptr = gpu.alloc(PACKED_BYTES)?;
        let mut result = Self {
            gpu,
            scratch: Some(Span {
                ptr,
                bytes: PACKED_BYTES,
            }),
            stream,
            poisoned: false,
        };
        if let Err(error) = Span::new(ptr, PACKED_BYTES, 16) {
            result.poisoned = true;
            return Err(error);
        }
        Ok(result)
    }
    fn repack<'s>(
        &mut self,
        source: NativeGateUpLayer<'s, 'a>,
        family: &kernels::KernelFamily<'a>,
    ) -> Result<UnpublishedBTileLayer<'s, 'a>> {
        ensure!(
            !self.poisoned,
            "poisoned repack workspace; abandon construction"
        );
        ensure!(
            std::ptr::addr_eq(self.gpu, source.gpu())
                && std::ptr::addr_eq(self.gpu, family.gpu)
                && self.stream == source.stream(),
            "repack backend/stream mismatch"
        );
        let [packed, transpose] = [family.handles[0], family.handles[1]];
        ensure!(
            !self.gpu.stream_is_capturing(self.stream),
            "repack during capture"
        );
        ensure!(
            packed.0 != 0 && transpose.0 != 0,
            "missing byte repack handle"
        );
        let scratch = self
            .scratch
            .ok_or_else(|| anyhow::anyhow!("closed repack workspace"))?;
        ensure!(
            scratch.bytes == PACKED_BYTES,
            "repack scratch capacity mismatch"
        );
        Span::new(scratch.ptr, scratch.bytes, 16)?;
        if !source.scratch_is_disjoint(scratch) {
            // A broken allocator returned an original tensor span. Do not
            // attempt to free that owner under the name of scratch cleanup.
            self.poisoned = true;
            self.scratch = None;
            anyhow::bail!(
                "scratch allocator/source overlap at {:?}; abandon construction; ownership uncertain",
                scratch.ptr
            );
        }
        let result = (|| -> Result<()> {
            for projection in source.projections() {
                self.gpu.copy_d2d_async(
                    projection.packed.ptr,
                    scratch.ptr,
                    PACKED_BYTES,
                    self.stream,
                )?;
                crate::layers::ops::moe_gate_up_repack::glm_native_to_btile(
                    self.gpu,
                    packed,
                    scratch.ptr,
                    projection.packed.ptr,
                    self.stream,
                )?;
                self.gpu.synchronize(self.stream)?;
                self.gpu.copy_d2d_async(
                    projection.scales.ptr,
                    scratch.ptr,
                    native_source::SCALE_BYTES,
                    self.stream,
                )?;
                crate::layers::ops::transpose_u8(
                    self.gpu,
                    transpose,
                    scratch.ptr,
                    projection.scales.ptr,
                    2048,
                    256,
                    self.stream,
                )?;
                self.gpu.synchronize(self.stream)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            // Even a failed enqueue can leave work in flight. Poison before
            // cleanup, and never reuse this workspace on any error path.
            self.poisoned = true;
            let cleanup = self.cleanup();
            let message = match cleanup {
                Ok(()) => "abandon partially converted construction; scratch cleaned".to_owned(),
                Err(cleanup) => format!(
                    "abandon partially converted construction; cleanup failure: {cleanup:#}"
                ),
            };
            return Err(error.context(message));
        }
        Ok(UnpublishedBTileLayer { source })
    }
    pub(super) fn close(mut self) -> Result<()> {
        self.cleanup()
    }
    fn cleanup(&mut self) -> Result<()> {
        if self.scratch.is_none() {
            return Ok(());
        }
        ensure!(
            !self.gpu.stream_is_capturing(self.stream),
            "cannot clean repack scratch during capture"
        );
        let span = self.scratch.take().expect("checked scratch presence");
        let sync = self.gpu.synchronize(self.stream);
        // CudaBackend removes its allocation ledger entry BEFORE cuMemFree.
        // A failed free is not recoverable by claiming a later ledger sweep.
        let free = self.gpu.free(span.ptr);
        if sync.is_err() || free.is_err() {
            self.poisoned = true;
        }
        match (sync, free) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error.context("repack cleanup synchronization failed")),
            (Ok(()), Err(error)) => Err(error.context(format!("scratch free failed at {:?}; CUDA context/process teardown may be required", span.ptr))),
            (Err(sync), Err(free)) => Err(sync.context(format!("scratch free failed at {:?}: {free:#}; CUDA context/process teardown may be required", span.ptr))),
        }
    }
}
impl Drop for RepackWorkspace<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.cleanup() {
            tracing::error!(error = %format!("{error:#}"), scratch = ?self.scratch.map(|s| s.ptr),
                "unpublished repack cleanup failed; do not resume construction");
        }
    }
}
#[cfg(test)]
#[path = "gate_up_repack_tests.rs"]
mod tests;
