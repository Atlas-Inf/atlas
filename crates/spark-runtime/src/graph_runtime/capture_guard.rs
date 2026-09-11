// SPDX-License-Identifier: AGPL-3.0-only

//! RAII guard that aborts an in-progress capture if the capturing body returns
//! early. Split out of `graph_runtime.rs` for the 500-LoC cap.

use crate::gpu::{GpuBackend, GraphHandle};

pub(super) struct CaptureGuard<'a> {
    backend: &'a dyn GpuBackend,
    stream: u64,
    active: bool,
}

impl<'a> CaptureGuard<'a> {
    pub(super) fn begin(backend: &'a dyn GpuBackend, stream: u64) -> anyhow::Result<Self> {
        backend.begin_capture(stream)?;
        Ok(Self {
            backend,
            stream,
            active: true,
        })
    }

    pub(super) fn finish(
        mut self,
        dot_path: Option<&std::path::Path>,
    ) -> anyhow::Result<GraphHandle> {
        match self.backend.end_capture_with_dot(self.stream, dot_path) {
            Ok(handle) => {
                self.active = false;
                Ok(handle)
            }
            Err(error) => Err(error),
        }
    }
}

impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.backend.abort_capture_if_active(self.stream);
        }
    }
}
