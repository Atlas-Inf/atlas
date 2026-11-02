// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_DET_DUMP=<dir>`: the bytes of every traced tensor, so runs whose
//! bytes cannot match (another TP size, say) compare by value
//! (`scripts/dev/det_dump_compare.py`).

use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::OnceLock;

use super::At;

/// `ATLAS_GLM_DET_DUMP`: the directory traced tensors are written to.
fn dump_dir() -> Option<&'static std::path::Path> {
    static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("ATLAS_GLM_DET_DUMP").map(Into::into))
        .as_deref()
}

/// Write `bytes` at `ptr` (read after `stream`) as `stage` of `at`, rows from
/// `rows.0`, into the dump directory. Best effort: a debugging aid.
pub(super) fn dump(
    at: At,
    gpu: &dyn GpuBackend,
    stream: u64,
    stage: &str,
    ptr: DevicePtr,
    rows: (usize, usize),
    bytes: usize,
) {
    let Some(dir) = dump_dir() else { return };
    let mut host = vec![0u8; bytes];
    if gpu.synchronize(stream).is_err() || gpu.copy_d2h(ptr, &mut host).is_err() {
        return;
    }
    let name = format!(
        "r{}_q{}_c{}_L{}_{stage}_r{}_n{}.bin",
        at.rank, at.request, at.chunk_start, at.layer, rows.0, rows.1
    );
    let _ = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(dir.join(name), host));
}
