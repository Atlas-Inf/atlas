// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_N35_L0_DUMP=<dir>` — one-shot stage dump of the layer-0 Mamba-2
//! decode step. Used on Windows to bisect the L0 hidden-state divergence
//! vs GB10: each stage's device bytes land in `<dir>/<stage>.bin` plus a
//! `manifest.json` describing them. Zero-cost when the env is unset.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(crate) struct L0Dump {
    dir: PathBuf,
    manifest: Vec<String>,
    armed: bool,
    stream: u64,
}

static FIRED: AtomicBool = AtomicBool::new(false);
static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

fn dump_dir() -> Option<&'static Path> {
    DIR.get_or_init(|| std::env::var_os("ATLAS_N35_L0_DUMP").map(PathBuf::from))
        .as_deref()
}

static LAYER: OnceLock<usize> = OnceLock::new();

/// Layer index to dump: `ATLAS_N35_STAGE_DUMP_LAYER` when set, else 0
/// (the `ATLAS_N35_L0_DUMP` behaviour).
pub(crate) fn target_layer() -> usize {
    *LAYER.get_or_init(|| {
        std::env::var("ATLAS_N35_STAGE_DUMP_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Arm the dump when the target layer's decode runs. Returns `None` when
/// disabled, already fired, or when the first stage's readback fails (e.g.
/// the call is being recorded into a CUDA graph, where D2H copies are
/// illegal) — the latch is left unfired so the first EAGER call still dumps.
pub(crate) fn arm(
    gpu: &dyn GpuBackend,
    layer_idx: usize,
    hidden: DevicePtr,
    elems: usize,
    stream: u64,
) -> Option<L0Dump> {
    if layer_idx != target_layer() || FIRED.load(Ordering::SeqCst) {
        return None;
    }
    let dir = dump_dir()?;
    let _ = std::fs::create_dir_all(dir);
    let mut d = L0Dump {
        dir: dir.to_path_buf(),
        manifest: Vec::new(),
        armed: false,
        stream,
    };
    // Probe stage: the layer input. If it can't be read back this call is
    // not a real (eager) execution — stay unarmed and unfired.
    if !d.stage(gpu, "01_layer_input", hidden, elems, "bf16", &[elems]) {
        return None;
    }
    d.armed = true;
    FIRED.store(true, Ordering::SeqCst);
    Some(d)
}

impl L0Dump {
    /// Read `elems` elements at `ptr` back to host and write `<dir>/<name>.bin`.
    pub(crate) fn stage(
        &mut self,
        gpu: &dyn GpuBackend,
        name: &str,
        ptr: DevicePtr,
        elems: usize,
        dtype: &str,
        shape: &[usize],
    ) -> bool {
        let esz = match dtype {
            "bf16" => 2,
            "f32" | "u32" | "i32" => 4,
            _ => 1,
        };
        // Order the readback against the launch stream — under HIP the
        // D2H copy does not join non-default streams, so without this sync
        // prefill stages read buffers before the kernels land (all-zero
        // dumps on winbox).
        if gpu.synchronize(self.stream).is_err() {
            return false;
        }
        let mut buf = vec![0u8; elems * esz];
        if gpu.copy_d2h(ptr, &mut buf).is_err() {
            return false;
        }
        let path = self.dir.join(format!("{name}.bin"));
        if std::fs::write(&path, &buf).is_err() {
            return false;
        }
        self.manifest.push(format!(
            "{{\"name\":\"{name}\",\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"elems\":{elems}}}"
        ));
        true
    }

    /// Write manifest.json; called after the last stage.
    pub(crate) fn finish(self) {
        if !self.armed {
            return;
        }
        let m = format!("{{\"stages\":[{}]}}", self.manifest.join(","));
        let _ = std::fs::write(self.dir.join("manifest.json"), m);
    }
}
