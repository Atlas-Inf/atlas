// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_MOE_ROUTE_DUMP=<file>` (diagnostic): the expert ids of
//! every rows-pair MoE launch (`forward_rows.rs`), one line of `rows * top_k`
//! space-separated ids per launch, appended to `<file>`. The unique local
//! experts per layer per step -- what a C8 step's routed MoE streams -- come
//! from these; `scripts/dev/qwen4exp_moe_c8_bench.cu` replays them
//! (`ROUTES=<file> ROUTE_ROWS=<rows>`).
//!
//! The first `ATLAS_QWEN4EXP_MOE_ROUTE_DUMP_MAX` launches (default 384: eight
//! steps of 48 layers) also append a record to `<file>.bin` with the launch's
//! MoE input rows, for `scripts/dev/qwen4exp_moe_fidelity.cu`. Record, little
//! endian: u32 magic 0x31444D51 ("QMD1"), u32 layer (order of the layer's
//! first launch), u32 rows, u32 top_k, u32 hidden, then u32 ids
//! [rows * top_k], f32 routing weights [rows * top_k], BF16 input
//! [rows * hidden].
//!
//! The text lines stop after `ATLAS_QWEN4EXP_MOE_ROUTE_DUMP_LINES` launches
//! (default 49152: 1024 steps of 48 layers).
//!
//! PRIVACY: `<file>.bin` holds the requests' activations (the MoE input
//! rows), from which their text can be recovered in part, and the ids trace
//! them too. Local diagnosis only: never on a server taking other people's
//! requests, and delete the files after use. Neither file is ever reused:
//! one that already exists is refused (`create_new`), so each rank and run
//! needs its own path. A file that cannot be created or written turns the
//! dump off with one warning; the forward goes on.
//!
//! Synchronizes the stream per launch: a measurement run, not a timing one.
//! Launches recorded into a CUDA graph cannot be read at capture and are
//! skipped (run with the decode graphs off to see every step).

use std::io::Write;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

struct Dump {
    text: Option<std::fs::File>,
    bin: Option<std::fs::File>,
    layers: Vec<u64>,
    lines: usize,
    records: usize,
    /// A file failed: the dump is off for the rest of the process.
    off: bool,
}

static DUMP: Mutex<Dump> = Mutex::new(Dump {
    text: None,
    bin: None,
    layers: Vec::new(),
    lines: 0,
    records: 0,
    off: false,
});

/// `name` as a count, read once into `cell`; `default` when unset or bad.
fn cap(cell: &'static OnceLock<usize>, name: &str, default: usize) -> usize {
    *cell.get_or_init(|| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    })
}

/// Writes `bytes` to the file in `slot`, creating it at `path` first; never
/// an existing file.
fn write(slot: &mut Option<std::fs::File>, path: &str, bytes: &[u8]) -> Result<()> {
    if slot.is_none() {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("ATLAS_QWEN4EXP_MOE_ROUTE_DUMP: create {path}"))?;
        *slot = Some(f);
    }
    if let Some(f) = slot.as_mut() {
        f.write_all(bytes)
            .with_context(|| format!("ATLAS_QWEN4EXP_MOE_ROUTE_DUMP: write {path}"))?;
    }
    Ok(())
}

/// `r` of a file operation: an error turns the dump off, with one warning.
fn file_ok(d: &mut Dump, r: Result<()>) -> bool {
    if let Err(why) = r {
        tracing::warn!("{why:#}; the route dump is off");
        d.off = true;
    }
    !d.off
}

fn read(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(ptr, &mut v)?;
    Ok(v)
}

/// The routing of one rows-pair launch: `rows` rows of `top_k` ids at
/// `indices` and weights at `weights`, its `[rows, hidden]` BF16 `input`;
/// `layer` identifies the layer (its expert table address).
pub(super) struct Launch {
    pub indices: DevicePtr,
    pub weights: DevicePtr,
    pub input: DevicePtr,
    pub rows: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub layer: u64,
}

/// Appends `l` (after `stream`'s work) to `path` and, within the record cap,
/// to `path.bin`.
pub(super) fn append(gpu: &dyn GpuBackend, l: &Launch, path: &str, stream: u64) -> Result<()> {
    static LINES: OnceLock<usize> = OnceLock::new();
    static RECORDS: OnceLock<usize> = OnceLock::new();
    if gpu.stream_is_capturing(stream) {
        return Ok(());
    }
    {
        let d = DUMP.lock().unwrap_or_else(|e| e.into_inner());
        let lines = cap(&LINES, "ATLAS_QWEN4EXP_MOE_ROUTE_DUMP_LINES", 49152);
        if d.off || d.lines >= lines {
            return Ok(());
        }
    }
    gpu.synchronize(stream)?;
    let slots = l.rows * l.top_k;
    let ids = read(gpu, l.indices, slots * 4)?;
    let mut line = String::with_capacity(slots * 4);
    for (i, id) in ids.chunks_exact(4).enumerate() {
        let id = u32::from_le_bytes([id[0], id[1], id[2], id[3]]);
        line.push_str(if i == 0 { "" } else { " " });
        line.push_str(&id.to_string());
    }
    line.push('\n');
    let mut d = DUMP.lock().unwrap_or_else(|e| e.into_inner());
    let wrote = write(&mut d.text, path, line.as_bytes());
    if !file_ok(&mut d, wrote) {
        return Ok(());
    }
    d.lines += 1;
    if d.records >= cap(&RECORDS, "ATLAS_QWEN4EXP_MOE_ROUTE_DUMP_MAX", 384) {
        return Ok(());
    }
    d.records += 1;
    let layer = match d.layers.iter().position(|&k| k == l.layer) {
        Some(i) => i,
        None => {
            d.layers.push(l.layer);
            d.layers.len() - 1
        }
    };
    let mut rec = Vec::with_capacity(20 + slots * 8 + l.rows * l.hidden * 2);
    for w in [
        0x3144_4D51u32,
        layer as u32,
        l.rows as u32,
        l.top_k as u32,
        l.hidden as u32,
    ] {
        rec.extend_from_slice(&w.to_le_bytes());
    }
    rec.extend_from_slice(&ids);
    rec.extend_from_slice(&read(gpu, l.weights, slots * 4)?);
    rec.extend_from_slice(&read(gpu, l.input, l.rows * l.hidden * 2)?);
    let wrote = write(&mut d.bin, &format!("{path}.bin"), &rec);
    file_ok(&mut d, wrote);
    Ok(())
}
