// SPDX-License-Identifier: AGPL-3.0-only

//! Native batch-1 EXL3 dense GEMV on the packed trellis
//! (`ATLAS_EXL3_DENSE_NATIVE=1`, off by default).
//!
//! The default EXL3 load materializes every dense linear to a BF16 `[out, in]`
//! copy (`exl3_materialize_dense`) and decode streams those 2 bytes/weight
//! through `dense_gemv_bf16`. With the switch on, the materializer also keeps
//! the packed trellis/suh/svh of the main-model linears (attention q/k/v/o,
//! GDN in_proj_qkv/in_proj_z/out_proj, lm_head) and registers them here,
//! keyed by the BF16 copy's device pointer (the pattern `moe_pack` uses).
//! [`exl3_dense_try`] is called at the top of `ops::dense_gemv` (the M = 1
//! BF16 GEMV): a registered weight runs `exl3_gemv_k<K>` + `exl3_gemv_post_bf16`
//! instead (K/8 bytes per weight). Everything else — prefill GEMMs, the M > 1
//! batched GEMVs, unregistered weights, a different GEMV kernel — keeps the
//! BF16 copy, which stays resident as the fallback.
//!
//! The GDN loader concatenates in_proj_qkv and in_proj_z into one fused
//! `[qkv; z]` BF16 buffer and frees the two sources, so
//! [`exl3_dense_register_concat`] moves their entries onto the fused pointer
//! as a two-part entry (each part writes its own row range).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::exl3::Exl3Weight;

/// Warps per GEMV block (k splits) and output columns per group — mirrors
/// `EXL3_GEMV_WARPS` / `EXL3_GEMV_COLS` in `exl3_gemv.cu`.
const WARPS: usize = 16;
const COLS: usize = 32;
const TILES_PER_WARP: usize = 2;

/// `ATLAS_EXL3_DENSE_NATIVE=1` — read once.
pub fn exl3_dense_native_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_DENSE_NATIVE").as_deref() == Ok("1"))
}

/// Routing switch (registration happens at load either way when the env is
/// on). `ATLAS_EXL3_DENSE_NATIVE_TOGGLE_FILE=<path>`: a poller re-reads the
/// file every 200 ms, `1` = native, anything else = the BF16 GEMV — A/B/A in
/// one serve, with the per-phase native call count logged at every flip.
static ACTIVE: AtomicBool = AtomicBool::new(true);

fn start_toggle_poller() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let Ok(path) = std::env::var("ATLAS_EXL3_DENSE_NATIVE_TOGGLE_FILE") else {
            return;
        };
        if let Ok(s) = std::fs::read_to_string(&path) {
            ACTIVE.store(s.trim() == "1", Ordering::Relaxed);
        }
        tracing::info!(
            "EXL3 dense native toggle file {path}: start {} (poll 200 ms)",
            if ACTIVE.load(Ordering::Relaxed) {
                "native"
            } else {
                "bf16"
            }
        );
        let _ = std::thread::Builder::new()
            .name("exl3-dense-toggle".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    let Ok(s) = std::fs::read_to_string(&path) else {
                        continue;
                    };
                    let v = s.trim() == "1";
                    if ACTIVE.swap(v, Ordering::Relaxed) != v {
                        tracing::info!(
                            "EXL3 dense native toggle -> {} (native GEMVs so far: {})",
                            if v { "native" } else { "bf16" },
                            CALLS.load(Ordering::Relaxed)
                        );
                    }
                }
            });
    });
}

/// Max GEMV grid (blocks); groups beyond it are grid-strided.
/// `ATLAS_EXL3_GEMV_GRID` overrides (default 96 = 2 x 48 SMs on GB10).
fn max_grid() -> usize {
    static G: OnceLock<usize> = OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("ATLAS_EXL3_GEMV_GRID")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &usize| v > 0)
            .unwrap_or(96)
    })
}

/// The `exl3_gemv` module's kernels.
pub struct Exl3GemvKernels {
    /// bf16 activations; index = `bits - 2` (K2..=K8).
    pub gemv: [KernelHandle; 7],
    /// Pre-transformed fp16 activations (tests); index = `bits - 2`.
    pub gemv_raw: [KernelHandle; 7],
    pub post: KernelHandle,
    /// M-row forms (index = `bits - 2`), the activation pre-kernel and the
    /// row-strided post.
    pub gemv_m: [KernelHandle; 7],
    pub xh_rows: KernelHandle,
    pub post_rows: KernelHandle,
}

impl Exl3GemvKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: String| gpu.kernel("exl3_gemv", &name);
        let mut gemv = [KernelHandle(0); 7];
        let mut gemv_raw = [KernelHandle(0); 7];
        let mut gemv_m = [KernelHandle(0); 7];
        for bits in 2..=8usize {
            gemv[bits - 2] = k(format!("exl3_gemv_k{bits}"))?;
            gemv_raw[bits - 2] = k(format!("exl3_gemv_raw_k{bits}"))?;
            gemv_m[bits - 2] = k(format!("exl3_gemv_m_k{bits}"))?;
        }
        Ok(Self {
            gemv,
            gemv_raw,
            post: k("exl3_gemv_post_bf16".to_string())?,
            gemv_m,
            xh_rows: k("exl3_gemv_xh_rows".to_string())?,
            post_rows: k("exl3_gemv_post_rows_bf16".to_string())?,
        })
    }
}

/// Dynamic shared memory of one GEMV block: fp16 x, the staged tiles, the
/// cross-warp reduction.
fn gemv_smem_bytes(in_features: usize, bits: u32) -> usize {
    in_features * 2 + WARPS * TILES_PER_WARP * 8 * bits as usize * 4 + WARPS * COLS * 4
}

fn check_shape(w: &Exl3Weight) -> Result<()> {
    let sh = &w.shape;
    ensure!(
        (2..=8).contains(&sh.bits),
        "EXL3 GEMV: bits {} has no kernel (K2..=K8)",
        sh.bits
    );
    ensure!(
        sh.in_features.is_multiple_of(128) && sh.out_features.is_multiple_of(128),
        "EXL3 GEMV: {}x{} (in x out) must be multiples of 128",
        sh.in_features,
        sh.out_features
    );
    ensure!(
        w.trellis.0.is_multiple_of(16),
        "EXL3 GEMV: trellis {:#x} is not 16-byte aligned",
        w.trellis.0
    );
    ensure!(
        w.suh.0.is_multiple_of(8) && w.svh.0.is_multiple_of(8),
        "EXL3 GEMV: suh/svh {:#x}/{:#x} are not 8-byte aligned",
        w.suh.0,
        w.svh.0
    );
    ensure!(
        gemv_smem_bytes(sh.in_features, sh.bits) <= 48 * 1024,
        "EXL3 GEMV: in_features {} needs more than 48 KiB of shared memory",
        sh.in_features
    );
    Ok(())
}

/// `y_inner[out] (fp32) = had_r128(x * suh) · W_inner`. `raw = false`: `x` is
/// bf16 `[in]` and the kernel applies suh + the input Hadamard; `raw = true`:
/// `x` is the already-transformed fp16 `xh` (tests).
pub fn exl3_gemv_inner(
    gpu: &dyn GpuBackend,
    k: &Exl3GemvKernels,
    raw: bool,
    x: DevicePtr,
    w: &Exl3Weight,
    y_f32: DevicePtr,
    stream: u64,
) -> Result<()> {
    exl3_gemv_inner_n(gpu, k, raw, x, w, y_f32, w.shape.out_features, stream)
}

/// [`exl3_gemv_inner`] over the leading `n_act` output columns only
/// (`n_act` a multiple of 128, so every Hadamard block the post kernel reads
/// is complete). Each 32-column group is computed exactly as in the full
/// call; groups past `n_act` are skipped. Used by the MTP drafter lm_head,
/// whose rows are capped at `--mtp-vocab`.
#[allow(clippy::too_many_arguments)]
pub fn exl3_gemv_inner_n(
    gpu: &dyn GpuBackend,
    k: &Exl3GemvKernels,
    raw: bool,
    x: DevicePtr,
    w: &Exl3Weight,
    y_f32: DevicePtr,
    n_act: usize,
    stream: u64,
) -> Result<()> {
    check_shape(w)?;
    let sh = &w.shape;
    ensure!(
        n_act > 0 && n_act.is_multiple_of(128) && n_act <= sh.out_features,
        "EXL3 GEMV: n_act {n_act} vs out_features {}",
        sh.out_features
    );
    let groups = n_act / COLS;
    let grid = groups.min(max_grid()) as u32;
    let table = if raw { &k.gemv_raw } else { &k.gemv };
    KernelLaunch::new(gpu, table[sh.bits as usize - 2])
        .grid([grid, 1, 1])
        .block([(WARPS * 32) as u32, 1, 1])
        .shared_mem(gemv_smem_bytes(sh.in_features, sh.bits) as u32)
        .arg_ptr(x)
        .arg_ptr(w.suh)
        .arg_ptr(w.trellis)
        .arg_ptr(y_f32)
        .arg_i32(sh.in_features as i32)
        .arg_i32(sh.out_features as i32)
        .arg_i32(n_act as i32)
        .launch(stream)
}

/// `out[c] = bf16(had_r128(y_inner)[c] * svh[c])` for `c < n_valid`.
#[allow(clippy::too_many_arguments)]
pub fn exl3_gemv_post(
    gpu: &dyn GpuBackend,
    k: &Exl3GemvKernels,
    y_f32: DevicePtr,
    out_bf16: DevicePtr,
    svh: DevicePtr,
    n: usize,
    n_valid: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        n.is_multiple_of(128) && n_valid <= n,
        "EXL3 GEMV post: n {n}, n_valid {n_valid}"
    );
    KernelLaunch::new(gpu, k.post)
        .grid([div_ceil((n / 128) as u32, 4), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(y_f32)
        .arg_ptr(out_bf16)
        .arg_ptr(svh)
        .arg_i32(n as i32)
        .arg_i32(n_valid as i32)
        .launch(stream)
}

/// The whole batch-1 linear: bf16 `x [in]` -> bf16 `out [n_valid]`.
/// `y_scratch` holds `out_features` f32.
#[allow(clippy::too_many_arguments)]
pub fn exl3_gemv_bf16(
    gpu: &dyn GpuBackend,
    k: &Exl3GemvKernels,
    x_bf16: DevicePtr,
    w: &Exl3Weight,
    y_scratch: DevicePtr,
    out_bf16: DevicePtr,
    n_valid: usize,
    stream: u64,
) -> Result<()> {
    exl3_gemv_inner(gpu, k, false, x_bf16, w, y_scratch, stream)?;
    exl3_gemv_post(
        gpu,
        k,
        y_scratch,
        out_bf16,
        w.svh,
        w.shape.out_features,
        n_valid,
        stream,
    )
}

// ---------------------------------------------------------------------------
// Registry: BF16 copy pointer -> packed parts.

struct Part {
    w: Exl3Weight,
    /// First output row of this part inside the (possibly fused) BF16 weight.
    row0: usize,
}

struct Entry {
    parts: Vec<Part>,
    n: usize,
    k: usize,
}

#[derive(Default)]
struct State {
    map: HashMap<u64, Entry>,
    kernels: Option<Exl3GemvKernels>,
    bf16_gemv: u64,
    /// Per-stream fp32 y scratch: (ptr, elems).
    scratch: HashMap<u64, (DevicePtr, usize)>,
    scratch_elems: usize,
    packed_bytes: u64,
    warned: bool,
    /// M-row scratch, preallocated at registration so the first verify
    /// call can run under CUDA-graph capture: y f32 [ROWS_MAX, max n],
    /// xh fp16 [ROWS_MAX, max k]. `rows_free` is claimed by the first
    /// stream that asks; `rows_scratch` maps stream -> (y, y elems, xh, xh elems).
    rows_free: Option<(DevicePtr, usize, DevicePtr, usize)>,
    rows_scratch: HashMap<u64, (DevicePtr, usize, DevicePtr, usize)>,
    max_k: usize,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));
static ON: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Native GEMV calls served so far (all streams).
pub fn exl3_dense_native_calls() -> u64 {
    CALLS.load(Ordering::Relaxed)
}

/// Number of registered (possibly fused) BF16 weights and the packed bytes
/// they keep resident.
pub fn exl3_dense_registered() -> (usize, u64) {
    let st = state();
    (st.map.len(), st.packed_bytes)
}

/// Register the packed form of the BF16 copy at `bf16_ptr` (`[out, in]`).
/// A trellis/suh/svh that is not aligned for the kernel's vector loads is
/// copied once into an aligned buffer owned by the registry (the store keeps
/// the original).
pub fn exl3_dense_register(gpu: &dyn GpuBackend, bf16_ptr: DevicePtr, w: Exl3Weight) -> Result<()> {
    let mut w = w;
    let (n0, k0) = (w.shape.out_features, w.shape.in_features);
    let realign = |p: DevicePtr, bytes: usize, align: u64| -> Result<DevicePtr> {
        if p.0.is_multiple_of(align) {
            return Ok(p);
        }
        let dst = gpu.alloc(bytes)?;
        gpu.copy_d2d(p, dst, bytes)?;
        Ok(dst)
    };
    w.trellis = realign(w.trellis, n0 * k0 * w.shape.bits as usize / 8, 16)?;
    w.suh = realign(w.suh, k0 * 2, 8)?;
    w.svh = realign(w.svh, n0 * 2, 8)?;
    check_shape(&w)?;
    let mut st = state();
    if st.kernels.is_none() {
        st.kernels = Some(Exl3GemvKernels::resolve(gpu)?);
        st.bf16_gemv = gpu.kernel("gemv", "dense_gemv_bf16")?.0;
    }
    let (n, k) = (w.shape.out_features, w.shape.in_features);
    st.packed_bytes += (n * k) as u64 * w.shape.bits as u64 / 8 + 2 * (n + k) as u64;
    st.scratch_elems = st.scratch_elems.max(n);
    st.max_k = st.max_k.max(k);
    grow_rows_free(gpu, &mut st)?;
    st.map.insert(
        bf16_ptr.0,
        Entry {
            parts: vec![Part { w, row0: 0 }],
            n,
            k,
        },
    );
    ON.store(true, Ordering::Release);
    start_toggle_poller();
    Ok(())
}

/// Rows one native M-row call carries (mirrors `EXL3_GEMV_MROWS`).
pub const EXL3_GEMV_ROWS_MAX: usize = 8;

/// Grow the unclaimed M-row scratch to the current maxima (load time only:
/// registration never runs under graph capture).
fn grow_rows_free(gpu: &dyn GpuBackend, st: &mut State) -> Result<()> {
    let need_y = EXL3_GEMV_ROWS_MAX * st.scratch_elems;
    let need_x = EXL3_GEMV_ROWS_MAX * st.max_k;
    if let Some((y, ye, x, xe)) = st.rows_free
        && ye >= need_y
        && xe >= need_x
    {
        let _ = (y, x);
        return Ok(());
    }
    if let Some((y, _, x, _)) = st.rows_free.take() {
        let _ = gpu.free(y);
        let _ = gpu.free(x);
    }
    let y = gpu.alloc(need_y * 4)?;
    let x = gpu.alloc(need_x * 2)?;
    st.rows_free = Some((y, need_y, x, need_x));
    Ok(())
}

/// `ATLAS_EXL3_DENSE_ROWS=1` (off by default): M = 2..=8 row GEMVs on a
/// registered weight (`dense_gemv_batch2` / `dense_gemv_batchm`) run on the
/// packed trellis, reading it once for all rows.
pub fn exl3_dense_rows_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_DENSE_ROWS").as_deref() == Ok("1"))
}

static ROWS_CALLS: AtomicU64 = AtomicU64::new(0);

/// Whether `weight` has a packed entry the M-row path can serve.
pub fn exl3_dense_is_registered(weight: DevicePtr) -> bool {
    ON.load(Ordering::Relaxed) && state().map.contains_key(&weight.0)
}

/// Called by `ops::dense_gemv_batch2` / `ops::dense_gemv_batchm`:
/// `input` [m, k] bf16 contiguous, output row t at `output + t * out_stride`
/// (bf16 elements). `Some` when the native path ran (or failed); `None` means
/// "use the BF16 kernel". Launch-only (graph-capture safe once the stream owns
/// its scratch, which the first call takes without allocating).
#[allow(clippy::too_many_arguments)]
pub fn exl3_dense_try_rows(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    m: usize,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Option<Result<()>> {
    if !ON.load(Ordering::Relaxed) || !ACTIVE.load(Ordering::Relaxed) || !exl3_dense_rows_env() {
        return None;
    }
    if !(2..=EXL3_GEMV_ROWS_MAX).contains(&m) {
        return None;
    }
    let mut st = state();
    let (n, k) = (n as usize, k as usize);
    let (entry_n, entry_k) = {
        let e = st.map.get(&weight.0)?;
        (e.n, e.k)
    };
    if k != entry_k || n > entry_n {
        return None;
    }
    let (y_scr, y_elems, x_scr, x_elems) = match st.rows_scratch.get(&stream).copied() {
        Some(s) => s,
        None => {
            let s = match st.rows_free.take() {
                Some(s) => s,
                None => {
                    // A second stream: allocate (never the verify stream,
                    // which claimed the preallocated slot first).
                    let ye = EXL3_GEMV_ROWS_MAX * st.scratch_elems;
                    let xe = EXL3_GEMV_ROWS_MAX * st.max_k;
                    let y = match gpu.alloc(ye * 4) {
                        Ok(p) => p,
                        Err(e) => return Some(Err(e)),
                    };
                    let x = match gpu.alloc(xe * 2) {
                        Ok(p) => p,
                        Err(e) => return Some(Err(e)),
                    };
                    (y, ye, x, xe)
                }
            };
            st.rows_scratch.insert(stream, s);
            s
        }
    };
    if y_elems < m * entry_n || x_elems < m * entry_k {
        return None;
    }
    let st = &*st;
    let kernels = st.kernels.as_ref()?;
    let e = st.map.get(&weight.0)?;
    let run = || -> Result<()> {
        for p in &e.parts {
            let sh = &p.w.shape;
            let n_part = sh.out_features;
            let valid = n.saturating_sub(p.row0).min(n_part);
            if valid == 0 {
                continue;
            }
            check_shape(&p.w)?;
            KernelLaunch::new(gpu, kernels.xh_rows)
                .grid([div_ceil((k / 128) as u32, 4), m as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(input)
                .arg_ptr(p.w.suh)
                .arg_ptr(x_scr)
                .arg_i32(k as i32)
                .arg_i32(k as i32)
                .launch(stream)?;
            let groups = n_part / COLS;
            let grid = groups.min(max_grid()) as u32;
            let smem = WARPS * TILES_PER_WARP * 8 * sh.bits as usize * 4
                + WARPS * EXL3_GEMV_ROWS_MAX * COLS * 4;
            let y_part = y_scr.offset(p.row0 * 4);
            KernelLaunch::new(gpu, kernels.gemv_m[sh.bits as usize - 2])
                .grid([grid, 1, 1])
                .block([(WARPS * 32) as u32, 1, 1])
                .shared_mem(smem as u32)
                .arg_ptr(x_scr)
                .arg_ptr(p.w.trellis)
                .arg_ptr(y_part)
                .arg_i32(k as i32)
                .arg_i32(n_part as i32)
                .arg_i32(m as i32)
                .arg_i32(entry_n as i32)
                .launch(stream)?;
            KernelLaunch::new(gpu, kernels.post_rows)
                .grid([div_ceil((n_part / 128) as u32, 4), m as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(y_part)
                .arg_ptr(output.offset(p.row0 * 2))
                .arg_ptr(p.w.svh)
                .arg_i32(n_part as i32)
                .arg_i32(valid as i32)
                .arg_i32(entry_n as i32)
                .arg_i32(out_stride as i32)
                .launch(stream)?;
        }
        Ok(())
    };
    let r = run();
    let c = ROWS_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if c == 1 || c == 10_000 || c == 100_000 {
        tracing::info!(
            "EXL3 dense native rows: {c} M-row GEMVs on the packed trellis so far (m = {m})"
        );
    }
    Some(r)
}

/// `fused` = the row concatenation of `sources` (in order). Moves their
/// entries onto `fused`; the sources are dropped from the map (the loader
/// frees them). No-op when nothing is registered.
pub fn exl3_dense_register_concat(fused: DevicePtr, sources: &[DevicePtr]) {
    if !ON.load(Ordering::Acquire) {
        return;
    }
    let mut st = state();
    let taken: Vec<Option<Entry>> = sources.iter().map(|p| st.map.remove(&p.0)).collect();
    if taken.iter().any(Option::is_none) {
        if taken.iter().any(Option::is_some) {
            tracing::warn!(
                "EXL3 dense native: fused weight {:#x} has an unregistered source; it stays BF16",
                fused.0
            );
        }
        return;
    }
    let mut parts = Vec::new();
    let (mut row0, mut k) = (0usize, None);
    for e in taken.into_iter().flatten() {
        if *k.get_or_insert(e.k) != e.k {
            tracing::warn!("EXL3 dense native: fused sources disagree on in_features; stays BF16");
            return;
        }
        for p in e.parts {
            parts.push(Part {
                w: p.w,
                row0: row0 + p.row0,
            });
        }
        row0 += e.n;
    }
    st.scratch_elems = st.scratch_elems.max(row0);
    st.map.insert(
        fused.0,
        Entry {
            parts,
            n: row0,
            k: k.unwrap_or(0),
        },
    );
}

/// Called by `ops::dense_gemv` (M = 1). `Some` when the native path ran (or
/// failed); `None` means "use the BF16 GEMV".
#[allow(clippy::too_many_arguments)]
pub fn exl3_dense_try(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Option<Result<()>> {
    if !ON.load(Ordering::Relaxed) || !ACTIVE.load(Ordering::Relaxed) {
        return None;
    }
    let mut st = state();
    if kernel.0 != st.bf16_gemv {
        return None;
    }
    let (n, k) = (n as usize, k as usize);
    let (entry_n, entry_k) = {
        let e = st.map.get(&weight.0)?;
        (e.n, e.k)
    };
    // `n` may be short of the packed rows (lm_head: the vocab is capped to the
    // tokenizer); the rows are the leading rows of the same [out, in] weight.
    if k != entry_k || n > entry_n {
        if !st.warned {
            st.warned = true;
            tracing::warn!(
                "EXL3 dense native: GEMV on {:#x} asks {n}x{k}, packed is {entry_n}x{entry_k}; using BF16",
                weight.0
            );
        }
        return None;
    }
    let need = st.scratch_elems;
    let scratch = match st.scratch.get(&stream).copied() {
        Some((p, elems)) if elems >= need => p,
        old => {
            if let Some((p, _)) = old {
                let _ = gpu.synchronize(stream);
                let _ = gpu.free(p);
            }
            match gpu.alloc(need * 4) {
                Ok(p) => {
                    st.scratch.insert(stream, (p, need));
                    p
                }
                Err(e) => return Some(Err(e)),
            }
        }
    };
    let st = &*st;
    let kernels = st.kernels.as_ref()?;
    let e = st.map.get(&weight.0)?;
    let run = || -> Result<()> {
        for p in &e.parts {
            let n_part = p.w.shape.out_features;
            let valid = n.saturating_sub(p.row0).min(n_part);
            if valid == 0 {
                continue;
            }
            // Only the leading 128-aligned block span that covers `valid`
            // (lm_head capped to the draft vocab); all of it otherwise.
            let n_act = valid.div_ceil(128).saturating_mul(128).min(n_part);
            exl3_gemv_inner_n(
                gpu,
                kernels,
                false,
                input,
                &p.w,
                scratch.offset(p.row0 * 4),
                n_act,
                stream,
            )?;
            exl3_gemv_post(
                gpu,
                kernels,
                scratch.offset(p.row0 * 4),
                output.offset(p.row0 * 2),
                p.w.svh,
                n_act,
                valid,
                stream,
            )?;
        }
        Ok(())
    };
    let r = run();
    let c = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if c == 1 || c == 10_000 || c == 100_000 {
        tracing::info!(
            "EXL3 dense native: {c} batch-1 GEMVs on the packed trellis so far ({} weights registered)",
            st.map.len()
        );
    }
    Some(r)
}
