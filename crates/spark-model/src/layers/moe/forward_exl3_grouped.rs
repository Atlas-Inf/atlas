// SPDX-License-Identifier: AGPL-3.0-only

//! Grouped packed-EXL3 MoE prefill (`ATLAS_EXL3_PREFILL_GROUPED=1`, off by
//! default).
//!
//! The per-token arms in `forward_exl3.rs` (interim reconstruct + hgemm, or
//! CoopMK one row per launch) read every selected expert's trellis once per
//! token. Here a prefill chunk of one layer is routed once, its (token, slot)
//! rows are sorted by expert on the host (one top-k readback per layer per
//! chunk), and each selected expert's trellis is decoded once per 64 of its
//! rows, straight into tensor-core B fragments (`exl3_prefill.cu`
//! `exl3_pf_mma_k*`). The shared expert rides as one more local expert with
//! every token as a row. Nothing is requantized and no fp16/BF16 copy of any
//! expert is stored: the only scratch is activations, sized by the chunk.
//!
//! Per row the arithmetic is the interim's (same routing kernels, same
//! Hadamard / conversion / SiLU / blend code); only the fp16 GEMM's summation
//! order differs (tensor-core mma vs the scalar hgemm), which the standalone
//! harness measured at <= 1 fp16 ulp, with identical error against fp64.
//!
//! Knobs:
//! * `ATLAS_EXL3_PREFILL_GROUPED=1` enables it for every multi-row call that
//!   reaches `try_forward_exl3` (prefill chunks and M = 2..8 MTP verify alike);
//!   `ATLAS_EXL3_PREFILL_GROUPED_MIN` (default 2) is the smallest row count it
//!   takes, so single-token decode stays on CoopMK / interim. CUDA-graph
//!   capture always falls through (the grouped arm reads routing on the host).
//! * `ATLAS_EXL3_PREFILL_GROUPED_BENCH=<n>`: on the first call with >= n rows,
//!   every MoE layer times the per-token arm and the grouped arm on its first
//!   m = 1, 2, 3, 4, 8, ... rows (same input, same weights; median of 3) and
//!   logs `EXL3 grouped bench` lines (per-layer MoE ms). Debug only.
//! * `ATLAS_EXL3_PREFILL_GROUPED_TOGGLE_FILE=<path>`: re-read at most every
//!   200 ms; `1` = grouped, anything else = the per-token arm (A/B in one
//!   serve).
//! * `ATLAS_EXL3_PREFILL_GROUPED_CHECK=N`: the first N grouped calls also run
//!   the per-token arm on the same input and log max |diff| / cosine (debug).
//! * `ATLAS_EXL3_PREFILL_GROUPED_PROFILE=1`: synchronize and time each call.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::exl3::moe_pack::{Exl3MoePack, with_exl3_moe};

use super::*;

const HAD_SCALE: f32 = 0.088_388_346;
/// Rows per mma item (`exl3_pf::ROWS`).
const ITEM_ROWS: usize = 64;
const ITEM_BYTES: usize = 32;

static LOGGED: AtomicBool = AtomicBool::new(false);
/// Set while CHECK runs the per-token reference, so the hook passes through.
static BYPASS: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static ROWS_TOTAL: AtomicU64 = AtomicU64::new(0);
static CHECK_DONE: AtomicU64 = AtomicU64::new(0);
static PROF_NS: AtomicU64 = AtomicU64::new(0);

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|s| s.trim().to_string())
}

fn env_num(name: &str, default: u64) -> u64 {
    env_str(name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn min_rows() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| env_num("ATLAS_EXL3_PREFILL_GROUPED_MIN", 2).max(1) as usize)
}

fn check_budget() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| env_num("ATLAS_EXL3_PREFILL_GROUPED_CHECK", 0))
}

fn bench_min() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| env_num("ATLAS_EXL3_PREFILL_GROUPED_BENCH", 0) as usize)
}

static BENCHED: LazyLock<Mutex<std::collections::HashSet<u64>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashSet::new()));

fn profile() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_str("ATLAS_EXL3_PREFILL_GROUPED_PROFILE").as_deref() == Some("1"))
}

/// Whether the grouped arm is selected right now.
fn grouped_selected() -> bool {
    static ENV: OnceLock<(bool, Option<String>)> = OnceLock::new();
    static TOGGLE: Mutex<Option<(Instant, bool)>> = Mutex::new(None);
    let (on, file) = ENV.get_or_init(|| {
        (
            env_str("ATLAS_EXL3_PREFILL_GROUPED").as_deref() == Some("1"),
            env_str("ATLAS_EXL3_PREFILL_GROUPED_TOGGLE_FILE"),
        )
    });
    let Some(path) = file else {
        return *on;
    };
    let mut g = TOGGLE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    if let Some((t, v)) = *g {
        if now.duration_since(t) >= Duration::from_millis(200) {
            // refresh
        } else {
            return v;
        }
    }
    let v = std::fs::read_to_string(path)
        .map(|s| s.trim() == "1")
        .unwrap_or(*on);
    if g.map(|(_, old)| old != v).unwrap_or(true) {
        tracing::info!(
            "EXL3 grouped prefill toggle {path}: {} (grouped calls so far {}, rows {})",
            if v { "grouped" } else { "per-token" },
            CALLS.load(Ordering::Relaxed),
            ROWS_TOTAL.load(Ordering::Relaxed)
        );
    }
    *g = Some((now, v));
    v
}

#[derive(Clone, Copy)]
struct PfKernels {
    gemv_rows: KernelHandle,
    topk_rows: KernelHandle,
    gather_had: KernelHandle,
    had_post_bf16: KernelHandle,
    /// index = bits - 2 (k2..=k8)
    mma: [KernelHandle; 7],
}

impl PfKernels {
    fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| gpu.kernel("exl3_prefill", name);
        Ok(Self {
            gemv_rows: k("exl3_pf_gemv_rows")?,
            topk_rows: k("exl3_pf_topk_rows")?,
            gather_had: k("exl3_pf_gather_had")?,
            had_post_bf16: k("exl3_pf_had_post_bf16")?,
            mma: [
                k("exl3_pf_mma_k2")?,
                k("exl3_pf_mma_k3")?,
                k("exl3_pf_mma_k4")?,
                k("exl3_pf_mma_k5")?,
                k("exl3_pf_mma_k6")?,
                k("exl3_pf_mma_k7")?,
                k("exl3_pf_mma_k8")?,
            ],
        })
    }
}

/// Per-layer device tables: fp16 scale-vector addresses per local expert
/// (routed then shared), `[g_suh | u_suh | d_suh | g_svh | u_svh | d_svh]`.
#[derive(Clone, Copy)]
struct LayerTables {
    tabs: DevicePtr,
    n_local: usize,
}

impl LayerTables {
    fn tab(&self, i: usize) -> DevicePtr {
        self.tabs.offset(i * self.n_local * 8)
    }
}

/// `None` = this layer cannot take the grouped arm (logged once at build).
static TABLES: LazyLock<Mutex<HashMap<u64, Option<LayerTables>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Activation scratch shared by every layer (layers run in stream order).
struct Scratch {
    kernels: PfKernels,
    rows_cap: usize,
    h: usize,
    inter: usize,
    /// `[rows, H]` fp16: gate input, later the down output.
    xg: DevicePtr,
    /// `[rows, H]`: up input, later the bf16 per-(token, slot) down rows.
    xu: DevicePtr,
    /// `[rows, I]` fp16: gate output -> bf16 gate -> SiLU·up -> down input.
    yg: DevicePtr,
    /// `[rows, I]` fp16: up output -> bf16 up.
    yu: DevicePtr,
    meta: DevicePtr,
    meta_cap: usize,
}

static SCRATCH: Mutex<Option<Scratch>> = Mutex::new(None);

fn free_all(gpu: &dyn GpuBackend, ptrs: &[DevicePtr]) {
    for p in ptrs {
        if !p.is_null() {
            let _ = gpu.free(*p);
        }
    }
}

fn ensure_scratch<'a>(
    slot: &'a mut Option<Scratch>,
    gpu: &dyn GpuBackend,
    rows: usize,
    meta_bytes: usize,
    h: usize,
    inter: usize,
    stream: u64,
) -> Result<&'a mut Scratch> {
    let fits = slot
        .as_ref()
        .map(|s| s.rows_cap >= rows && s.meta_cap >= meta_bytes && s.h == h && s.inter == inter)
        .unwrap_or(false);
    if !fits {
        let kernels = match slot.as_ref() {
            Some(s) => s.kernels,
            None => PfKernels::resolve(gpu)?,
        };
        if let Some(old) = slot.take() {
            gpu.synchronize(stream)?;
            free_all(gpu, &[old.xg, old.xu, old.yg, old.yu, old.meta]);
        }
        // Grow in steps so a slowly lengthening prompt does not realloc per call.
        let rows_cap = rows.next_power_of_two().max(1024);
        let meta_cap = meta_bytes.next_power_of_two().max(1 << 20);
        let mut got: Vec<DevicePtr> = Vec::new();
        let mut alloc = |bytes: usize| -> Result<DevicePtr> {
            match gpu.alloc(bytes) {
                Ok(p) => {
                    got.push(p);
                    Ok(p)
                }
                Err(e) => {
                    free_all(gpu, &got);
                    Err(e)
                }
            }
        };
        let xg = alloc(rows_cap * h * 2)?;
        let xu = alloc(rows_cap * h * 2)?;
        let yg = alloc(rows_cap * inter * 2)?;
        let yu = alloc(rows_cap * inter * 2)?;
        let meta = alloc(meta_cap)?;
        tracing::info!(
            "EXL3 grouped prefill scratch: {rows_cap} rows ({:.1} MB activations + {:.1} MB metadata)",
            (rows_cap * (2 * h + 2 * inter) * 2) as f64 / 1e6,
            meta_cap as f64 / 1e6
        );
        *slot = Some(Scratch {
            kernels,
            rows_cap,
            h,
            inter,
            xg,
            xu,
            yg,
            yu,
            meta,
            meta_cap,
        });
    }
    Ok(slot.as_mut().unwrap())
}

fn build_tables(
    gpu: &dyn GpuBackend,
    pack: &Exl3MoePack,
    h: usize,
    inter: usize,
) -> Result<Option<LayerTables>> {
    let n_routed = pack.experts.len();
    let n_local = n_routed + 1;
    let mut t: Vec<Vec<u64>> = vec![vec![0u64; n_local]; 6];
    let shape_ok = |g: &crate::weight_map::exl3::Exl3Weight,
                    u: &crate::weight_map::exl3::Exl3Weight,
                    d: &crate::weight_map::exl3::Exl3Weight|
     -> bool {
        let ok_k = |b: u32| (2..=8).contains(&b);
        g.shape.in_features == h
            && g.shape.out_features == inter
            && u.shape.in_features == h
            && u.shape.out_features == inter
            && d.shape.in_features == inter
            && d.shape.out_features == h
            && ok_k(g.shape.bits)
            && ok_k(u.shape.bits)
            && ok_k(d.shape.bits)
    };
    let all = pack
        .experts
        .iter()
        .chain(std::iter::once(&pack.shared))
        .enumerate();
    for (e, x) in all {
        if !shape_ok(&x.gate, &x.up, &x.down) {
            tracing::warn!(
                "EXL3 grouped prefill off at {}: local expert {e} shape/bitrate outside the grouped kernels \
                 (H {h} I {inter}, K 2..8); per-token arm stays",
                pack.name
            );
            return Ok(None);
        }
        for (i, p) in [
            x.gate.suh, x.up.suh, x.down.suh, x.gate.svh, x.up.svh, x.down.svh,
        ]
        .into_iter()
        .enumerate()
        {
            t[i][e] = p.0;
        }
    }
    let bytes: Vec<u8> = t.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    let tabs = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, tabs)?;
    Ok(Some(LayerTables { tabs, n_local }))
}

fn push_item(buf: &mut Vec<u8>, trellis: u64, a: u64, c: u64, nrows: usize) {
    buf.extend_from_slice(&trellis.to_le_bytes());
    buf.extend_from_slice(&a.to_le_bytes());
    buf.extend_from_slice(&c.to_le_bytes());
    buf.extend_from_slice(&(nrows as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
}

fn bf16_to_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
        .collect()
}

impl MoeLayer {
    /// Hook (called from `try_forward_exl3`, so prefill chunks and the
    /// forward_k2/k3/batched verify paths all reach it): run `num_tokens` rows on the grouped arm when it is selected and
    /// applicable. Returns whether it ran (else the per-token arms run).
    pub(super) fn try_forward_exl3_grouped(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if num_tokens < min_rows()
            || ctx.graph_capture
            || BYPASS.load(Ordering::Relaxed)
            || !grouped_selected()
            || self.tid2eid_dev.is_some()
            || self.correction_bias_dev.is_some()
        {
            return Ok(false);
        }
        let cfg = ctx.config;
        let (h, inter, top_k) = (
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
        );
        if h % 128 != 0 || inter % 128 != 0 || top_k == 0 {
            return Ok(false);
        }
        let n = num_tokens;
        if ctx.buffers.scratch_bytes() < n * top_k * 8 {
            bail!(
                "EXL3 grouped prefill: routing scratch {} B < {} B for {n} tokens",
                ctx.buffers.scratch_bytes(),
                n * top_k * 8
            );
        }
        let gate_key = self.weights.shared_expert_gate.weight.0;
        let tables = {
            let mut map = TABLES
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match map.get(&gate_key) {
                Some(t) => *t,
                None => {
                    let t = with_exl3_moe(gate_key, |pack| build_tables(ctx.gpu, pack, h, inter))
                        .ok_or_else(|| {
                            anyhow::anyhow!("EXL3 MoE pack missing for gate {gate_key:#x}")
                        })??;
                    map.insert(gate_key, t);
                    t
                }
            }
        };
        let Some(tables) = tables else {
            return Ok(false);
        };
        let bm = bench_min();
        if bm > 0 && n >= bm {
            let first = BENCHED
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(gate_key);
            if first {
                self.grouped_bench(input, n, tables, ctx, stream)?;
            }
        }
        let t0 = profile().then(|| {
            let _ = ctx.gpu.synchronize(stream);
            Instant::now()
        });
        let rows = self.grouped_run(input, n, tables, ctx, stream)?;

        let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        ROWS_TOTAL.fetch_add(rows as u64, Ordering::Relaxed);
        if LOGGED
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            tracing::info!(
                "EXL3 MoE grouped prefill: {n} tokens -> {rows} expert rows; each selected expert decoded once \
                 per 64 rows straight into mma fragments (no reconstruct, no NVFP4, no expert copies)"
            );
        }
        if let Some(t0) = t0 {
            ctx.gpu.synchronize(stream)?;
            let dt = t0.elapsed().as_nanos() as u64;
            let ns = PROF_NS.fetch_add(dt, Ordering::Relaxed) + dt;
            if calls.is_multiple_of(cfg.num_hidden_layers.max(1) as u64) {
                tracing::info!(
                    "EXL3 grouped prefill profile: {calls} calls, {:.1} ms total MoE (last chunk {n} tokens, {rows} rows)",
                    ns as f64 / 1e6
                );
            }
        }
        if CHECK_DONE.load(Ordering::Relaxed) < check_budget() {
            self.grouped_check(input, n, ctx, stream)?;
        }
        Ok(true)
    }

    /// Router GEMV + top-k softmax for `n` rows: u32 ids at `idx_dev`
    /// [n, top_k], f32 weights at `w_dev` [n, top_k]. The kernels are the
    /// grouped path's bit-identical copies of `dense_gemv_bf16` +
    /// `moe_topk_softmax`, so every row routes exactly as the per-token path
    /// routes it. Launch-only (CUDA-graph capture safe).
    pub(super) fn exl3_route_rows(
        &self,
        input: DevicePtr,
        n: usize,
        idx_dev: DevicePtr,
        w_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        static K: OnceLock<PfKernels> = OnceLock::new();
        let kernels = match K.get() {
            Some(k) => *k,
            None => {
                let k = PfKernels::resolve(ctx.gpu)?;
                let _ = K.set(k);
                k
            }
        };
        let cfg = ctx.config;
        let (h, top_k) = (cfg.hidden_size, cfg.num_experts_per_tok);
        let router_in = self.router_input(input, n as u32, h as u32, ctx, stream)?;
        let logits = ctx.buffers.gate_logits();
        let n_logits = self.router_logits_n;
        KernelLaunch::new(ctx.gpu, kernels.gemv_rows)
            .grid([div_ceil(n_logits, 4), n as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(router_in)
            .arg_ptr(self.weights.gate.weight)
            .arg_ptr(logits)
            .arg_u32(n_logits)
            .arg_u32(h as u32)
            .arg_u32(n_logits)
            .launch(stream)?;
        KernelLaunch::new(ctx.gpu, kernels.topk_rows)
            .grid([n as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(idx_dev)
            .arg_ptr(w_dev)
            .arg_u32(cfg.num_experts as u32)
            .arg_u32(top_k as u32)
            .arg_u32(u32::from(cfg.norm_topk_prob))
            .arg_u32(n_logits)
            .launch(stream)?;
        let _ = top_k;
        Ok(())
    }

    /// Route `n` rows once and run the grouped experts; output in moe_output.
    fn grouped_run(
        &self,
        input: DevicePtr,
        n: usize,
        tables: LayerTables,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let cfg = ctx.config;
        let (h, top_k) = (cfg.hidden_size, cfg.num_experts_per_tok);
        let gate_key = self.weights.shared_expert_gate.weight.0;
        let mut guard = SCRATCH
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let kernels = match guard.as_ref() {
            Some(s) => s.kernels,
            None => PfKernels::resolve(ctx.gpu)?,
        };

        // ── Routing: bit-identical copies of dense_gemv_bf16 + moe_topk_softmax.
        let router_in = self.router_input(input, n as u32, h as u32, ctx, stream)?;
        let logits = ctx.buffers.gate_logits();
        let n_logits = self.router_logits_n;
        KernelLaunch::new(ctx.gpu, kernels.gemv_rows)
            .grid([div_ceil(n_logits, 4), n as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(router_in)
            .arg_ptr(self.weights.gate.weight)
            .arg_ptr(logits)
            .arg_u32(n_logits)
            .arg_u32(h as u32)
            .arg_u32(n_logits)
            .launch(stream)?;
        let idx_dev = ctx.buffers.scratch();
        let w_dev = idx_dev.offset(n * top_k * 4);
        KernelLaunch::new(ctx.gpu, kernels.topk_rows)
            .grid([n as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(idx_dev)
            .arg_ptr(w_dev)
            .arg_u32(cfg.num_experts as u32)
            .arg_u32(top_k as u32)
            .arg_u32(u32::from(cfg.norm_topk_prob))
            .arg_u32(n_logits)
            .launch(stream)?;
        let mut ib = vec![0u8; n * top_k * 4];
        ctx.gpu.copy_d2h_on_stream(idx_dev, &mut ib, stream)?;
        let ids: Vec<u32> = ib
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        with_exl3_moe(gate_key, |pack| -> Result<usize> {
            self.grouped_body(pack, &mut guard, tables, input, n, &ids, w_dev, ctx, stream)
        })
        .ok_or_else(|| anyhow::anyhow!("EXL3 MoE pack missing for gate {gate_key:#x}"))?
    }

    /// Debug: per-layer MoE wall time (sync-bounded) of the per-token arm vs
    /// the grouped arm on the first m rows of this call.
    fn grouped_bench(
        &self,
        input: DevicePtr,
        n: usize,
        tables: LayerTables,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let name = with_exl3_moe(self.weights.shared_expert_gate.weight.0, |p| p.name.clone())
            .unwrap_or_default();
        let time = |f: &dyn Fn() -> Result<()>, reps: usize| -> Result<f64> {
            let mut v = Vec::with_capacity(reps);
            for _ in 0..reps {
                ctx.gpu.synchronize(stream)?;
                let t = Instant::now();
                f()?;
                ctx.gpu.synchronize(stream)?;
                v.push(t.elapsed().as_secs_f64() * 1e3);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(v[v.len() / 2])
        };
        let per_token = |m: usize| -> Result<()> {
            BYPASS.store(true, Ordering::Relaxed);
            let r = self.try_forward_exl3(input, m, ctx, stream);
            BYPASS.store(false, Ordering::Relaxed);
            r.map(|_| ())
        };
        let grouped = |m: usize| -> Result<()> {
            self.grouped_run(input, m, tables, ctx, stream).map(|_| ())
        };
        // warm both (scratch sizing, first launches)
        per_token(1)?;
        grouped(n.min(2))?;
        let mut line = String::new();
        for m in [1usize, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 512] {
            if m > n {
                break;
            }
            let reps = if m <= 64 { 3 } else { 1 };
            let a = time(&|| per_token(m), reps)?;
            let b = time(&|| grouped(m), reps)?;
            line.push_str(&format!(" m{m}={a:.3}/{b:.3}"));
        }
        tracing::info!("EXL3 grouped bench {name} (per-token ms / grouped ms):{line}");
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn grouped_body(
        &self,
        pack: &Exl3MoePack,
        guard: &mut Option<Scratch>,
        tables: LayerTables,
        input: DevicePtr,
        n: usize,
        ids: &[u32],
        w_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let gpu = ctx.gpu;
        let cfg = ctx.config;
        let (h, inter, k) = (
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
        );
        let n_routed = pack.experts.len();
        let shared = n_routed;
        let has_shared = !pack.shared.is_null();
        let n_local = tables.n_local;

        // ── Sort (token, slot) rows by expert: counting sort, (t, s) order kept.
        let mut count = vec![0usize; n_local];
        for &id in ids {
            let e = id as usize;
            if e >= n_routed || pack.experts[e].is_null() {
                bail!(
                    "EXL3 grouped prefill: selected expert {id} has no packed weights ({n_routed} routed)"
                );
            }
            count[e] += 1;
        }
        if has_shared {
            count[shared] = n;
        }
        let mut off = vec![0usize; n_local + 1];
        for e in 0..n_local {
            off[e + 1] = off[e] + count[e];
        }
        let rows = off[n_local];
        // D (bf16 down rows) holds n*k routed rows then n shared rows.
        let d_rows = n * k + n;
        let mut cur = off.clone();
        let mut row_src = vec![0u32; rows];
        let mut row_exp = vec![0u32; rows];
        let mut row_dst = vec![0u32; rows];
        for t in 0..n {
            for s in 0..k {
                let e = ids[t * k + s] as usize;
                let p = cur[e];
                cur[e] += 1;
                row_src[p] = t as u32;
                row_exp[p] = e as u32;
                row_dst[p] = (t * k + s) as u32;
            }
        }
        if has_shared {
            for t in 0..n {
                let p = cur[shared];
                cur[shared] += 1;
                row_src[p] = t as u32;
                row_exp[p] = shared as u32;
                row_dst[p] = (n * k + t) as u32;
            }
        }

        let n_items_max = 2 * (n_local + rows / ITEM_ROWS + 1);
        let meta_bytes = rows * 12 + 16 + n_items_max * ITEM_BYTES * 2 + 64;
        let sc = ensure_scratch(guard, gpu, rows.max(d_rows), meta_bytes, h, inter, stream)?;

        // ── Items, bucketed by bit rate (one mma launch per K per stage).
        let (xg, xu, yg, yu) = (sc.xg.0, sc.xu.0, sc.yg.0, sc.yu.0);
        let mut gu: Vec<Vec<u8>> = vec![Vec::new(); 7];
        let mut dn: Vec<Vec<u8>> = vec![Vec::new(); 7];
        for e in 0..n_local {
            if count[e] == 0 {
                continue;
            }
            let x = if e == shared {
                &pack.shared
            } else {
                &pack.experts[e]
            };
            let mut r0 = off[e];
            while r0 < off[e + 1] {
                let nr = (off[e + 1] - r0).min(ITEM_ROWS);
                let (ah, ai) = ((r0 * h * 2) as u64, (r0 * inter * 2) as u64);
                let bg = gu.get_mut(x.gate.shape.bits as usize - 2).unwrap();
                push_item(bg, x.gate.trellis.0, xg + ah, yg + ai, nr);
                let bu = gu.get_mut(x.up.shape.bits as usize - 2).unwrap();
                push_item(bu, x.up.trellis.0, xu + ah, yu + ai, nr);
                let bd = dn.get_mut(x.down.shape.bits as usize - 2).unwrap();
                push_item(bd, x.down.trellis.0, yg + ai, xg + ah, nr);
                r0 += nr;
            }
        }
        let mut meta: Vec<u8> = Vec::with_capacity(meta_bytes);
        for v in [&row_src, &row_exp, &row_dst] {
            meta.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        while !meta.len().is_multiple_of(16) {
            meta.push(0);
        }
        let mut gu_launch: Vec<(usize, usize, u32)> = Vec::new(); // (byte off, items, bits)
        let mut dn_launch: Vec<(usize, usize, u32)> = Vec::new();
        for (list, out) in [(&gu, &mut gu_launch), (&dn, &mut dn_launch)] {
            for (i, b) in list.iter().enumerate() {
                if !b.is_empty() {
                    out.push((meta.len(), b.len() / ITEM_BYTES, i as u32 + 2));
                    meta.extend_from_slice(b);
                }
            }
        }
        if meta.len() > sc.meta_cap {
            bail!(
                "EXL3 grouped prefill: metadata {} B > scratch {} B",
                meta.len(),
                sc.meta_cap
            );
        }
        gpu.copy_h2d_async(&meta, sc.meta, stream)?;
        let d_src = sc.meta;
        let d_exp = sc.meta.offset(rows * 4);
        let d_dst = sc.meta.offset(rows * 8);
        let kn = sc.kernels;
        let r = rows as u32;

        // 1. xg / xu = had(f16(x[token]) * suh_gate / suh_up)
        KernelLaunch::new(gpu, kn.gather_had)
            .grid([r, div_ceil(h as u32 / 128, 4), 2])
            .block([32, 4, 1])
            .arg_ptr(input)
            .arg_ptr(d_src)
            .arg_ptr(d_exp)
            .arg_ptr(sc.xg)
            .arg_ptr(sc.xu)
            .arg_ptr(tables.tab(0))
            .arg_ptr(tables.tab(1))
            .arg_u32(h as u32)
            .arg_f32(HAD_SCALE)
            .launch(stream)?;
        // 2. yg / yu = xg / xu * W_inner(gate / up), per expert group
        for &(o, items, bits) in &gu_launch {
            KernelLaunch::new(gpu, kn.mma[bits as usize - 2])
                .grid([inter as u32 / 128, items as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(sc.meta.offset(o))
                .arg_i32(h as i32)
                .arg_i32(inter as i32)
                .arg_i32((inter / 16) as i32)
                .arg_i32((h / 16) as i32)
                .launch(stream)?;
        }
        // 3. bf16(had(y) * svh), in place
        KernelLaunch::new(gpu, kn.had_post_bf16)
            .grid([r, div_ceil(inter as u32 / 128, 4), 2])
            .block([32, 4, 1])
            .arg_ptr(sc.yg)
            .arg_ptr(sc.yu)
            .arg_ptr(sc.yg)
            .arg_ptr(sc.yu)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(d_exp)
            .arg_ptr(tables.tab(3))
            .arg_ptr(tables.tab(4))
            .arg_u32(inter as u32)
            .arg_f32(HAD_SCALE)
            .launch(stream)?;
        // 4. SiLU(gate) * up -> xu as bf16 [rows, I] (the interim's kernel;
        //    xu is free once the up GEMM ran, and its args are __restrict__).
        ops::moe_silu_mul(
            gpu,
            self.moe_silu_mul,
            sc.yg,
            sc.yu,
            sc.xu,
            r * inter as u32,
            stream,
        )?;
        // 5. down input: yg = had(f16(act) * suh_down)
        KernelLaunch::new(gpu, kn.gather_had)
            .grid([r, div_ceil(inter as u32 / 128, 4), 1])
            .block([32, 4, 1])
            .arg_ptr(sc.xu)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(d_exp)
            .arg_ptr(sc.yg)
            .arg_ptr(sc.yg)
            .arg_ptr(tables.tab(2))
            .arg_ptr(tables.tab(2))
            .arg_u32(inter as u32)
            .arg_f32(HAD_SCALE)
            .launch(stream)?;
        // 6. down GEMM -> xg
        for &(o, items, bits) in &dn_launch {
            KernelLaunch::new(gpu, kn.mma[bits as usize - 2])
                .grid([h as u32 / 128, items as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(sc.meta.offset(o))
                .arg_i32(inter as i32)
                .arg_i32(h as i32)
                .arg_i32((h / 16) as i32)
                .arg_i32((inter / 16) as i32)
                .launch(stream)?;
        }
        // 7. bf16(had(y) * svh_down) scattered to D[t * k + s] / D[n * k + t] (= xu)
        KernelLaunch::new(gpu, kn.had_post_bf16)
            .grid([r, div_ceil(h as u32 / 128, 4), 1])
            .block([32, 4, 1])
            .arg_ptr(sc.xg)
            .arg_ptr(sc.xg)
            .arg_ptr(sc.xu)
            .arg_ptr(sc.xu)
            .arg_ptr(d_dst)
            .arg_ptr(d_exp)
            .arg_ptr(tables.tab(5))
            .arg_ptr(tables.tab(5))
            .arg_u32(h as u32)
            .arg_f32(HAD_SCALE)
            .launch(stream)?;
        let shared_out = sc.xu.offset(n * k * h * 2);
        if !has_shared {
            gpu.memset_async(shared_out, 0, n * h * 2, stream)?;
        }
        // 8. weighted sum + sigmoid(x . gate) * shared: the per-token blend's
        //    arithmetic, one block row per token.
        ops::moe_weighted_sum_blend_prefill(
            gpu,
            self.moe_weighted_sum_blend_token_major,
            ctx.buffers.moe_output(),
            sc.xu,
            w_dev,
            shared_out,
            input,
            self.weights.shared_expert_gate.weight,
            h as u32,
            k as u32,
            h as u32,
            n as u32,
            stream,
        )?;
        Ok(rows)
    }

    /// Debug: the per-token arm on the same input; log the difference. The
    /// grouped result is restored afterwards (the trajectory stays grouped).
    fn grouped_check(
        &self,
        input: DevicePtr,
        n: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let id = CHECK_DONE.fetch_add(1, Ordering::Relaxed);
        let h = ctx.config.hidden_size;
        let out = ctx.buffers.moe_output();
        let bytes = n * h * 2;
        let mut a = vec![0u8; bytes];
        ctx.gpu.copy_d2h_on_stream(out, &mut a, stream)?;
        BYPASS.store(true, Ordering::Relaxed);
        let r = self.try_forward_exl3(input, n, ctx, stream);
        BYPASS.store(false, Ordering::Relaxed);
        r?;
        let mut b = vec![0u8; bytes];
        ctx.gpu.copy_d2h_on_stream(out, &mut b, stream)?;
        ctx.gpu.copy_h2d(&a, out)?;
        let (x, y) = (bf16_to_f64(&a), bf16_to_f64(&b));
        let (mut dot, mut nx, mut ny, mut maxd, mut maxy, mut same) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0usize);
        let mut row_cos_min = 1f64;
        for t in 0..n {
            let (mut d, mut p, mut q) = (0f64, 0f64, 0f64);
            for j in 0..h {
                let (u, v) = (x[t * h + j], y[t * h + j]);
                d += u * v;
                p += u * u;
                q += v * v;
                maxd = maxd.max((u - v).abs());
                maxy = maxy.max(v.abs());
                same += usize::from(
                    a[2 * (t * h + j)..2 * (t * h + j) + 2]
                        == b[2 * (t * h + j)..2 * (t * h + j) + 2],
                );
            }
            dot += d;
            nx += p;
            ny += q;
            row_cos_min = row_cos_min.min(d / (p.sqrt() * q.sqrt()).max(1e-300));
        }
        let finite = x.iter().all(|v| v.is_finite());
        let name = with_exl3_moe(self.weights.shared_expert_gate.weight.0, |p| p.name.clone())
            .unwrap_or_default();
        tracing::info!(
            "EXL3 grouped prefill check #{id} {name} ({n} tokens) vs per-token arm: max|d| {:.4e} (max|ref| {:.4e}, rel {:.3e}) \
             cos {:.8} min row cos {:.8} bitwise-equal {:.4} finite {finite}",
            maxd,
            maxy,
            maxd / maxy.max(1e-30),
            dot / (nx.sqrt() * ny.sqrt()).max(1e-300),
            row_cos_min,
            same as f64 / (n * h) as f64
        );
        Ok(())
    }
}
