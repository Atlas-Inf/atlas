// SPDX-License-Identifier: AGPL-3.0-only

//! Fast safetensors loader (InstantTensor-style) — pure Rust.
//!
//! Two wins over the mmap-based loader in [`crate::weights`]:
//!
//! 1. **`O_DIRECT`** reads. Bypasses the OS page cache, so the bytes never
//!    compete with GPU allocations on GB10 unified memory. The mmap path
//!    already works around this with `POSIX_FADV_DONTNEED` post-load; here
//!    we avoid the pollution in the first place.
//! 2. **Pipelined read/copy**. One background reader thread fetches the
//!    next tensor while the main thread does `copy_h2d` for the current
//!    one. Overlaps disk I/O with the host→device memcpy.
//!
//! Behavioural parity with [`crate::weights::SafetensorsLoader`] is
//! preserved — same EP filtering, same OOM pre-flight, same UVM fallback
//! on GPU allocation failure, same extra-weights handling.

use crate::gpu::GpuBackend;
use crate::weights::{
    WeightLoader, WeightStore, WeightTensor, check_oom_guard, estimate_has_fp8, estimate_load_bytes,
};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

mod direct_io;
mod header;
mod shard;

use header::resolve_shards;
use shard::load_shard_fast;

/// Run `f` over `len` bytes at `start` within a deferred tensor, read with
/// `O_DIRECT` when the filesystem allows it (so the bytes never sit in the
/// page cache on GB10 unified memory), else buffered.
pub fn with_deferred_bytes<T>(
    d: &crate::weights::DeferredTensor,
    start: usize,
    len: usize,
    f: impl FnOnce(&[u8]) -> Result<T>,
) -> Result<T> {
    use std::os::unix::io::AsRawFd;
    let (file, direct) = match direct_io::open_direct(&d.path) {
        Ok(file) => (file, true),
        Err(_) => (std::fs::File::open(&d.path)?, false),
    };
    let (buf, off) =
        direct_io::read_tensor_aligned(file.as_raw_fd(), d.offset + start as u64, len, direct)?;
    f(&buf.as_slice()[off..off + len])
}

/// Pure-Rust InstantTensor-style loader. Same public shape as
/// [`crate::weights::SafetensorsLoader`].
pub struct FastSafetensorsLoader {
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
    /// Optional layer-name fragment whose expert tensors are loaded only on
    /// EP rank 0, with all experts replicated there. Used by model-specific
    /// draft modules that execute entirely on the coordinator rank.
    pub rank0_only_expert_prefix: Option<String>,
    /// Optional layer-name fragment whose expert tensors bypass EP filtering
    /// and are replicated on every rank. Used when a small draft module keeps
    /// identical local arithmetic while distributing only its output head.
    pub replicated_expert_prefix: Option<String>,
    pub peak_memory_multiplier: Option<f64>,
    /// Skip the W4A4 `*.input_scale` activation scales at load.
    ///
    /// ModelOpt NVFP4 checkpoints ship one 0-dim F32 scalar per quantized
    /// projection. On a 512-expert model that is ~74k four-byte allocations,
    /// each taking a full allocation granule — GBs of padding for values
    /// Atlas never reads, because it serves w4a16 (BF16 activations) and the
    /// NVFP4 loader already treats the key as optional.
    ///
    /// OPT-IN: `step3p7` reads this key on its own path, so it must stay off
    /// unless the model's loader is known not to need it.
    pub skip_activation_scales: bool,
    /// Skip `mtp.*` tensors at load.
    ///
    /// For models whose loader deliberately does not build an MTP head,
    /// uploading its weights is pure waste — on Qwen3.8-Flash-Next that is a
    /// 1.49 GB expert shard plus the MTP backbone, held resident while the KV
    /// cache goes without.
    ///
    /// OPT-IN: a model that DOES build an MTP head must keep them, so this is
    /// set only where `load_mtp_weights` is known to return `None`.
    pub skip_mtp: bool,
    /// Exact tensor prefix of an unused appended predictor layer; default retains it.
    pub skip_layer_prefix: Option<String>,
    /// When true (default), attempt `O_DIRECT`; fall back to buffered reads if
    /// the filesystem rejects it (tmpfs, overlayfs, some FUSE backends).
    pub try_direct_io: bool,
    /// Per-shard heuristic cap: if a shard's tensor count exceeds this,
    /// we skip `O_DIRECT` for that shard and fall back to buffered +
    /// pipelined reads even when [`Self::try_direct_io`] is `true`.
    ///
    /// Motivation: `O_DIRECT`'s 4 KiB-aligned per-tensor `pread` has a
    /// fixed syscall + copy overhead that kernel readahead amortises for
    /// free on the buffered path. Benchmarks on GB10 showed buffered wins
    /// above ~5k tensors/shard; O_DIRECT wins below. Set to [`usize::MAX`]
    /// to disable.
    pub direct_io_tensor_cap: usize,
    /// When true, advise the kernel to read a whole buffered shard
    /// sequentially before the per-tensor copy loop starts. This helps NFS
    /// mounts where many small tensor reads defeat normal readahead.
    pub prefetch_shards: bool,
    /// Name substrings marking tensors that are **read from disk at use time
    /// and never made resident**. They are excluded from the load AND from the
    /// pre-flight estimate, since counting bytes we will not allocate refuses
    /// models that would in fact fit.
    ///
    /// This exists for embedding tables that are gathered by row rather than
    /// multiplied. Qwen3.8-Flash-Next's n-gram table is 51.2 B parameters —
    /// 41% of that checkpoint — and one token touches ~2.5 KB of it. Loading
    /// it puts the pre-flight at 163 GB on a 119 GB box; skipping it puts the
    /// same model at ~96 GB.
    ///
    /// A tensor named here MUST have a reader that goes to disk
    /// (`atlas_core::ngram_table` is the one that does). Nothing checks that
    /// from here — a name listed without a reader is simply absent from the
    /// store, and the model loader will fail on it by name.
    pub demand_paged_patterns: Vec<String>,
    /// Expert TP: keep every routed expert (no EP filter) but defer their
    /// packed weights and block scales, which the model loader reads from
    /// disk as this rank's slice (see `should_defer_tensor`).
    pub expert_tp: bool,
}

/// Default tensor-count cap for per-shard `O_DIRECT`. Above this, the fast
/// loader uses buffered reads even when `try_direct_io = true`. See the
/// field doc on [`FastSafetensorsLoader::direct_io_tensor_cap`].
pub const DEFAULT_DIRECT_IO_TENSOR_CAP: usize = 5000;

impl Default for FastSafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[path = "skip.rs"]
mod skip;

impl FastSafetensorsLoader {
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            rank0_only_expert_prefix: None,
            replicated_expert_prefix: None,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            skip_layer_prefix: None,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            prefetch_shards: false,
            demand_paged_patterns: Vec::new(),
            expert_tp: false,
        }
    }

    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            rank0_only_expert_prefix: None,
            replicated_expert_prefix: None,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            skip_layer_prefix: None,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            prefetch_shards: false,
            demand_paged_patterns: Vec::new(),
            expert_tp: false,
        }
    }
}

impl WeightLoader for FastSafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let skip_fn = |name: &str| self.should_skip_tensor(name);

        // Resolve shard list (sharded index, single file, or unindexed shards).
        let (shard_files, tensor_to_shard): (Vec<PathBuf>, Option<HashMap<String, String>>) =
            resolve_shards(model_dir)?;

        // Pre-flight OOM estimate (identical to SafetensorsLoader).
        //
        // The n-gram tables are DEFERRED further down — they are never
        // uploaded, so counting them here refuses a model that fits. On
        // LongCat-Flash-Lite they are 62.8 of the checkpoint's 138 GB, which
        // is the difference between a 167 GB "peak" and a 98 GB one.
        let defer_fn = |name: &str| self.should_defer_tensor(name);
        let preflight_skip =
            |name: &str| skip_fn(name) || defer_fn(name) || crate::weights::is_ngram_table(name);
        {
            let estimated = estimate_load_bytes(&shard_files, &preflight_skip)?;
            let has_fp8 = estimate_has_fp8(&shard_files, &preflight_skip)?;
            let mult = self
                .peak_memory_multiplier
                .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            let peak = (estimated as f64 * mult) as usize;
            let free = gpu.free_memory()?;
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Fast-load pre-flight: {:.2} GB on-disk, {:.1}x overhead = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8: {})",
                gib(estimated),
                mult,
                gib(peak),
                gib(free),
                gib(oom_reserve_bytes),
                has_fp8,
            );
            crate::progress::preflight(gib(estimated), gib(free));
            if peak + oom_reserve_bytes > free {
                bail!(
                    "OOM pre-flight: peak {:.2} GB + {:.2} GB reserve exceeds {:.2} GB free. \
                     Use a smaller quantization or add more GPUs for EP.",
                    gib(peak),
                    gib(oom_reserve_bytes),
                    gib(free),
                );
            }
        }

        // Load each shard. Loaded tensors filtered by EP rules upstream.
        let mut weights: HashMap<String, WeightTensor> = HashMap::new();
        // Locations of tensors deliberately NOT uploaded (the n-gram tables).
        let mut deferred: HashMap<String, crate::weights::DeferredTensor> = HashMap::new();
        let total_shards = shard_files.len();
        let initial_free = gpu.free_memory()?;
        let mut offload_logged = false;

        for (i, shard_path) in shard_files.iter().enumerate() {
            // When an index is present, only load the tensors it routes here;
            // otherwise load everything in the shard. `None` means "load all".
            let shard_name = shard_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let tensor_filter: Option<Vec<String>> = tensor_to_shard.as_ref().map(|map| {
                map.iter()
                    .filter(|(_, s)| *s == shard_name)
                    .map(|(t, _)| t.clone())
                    .collect()
            });

            tracing::info!(
                "Fast-loading shard {}/{}: {}{}",
                i + 1,
                total_shards,
                shard_name,
                tensor_filter
                    .as_ref()
                    .map(|v| format!(" ({} tensors)", v.len()))
                    .unwrap_or_default(),
            );
            crate::progress::shard_start(i + 1, total_shards, shard_name);

            load_shard_fast(
                shard_path,
                tensor_filter.as_deref(),
                gpu,
                &skip_fn,
                &defer_fn,
                self.try_direct_io,
                self.direct_io_tensor_cap,
                self.prefetch_shards,
                &mut weights,
                &mut deferred,
                &mut offload_logged,
            )?;

            let free_now = gpu.free_memory().unwrap_or(0);
            let used = initial_free.saturating_sub(free_now);
            tracing::info!(
                "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            crate::progress::shard_done(
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            if !offload_logged {
                check_oom_guard(
                    gpu,
                    oom_reserve_bytes,
                    &format!("fast weight loading (shard {}/{})", i + 1, total_shards),
                )?;
            }
        }

        // Extra weights (e.g. MTP grafted from another quantization).
        let skip_unused = |name: &str| {
            self.skip_layer_prefix
                .as_ref()
                .is_some_and(|prefix| name.starts_with(prefix))
        };
        let extra = model_dir.join("extra_weights.safetensors");
        if extra.exists() {
            tracing::info!("Fast-loading extra_weights.safetensors");
            let mut extra_offload = false;
            load_shard_fast(
                &extra,
                None,
                gpu,
                &skip_unused,
                &|_| false,
                self.try_direct_io,
                self.direct_io_tensor_cap,
                self.prefetch_shards,
                &mut weights,
                &mut deferred,
                &mut extra_offload,
            )?;
        }

        tracing::info!("Fast-loaded {} weight tensors", weights.len());
        let mut store = WeightStore::from_map(weights);
        for (name, d) in deferred {
            store.defer(name, d);
        }
        Ok(store)
    }
}

#[cfg(test)]
mod expert_filter_tests {
    use super::FastSafetensorsLoader;

    #[test]
    fn replicated_prefix_overrides_ep_expert_filter() {
        let mut loader = FastSafetensorsLoader::with_ep(1, 2, 288);
        let appended = "model.language_model.layers.45.mlp.experts.1.gate_proj.weight";
        let target = "model.language_model.layers.44.mlp.experts.1.gate_proj.weight";
        assert!(loader.should_skip_tensor(appended));
        loader.replicated_expert_prefix = Some(".layers.45.".into());
        assert!(!loader.should_skip_tensor(appended));
        assert!(loader.should_skip_tensor(target));
    }

    #[test]
    fn expert_tp_keeps_every_expert_and_defers_only_its_bulk() {
        let mut loader = FastSafetensorsLoader::with_ep(1, 2, 288);
        loader.expert_tp = true;
        loader.rank0_only_expert_prefix = Some(".layers.45.".into());
        let p = "model.language_model.layers.44.mlp.experts.1.gate_proj";
        for suffix in ["weight", "weight_scale"] {
            let name = format!("{p}.{suffix}");
            assert!(!loader.should_skip_tensor(&name));
            assert!(loader.should_defer_tensor(&name));
        }
        for suffix in ["weight_scale_2", "input_scale"] {
            let name = format!("{p}.{suffix}");
            assert!(!loader.should_skip_tensor(&name));
            assert!(!loader.should_defer_tensor(&name));
        }
        let appended = "model.language_model.layers.45.mlp.experts.1.gate_proj.weight";
        assert!(loader.should_skip_tensor(appended));
        assert!(!loader.should_defer_tensor(appended));
        let shared = "model.language_model.layers.44.mlp.shared_experts.gate_proj.weight";
        assert!(!loader.should_defer_tensor(shared));
    }
}
