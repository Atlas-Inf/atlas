// SPDX-License-Identifier: AGPL-3.0-only

//! Row tiles of the prequant routed FFN for GLM verify decode, and the M16
//! twins of the K128W kernels (`ATLAS_GLM_MOE_DECODE_M16=1`, default off).
//!
//! Top-k picks an expert at most once per row, so a batch of at most
//! [`MAX_ROWS`] rows leaves no expert more than one m16 MMA slab. The compact
//! k64 gate/up, `silu_mul_quant_nvfp4` and dense K128 down that verify decode
//! launches otherwise pad every tile to 64 rows; the twins compute the same
//! MMAs per output element over 16-row tiles with the SiLU in the gate/up
//! epilogue, so the bytes are identical (`scripts/moe-decode-bench`).
//!
//! Coverage is the verify-decode batches only ([`MoeLayer::verify_decode_batch`]):
//! one DFlash verify block or an owner-batched verify of 2..=8 rows in total
//! (`independent_grouped`), owner-batched 3-row blocks (6, 9 or 12 rows, C3
//! grouped), and the C2/C4/K5 shapes. One-row decode does not reach the
//! grouped FFN, and an owner-batched verify of more than 8 rows at another
//! width is a plain `forward_prefill` batch, which keeps the K128W grid.
//!
//! `ATLAS_GLM_MOE_DOWN_ZSKIP=1` (with the flag above) launches the down twin
//! that does not load weight rows whose activations are E2M1 zeros in every
//! row of the tile: those MMA terms are exact zeros, so again the same bytes.
//!
//! `ATLAS_GLM_MOE_DECODE_K128W=1` is the control with no new kernel: verify
//! decode batches the twins do not take run the prefill K128W grid (M64
//! tiles, same bytes) in place of the compact worklist.
//!
//! `ATLAS_GLM_MOE_DECODE_STREAM=1` (with the M16 flag) launches the twins'
//! stream-loaded versions (glm_moe_decode_stream.cuh: streaming loads in place
//! of cp.async, same bytes), and lets them take every prequant routed FFN of
//! up to [`STREAM_MAX_ROWS`] rows, verify decode or not: batches of more than
//! 16 rows (owner batches of two to four streams) run the two-slab kernels,
//! and so do prefill chunks of up to 32 tokens. A target without all of the
//! stream kernels runs the M16 pair, as without this flag.
//!
//! `ATLAS_GLM_MOE_DECODE_L2PF=1` (with the stream twins) launches their
//! `_l2pf` twins, whose CTAs also ask each K stage's whole table rows into L2
//! a few stages ahead, every CTA of an expert its share of the rows, so DRAM
//! serves the expert's stages as contiguous runs instead of 128- and 256-byte
//! strips (glm_moe_decode_stream.cuh); `=2` takes the prefetching gate/up
//! only. Prefetches only, so the same bytes. A target lacking any of the
//! `_l2pf` twins keeps the stream kernels.
//!
//! `ATLAS_GLM_MOE_DECODE_PERSIST=1` (with `ATLAS_GLM_MOE_DECODE_L2PF=1` and
//! the zero-row skip) launches the prefetching pair's persistent twins: one
//! CTA per SM, each striding over the worklist's (expert, column tile) items
//! and asking its next item's first stages into L2 while it finishes the
//! last, so fewer experts' tables stream at once and an item's start is
//! hidden. Each item's loads, MMAs and epilogue are the grid kernels', so the
//! same bytes (glm_moe_decode_stream.cuh). A target lacking any of them keeps
//! the prefetching twins.
//!
//! `ATLAS_GLM_MOE_STREAM_NOSYNC=1` (with the stream twins) skips the exact
//! M64 tile count (`ATLAS_MOE_PREFILL_EXACT_TILES`, default on for NVFP4) for
//! the batches the stream twins take ([`MoeLayer::offsets_unread`]). That
//! count is a host copy of the expert offsets, which drains the stream in
//! every MoE layer of a batch of 9 to 32 rows (owner-batched verify of two to
//! four streams), and only sizes the M64 grids, which those batches never
//! launch: the twins' grid bound is fixed on the host. The same kernels run
//! with the same arguments, so the bytes are the same.

use super::prequant_fp4::{CompactMoeWorklist, MtileGrid};
use super::*;

/// Rows of a routed FFN batch the M16 tiles cover: one m16 MMA slab.
pub(super) const MAX_ROWS: u32 = 16;

/// Rows the two-slab stream twins cover.
pub(super) const STREAM_MAX_ROWS: u32 = 2 * MAX_ROWS;

/// Widest batch whose tile choice is logged: an owner-batched verify.
const NOTE_ROWS: u32 = crate::layer::glm_long_owner::MAX_ROWS as u32;

/// The verify-decode tile selection: the M16 fused gate/up and down kernels
/// or, under ATLAS_GLM_MOE_DECODE_STREAM, their one-slab stream twins (both
/// null unless the flag is on and the target ships them), the two-slab stream
/// pair (null unless the stream twins are loaded), and the K128W control.
#[derive(Clone, Copy)]
pub(super) struct DecodeM16 {
    gate_up_silu: KernelHandle,
    down: KernelHandle,
    wide: [KernelHandle; 2],
    zskip: bool,
    k128w: bool,
    /// `ATLAS_GLM_MOE_STREAM_NOSYNC`.
    nosync: bool,
    /// CTAs of the persistent twins (`ATLAS_GLM_MOE_DECODE_PERSIST`, the
    /// SMs), 0 for the grid kernels.
    ctas: u32,
}

impl DecodeM16 {
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        let toggle = |name: &str| match std::env::var(name).as_deref() {
            Err(_) | Ok("0") => Ok(false),
            Ok("1") => Ok(true),
            Ok(value) => anyhow::bail!("{name} requires 0 or 1, got {value:?}"),
        };
        let requested = toggle("ATLAS_GLM_MOE_DECODE_M16")?;
        let zskip = toggle("ATLAS_GLM_MOE_DOWN_ZSKIP")?;
        let stream = toggle("ATLAS_GLM_MOE_DECODE_STREAM")?;
        // Some(whether the downs prefetch too).
        let l2pf = match std::env::var("ATLAS_GLM_MOE_DECODE_L2PF").as_deref() {
            Err(_) | Ok("0") => None,
            Ok("1") => Some(true),
            Ok("2") => Some(false),
            Ok(value) => {
                anyhow::bail!("ATLAS_GLM_MOE_DECODE_L2PF requires 0, 1 or 2, got {value:?}")
            }
        };
        let glm = config.model_type == "glm5_next";
        let on = requested && glm;
        let kernel = |on, name: &str| super::super::try_kernel_gated(on, gpu, "moe_w4a16", name);
        // A family's fused gate/up and (zero-skipping) down.
        let pair = |on, family: &str, suffix: &str| {
            let skip = if zskip { "_zskip" } else { "" };
            [
                kernel(
                    on,
                    &format!("glm_moe_decode_{family}_gate_up_silu_k128w{suffix}"),
                ),
                kernel(on, &format!("glm_moe_decode_{family}_k128w{skip}{suffix}")),
            ]
        };
        // A set of twins of both slabs, all or none: a target lacking any of
        // the stream twins runs the M16 pair, as without the flag, and one
        // lacking any of their prefetching twins the stream twins.
        let twins = |on, suffix| {
            let twins = [pair(on, "m16s", suffix), pair(on, "m32s", suffix)];
            twins.iter().flatten().all(|k| k.0 != 0).then_some(twins)
        };
        let streams = twins(on && stream, "");
        let prefetch = twins(streams.is_some() && l2pf.is_some(), "_l2pf");
        let persist = toggle("ATLAS_GLM_MOE_DECODE_PERSIST")?;
        let persistent = twins(
            persist && prefetch.is_some() && l2pf == Some(true) && zskip,
            "_l2pf_p",
        );
        let [[mut gate_up_silu, mut down], mut wide] =
            streams.unwrap_or_else(|| [pair(on, "m16", ""), [KernelHandle(0); 2]]);
        if let Some([[gate_up16, down16], [gate_up32, down32]]) = persistent.or(prefetch) {
            [gate_up_silu, wide[0]] = [gate_up16, gate_up32];
            if l2pf == Some(true) {
                [down, wide[1]] = [down16, down32];
            }
        }
        let ctas = match persistent {
            Some(_) => gpu.sm_count()?,
            None => 0,
        };
        let mut this = Self {
            gate_up_silu,
            down,
            wide,
            zskip,
            k128w: toggle("ATLAS_GLM_MOE_DECODE_K128W")? && glm,
            nosync: toggle("ATLAS_GLM_MOE_STREAM_NOSYNC")? && glm,
            ctas,
        };
        if !this.loaded() {
            // Both or neither: half a pair never launches.
            this.gate_up_silu = KernelHandle(0);
        }
        if this.nosync && gpu.op_cache().once("moe:stream_nosync") {
            if this.stream() {
                tracing::info!(
                    "ATLAS_GLM_MOE_STREAM_NOSYNC: no expert-offset read for the stream twins' batches"
                );
            } else {
                tracing::warn!(
                    "ATLAS_GLM_MOE_STREAM_NOSYNC=1 ignored: it needs the stream twins (ATLAS_GLM_MOE_DECODE_M16=1 ATLAS_GLM_MOE_DECODE_STREAM=1)"
                );
            }
        }
        if (requested || zskip || stream || l2pf.is_some() || persist)
            && gpu.op_cache().once("moe:decode_m16")
        {
            if this.loaded() {
                tracing::info!(
                    "ATLAS_GLM_MOE_DECODE_M16: M16 routed gate/up+SiLU and down for verify decode (zero-row skip: {zskip})"
                );
                if stream && this.stream() {
                    tracing::info!(
                        "ATLAS_GLM_MOE_DECODE_STREAM: stream-loaded M16 twins for every routed FFN of up to {STREAM_MAX_ROWS} rows"
                    );
                } else if stream {
                    tracing::warn!(
                        "ATLAS_GLM_MOE_DECODE_STREAM=1 ignored: target lacks the stream kernels"
                    );
                }
                if let Some(downs) = l2pf {
                    if prefetch.is_some() {
                        tracing::info!(
                            "ATLAS_GLM_MOE_DECODE_L2PF: stream twins with the L2 row prefetch (gate/up, downs: {downs})"
                        );
                    } else {
                        tracing::warn!(
                            "ATLAS_GLM_MOE_DECODE_L2PF ignored: needs the stream twins and the target's _l2pf twins"
                        );
                    }
                }
                if persistent.is_some() {
                    tracing::info!(
                        "ATLAS_GLM_MOE_DECODE_PERSIST: persistent prefetching twins over {ctas} CTAs"
                    );
                } else if persist {
                    tracing::warn!(
                        "ATLAS_GLM_MOE_DECODE_PERSIST=1 ignored: needs ATLAS_GLM_MOE_DECODE_L2PF=1, ATLAS_GLM_MOE_DOWN_ZSKIP=1 and the target's _l2pf_p twins"
                    );
                }
            } else if on {
                tracing::warn!("ATLAS_GLM_MOE_DECODE_M16=1 ignored: target lacks the M16 kernels");
            } else if !requested {
                tracing::warn!(
                    "ATLAS_GLM_MOE_DOWN_ZSKIP / _DECODE_STREAM / _DECODE_L2PF / _DECODE_PERSIST ignored: they need ATLAS_GLM_MOE_DECODE_M16=1"
                );
            }
        }
        Ok(this)
    }

    fn loaded(&self) -> bool {
        self.gate_up_silu.0 != 0 && self.down.0 != 0
    }

    /// Whether the stream twins are loaded: then they take every batch they
    /// fit, up to [`STREAM_MAX_ROWS`] rows.
    fn stream(&self) -> bool {
        self.wide[0].0 != 0
    }

    /// The fused gate/up and down kernels of a batch of `rows` rows: one slab
    /// up to [`MAX_ROWS`], else the stream pair's two.
    fn pair(&self, rows: u32) -> [KernelHandle; 2] {
        if rows <= MAX_ROWS {
            [self.gate_up_silu, self.down]
        } else {
            self.wide
        }
    }

    /// Log, once per backend and row count, whether a routed FFN batch of
    /// `rows` rows ran the twins: a rank's proof that the flag engaged, and
    /// the batches it does not cover.
    fn note(&self, gpu: &dyn GpuBackend, rows: u32, engaged: bool) {
        if !self.loaded() || rows > NOTE_ROWS {
            return;
        }
        if gpu
            .op_cache()
            .first_shape("moe:decode_m16", rows, engaged as u32, 0)
        {
            if engaged {
                tracing::info!(
                    "ATLAS_GLM_MOE_DECODE_M16 engaged: rows={rows} zskip={}",
                    self.zskip
                );
            } else {
                tracing::info!("ATLAS_GLM_MOE_DECODE_M16 not used: rows={rows}");
            }
        }
    }
}

/// Whether `rows` routed rows of `[inter, h]` experts fit M16 tiles of at
/// most `max_rows` rows: K128 stages, and 128 gate/up and 256 down columns per
/// tile.
pub(super) fn m16_shape(rows: u32, max_rows: u32, h: u32, inter: u32) -> bool {
    (1..=max_rows).contains(&rows) && inter.is_multiple_of(128) && h.is_multiple_of(256)
}

impl MoeLayer {
    /// Whether a routed FFN of `rows` rows is a verify-decode batch: the K5
    /// compact shape or one of the exact grouped decode entries.
    pub(super) fn verify_decode_batch(
        &self,
        compact_k5: bool,
        ctx: &ForwardContext,
        rows: u32,
    ) -> bool {
        compact_k5
            || self.glm_c2_grouped(ctx, rows)
            || self.glm_c3_grouped(ctx, rows)
            || self.glm_c4_grouped(ctx, rows)
            || self.independent_grouped(ctx, rows)
    }

    /// `ATLAS_GLM_MOE_STREAM_NOSYNC`: whether a routed FFN of `rows` rows
    /// runs the stream twins on the prequant path whatever its kind, so
    /// nothing reads a host copy of its expert offsets. This mirrors the
    /// dispatch of `run_routed_grouped_gemm`: no MMQ layout, no CUTLASS
    /// grouped GEMM, the transposed NVFP4 tables and the prequant kernel,
    /// then [`Self::prequant_tiles`] taking [`Self::decode_m16_grid`], whose
    /// gate/up and down both launch over its grid.
    pub(super) fn offsets_unread(&self, rows: u32, h: u32, inter: u32) -> bool {
        self.decode_m16.nosync
            && self.decode_m16.stream()
            && self.m16_grid_fits(rows, STREAM_MAX_ROWS, h, inter)
            && !self.nvfp4_mmq_layout
            && !(super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
                && self.cutlass_grouped_host.is_some())
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.nvfp4_prequant_moe
            && self.moe_w4a4_prequant_t_k64.0 != 0
    }

    /// Whether the M16 kernels can take `rows` routed rows (at most
    /// `max_rows`): loaded, the shape tiled, the worklist builder present and
    /// the fused SiLU quantization the one serving would run.
    fn m16_grid_fits(&self, rows: u32, max_rows: u32, h: u32, inter: u32) -> bool {
        self.decode_m16.loaded()
            && m16_shape(rows, max_rows, h, inter)
            && self.moe_build_tile_worklist_k.0 != 0
            && self.nvfp4_fused_silu_quant
            && self.silu_mul_quant_nvfp4_k.0 != 0
            && self.lora.is_none()
    }

    /// The row tiles of one prequant routed FFN of `rows` rows. A verify
    /// decode batch (`decode`) takes the M16 grid when selected, else the
    /// K128W grid under `ATLAS_GLM_MOE_DECODE_K128W`, else the compact M64
    /// worklist; any other batch the stream twins' grid when they are loaded
    /// and fit, else the K128W grid when it is loaded.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prequant_tiles(
        &self,
        decode: bool,
        expert_offsets: DevicePtr,
        local_ptrs: DevicePtr,
        [rows, top_k]: [u32; 2],
        [h, inter]: [u32; 2],
        num_experts: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(Option<CompactMoeWorklist>, Option<MtileGrid>)> {
        let total_expanded = rows * top_k;
        let k128w = || {
            self.mtile_grid(
                expert_offsets,
                local_ptrs,
                total_expanded,
                num_experts,
                ctx,
                stream,
            )
        };
        let m16 = if decode || self.decode_m16.stream() {
            self.decode_m16_grid(
                expert_offsets,
                local_ptrs,
                rows,
                total_expanded,
                [h, inter],
                num_experts,
                ctx,
                stream,
            )?
        } else {
            None
        };
        self.decode_m16.note(ctx.gpu, rows, m16.is_some());
        if m16.is_some() {
            return Ok((None, m16));
        }
        if !decode {
            return Ok((None, k128w()?));
        }
        if self.decode_m16.k128w {
            let grid = k128w()?;
            if ctx
                .gpu
                .op_cache()
                .first_shape("moe:decode_k128w", rows, grid.is_some() as u32, 0)
            {
                tracing::info!(
                    "ATLAS_GLM_MOE_DECODE_K128W {}: rows={rows}",
                    if grid.is_some() {
                        "engaged"
                    } else {
                        "not used (K128W kernels not loaded)"
                    }
                );
            }
            if grid.is_some() {
                return Ok((None, grid));
            }
        }
        let total_tiles = ctx.buffers.moe_router_in_f32();
        let worklist = total_tiles.offset(16);
        let n_tiles = inter.div_ceil(128);
        anyhow::ensure!(
            ctx.buffers.sizes().moe_router_in_f32
                >= super::prequant_fp4::compact_gate_up_worklist_bytes(rows, top_k, inter),
            "compact native-FP4 gate/up worklist exceeds router scratch"
        );
        ops::moe_build_tile_worklist(
            ctx.gpu,
            self.moe_build_tile_worklist_k,
            expert_offsets,
            local_ptrs,
            worklist,
            total_tiles,
            num_experts,
            n_tiles,
            64,
            stream,
        )?;
        let compact = CompactMoeWorklist {
            worklist,
            total_tiles,
            max_tiles: total_expanded * n_tiles,
        };
        Ok((Some(compact), None))
    }

    /// The row-tile grid of the M16 kernels over a verify batch of `rows`
    /// rows (`total_expanded` sorted rows), when they are loaded and the
    /// fused SiLU quantization they apply is the one serving would run. Its
    /// "prefix" is the compact worklist scratch with one N tile per routed
    /// local expert, which the twins index by grid row. The grid bound is
    /// host-static: there are at most `total_expanded` routed experts.
    #[allow(clippy::too_many_arguments)]
    fn decode_m16_grid(
        &self,
        expert_offsets: DevicePtr,
        local_ptrs: DevicePtr,
        rows: u32,
        total_expanded: u32,
        [h, inter]: [u32; 2],
        num_experts: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<MtileGrid>> {
        let k = self.decode_m16;
        let max_rows = if k.stream() {
            STREAM_MAX_ROWS
        } else {
            MAX_ROWS
        };
        if !self.m16_grid_fits(rows, max_rows, h, inter) {
            return Ok(None);
        }
        let total_tiles = ctx.buffers.moe_router_in_f32();
        anyhow::ensure!(
            ctx.buffers.sizes().moe_router_in_f32 >= 16 + total_expanded as usize * 8,
            "M16 decode worklist exceeds router scratch"
        );
        ops::moe_build_tile_worklist(
            ctx.gpu,
            self.moe_build_tile_worklist_k,
            expert_offsets,
            local_ptrs,
            total_tiles.offset(16),
            total_tiles,
            num_experts,
            1,
            64,
            stream,
        )?;
        let grid_only = |grid| ops::K128wKernel {
            grid,
            persist: KernelHandle(0),
        };
        let [gate_up_silu, down] = k.pair(rows);
        // The persistent twins stride over the worklist's items themselves.
        let schedule = match k.ctas {
            0 => ops::K128wSchedule::Grid {
                bound: total_expanded.min(num_experts),
            },
            ctas => ops::K128wSchedule::Stride { ctas },
        };
        Ok(Some(MtileGrid {
            prefix: total_tiles,
            schedule,
            rows: total_expanded,
            gate_up_silu: grid_only(gate_up_silu),
            down: grid_only(down),
        }))
    }
}

#[cfg(test)]
#[path = "decode_m16_test_gpu.rs"]
mod gpu_tests;
