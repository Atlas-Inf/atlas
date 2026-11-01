// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off, bounded GLM gate/up M16 selection and eager numerical oracle.

use super::*;
use atlas_core::config::ModelConfig;
use std::sync::atomic::{AtomicU32, Ordering};

use super::prequant_fp4::CompactMoeWorklist;

fn reject_graphs(model_type: &str, verify: bool, use_graphs: bool) -> Result<()> {
    anyhow::ensure!(
        model_type != "glm5_next" || !verify || !use_graphs,
        "ATLAS_GLM_MOE_GATE_UP_M16_VERIFY requires eager execution; disable actual decode/verify graphs before capture"
    );
    Ok(())
}

/// Call at the model's actual graph decision, before graph lookup or capture.
pub(crate) fn validate_m16_gate_up_graphs(model_type: &str, use_graphs: bool) -> Result<()> {
    reject_graphs(
        model_type,
        std::env::var("ATLAS_GLM_MOE_GATE_UP_M16_VERIFY").as_deref() == Ok("1"),
        use_graphs,
    )
}

#[allow(clippy::too_many_arguments)]
fn eligible(
    config: &ModelConfig,
    rows: u32,
    n: u32,
    k: u32,
    experts: u32,
    work: CompactMoeWorklist,
    native_resources: bool,
) -> bool {
    // C3 and generic prefill remain untouched. The existing fused-path caller
    // is responsible for reaching this selection only with compact metadata.
    matches!(rows, 4 | 5)
        && super::prequant_fp4::glm_grouped_shape(config)
        && (n, k, experts) == (2048, 4096, 288)
        && native_resources
        && work.max_tiles == rows * 8 * 16
        && !work.worklist.is_null()
        && !work.total_tiles.is_null()
        && work.worklist.0.is_multiple_of(4)
        && work.total_tiles.0.is_multiple_of(4)
}

fn parse_toggle(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => anyhow::bail!("GLM M16 gate/up toggles require explicit 0 or 1"),
    }
}

fn validate_toggles(enabled: bool, verify: bool) -> Result<()> {
    anyhow::ensure!(
        !verify || enabled,
        "ATLAS_GLM_MOE_GATE_UP_M16_VERIFY requires ATLAS_GLM_MOE_GATE_UP_M16=1"
    );
    Ok(())
}

fn span(ptr: DevicePtr, bytes: usize, alignment: u64) -> Result<std::ops::Range<u64>> {
    anyhow::ensure!(
        !ptr.is_null() && ptr.0.is_multiple_of(alignment),
        "M16 null/misaligned buffer"
    );
    let end = ptr
        .0
        .checked_add(bytes as u64)
        .ok_or_else(|| anyhow::anyhow!("M16 buffer address overflow"))?;
    Ok(ptr.0..end)
}

fn checked_output_bytes(
    rows: u32,
    outputs: [DevicePtr; 2],
    capacities: [usize; 2],
    packed_a: DevicePtr,
) -> Result<usize> {
    anyhow::ensure!(matches!(rows, 4 | 5), "M16 output row bound");
    let bytes = rows as usize * 8 * 2048 * 2;
    anyhow::ensure!(
        capacities.iter().all(|&cap| cap >= bytes),
        "M16 gate/up output capacity"
    );
    let a = span(packed_a, rows as usize * (4096 / 2 + 4096 / 16), 16)?;
    let gate = span(outputs[0], bytes, 2)?;
    let up = span(outputs[1], bytes, 2)?;
    for (x, y) in [(&gate, &up), (&gate, &a), (&up, &a)] {
        anyhow::ensure!(
            x.end <= y.start || y.end <= x.start,
            "M16 live input/output alias"
        );
    }
    Ok(bytes)
}

fn compare_output(label: &str, expected: &[u8], actual: &[u8]) -> Result<()> {
    anyhow::ensure!(
        expected.len() == actual.len(),
        "M16 {label} comparison extent"
    );
    if let Some(index) = expected.iter().zip(actual).position(|(a, b)| a != b) {
        anyhow::bail!(
            "M16 {label} native oracle mismatch byte={index}: old={} new={}",
            expected[index],
            actual[index]
        );
    }
    Ok(())
}

fn verify_outputs(
    gpu: &dyn GpuBackend,
    stream: u64,
    outputs: [DevicePtr; 2],
    bytes: usize,
    original: KernelHandle,
    candidate: KernelHandle,
    launch: impl Fn(KernelHandle) -> Result<()>,
) -> Result<()> {
    let poison = || -> Result<()> {
        for output in outputs {
            gpu.memset_async(output, 0x5a, bytes, stream)?;
        }
        Ok(())
    };
    let snapshot = || -> Result<[Vec<u8>; 2]> {
        let mut values = [vec![0u8; bytes], vec![0u8; bytes]];
        for (output, dst) in outputs.into_iter().zip(&mut values) {
            gpu.copy_d2h_on_stream(output, dst, stream)?;
        }
        Ok(values)
    };
    poison()?;
    launch(original)?;
    let expected = snapshot()?;
    poison()?; // Never accept stale output from the reference launch.
    launch(candidate)?;
    let actual = snapshot()?;
    compare_output("gate", &expected[0], &actual[0])?;
    compare_output("up", &expected[1], &actual[1])
}

pub(super) struct M16GateUp {
    enabled: bool,
    verify: bool,
    scalar: KernelHandle,
    vector: KernelHandle,
    verified_rows: AtomicU32,
    selected_rows: AtomicU32,
}

pub(super) struct GateUpCall {
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    pub experts: u32,
    pub work: CompactMoeWorklist,
    pub native_resources: bool,
    pub vector: bool,
    pub original: KernelHandle,
    pub outputs: [DevicePtr; 2],
    pub sorted_tokens: DevicePtr,
    pub gate_table: DevicePtr,
}

impl M16GateUp {
    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn new(gpu: &dyn GpuBackend, config: &ModelConfig) -> Result<Self> {
        let requested = parse_toggle(std::env::var("ATLAS_GLM_MOE_GATE_UP_M16").ok().as_deref())?;
        let verify = parse_toggle(
            std::env::var("ATLAS_GLM_MOE_GATE_UP_M16_VERIFY")
                .ok()
                .as_deref(),
        )?;
        validate_toggles(requested, verify)?;
        let enabled = requested && config.model_type == "glm5_next";
        let kernel = |symbol| {
            if enabled {
                gpu.kernel("moe_w4a16", symbol)
            } else {
                Ok(KernelHandle(0))
            }
        };
        Ok(Self {
            enabled,
            verify: enabled && verify,
            scalar: kernel("glm_moe_gate_up_m16n128")?,
            vector: kernel("glm_moe_gate_up_m16n128_vecscale")?,
            verified_rows: AtomicU32::new(0),
            selected_rows: AtomicU32::new(0),
        })
    }

    pub(super) fn run(
        &self,
        call: GateUpCall,
        ctx: &ForwardContext,
        stream: u64,
        launch: impl Fn(KernelHandle) -> Result<()>,
    ) -> Result<()> {
        if !self.enabled
            || !eligible(
                ctx.config,
                call.rows,
                call.n,
                call.k,
                call.experts,
                call.work,
                call.native_resources,
            )
        {
            return launch(call.original);
        }
        let candidate = if call.vector {
            self.vector
        } else {
            self.scalar
        };
        anyhow::ensure!(
            candidate.0 != 0 && call.original.0 != 0,
            "M16 missing selected/reference kernel"
        );
        anyhow::ensure!(
            !call.sorted_tokens.is_null() && call.sorted_tokens.0.is_multiple_of(4),
            "M16 requires aligned token gather"
        );
        // These destinations and packed-A owner are fixed by prequant_fp4.
        anyhow::ensure!(
            call.outputs == [ctx.buffers.expert_gate_out(), ctx.buffers.expert_up_out()],
            "M16 unexpected output ownership"
        );
        let sizes = ctx.buffers.sizes();
        let bytes = checked_output_bytes(
            call.rows,
            call.outputs,
            [sizes.expert_gate_out, sizes.expert_up_out],
            ctx.buffers.expert_down_out(),
        )?;
        anyhow::ensure!(
            sizes.expert_down_out >= call.rows as usize * 2304,
            "M16 packed-A scratch capacity"
        );
        let bit = 1u32 << call.rows;
        if self.verify {
            // Model-side hooks reject use_graphs BEFORE lookup/capture. This
            // defense performs no D2H even if a future caller omits that hook.
            anyhow::ensure!(
                !ctx.graph_capture && !ctx.gpu.stream_is_capturing(stream),
                "M16 VERIFY cannot run during graph capture"
            );
        }
        if self.verify && self.verified_rows.load(Ordering::Acquire) & bit == 0 {
            // Poisoned remote holes cannot contribute downstream: native
            // eligibility requires EP2 + comm + indexed-EP reduction + no
            // LoRA. SiLU quantizes each row independently, dense down skips
            // null remote weights, and EP reduction checks the expert range
            // before reading its row. Shared expert scratch is separate.
            verify_outputs(
                ctx.gpu,
                stream,
                call.outputs,
                bytes,
                call.original,
                candidate,
                &launch,
            )?;
            let mut raw_tiles = [0u8; 4];
            ctx.gpu
                .copy_d2h_on_stream(call.work.total_tiles, &mut raw_tiles, stream)?;
            let tiles = i32::from_le_bytes(raw_tiles);
            anyhow::ensure!(
                tiles >= 0 && tiles as u32 <= call.work.max_tiles,
                "M16 oracle work count bound"
            );
            // An empty local rank is a valid pass, but cannot establish that
            // this layer's actual weights/arithmetic were exercised. Retry it.
            if tiles > 0 {
                self.verified_rows.fetch_or(bit, Ordering::Release);
            }
            tracing::info!(
                rank = ctx.config.ep_rank,
                rows = call.rows,
                gate_table = call.gate_table.0,
                tiles,
                bytes_per_projection = bytes,
                "GLM M16 gate/up native oracle BITEXACT (useful only when tiles>0)"
            );
            return Ok(());
        }
        if self.selected_rows.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            tracing::info!(
                rank = ctx.config.ep_rank,
                rows = call.rows,
                gate_table = call.gate_table.0,
                "GLM M16 gate/up selected (default-off A/B)"
            );
        }
        launch(candidate)
    }
}

#[cfg(test)]
#[path = "gate_up_m16_tests.rs"]
mod tests;
