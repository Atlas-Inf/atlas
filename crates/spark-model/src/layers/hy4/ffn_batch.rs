// SPDX-License-Identifier: AGPL-3.0-only

use super::ffn::Hy4Ffn;
use super::timing;
use anyhow::Result;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

/// Tensor-core expert path, gated by ATLAS_HY4_EXPERT_MMA=1: native-FP4
/// (mxf4nvf4). Each phase's bf16 activations are quantized into a packed-FP4
/// side buffer ([a_rows*k/2 e2m1 nibbles | a_rows*k/16 e4m3 scales], built by
/// `hy4_expert_quantize` in the same module) and consumed by
/// `hy4_batch_expert_mma`.
///
/// When the gate is unset — or a phase's K/N is not a multiple of the
/// kernel's 128-wide stage — the proven `hy4_batch_expert` kernel keeps
/// serving the phase unchanged. VERIFY rows (≤6) stay on that kernel too:
/// the MMA CTA is 128 columns × 64 rows, so a 6-row VERIFY launches mostly
/// padding and measured slower than the scalar expert kernel on GB10.
const MMA_K_STAGE: usize = 128;
const MMA_MIN_ROWS: u32 = 32;

impl Hy4Ffn<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn batch_mlp(
        &self,
        x: DevicePtr,
        activated: DevicePtr,
        out: DevicePtr,
        ids: DevicePtr,
        weights: DevicePtr,
        dims: [usize; 4],
        limit: f32,
        rows: u32,
        batch: [DevicePtr; 2],
        stream: u64,
    ) -> Result<()> {
        let [h, i, slots, group] = dims;
        let [groups, partials] = batch;
        // Activation row counts per phase: gate/up consumes the `rows` token
        // rows (x); down consumes the `rows*slots` activated slot rows.
        let phase_rows = [rows as usize, rows as usize * slots];
        let phase_k = [h, i];
        let phase_native = rows >= MMA_MIN_ROWS
            && std::env::var("ATLAS_HY4_EXPERT_MMA").ok().as_deref() == Some("1");
        // Packed-FP4 prequant buffer per phase. Two ops-cache tags keep the
        // gate/up and down buffers independent (they overlap in time only
        // within this call, but grow-only scratch is keyed per phase anyway).
        let mut packed = [DevicePtr::NULL; 2];
        if phase_native {
            for phase in 0..2 {
                let bytes = phase_rows[phase] * (phase_k[phase] / 2 + phase_k[phase] / 16);
                let tag: &'static str = if phase == 0 {
                    "hy4_mma_quant_gate_up"
                } else {
                    "hy4_mma_quant_down"
                };
                packed[phase] = self.gpu.op_cache().scratch(self.gpu, tag, bytes)?;
            }
        }
        let experts = if ids.is_null() { 1 } else { self.ep.experts };
        let grouping = timing::span(
            if ids.is_null() {
                "dense_grouping"
            } else {
                "expert_grouping"
            },
            stream,
        )?;
        KernelLaunch::new(
            self.gpu,
            self.gpu
                .op_cache()
                .kernel(self.gpu, "hy4_ffn", "hy4_group_init")?,
        )
        .grid([1, 1, 1])
        .block([1, 1, 1])
        .arg_ptr(groups)
        .arg_u32(rows)
        .arg_u32(experts as u32)
        .launch(stream)?;
        KernelLaunch::new(
            self.gpu,
            self.gpu
                .op_cache()
                .kernel(self.gpu, "hy4_ffn", "hy4_group_rows")?,
        )
        .grid([experts as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ids)
        .arg_ptr(groups)
        .arg_u32(rows)
        .arg_u32(slots as u32)
        .launch(stream)?;
        drop(grouping);
        for (phase, (input, output, n, k, down)) in [
            (0usize, (x, activated, i, h, 0u32)),
            (1, (activated, partials, h, i, 1)),
        ] {
            let _phase = timing::span(
                match (ids.is_null(), down) {
                    (true, 0) => "dense_gate_up",
                    (true, _) => "dense_down",
                    (false, 0) => "expert_gate_up",
                    _ => "expert_down",
                },
                stream,
            )?;
            // A phase runs the native kernel only when its tile sizes fit
            // (K must be a multiple of the 128-wide K stage, N of the
            // 128-column tile); otherwise the proven kernel reads the bf16
            // `input` directly and the side buffer stays unused.
            let native = phase_native && k % MMA_K_STAGE == 0 && n % MMA_K_STAGE == 0;
            if native {
                let a_rows = phase_rows[phase];
                let packed_bytes = a_rows * (k / 2);
                KernelLaunch::new(
                    self.gpu,
                    self.gpu
                        .op_cache()
                        .kernel(self.gpu, "hy4_ffn", "hy4_expert_quantize")?,
                )
                .grid([a_rows as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(packed[phase])
                .arg_ptr(packed[phase].offset(packed_bytes))
                .arg_u32(a_rows as u32)
                .arg_u32(k as u32)
                .launch(stream)?;
                KernelLaunch::new(
                    self.gpu,
                    self.gpu
                        .op_cache()
                        .kernel(self.gpu, "hy4_ffn", "hy4_batch_expert_mma")?,
                )
                .grid([
                    n.div_ceil(MMA_K_STAGE) as u32,
                    1,
                    experts.min(rows as usize * slots) as u32,
                ])
                .block([256, 1, 1])
                .arg_ptr(packed[phase])
                .arg_ptr(self.table)
                .arg_ptr(groups)
                .arg_ptr(output)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(rows)
                .arg_u32(slots as u32)
                .arg_u32(group as u32)
                .arg_u32(down)
                .arg_f32(limit)
                .arg_u32(experts as u32)
                .launch(stream)?;
            } else {
                KernelLaunch::new(
                    self.gpu,
                    self.gpu
                        .op_cache()
                        .kernel(self.gpu, "hy4_ffn", "hy4_batch_expert")?,
                )
                .grid([
                    n.div_ceil(8) as u32,
                    1,
                    experts.min(rows as usize * slots) as u32,
                ])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(self.table)
                .arg_ptr(groups)
                .arg_ptr(output)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(rows)
                .arg_u32(slots as u32)
                .arg_u32(group as u32)
                .arg_u32(down)
                .arg_f32(limit)
                .arg_u32(experts as u32)
                .launch(stream)?;
            }
        }
        let _combine = timing::span(
            if ids.is_null() {
                "dense_combine"
            } else {
                "expert_combine"
            },
            stream,
        )?;
        KernelLaunch::new(
            self.gpu,
            self.gpu
                .op_cache()
                .kernel(self.gpu, "hy4_ffn", "hy4_batch_combine")?,
        )
        .grid([h.div_ceil(256) as u32, rows, 1])
        .block([256, 1, 1])
        .arg_ptr(partials)
        .arg_ptr(weights)
        .arg_ptr(out)
        .arg_u32(h as u32)
        .arg_u32(slots as u32)
        .launch(stream)
    }
}
