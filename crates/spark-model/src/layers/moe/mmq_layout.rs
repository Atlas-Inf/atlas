// SPDX-License-Identifier: AGPL-3.0-only

//! Equal-memory routed expert repack for the grouped NVFP4 MMQ path.

use super::*;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_nvfp4_mmq_decode(
        &self,
        ctx: &ForwardContext,
        expert_input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_out: DevicePtr,
        indices_dev: DevicePtr,
        h: u32,
        inter: u32,
        top_k: u32,
        single_seq_decode: bool,
        stream: u64,
    ) -> Result<()> {
        ops::moe_expert_gate_up_shared(
            ctx.gpu,
            self.moe_expert_gate_up_shared_mmq_k,
            expert_input,
            self.gate_ptrs.packed_ptrs,
            self.gate_ptrs.scale_ptrs,
            self.gate_ptrs.scale2_vals,
            expert_gate_out,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            &self.weights.shared_expert.gate_proj,
            shared_gate_scratch,
            &self.weights.shared_expert.up_proj,
            shared_up_scratch,
            inter,
            h,
            top_k,
            stream,
        )?;
        if single_seq_decode {
            self.apply_expert_lora_decode_gateup(
                expert_gate_out,
                expert_up_out,
                expert_input,
                indices_dev,
                top_k,
                top_k,
                DevicePtr::NULL,
                ctx,
                stream,
            )?;
        }
        ops::moe_expert_silu_down_shared(
            ctx.gpu,
            self.moe_expert_silu_down_shared_mmq_k,
            expert_gate_out,
            expert_up_out,
            self.down_ptrs.packed_ptrs,
            self.down_ptrs.scale_ptrs,
            self.down_ptrs.scale2_vals,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            &self.weights.shared_expert.down_proj,
            shared_out,
            h,
            inter,
            top_k,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_nvfp4_mmq_gate_up(
        &self,
        expert_input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        total_expanded: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        macro_rules! prof_step {
            ($started:expr, $label:expr) => {
                if let Some(started) = $started.as_mut() {
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!(
                        "    NVFP4 MMQ [{}] rows={}: {}us",
                        $label,
                        total_expanded,
                        started.elapsed().as_micros()
                    );
                    *started = std::time::Instant::now();
                }
            };
        }
        let mut started = ctx.profile.then(std::time::Instant::now);
        let input_fp4 = ctx.buffers.expert_down_out();
        ops::moe_nvfp4_mmq_quantize(
            ctx.gpu,
            self.moe_nvfp4_mmq_quantize_k,
            expert_input,
            sorted_token_ids,
            input_fp4,
            total_expanded,
            h,
            stream,
        )?;
        prof_step!(started, "quantize_gate_up");
        ops::moe_nvfp4_mmq_gate_up(
            ctx.gpu,
            self.moe_nvfp4_mmq_gate_up_k,
            self.gate_ptrs.packed_ptrs,
            self.up_ptrs.packed_ptrs,
            input_fp4,
            expert_gate_out,
            expert_up_out,
            expert_offsets,
            num_experts,
            total_expanded,
            inter,
            h,
            max_m_tiles,
            stream,
        )?;
        prof_step!(started, "gate_up");
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_nvfp4_mmq_silu_down(
        &self,
        expert_input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        total_expanded: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        max_m_tiles: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        macro_rules! prof_step {
            ($started:expr, $label:expr) => {
                if let Some(started) = $started.as_mut() {
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!(
                        "    NVFP4 MMQ [{}] rows={}: {}us",
                        $label,
                        total_expanded,
                        started.elapsed().as_micros()
                    );
                    *started = std::time::Instant::now();
                }
            };
        }
        let mut started = ctx.profile.then(std::time::Instant::now);
        let max_rows = max_m_tiles * ops::MOE_NVFP4_MMQ_M_TILE;
        if self.lora.is_some() {
            for (data, scale) in [
                (expert_gate_out, self.gate_ptrs.scale2_vals),
                (expert_up_out, self.up_ptrs.scale2_vals),
            ] {
                ops::moe_nvfp4_mmq_scale2_rows(
                    ctx.gpu,
                    self.moe_nvfp4_mmq_scale2_rows_k,
                    data,
                    scale,
                    expert_offsets,
                    inter,
                    max_rows,
                    num_experts,
                    stream,
                )?;
            }
            self.apply_expert_lora_prefill_gateup(
                expert_gate_out,
                expert_up_out,
                expert_input,
                expert_offsets,
                sorted_token_ids,
                total_expanded,
                ctx,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                total_expanded * inter,
                stream,
            )?;
            prof_step!(started, "lora_silu");
        } else {
            ops::moe_nvfp4_mmq_silu_scale2(
                ctx.gpu,
                self.moe_nvfp4_mmq_silu_scale2_k,
                expert_gate_out,
                expert_up_out,
                expert_gate_out,
                self.gate_ptrs.scale2_vals,
                self.up_ptrs.scale2_vals,
                expert_offsets,
                inter,
                max_rows,
                num_experts,
                stream,
            )?;
            prof_step!(started, "silu_scale2");
        }

        // Up output is dead after SiLU. Reuse it for the compact MMQ input.
        ops::moe_nvfp4_mmq_quantize(
            ctx.gpu,
            self.moe_nvfp4_mmq_quantize_k,
            expert_gate_out,
            DevicePtr::NULL,
            expert_up_out,
            total_expanded,
            inter,
            stream,
        )?;
        prof_step!(started, "quantize_down");
        ops::moe_nvfp4_mmq_down(
            ctx.gpu,
            self.moe_nvfp4_mmq_down_k,
            self.down_ptrs.packed_ptrs,
            expert_up_out,
            expert_down_out,
            expert_offsets,
            num_experts,
            total_expanded,
            h,
            inter,
            max_m_tiles,
            stream,
        )?;
        prof_step!(started, "down");
        ops::moe_nvfp4_mmq_scale2_rows(
            ctx.gpu,
            self.moe_nvfp4_mmq_scale2_rows_k,
            expert_down_out,
            self.down_ptrs.scale2_vals,
            expert_offsets,
            h,
            max_rows,
            num_experts,
            stream,
        )?;
        prof_step!(started, "down_scale2");
        Ok(())
    }

    /// Replace checkpoint routed weights with Atlas's block_nvfp4 MMQ layout.
    ///
    /// Each K64 consumes 36 bytes in both representations, so the steady-state
    /// model footprint is unchanged. Repack and free one projection at a time
    /// to bound peak load-time memory on a 128 GiB Spark.
    pub fn repack_nvfp4_mmq_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.btile_storage.require_legacy()?;
        self.experts_scale_kind.expect(
            crate::weight_map::WeightQuantFormat::Nvfp4,
            "NVFP4 MMQ routed-weight repack",
        );
        anyhow::ensure!(
            self.moe_nvfp4_mmq_gate_up_k.0 != 0
                && self.moe_nvfp4_mmq_down_k.0 != 0
                && self.moe_nvfp4_mmq_quantize_k.0 != 0
                && self.moe_nvfp4_mmq_repack_k.0 != 0
                && self.moe_nvfp4_mmq_silu_scale2_k.0 != 0
                && self.moe_nvfp4_mmq_scale2_rows_k.0 != 0
                && self.moe_expert_gate_up_shared_mmq_k.0 != 0
                && self.moe_expert_silu_down_shared_mmq_k.0 != 0,
            "NVFP4 MMQ requested but its complete kernel family did not resolve"
        );
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let stream = gpu.default_stream();

        // Keep the small shared expert checkpoint-native for decode, and make
        // transposed twins for its existing prefill path.
        let shared_inter = config.shared_expert_intermediate_size;
        if shared_inter > 0 && !self.weights.shared_expert.gate_proj.is_null() {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                h,
                shared_inter,
            )?);
        }

        let gate_src: Vec<_> = self.weights.experts.iter().map(|e| e.gate_proj).collect();
        let (gate, gate_slab) =
            self.repack_mmq_projection(gpu, &gate_src, &self.gate_ptrs, inter, h, stream)?;
        for expert in &mut self.weights.experts {
            free_quantized_payload(gpu, &mut expert.gate_proj)?;
        }
        free_expert_table(gpu, &self.gate_ptrs)?;
        self.gate_ptrs = gate;
        self._nvfp4_mmq_owned.push(gate_slab);

        let up_src: Vec<_> = self.weights.experts.iter().map(|e| e.up_proj).collect();
        let (up, up_slab) =
            self.repack_mmq_projection(gpu, &up_src, &self.up_ptrs, inter, h, stream)?;
        for expert in &mut self.weights.experts {
            free_quantized_payload(gpu, &mut expert.up_proj)?;
        }
        free_expert_table(gpu, &self.up_ptrs)?;
        self.up_ptrs = up;
        self._nvfp4_mmq_owned.push(up_slab);

        let down_src: Vec<_> = self.weights.experts.iter().map(|e| e.down_proj).collect();
        let (down, down_slab) =
            self.repack_mmq_projection(gpu, &down_src, &self.down_ptrs, h, inter, stream)?;
        for expert in &mut self.weights.experts {
            free_quantized_payload(gpu, &mut expert.down_proj)?;
        }
        free_expert_table(gpu, &self.down_ptrs)?;
        self.down_ptrs = down;
        self._nvfp4_mmq_owned.push(down_slab);

        self.nvfp4_mmq_layout = true;
        self.unified_layout = false;
        tracing::info!("NVFP4 MMQ: replaced routed gate/up/down with equal-size block_nvfp4 slabs");
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn repack_mmq_projection(
        &self,
        gpu: &dyn GpuBackend,
        src: &[QuantizedWeight],
        src_table: &ExpertPtrTable,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<(ExpertPtrTable, DevicePtr)> {
        anyhow::ensure!(
            k.is_multiple_of(256),
            "MMQ projection K={k} must be a multiple of 256"
        );
        let (local_count, slots) =
            super::compact_layout::compact_slot_map(src.iter().map(|w| !w.is_null()));
        anyhow::ensure!(local_count > 0, "MMQ projection has no local experts");
        let bytes_each = ops::moe_nvfp4_mmq_weight_bytes(n as u32, k as u32);
        anyhow::ensure!(
            bytes_each == n * (k / 2 + k / 16),
            "MMQ representation must be byte-equal to packed+scale checkpoint storage"
        );
        let slab = gpu.alloc(local_count * bytes_each)?;
        let out: Vec<_> = src
            .iter()
            .zip(slots)
            .map(|(source, slot)| match slot {
                Some(slot) => QuantizedWeight {
                    weight: slab.offset(slot * bytes_each),
                    weight_scale: DevicePtr::NULL,
                    weight_scale_2: source.weight_scale_2,
                    input_scale: source.input_scale,
                    weight_scale_2_vec: source.weight_scale_2_vec,
                },
                None => QuantizedWeight::null(),
            })
            .collect();
        let dst_table = build_ptr_table_from_qw(&out, gpu)?;
        ops::moe_nvfp4_mmq_repack_batched(
            gpu,
            self.moe_nvfp4_mmq_repack_k,
            src_table.packed_ptrs,
            src_table.scale_ptrs,
            dst_table.packed_ptrs,
            n as u32,
            k as u32,
            src.len() as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        Ok((dst_table, slab))
    }
}

fn free_quantized_payload(gpu: &dyn GpuBackend, weight: &mut QuantizedWeight) -> Result<()> {
    if !weight.weight.is_null() {
        gpu.free(weight.weight)?;
        gpu.free(weight.weight_scale)?;
        weight.weight = DevicePtr::NULL;
        weight.weight_scale = DevicePtr::NULL;
    }
    Ok(())
}

fn free_expert_table(gpu: &dyn GpuBackend, table: &ExpertPtrTable) -> Result<()> {
    gpu.free(table.packed_ptrs)?;
    gpu.free(table.scale_ptrs)?;
    gpu.free(table.scale2_vals)?;
    Ok(())
}
