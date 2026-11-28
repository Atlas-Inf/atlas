// SPDX-License-Identifier: AGPL-3.0-only

//! One-token native FFN for the original Ling 3.2 air checkpoint.
//!
//! The router and shared expert stay resident; deferred FP8 matrices use a
//! bounded mmap staging fallback when direct host-page access is unavailable.

use std::sync::{Arc, Mutex};

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::weights::ling_paging::LingPagingReader;
use spark_runtime::weights::{DeferredTensor, WeightDtype};

use super::ops;
use crate::layer::ForwardContext;
use crate::layers::moe::ExpertPtrTable;
use crate::weight_map::QuantizedWeight;
use crate::weight_map::{DenseWeight, Fp8Weight, WeightQuantFormat};

const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 768;
const EXPERTS: usize = 512;
const TOP_K: usize = 8;
const GROUP: usize = 128;

#[path = "ling_moe_a16.rs"]
mod a16;
#[path = "ling_moe_fast.rs"]
mod fast;
#[path = "ling_moe_load.rs"]
mod load;

enum LingProjection {
    Dense {
        weight: DenseWeight,
        n: usize,
        k: usize,
    },
    Nvfp4 {
        weight: QuantizedWeight,
        n: usize,
        k: usize,
    },
    Resident(Fp8Weight),
    Paged {
        weight: DeferredTensor,
        scale: DeferredTensor,
        n: usize,
        k: usize,
    },
}

impl LingProjection {
    fn dims(&self) -> (usize, usize) {
        match self {
            Self::Dense { n, k, .. } => (*n, *k),
            Self::Nvfp4 { n, k, .. } => (*n, *k),
            Self::Resident(w) => (w.n as usize, w.k as usize),
            Self::Paged { n, k, .. } => (*n, *k),
        }
    }

    fn resident(&self) -> Option<Fp8Weight> {
        match self {
            Self::Resident(w) => Some(*w),
            Self::Dense { .. } | Self::Nvfp4 { .. } | Self::Paged { .. } => None,
        }
    }

    fn dense(&self) -> Option<DenseWeight> {
        match self {
            Self::Dense { weight, .. } => Some(*weight),
            Self::Nvfp4 { .. } | Self::Resident(_) | Self::Paged { .. } => None,
        }
    }

    fn nvfp4(&self) -> Option<(QuantizedWeight, usize, usize)> {
        match self {
            Self::Nvfp4 { weight, n, k } => Some((*weight, *n, *k)),
            Self::Dense { .. } | Self::Resident(_) | Self::Paged { .. } => None,
        }
    }
}

struct Stage {
    weight: DevicePtr,
    raw_scale: DevicePtr,
    scale_f32: DevicePtr,
    weight_bytes: usize,
    raw_scale_bytes: usize,
    scale_elements: usize,
}

#[derive(Clone, Copy)]
enum UnionProjection {
    Scalar(KernelHandle),
    Tensor(KernelHandle),
}

struct Kernels {
    router_gemv: KernelHandle,
    router_batch_gemv: Option<KernelHandle>,
    router: KernelHandle,
    quant: KernelHandle,
    gemm: KernelHandle,
    widen: KernelHandle,
    silu_quant: KernelHandle,
    fp8_gate_up: Option<KernelHandle>,
    ling_down_fp8: Option<KernelHandle>,
    silu_bf16: Option<KernelHandle>,
    moe_gate_up: Option<KernelHandle>,
    moe_down: Option<KernelHandle>,
    moe_wide: Option<KernelHandle>,
    decode_union: Option<(KernelHandle, UnionProjection)>,
    grouped_w4a4: Option<[KernelHandle; 3]>,
    grouped_a16: Option<[KernelHandle; 3]>,
    blend_rows: Option<KernelHandle>,
    weighted_sum: KernelHandle,
    residual_add: KernelHandle,
}

struct A16Tables {
    gate: ExpertPtrTable,
    up: ExpertPtrTable,
    down: ExpertPtrTable,
}

/// Original Ling 3.2 routed + shared expert FFN.
pub struct LingMoeFfn {
    layer_idx: usize,
    hidden: usize,
    intermediate: usize,
    router: DenseWeight,
    expert_bias: DevicePtr,
    shared_gate: LingProjection,
    shared_up: LingProjection,
    shared_down: LingProjection,
    routed: Vec<[LingProjection; 3]>,
    reader: Arc<Mutex<LingPagingReader>>,
    stage: Stage,
    input_fp8: DevicePtr,
    input_scale: DevicePtr,
    activation_fp8: DevicePtr,
    activation_scale: DevicePtr,
    activation_bf16: DevicePtr,
    a16_gate_up: DevicePtr,
    a16_tables: Option<A16Tables>,
    shared_decode: Option<[super::ling_projection::LingProjection; 3]>,
    a16: bool,
    strict_fast: Option<fast::StrictFastTables>,
    kernels: Kernels,
}

impl LingMoeFfn {
    pub(crate) fn supports_batched_prefill(&self) -> bool {
        self.a16 && self.kernels.moe_wide.is_some() && self.kernels.blend_rows.is_some()
    }

    /// One-token decode forward. The explicit graph-capture refusal protects
    /// the host route-id read and mmap staging from accidental graph replay.
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<DevicePtr> {
        if self.a16 {
            return self.forward_a16(input, ctx, stream);
        }
        if self.strict_fast.is_some() {
            return self.forward_strict_fast(input, ctx, stream);
        }
        ensure!(ctx.config.model_type == "bailing_moe_v3_vl");
        ensure!(
            !ctx.graph_capture,
            "Ling native FFN does not support CUDA graph capture yet"
        );
        ensure!(ctx.config.hidden_size == self.hidden && self.hidden.is_multiple_of(GROUP));
        let one = 1u32;
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.kernels.quant,
            input,
            self.input_fp8,
            self.input_scale,
            one,
            self.hidden as u32,
            stream,
        )?;

        let shared_gate = self
            .shared_gate
            .resident()
            .expect("validated resident shared gate");
        let shared_up = self
            .shared_up
            .resident()
            .expect("validated resident shared up");
        let shared_down = self
            .shared_down
            .resident()
            .expect("validated resident shared down");
        self.gemm(
            ctx,
            &shared_gate,
            self.input_fp8,
            self.input_scale,
            ctx.buffers.expert_gate_out(),
            stream,
        )?;
        self.gemm(
            ctx,
            &shared_up,
            self.input_fp8,
            self.input_scale,
            ctx.buffers.expert_up_out(),
            stream,
        )?;
        ops::ling_silu_mul_quant_fp8(
            ctx.gpu,
            self.kernels.silu_quant,
            ctx.buffers.expert_gate_out(),
            ctx.buffers.expert_up_out(),
            self.activation_fp8,
            self.activation_scale,
            DevicePtr::NULL,
            one,
            self.intermediate as u32,
            shared_swiglu_limit(self.layer_idx),
            stream,
        )?;
        self.gemm(
            ctx,
            &shared_down,
            self.activation_fp8,
            self.activation_scale,
            ctx.buffers.attn_output(),
            stream,
        )?;

        let ids = ctx.buffers.scratch();
        let weights = ids.offset(TOP_K * std::mem::size_of::<i32>());
        ensure!(ctx.buffers.scratch_bytes() >= TOP_K * 8 * std::mem::size_of::<u32>());
        ops::dense_gemv_bf16_fp32out(
            ctx.gpu,
            self.kernels.router_gemv,
            input,
            &self.router,
            ctx.buffers.gate_logits_f32(),
            EXPERTS as u32,
            self.hidden as u32,
            stream,
        )?;
        ops::ling_grouped_top8(
            ctx.gpu,
            self.kernels.router,
            ctx.buffers.gate_logits_f32(),
            self.expert_bias,
            ids,
            weights,
            stream,
        )?;
        let mut id_bytes = vec![0u8; TOP_K * std::mem::size_of::<i32>()];
        ctx.gpu.copy_d2h_on_stream(ids, &mut id_bytes, stream)?;
        for slot in 0..TOP_K {
            let start = slot * std::mem::size_of::<i32>();
            let id = i32::from_ne_bytes(id_bytes[start..start + 4].try_into().unwrap());
            ensure!(
                (0..EXPERTS as i32).contains(&id),
                "Ling router produced expert id {id}"
            );
            let expert = id as usize;
            let gate = self.stage_projection(ctx, &self.routed[expert][0], stream)?;
            self.gemm(
                ctx,
                &gate,
                self.input_fp8,
                self.input_scale,
                ctx.buffers.expert_gate_out(),
                stream,
            )?;
            let up = self.stage_projection(ctx, &self.routed[expert][1], stream)?;
            self.gemm(
                ctx,
                &up,
                self.input_fp8,
                self.input_scale,
                ctx.buffers.expert_up_out(),
                stream,
            )?;
            ops::ling_silu_mul_quant_fp8(
                ctx.gpu,
                self.kernels.silu_quant,
                ctx.buffers.expert_gate_out(),
                ctx.buffers.expert_up_out(),
                self.activation_fp8,
                self.activation_scale,
                DevicePtr::NULL,
                one,
                self.intermediate as u32,
                routed_swiglu_limit(self.layer_idx),
                stream,
            )?;
            let down = self.stage_projection(ctx, &self.routed[expert][2], stream)?;
            let output = ctx
                .buffers
                .expert_down_out()
                .offset(slot * self.hidden * std::mem::size_of::<u16>());
            self.gemm(
                ctx,
                &down,
                self.activation_fp8,
                self.activation_scale,
                output,
                stream,
            )?;
        }
        ops::ling_weighted_sum(
            ctx.gpu,
            self.kernels.weighted_sum,
            ctx.buffers.expert_down_out(),
            weights,
            ctx.buffers.moe_output(),
            self.hidden as u32,
            TOP_K as u32,
            stream,
        )?;
        ops::residual_add(
            ctx.gpu,
            self.kernels.residual_add,
            ctx.buffers.moe_output(),
            ctx.buffers.attn_output(),
            self.hidden as u32,
            stream,
        )?;
        Ok(ctx.buffers.moe_output())
    }

    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        if self.supports_batched_prefill() {
            return self.forward_a16_prefill(input, num_tokens, ctx, stream);
        }
        ensure!(
            num_tokens == 1,
            "Ling native FFN currently supports one token, got {num_tokens}"
        );
        self.forward(input, ctx, stream).map(|_| ())
    }

    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }

    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }

    pub fn forward_atomic_c4_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }

    pub fn forward_k2(
        &self,
        _input: DevicePtr,
        _ctx: &ForwardContext<'_>,
        _stream: u64,
    ) -> Result<()> {
        bail!("Ling native FFN does not support K=2 verify yet")
    }

    pub fn forward_k3(
        &self,
        _input: DevicePtr,
        _ctx: &ForwardContext<'_>,
        _stream: u64,
    ) -> Result<()> {
        bail!("Ling native FFN does not support K=3 verify yet")
    }

    fn gemm(
        &self,
        ctx: &ForwardContext<'_>,
        weight: &Fp8Weight,
        input: DevicePtr,
        scale: DevicePtr,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.kernels.gemm,
            input,
            scale,
            weight.weight,
            weight.row_scale,
            output,
            1,
            weight.n,
            weight.k,
            stream,
        )
    }

    fn stage_projection(
        &self,
        ctx: &ForwardContext<'_>,
        source: &LingProjection,
        stream: u64,
    ) -> Result<Fp8Weight> {
        if let Some(weight) = source.resident() {
            return Ok(weight);
        }
        let LingProjection::Paged {
            weight,
            scale,
            n,
            k,
        } = source
        else {
            unreachable!()
        };
        let mut reader = self
            .reader
            .lock()
            .map_err(|_| anyhow::anyhow!("Ling paging reader poisoned"))?;
        let weight_page = reader.e4m3_weight(weight)?;
        ensure!(weight_page.len() == n * k && weight_page.len() <= self.stage.weight_bytes);
        ctx.gpu.copy_h2d(weight_page.bytes(), self.stage.weight)?;
        let scale_page = reader.page(scale)?;
        let input_dtype = match scale.dtype {
            WeightDtype::BF16 => 0,
            WeightDtype::FP32 => 1,
            other => bail!("Ling deferred scale has unsupported dtype {other:?}"),
        };
        ensure!(scale_page.len() <= self.stage.raw_scale_bytes);
        ctx.gpu.copy_h2d(scale_page.bytes(), self.stage.raw_scale)?;
        let scale_elements = (n / GROUP) * (k / GROUP);
        ensure!(scale_elements == self.stage.scale_elements);
        ops::widen_block_scale_f32(
            ctx.gpu,
            self.kernels.widen,
            self.stage.raw_scale,
            self.stage.scale_f32,
            scale_elements as u32,
            input_dtype,
            stream,
        )?;
        Ok(Fp8Weight {
            weight: self.stage.weight,
            row_scale: self.stage.scale_f32,
            n: (*n).try_into()?,
            k: (*k).try_into()?,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        })
    }
}

pub(crate) fn routed_swiglu_limit(layer_idx: usize) -> f32 {
    if (35..=41).contains(&layer_idx) {
        4.0
    } else {
        0.0
    }
}

pub(crate) fn shared_swiglu_limit(layer_idx: usize) -> f32 {
    if (40..=41).contains(&layer_idx) {
        7.0
    } else if (34..=39).contains(&layer_idx) {
        5.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{routed_swiglu_limit, shared_swiglu_limit};

    #[test]
    fn ling_layer_limits_match_original_lists() {
        assert_eq!(routed_swiglu_limit(34), 0.0);
        assert_eq!(routed_swiglu_limit(35), 4.0);
        assert_eq!(routed_swiglu_limit(41), 4.0);
        assert_eq!(routed_swiglu_limit(42), 0.0);
        assert_eq!(shared_swiglu_limit(33), 0.0);
        assert_eq!(shared_swiglu_limit(34), 5.0);
        assert_eq!(shared_swiglu_limit(39), 5.0);
        assert_eq!(shared_swiglu_limit(40), 7.0);
        assert_eq!(shared_swiglu_limit(41), 7.0);
        assert_eq!(shared_swiglu_limit(42), 0.0);
    }
}
