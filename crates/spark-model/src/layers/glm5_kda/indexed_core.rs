// SPDX-License-Identifier: AGPL-3.0-only

//! Independent indexed core selection and pre-mutation validation.

use super::Glm5KdaLayer;
use crate::layer::ForwardContext;
use crate::layer::ssm_batch::{SsmBatchLayer, SsmBatchView};
use crate::layers::ops::{self, KdaBuffer, KdaIndexedConv, KdaIndexedRecurrent, KdaIndexedShape};

pub(super) fn ordinal(config: &atlas_core::config::ModelConfig, global: usize) -> Result<usize> {
    let kinds: Vec<_> = (0..config.num_hidden_layers)
        .map(|layer| config.layer_type(layer))
        .collect();
    crate::model::ssm_indexed_decode::ssm_ordinal(&kinds, global)
}
use anyhow::{Result, ensure};

fn select(
    view: Option<SsmBatchView<'_>>,
    ordinal: usize,
    rows: usize,
    geometry: (usize, usize, usize),
    handles: [u64; 2],
) -> Result<Option<(KdaIndexedShape, SsmBatchLayer)>> {
    let Some(view) = view else { return Ok(None) };
    let layer = view.layer(ordinal)?;
    ensure!(
        layer.rows() as usize == rows,
        "indexed KDA present row metadata disagrees with decode width"
    );
    let Some(shape) = KdaIndexedShape::select(rows, geometry.0, geometry.1, geometry.2) else {
        return Ok(None);
    };
    ensure!(
        layer.h_stride_elements() >= 32 * 128 * 128
            && layer.conv_stride_elements() >= 3 * 32 * 128 * 4,
        "indexed KDA present FP32 pool cannot hold the supported state geometry"
    );
    if handles.contains(&0) {
        return Ok(None);
    }
    Ok(Some((shape, layer)))
}

pub(super) struct IndexedCore {
    shape: KdaIndexedShape,
    layer: SsmBatchLayer,
    conv: KdaIndexedConv,
    recurrent: KdaIndexedRecurrent,
}

impl Glm5KdaLayer {
    /// Runs before mHC/projections as well as before either stateful operation.
    /// No allocation, metadata upload, kernel lookup, or device work here.
    pub(super) fn prepare_indexed_core(
        &self,
        rows: usize,
        ctx: &ForwardContext,
    ) -> Result<Option<IndexedCore>> {
        let Some((shape, layer)) = select(
            ctx.ssm_batch,
            self.ssm_ordinal,
            rows,
            (self.heads, self.dim, self.conv_width),
            [self.conv_indexed_k.0, self.recurrent_indexed_k.0],
        )?
        else {
            return Ok(None);
        };
        let sizes = ctx.buffers.sizes();
        let p = self.heads * self.dim;
        let beta_offset = rows * 3 * p * 2;
        let beta_bytes = sizes
            .qkv_output
            .checked_sub(beta_offset)
            .ok_or_else(|| anyhow::anyhow!("indexed KDA beta offset exceeds QKV arena"))?;
        let convolved = KdaBuffer {
            ptr: ctx.buffers.ssm_conv_out_f32(),
            bytes: sizes.ssm_conv_out_f32,
        };
        // DenseWeight retains only pointers; these fixed extents are established
        // by glm5/components.rs's explicit conv allocation and FP32 TP sharding.
        let conv = KdaIndexedConv {
            input: KdaBuffer {
                ptr: ctx.buffers.ssm_qkvz(),
                bytes: sizes.ssm_qkvz,
            },
            weight: KdaBuffer {
                ptr: self.weights.conv.weight,
                bytes: self.conv_state_bytes / 2,
            },
            bias: None,
            output: convolved,
            input_stride: 3 * p,
            output_stride: 3 * p,
        };
        let recurrent = KdaIndexedRecurrent {
            qkv: convolved,
            gate: KdaBuffer {
                ptr: ctx.buffers.ssm_deinterleaved(),
                bytes: sizes.ssm_deinterleaved,
            },
            beta: KdaBuffer {
                ptr: ctx.buffers.qkv_output().offset(beta_offset),
                bytes: beta_bytes,
            },
            a_log: KdaBuffer {
                ptr: self.weights.a_log.weight,
                bytes: self.heads * 4,
            },
            dt_bias: KdaBuffer {
                ptr: self.weights.dt_bias.weight,
                bytes: p * 4,
            },
            output: KdaBuffer {
                ptr: ctx.buffers.attn_output(),
                bytes: sizes.attn_output,
            },
            lower_bound: self.lower_bound,
        };
        ops::validate_kda_indexed_pair(
            shape,
            layer,
            self.conv_indexed_k,
            self.recurrent_indexed_k,
            conv,
            recurrent,
        )?;
        Ok(Some(IndexedCore {
            shape,
            layer,
            conv,
            recurrent,
        }))
    }

    pub(super) fn run_indexed_core(
        &self,
        core: IndexedCore,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.ssm_ordinal == 0
            && !self
                .indexed_trace_logged
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::info!(
                rows = core.layer.rows(),
                ssm_ordinal = self.ssm_ordinal,
                capture = ctx.graph_capture,
                "GLM KDA selected indexed convolution/recurrent pair"
            );
        }
        ops::kda_conv_indexed(
            ctx.gpu,
            self.conv_indexed_k,
            core.shape,
            core.layer,
            core.conv,
            stream,
        )?;
        ops::kda_recurrent_indexed(
            ctx.gpu,
            self.recurrent_indexed_k,
            core.shape,
            core.layer,
            core.recurrent,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ssm_batch::SsmPoolView;
    use spark_runtime::gpu::DevicePtr;

    #[test]
    fn absent_views_missing_handles_and_unsupported_shapes_preserve_fallback() {
        let h = [DevicePtr(0x10000000)];
        let c = [DevicePtr(0x20000000)];
        let pool = SsmPoolView::new(&h, &c, 2097152, 2097152, 196608, 4).unwrap();
        for rows in 1..=4 {
            let ids = [3, 0, 2, 1];
            let view = SsmBatchView::new(pool, DevicePtr(0x1000), &ids[..rows]).unwrap();
            assert_eq!(
                select(Some(view), 0, rows, (32, 128, 4), [1, 2])
                    .unwrap()
                    .is_some(),
                rows >= 2
            );
            assert!(
                select(None, 0, rows, (32, 128, 4), [1, 2])
                    .unwrap()
                    .is_none()
            );
            for handles in [[0, 2], [1, 0], [0, 0]] {
                assert!(
                    select(Some(view), 0, rows, (32, 128, 4), handles)
                        .unwrap()
                        .is_none()
                );
            }
            assert!(
                select(Some(view), 0, rows, (64, 128, 4), [1, 2])
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn malformed_present_view_does_not_hide_behind_missing_kernels() {
        let h = [DevicePtr(0x10000000)];
        let c = [DevicePtr(0x20000000)];
        let pool = SsmPoolView::new(&h, &c, 2097152, 2097152, 196608, 4).unwrap();
        let view = SsmBatchView::new(pool, DevicePtr(0x1000), &[3, 0, 2]).unwrap();
        assert!(select(Some(view), 0, 4, (32, 128, 4), [0, 0]).is_err());
        assert!(select(Some(view), 1, 3, (32, 128, 4), [0, 0]).is_err());
        let small = SsmPoolView::new(&h, &c, 64, 64, 32, 4).unwrap();
        let view = SsmBatchView::new(small, DevicePtr(0x1000), &[3, 0]).unwrap();
        assert!(select(Some(view), 0, 2, (32, 128, 4), [0, 0]).is_err());
    }

    #[test]
    fn initialization_maps_mixed_global_layers_to_ssm_ordinals() {
        use atlas_core::config::LayerType::{FullAttention as F, LinearAttention as L};
        let kinds = [L, F, L, L, F, L];
        for (global, ordinal) in [(0, 0), (2, 1), (3, 2), (5, 3)] {
            assert_eq!(
                crate::model::ssm_indexed_decode::ssm_ordinal(&kinds, global).unwrap(),
                ordinal
            );
        }
        let source = include_str!("init.rs");
        assert!(source.contains("indexed_core::ordinal(config, layer_idx)?"));
        assert!(source.contains("\"glm_kda_conv_indexed\""));
        assert!(source.contains("\"glm_kda_recurrent_indexed\""));
    }

    #[test]
    fn ordinal_uses_the_same_implicit_and_partial_layer_map_as_metadata() {
        use atlas_core::config::{LayerType, ModelConfig};
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.num_hidden_layers = 6;
        config.full_attention_interval = 4;
        config.layer_types.clear();
        for (global, expected) in [(0, 0), (2, 2), (4, 3), (5, 4)] {
            assert_eq!(ordinal(&config, global).unwrap(), expected);
        }
        assert!(ordinal(&config, 3).is_err());
        assert!(ordinal(&config, 6).is_err());
        config.layer_types = vec![LayerType::LinearAttention];
        assert_eq!(ordinal(&config, 0).unwrap(), 0);
        assert!(ordinal(&config, 1).is_err());
    }
}
