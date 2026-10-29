// SPDX-License-Identifier: AGPL-3.0-only
//! Real constructor lookups: GLM-only projection probes must not poison other boots.
use super::Qwen3AttentionLayer;
use crate::layers::FfnComponent;
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight};
use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use std::sync::Mutex;

#[derive(Default)]
struct Recorder {
    gpu: MockGpuBackend,
    lookups: Mutex<Vec<String>>,
}
impl GpuBackend for Recorder {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.gpu.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.gpu.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.gpu.free(p)
    }
    fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
        self.gpu.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.gpu.copy_d2h(s, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.gpu.copy_d2d(s, d, n)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.gpu.synchronize(s)
    }
    fn memset(&self, p: DevicePtr, value: u8, n: usize) -> Result<()> {
        self.gpu.memset(p, value, n)
    }
    fn memset_async(&self, p: DevicePtr, value: u8, n: usize, stream: u64) -> Result<()> {
        self.gpu.memset_async(p, value, n, stream)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn total_memory(&self) -> Result<usize> {
        self.gpu.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.gpu.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.gpu.sm_count()
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.gpu.op_cache()
    }
    fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
        if module == "mxfp8_gemv" || name == "dense_gemv_bf16_batch5" {
            self.lookups
                .lock()
                .unwrap()
                .push(format!("{module}::{name}"));
        }
        self.gpu.kernel(module, name)
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("constructor must not launch a kernel")
    }
}

#[test]
fn real_constructor_limits_glm_projection_probes_to_glm_mla() {
    for (model, latent, rope, expected) in [
        ("qwen3_next", 0, 0, false),
        ("qwen3_6_moe", 0, 0, false),
        ("deepseek_v3", 512, 64, false),
        ("glm5_next", 512, 0, true),
    ] {
        let gpu = Recorder::default();
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = model.into();
        config.kv_lora_rank = latent;
        config.qk_rope_head_dim = rope;
        if expected {
            config.index_kpool = 4;
            config.index_head_dim = 128;
        }
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };
        let attn = AttentionWeights {
            q_proj: null,
            k_proj: null,
            v_proj: null,
            o_proj: QuantizedWeight::null(),
            q_norm: null,
            k_norm: null,
            q_norm_full: None,
            k_norm_full: None,
            k_scale: 1.0,
            v_scale: 1.0,
        };
        let layer = Qwen3AttentionLayer::new_ungated(
            null,
            attn,
            null,
            FfnComponent::None,
            0,
            None,
            None,
            None,
            &gpu,
            KvCacheDtype::Bf16,
            0,
            &config,
        )
        .unwrap();
        let mut actual = gpu.lookups.into_inner().unwrap();
        actual.sort();
        let mut wanted = if expected {
            vec![
                "dense_gemv_bf16_batchm::dense_gemv_bf16_batch5",
                "mxfp8_gemv::mxfp8_gemv_tc8",
                "mxfp8_gemv::mxfp8_gemv_tc16",
                "mxfp8_gemv::mxfp8_gemv_tc32",
                "mxfp8_gemv::mxfp8_gemv_tc8_grouped",
                "mxfp8_gemv::mxfp8_gemv_tc16_grouped",
                "mxfp8_gemv::mxfp8_quantize_bf16",
            ]
        } else {
            vec![]
        };
        wanted.sort();
        assert_eq!(
            actual, wanted,
            "{model}: unused architecture must issue no lookup"
        );
        assert_eq!(layer.dense_gemv_batch5_k.0 != 0, expected);
        assert!(layer.mxfp8_gemv_k.iter().all(|k| (k.0 != 0) == expected));
        assert!(
            layer
                .mxfp8_gemv_grouped_k
                .iter()
                .all(|k| (k.0 != 0) == expected)
        );
        assert_eq!(layer.mxfp8_quantize_k.0 != 0, expected);
    }
}
