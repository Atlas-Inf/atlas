// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::{DevicePtr, KernelArg, KernelHandle, mock::MockGpuBackend};
use std::{ffi::c_void, sync::Mutex};

#[derive(Default)]
struct Capture {
    inner: MockGpuBackend,
    arguments: Mutex<Vec<Vec<u8>>>,
    lookups: std::sync::atomic::AtomicUsize,
    streams: Mutex<Vec<u64>>,
}

impl GpuBackend for Capture {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, s: &[u8], d: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(s, d)
    }
    fn copy_d2h(&self, s: DevicePtr, d: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(s, d)
    }
    fn copy_d2d(&self, s: DevicePtr, d: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(s, d, n)
    }
    fn synchronize(&self, s: u64) -> Result<()> {
        self.streams.lock().unwrap().push(s);
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, symbol: &str) -> Result<KernelHandle> {
        assert_eq!(module, "glm_sparse_prefill_kv_reuse");
        assert_eq!(symbol, "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad");
        self.lookups
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(KernelHandle(0x777))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut c_void],
    ) -> Result<()> {
        bail!("expected the production typed launch")
    }
    fn launch_typed(
        &self,
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        assert_eq!(kernel.0, 0x777);
        assert_eq!(grid, [1, 1, 1]);
        assert_eq!(block, [256, 1, 1]);
        assert_eq!(shared, 69376);
        assert!(stream == 19 || stream == 0);
        *self.arguments.lock().unwrap() = args
            .iter()
            .map(|a| match a {
                KernelArg::Buffer(p) => p.0.to_ne_bytes().to_vec(),
                KernelArg::Bytes(b) => b.to_vec(),
            })
            .collect();
        Ok(())
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.inner.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, s)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.inner.sm_count()
    }
}

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c
}
fn args(c: &ModelConfig) -> GlmSparsePrefillTc<'_> {
    GlmSparsePrefillTc {
        config: c,
        dtype: KvCacheDtype::Bf16,
        identical_kv_latent: true,
        query: DevicePtr(16),
        k_cache: DevicePtr(32),
        v_cache: DevicePtr(48),
        indices: DevicePtr(64),
        output: DevicePtr(80),
        block_table: DevicePtr(96),
        rows: 1,
        heads: 32,
        head_dim: 512,
        index_width: 2051,
        block_size: 16,
        scale: 0.0625,
    }
}
fn values(gpu: &Capture) -> Vec<Vec<u8>> {
    gpu.arguments.lock().unwrap().clone()
}
#[test]
fn decode_tc_disabled_has_no_lookup_launch_sync_or_allocation() {
    let gpu = Capture::default();
    let c = config();
    assert!(!dispatch_decode(&gpu, &args(&c), 19, false).unwrap());
    initialize_decode(&gpu, &c, false).unwrap();
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert!(values(&gpu).is_empty());
    assert_eq!(gpu.inner.sync_count(), 0);
    assert_eq!(gpu.inner.alloc_count(), 0);
}
#[test]
fn decode_tc_launch_preserves_all_six_pointers_and_geometry() {
    let gpu = Capture::default();
    let c = config();
    assert!(dispatch_decode(&gpu, &args(&c), 19, true).unwrap());
    let actual = values(&gpu);
    let mut expected: Vec<Vec<u8>> = [16u64, 32, 48, 64, 80, 96]
        .into_iter()
        .map(|x| x.to_ne_bytes().to_vec())
        .collect();
    expected.extend(
        [1u32, 32, 512, 2051, 16]
            .into_iter()
            .map(|x| x.to_ne_bytes().to_vec()),
    );
    expected.push(0.0625f32.to_ne_bytes().to_vec());
    assert_eq!(actual, expected);
    assert_eq!(gpu.inner.sync_count(), 0);
}
#[test]
fn decode_tc_rejects_invalid_data_before_kernel_lookup() {
    let gpu = Capture::default();
    let c = config();
    for variant in 0..15 {
        let mut a = args(&c);
        match variant {
            0 => a.rows = 0,
            1 => a.rows = 2,
            2 => a.rows = 3,
            3 => a.identical_kv_latent = false,
            4 => a.dtype = KvCacheDtype::Fp8,
            5 => a.heads = 64,
            6 => a.head_dim = 256,
            7 => a.index_width = 2048,
            8 => a.block_size = 64,
            9 => a.scale = f32::NAN,
            10 => a.query = DevicePtr::NULL,
            11 => a.k_cache = DevicePtr::NULL,
            12 => a.v_cache = DevicePtr::NULL,
            13 => a.indices = DevicePtr::NULL,
            _ => a.output = DevicePtr::NULL,
        }
        assert!(dispatch_decode(&gpu, &a, 19, true).is_err());
    }
    let mut a = args(&c);
    a.block_table = DevicePtr::NULL;
    assert!(dispatch_decode(&gpu, &a, 19, true).is_err());
    for field in 0..6 {
        let mut a = args(&c);
        match field {
            0 => a.query = a.query.offset(2),
            1 => a.k_cache = a.k_cache.offset(2),
            2 => a.v_cache = a.v_cache.offset(2),
            3 => a.indices = a.indices.offset(2),
            4 => a.output = a.output.offset(2),
            _ => a.block_table = a.block_table.offset(2),
        }
        assert!(dispatch_decode(&gpu, &a, 19, true).is_err());
    }
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert!(values(&gpu).is_empty());
}
#[test]
fn decode_tc_startup_is_exact_zero_rows_null_storage_then_sync() {
    let gpu = Capture::default();
    let c = config();
    initialize_decode(&gpu, &c, true).unwrap();
    let mut expected = vec![0u64.to_ne_bytes().to_vec(); 6];
    expected.extend(
        [0u32, 32, 512, 2051, 16]
            .into_iter()
            .map(|x| x.to_ne_bytes().to_vec()),
    );
    expected.push(0.0625f32.to_ne_bytes().to_vec());
    assert_eq!(values(&gpu), expected);
    assert_eq!(*gpu.streams.lock().unwrap(), vec![0]);
    assert_eq!(gpu.inner.sync_count(), 1);
    assert_eq!(gpu.inner.alloc_count(), 0);
}
#[test]
fn decode_tc_config_and_flag_fail_closed_before_startup() {
    let gpu = Capture::default();
    for mutate in [
        (|c: &mut ModelConfig| c.model_type = "other".into()) as fn(&mut ModelConfig),
        |c| c.hidden_size = 2048,
        |c| c.kv_lora_rank = 256,
        |c| c.qk_rope_head_dim = 64,
        |c| c.index_topk = 1024,
        |c| c.index_kpool = 8,
    ] {
        let mut c = config();
        mutate(&mut c);
        assert!(initialize_decode(&gpu, &c, true).is_err());
        assert!(dispatch_decode(&gpu, &args(&c), 19, true).is_err());
    }
    for value in ["", "true", "2", " 1"] {
        assert!(parse("glm5_next", "ATLAS_GLM_SPARSE_DECODE_TC", Some(value)).is_err());
    }
    for value in [None, Some("0")] {
        assert!(!parse("glm5_next", "ATLAS_GLM_SPARSE_DECODE_TC", value).unwrap());
    }
    assert!(parse("glm5_next", "ATLAS_GLM_SPARSE_DECODE_TC", Some("1")).unwrap());
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
}
