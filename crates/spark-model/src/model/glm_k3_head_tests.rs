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
        assert_eq!(module, "dense_gemv_bf16_batchm");
        assert_eq!(symbol, "dense_gemv_bf16_batchm");
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
        assert!(kernel.0 == 0x777 || kernel.0 == 0x888);
        assert_eq!(
            grid,
            if kernel.0 == 0x777 {
                [38714, 1, 1]
            } else {
                [9679, 1, 1]
            }
        );
        assert_eq!(
            block,
            if kernel.0 == 0x777 {
                [256, 1, 1]
            } else {
                [16, 16, 1]
            }
        );
        assert_eq!(shared, 0);
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
    c.vocab_size = 154856;
    c.lm_head_bf16_override = Some(true);
    c
}
fn tensor() -> WeightTensor {
    WeightTensor {
        ptr: DevicePtr(0x2000),
        shape: vec![154856, 4096],
        dtype: WeightDtype::BF16,
    }
}
fn run(
    gpu: &Capture,
    c: &ModelConfig,
    rows: u32,
    enabled: bool,
    k5: bool,
    kernel: KernelHandle,
) -> Result<()> {
    dispatch(
        gpu,
        c,
        DevicePtr(0x1000),
        &DenseWeight {
            weight: DevicePtr(0x2000),
        },
        DevicePtr(0x3000),
        rows,
        kernel,
        KernelHandle(0x888),
        19,
        enabled,
        k5,
    )
}
fn expect_args(gpu: &Capture, rows: u32, batchm: bool) {
    let mut expected = vec![
        0x1000_u64.to_ne_bytes().to_vec(),
        0x2000_u64.to_ne_bytes().to_vec(),
        0x3000_u64.to_ne_bytes().to_vec(),
        rows.to_ne_bytes().to_vec(),
        154856_u32.to_ne_bytes().to_vec(),
        4096_u32.to_ne_bytes().to_vec(),
    ];
    if batchm {
        expected.push(154856_u32.to_ne_bytes().to_vec());
    }
    assert_eq!(*gpu.arguments.lock().unwrap(), expected);
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert!(gpu.streams.lock().unwrap().is_empty());
}
#[test]
fn k3_exact_resident_head_validation_and_preload() {
    let gpu = Capture::default();
    initialize(&gpu, &config(), &tensor(), true).unwrap();
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert!(gpu.arguments.lock().unwrap().is_empty());
    assert!(gpu.streams.lock().unwrap().is_empty());
}
#[test]
fn k3_dispatch_exact_pointers_dimensions_and_stride() {
    let gpu = Capture::default();
    run(&gpu, &config(), 3, true, false, KernelHandle(0x777)).unwrap();
    expect_args(&gpu, 3, true);
}
#[test]
fn k3_disabled_and_k5_default_still_use_gemm() {
    for (rows, enabled) in [
        (1, true),
        (3, false),
        (4, true),
        (5, false),
        (5, true),
        (8, true),
    ] {
        let gpu = Capture::default();
        run(&gpu, &config(), rows, enabled, false, KernelHandle(0x777)).unwrap();
        expect_args(&gpu, rows, false);
    }
}
#[test]
fn k5_existing_opt_in_remains_independent() {
    for enabled in [false, true] {
        let gpu = Capture::default();
        run(&gpu, &config(), 5, enabled, true, KernelHandle(0x777)).unwrap();
        expect_args(&gpu, 5, true);
    }
    let gpu = Capture::default();
    run(&gpu, &config(), 5, true, true, KernelHandle(0)).unwrap();
    expect_args(&gpu, 5, false);
}
#[test]
fn k3_rejects_wrong_actual_shape_dtype_or_null_before_lookup() {
    for (shape, dtype, ptr) in [
        (vec![154855, 4096], WeightDtype::BF16, 0x2000),
        (vec![154856, 2048], WeightDtype::BF16, 0x2000),
        (vec![154856, 4096], WeightDtype::UInt8, 0x2000),
        (vec![154856, 4096], WeightDtype::BF16, 0),
        (vec![154880, 4096], WeightDtype::BF16, 0x2002),
        (vec![154881, 4096], WeightDtype::BF16, 0x2000),
    ] {
        let gpu = Capture::default();
        let t = WeightTensor {
            ptr: DevicePtr(ptr),
            shape,
            dtype,
        };
        assert!(initialize(&gpu, &config(), &t, true).is_err());
        assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
        initialize(&gpu, &config(), &t, false).unwrap();
    }
}
#[test]
fn k3_config_mismatch_quantized_head_or_missing_kernel_fails_closed() {
    for change in 0..4 {
        let mut c = config();
        match change {
            0 => c.hidden_size = 2048,
            1 => c.vocab_size = 154880,
            2 => c.lm_head_bf16_override = Some(false),
            _ => c.model_type = "other".into(),
        }
        let gpu = Capture::default();
        assert!(initialize(&gpu, &c, &tensor(), true).is_err());
        assert!(gpu.arguments.lock().unwrap().is_empty());
    }
    let gpu = Capture::default();
    assert!(run(&gpu, &config(), 3, true, false, KernelHandle(0)).is_err());
    assert!(gpu.arguments.lock().unwrap().is_empty());
    let mut c = config();
    c.vocab_size = 154880;
    assert!(run(&gpu, &c, 3, true, false, KernelHandle(0x777)).is_err());
    assert!(gpu.arguments.lock().unwrap().is_empty());
}
#[test]
fn k3_flag_is_strict() {
    assert!(!parse_flag(None).unwrap());
    assert!(!parse_flag(Some("0")).unwrap());
    assert!(parse_flag(Some("1")).unwrap());
    for s in ["", "true", "false", "2", " 1"] {
        assert!(parse_flag(Some(s)).is_err());
    }
}

#[test]
fn k3_padded_resident_head_keeps_active_logits_geometry() {
    let gpu = Capture::default();
    let mut t = tensor();
    t.shape[0] = 154880;
    initialize(&gpu, &config(), &t, true).unwrap();
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 1);
    gpu.lookups.store(0, std::sync::atomic::Ordering::Relaxed);
    run(&gpu, &config(), 3, true, false, KernelHandle(0x777)).unwrap();
    expect_args(&gpu, 3, true);
}
#[test]
fn k3_vector_alignment_rejected_before_launch() {
    for (input, weight, output) in [
        (0x1002, 0x2000, 0x3000),
        (0x1000, 0x2008, 0x3000),
        (0x1000, 0x2000, 0x3001),
    ] {
        let gpu = Capture::default();
        assert!(
            dispatch(
                &gpu,
                &config(),
                DevicePtr(input),
                &DenseWeight {
                    weight: DevicePtr(weight)
                },
                DevicePtr(output),
                3,
                KernelHandle(0x777),
                KernelHandle(0x888),
                19,
                true,
                false
            )
            .is_err()
        );
        assert!(gpu.arguments.lock().unwrap().is_empty());
        assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
    // Scalar BF16 stores do not require a 16-byte-aligned output pointer.
    let gpu = Capture::default();
    dispatch(
        &gpu,
        &config(),
        DevicePtr(0x1000),
        &DenseWeight {
            weight: DevicePtr(0x2000),
        },
        DevicePtr(0x3002),
        3,
        KernelHandle(0x777),
        KernelHandle(0x888),
        19,
        true,
        false,
    )
    .unwrap();
    assert_eq!(gpu.arguments.lock().unwrap()[2], 0x3002_u64.to_ne_bytes());
}
