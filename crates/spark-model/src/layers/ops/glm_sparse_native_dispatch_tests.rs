// SPDX-License-Identifier: AGPL-3.0-only
use super::super::*;
use anyhow::bail;
use spark_runtime::gpu::{GpuBackend, KernelArg, KernelHandle, mock::MockGpuBackend};
use std::{ffi::c_void, sync::Mutex};
#[derive(Default)]
struct Capture {
    inner: MockGpuBackend,
    arguments: Mutex<Vec<Vec<Vec<u8>>>>,
    fail_kernel: Option<usize>,
    fail_launch: Option<usize>,
    launches: Mutex<Vec<(u64, [u32; 3], [u32; 3], u32, u64)>>,
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
        let index = self
            .lookups
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let expected = [
            ("glm_sparse_native_prep", "glm_sparse_native_prep_q"),
            ("glm_sparse_native_prep", "glm_sparse_native_prep_kv"),
            ("glm_sparse_native_split", "glm_sparse_native_split"),
            ("glm_sparse_native_attention", "glm_sparse_native_attention"),
            ("glm_sparse_native_merge", "glm_sparse_native_merge"),
        ];
        assert_eq!((module, symbol), expected[index % 5]);
        Ok(KernelHandle(if self.fail_kernel == Some(index) {
            0
        } else {
            index as u64 + 1
        }))
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
        let mut launches = self.launches.lock().unwrap();
        launches.push((kernel.0, grid, block, shared, stream));
        if self.fail_launch == Some(launches.len() - 1) {
            bail!("injected launch failure")
        }
        self.arguments.lock().unwrap().push(
            args.iter()
                .map(|a| match a {
                    KernelArg::Buffer(p) => p.0.to_ne_bytes().to_vec(),
                    KernelArg::Bytes(b) => b.to_vec(),
                })
                .collect(),
        );
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

fn plan() -> Plan {
    let spans = std::array::from_fn(|i| Span {
        ptr: (i as u64 + 1) * 0x100000000,
        bytes: 1 << 29,
    });
    Plan::new(2048, 16384, 1155, 1152, spans, 19).unwrap()
}
#[test]
fn all_symbols_resolve_before_any_sparse_write() {
    for index in 0..5 {
        let gpu = Capture {
            fail_kernel: Some(index),
            ..Default::default()
        };
        assert!(dispatch::execute(&plan(), &gpu).is_err());
        assert!(gpu.launches.lock().unwrap().is_empty());
        assert_eq!(gpu.inner.alloc_count(), 0);
    }
}
#[test]
fn six_native_launches_use_owned_stream_and_exact_geometry() {
    let gpu = Capture::default();
    dispatch::execute(&plan(), &gpu).unwrap();
    let launches = gpu.launches.lock().unwrap();
    assert_eq!(launches.len(), 6);
    assert_eq!(
        *launches,
        vec![
            (1, [147456, 1, 1], [256, 1, 1], 0, 19),
            (2, [18432, 1, 1], [128, 1, 1], 0, 19),
            (3, [2048, 1, 1], [256, 1, 1], 0, 19),
            (4, [2048, 1, 1], [384, 1, 1], 91920, 19),
            (4, [2048, 1, 1], [384, 1, 1], 91920, 19),
            (5, [2048, 32, 1], [256, 1, 1], 0, 19),
        ]
    );
    assert_eq!(gpu.inner.alloc_count(), 0);
    assert_eq!(gpu.inner.sync_count(), 0);
    let args = gpu.arguments.lock().unwrap();
    assert_eq!(
        args.iter().map(Vec::len).collect::<Vec<_>>(),
        [3, 4, 8, 7, 7, 9]
    );
    assert_eq!(args[0][0], (0x100000000u64).to_ne_bytes());
    assert_eq!(args[0][1], (0x500000000u64).to_ne_bytes());
    assert_eq!(args[3][3], (0x800000000u64).to_ne_bytes());
    assert_eq!(args[4][3], (0x900000000u64).to_ne_bytes());
    assert_eq!(args[5][5], (0xa00000000u64).to_ne_bytes());
    // Independently enumerated 2048-row metadata layout: ID planes, counts,
    // then three log2-LSE planes, all aligned to256 bytes.
    let m = [
        0u64, 16777216, 33554432, 33562624, 33570816, 33579008, 33841152, 34103296,
    ]
    .map(|offset| 0x700000000 + offset);
    assert_eq!(
        args[2][1..6],
        m[..5]
            .iter()
            .map(|p| p.to_ne_bytes().to_vec())
            .collect::<Vec<_>>()
    );
    for (launch, ids, lse, lengths) in [(3, 0, 5, 2), (4, 1, 6, 3)] {
        assert_eq!(args[launch][2], m[ids].to_ne_bytes());
        assert_eq!(args[launch][4], m[lse].to_ne_bytes());
        assert_eq!(args[launch][5], m[lengths].to_ne_bytes());
        assert_eq!(args[launch][6], 2048u32.to_ne_bytes());
    }
    assert_eq!(args[5][6], m[7].to_ne_bytes());
}
#[test]
fn submission_failure_stops_without_fallback_or_further_writes() {
    for failed in 0..6 {
        let gpu = Capture {
            fail_launch: Some(failed),
            ..Default::default()
        };
        assert!(
            dispatch::execute(&plan(), &gpu)
                .unwrap_err()
                .to_string()
                .contains("injected")
        );
        assert_eq!(gpu.launches.lock().unwrap().len(), failed + 1);
    }
}
#[test]
fn preloading_resolves_without_allocation_or_launch() {
    let gpu = Capture::default();
    dispatch::resolve(&gpu).unwrap();
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 5);
    assert!(gpu.launches.lock().unwrap().is_empty());
    assert_eq!(gpu.inner.alloc_count(), 0);
}
