// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::{KernelArg, mock::MockGpuBackend};
use std::{ffi::c_void, sync::Mutex};

/// A launch as the backend saw it: kernel, grid, block, argument bytes.
type Launch = (u64, [u32; 3], [u32; 3], Vec<Vec<u8>>);

/// Resolves the kernels in `shipped` (module, name) to 1-based handles in
/// that order and records every typed launch.
struct Capture {
    inner: MockGpuBackend,
    shipped: Vec<(&'static str, &'static str)>,
    launches: Mutex<Vec<Launch>>,
}

impl Capture {
    fn new(shipped: &[(&'static str, &'static str)]) -> Self {
        Self {
            inner: MockGpuBackend::new(),
            shipped: shipped.to_vec(),
            launches: Mutex::default(),
        }
    }
    fn handle(&self, module: &str, name: &str) -> u64 {
        1 + self
            .shipped
            .iter()
            .position(|&(m, n)| m == module && n == name)
            .unwrap() as u64
    }
    fn launches(&self) -> Vec<Launch> {
        self.launches.lock().unwrap().clone()
    }
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
        self.inner.synchronize(s)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, symbol: &str) -> Result<KernelHandle> {
        match self
            .shipped
            .iter()
            .position(|&(m, n)| m == module && n == symbol)
        {
            Some(i) => Ok(KernelHandle(1 + i as u64)),
            None => bail!("{module}::{symbol} not shipped"),
        }
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
        bail!("expected the typed launch")
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
        assert_eq!((shared, stream), (0, 7));
        let args = args
            .iter()
            .map(|a| match a {
                KernelArg::Buffer(p) => p.0.to_ne_bytes().to_vec(),
                KernelArg::Bytes(b) => b.to_vec(),
            })
            .collect();
        self.launches
            .lock()
            .unwrap()
            .push((kernel.0, grid, block, args));
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

const GLM: &[(&str, &str)] = &[
    ("hyper_connection", "hc_post"),
    ("hyper_connection", "hc_post_bf16"),
    ("hyper_connection", "hc_post_bf16_add_bf16"),
    ("glm_hc_prefill_vec", "glm_hc_decode_post_bf16"),
    ("glm_hc_prefill_vec", "glm_hc_decode_partial_rows_bf16"),
    ("glm_hc_prefill_vec", "glm_hc_decode_post_partial_rows_bf16"),
    ("moe", "moe_unpermute_blend_ep_vec8"),
    ("glm_hc_prefill_vec", "glm_hc_decode_partial_ilp_bf16"),
    ("glm_hc_prefill_vec", "glm_hc_decode_post_partial_ilp_bf16"),
    ("glm_hc_prefill_vec", "glm_hc_decode_finalize_ilp_bf16"),
];

fn ptr(p: u64) -> Vec<u8> {
    p.to_ne_bytes().to_vec()
}

fn word(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

#[test]
fn flag_is_explicit_and_mask_selects_groups() {
    assert_eq!(parse(None, None).unwrap(), 0);
    assert_eq!(parse(Some("0"), Some("31")).unwrap(), 0);
    assert_eq!(parse(Some("1"), None).unwrap(), ALL);
    assert_eq!(parse(Some("1"), Some("2")).unwrap(), HC_PARTIAL);
    assert_eq!(parse(Some("1"), Some("24")).unwrap(), MOE_SORT | RMS_NORM);
    assert_eq!(parse(Some("1"), Some("0")).unwrap(), 0);
    assert!(parse(Some("true"), None).is_err());
    assert!(parse(Some("1"), Some("32")).is_err());
    assert!(parse(Some("1"), Some("0x3")).is_err());
    assert_eq!(ALL, 31);
}

#[test]
fn handle_twins_swap_only_when_their_group_is_on_and_shipped() {
    let gpu = Capture::new(&[("moe", "moe_sort_by_expert_scan")]);
    let sort = KernelHandle(42);
    let scan = ("moe", "moe_sort_by_expert_scan");
    assert_eq!(twin_for(0, &gpu, MOE_SORT, sort, scan).0, 42);
    assert_eq!(twin_for(ALL & !MOE_SORT, &gpu, MOE_SORT, sort, scan).0, 42);
    assert_eq!(twin_for(MOE_SORT, &gpu, MOE_SORT, sort, scan).0, 1);
    // An absent original stays absent; an unshipped twin keeps the original.
    assert_eq!(twin_for(ALL, &gpu, MOE_SORT, KernelHandle(0), scan).0, 0);
    let regs = ("glm_rms_norm_regs", "rms_norm_vanilla_regs");
    assert_eq!(twin_for(ALL, &gpu, RMS_NORM, KernelHandle(7), regs).0, 7);
    assert!(gpu.launches().is_empty());
}

#[test]
fn hc_post_twin_takes_the_glm_bf16_post_and_its_peer_add() {
    let gpu = Capture::new(GLM);
    let (block, peer, streams, post, comb) = (0x1000, 0x2000, 0x3000, 0x4000, 0x5000);
    let hc_post_bf16 = KernelHandle(gpu.handle("hyper_connection", "hc_post_bf16"));
    let ptrs = |peer| [block, peer, streams, post, comb, streams].map(DevicePtr);
    assert!(hc_post_for(ALL, &gpu, hc_post_bf16, ptrs(0), [5, 4096, 4], 7).unwrap());
    let add = KernelHandle(gpu.handle("hyper_connection", "hc_post_bf16_add_bf16"));
    assert!(hc_post_for(HC_POST, &gpu, add, ptrs(peer), [32, 4096, 4], 7).unwrap());
    let twin = gpu.handle("glm_hc_prefill_vec", "glm_hc_decode_post_bf16");
    let launches = gpu.launches();
    assert_eq!(launches.len(), 2);
    for ((kernel, grid, block_dim, args), (rows, peer)) in launches.iter().zip([(5, 0), (32, peer)])
    {
        assert_eq!(
            (*kernel, *grid, *block_dim),
            (twin, [2 * rows, 1, 1], [256, 1, 1])
        );
        let want = [
            ptr(block),
            ptr(peer),
            ptr(streams),
            ptr(post),
            ptr(comb),
            word(rows),
        ];
        assert_eq!(args[..], want[..]);
    }
}

#[test]
fn hc_post_twin_leaves_every_other_post_alone() {
    let gpu = Capture::new(GLM);
    let bf16 = KernelHandle(gpu.handle("hyper_connection", "hc_post_bf16"));
    let fp32 = KernelHandle(gpu.handle("hyper_connection", "hc_post"));
    let ptrs = [0x1000, 0, 0x3000, 0x4000, 0x5000, 0x3000].map(DevicePtr);
    let off =
        |groups, kernel, ptrs, dims| !hc_post_for(groups, &gpu, kernel, ptrs, dims, 7).unwrap();
    assert!(off(0, bf16, ptrs, [4, 4096, 4]));
    assert!(off(HC_PARTIAL | MOE_POST, bf16, ptrs, [4, 4096, 4]));
    assert!(off(ALL, fp32, ptrs, [4, 4096, 4]));
    assert!(off(ALL, bf16, ptrs, [0, 4096, 4]));
    assert!(off(ALL, bf16, ptrs, [33, 4096, 4]));
    assert!(off(ALL, bf16, ptrs, [4, 2048, 4]));
    assert!(off(ALL, bf16, ptrs, [4, 4096, 2]));
    let mut out_of_place = ptrs;
    out_of_place[5] = DevicePtr(0x6000);
    assert!(off(ALL, bf16, out_of_place, [4, 4096, 4]));
    let mut unaligned = ptrs;
    unaligned[0] = DevicePtr(0x1008);
    assert!(off(ALL, bf16, unaligned, [4, 4096, 4]));
    assert!(gpu.launches().is_empty());
    // A target without the twin keeps its post.
    let old = Capture::new(&GLM[..3]);
    assert!(!hc_post_for(ALL, &old, bf16, ptrs, [4, 4096, 4], 7).unwrap());
}

#[test]
fn flag_off_hc_post_launch_is_unchanged() {
    let gpu = Capture::new(GLM);
    let bf16 = KernelHandle(gpu.handle("hyper_connection", "hc_post_bf16"));
    let [b, r, p, c] = [0x1000, 0x3000, 0x4000, 0x5000].map(DevicePtr);
    super::super::hc_post(&gpu, bf16, b, r, p, c, r, 4, 4096, 4, 7).unwrap();
    let launches = gpu.launches();
    let want = [
        ptr(b.0),
        ptr(r.0),
        ptr(p.0),
        ptr(c.0),
        ptr(r.0),
        word(4096),
        word(4),
    ];
    assert_eq!(launches.len(), 1);
    assert_eq!(
        (launches[0].0, launches[0].1, launches[0].2),
        (bf16.0, [4, 1, 1], [256, 1, 1])
    );
    assert_eq!(launches[0].3[..], want[..]);
}

#[test]
fn partial_twin_needs_its_group_a_bf16_highway_and_aligned_weights() {
    let gpu = Capture::new(GLM);
    let fn16 = DevicePtr(0x7000);
    let rows = |post| {
        let name = if post {
            "glm_hc_decode_post_partial_rows_bf16"
        } else {
            "glm_hc_decode_partial_rows_bf16"
        };
        Some(KernelHandle(gpu.handle("glm_hc_prefill_vec", name)))
    };
    for post in [false, true] {
        let twin = |groups, bf16, hc_fn| partial_twin(groups, &gpu, bf16, post, hc_fn).map(|k| k.0);
        assert_eq!(twin(HC_PARTIAL, true, fn16), rows(post).map(|k| k.0));
        assert_eq!(twin(HC_POST | MOE_POST, true, fn16), None);
        assert_eq!(twin(ALL, false, fn16), None);
        assert_eq!(twin(ALL, true, DevicePtr(0x7004)), None);
    }
    assert_eq!(
        partial_twin(ALL, &Capture::new(&GLM[..4]), true, true, fn16).map(|k| k.0),
        None
    );
}

#[test]
fn moe_twin_takes_the_split_blend_with_its_bytes_and_shape_guards() {
    let gpu = Capture::new(GLM);
    let ptrs = [0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800].map(DevicePtr);
    let dims = [4096, 5, 8, 0, 288];
    assert!(!moe_unpermute_blend_for(HC_POST | HC_PARTIAL, &gpu, ptrs, dims, 7).unwrap());
    assert!(!moe_unpermute_blend_for(ALL, &gpu, ptrs, [2048, 5, 8, 0, 288], 7).unwrap());
    assert!(!moe_unpermute_blend_for(ALL, &gpu, ptrs, [4096, 5, 9, 0, 288], 7).unwrap());
    assert!(!moe_unpermute_blend_for(ALL, &gpu, ptrs, [4096, 0, 8, 0, 288], 7).unwrap());
    let mut unaligned = ptrs;
    unaligned[7] = DevicePtr(0x802);
    assert!(!moe_unpermute_blend_for(ALL, &gpu, unaligned, dims, 7).unwrap());
    assert!(gpu.launches().is_empty());
    let mut ungated = ptrs;
    ungated[7] = DevicePtr::NULL;
    assert!(moe_unpermute_blend_for(MOE_POST, &gpu, ungated, dims, 7).unwrap());
    let launches = gpu.launches();
    let (kernel, grid, block, args) = &launches[0];
    assert_eq!(*kernel, gpu.handle("moe", "moe_unpermute_blend_ep_vec8"));
    assert_eq!((*grid, *block), ([5, 1, 1], [256, 1, 1]));
    let mut want: Vec<_> = ungated.iter().map(|p| ptr(p.0)).collect();
    want.extend([5, 8, 0, 288].map(word));
    assert_eq!(args[..], want[..]);
}

#[test]
fn seam_ilp_mask_is_explicit() {
    assert_eq!(parse_seam_ilp(None).unwrap(), 0);
    assert_eq!(parse_seam_ilp(Some("0")).unwrap(), 0);
    assert_eq!(parse_seam_ilp(Some("1")).unwrap(), SEAM_ILP_PARTIAL);
    assert_eq!(parse_seam_ilp(Some("2")).unwrap(), SEAM_ILP_FINALIZE);
    assert_eq!(parse_seam_ilp(Some("3")).unwrap(), 3);
    for bad in ["4", "-1", "on", "0x1", ""] {
        assert!(parse_seam_ilp(Some(bad)).is_err(), "{bad:?}");
    }
}

#[test]
fn seam_ilp_twins_need_their_bit_a_bf16_highway_and_the_bf16_weight() {
    let gpu = Capture::new(GLM);
    let fn16 = DevicePtr(0x7002);
    let handle = |name| Some(gpu.handle("glm_hc_prefill_vec", name));
    for (post, name) in [
        (false, "glm_hc_decode_partial_ilp_bf16"),
        (true, "glm_hc_decode_post_partial_ilp_bf16"),
    ] {
        let twin = |mask, bf16, w| seam_partial_twin(mask, &gpu, bf16, post, w).map(|k| k.0);
        assert_eq!(twin(SEAM_ILP_PARTIAL, true, fn16), handle(name));
        assert_eq!(twin(SEAM_ILP_FINALIZE, true, fn16), None);
        assert_eq!(twin(3, false, fn16), None);
        assert_eq!(twin(3, true, DevicePtr::NULL), None);
    }
    let finalize = |mask, bf16| seam_finalize_twin(mask, &gpu, bf16).map(|k| k.0);
    assert_eq!(
        finalize(SEAM_ILP_FINALIZE, true),
        handle("glm_hc_decode_finalize_ilp_bf16")
    );
    assert_eq!(finalize(SEAM_ILP_PARTIAL, true), None);
    assert_eq!(finalize(3, false), None);
    // A target without the twins keeps its kernels.
    let old = Capture::new(&GLM[..7]);
    assert!(seam_partial_twin(3, &old, true, true, fn16).is_none());
    assert!(seam_finalize_twin(3, &old, true).is_none());
    assert!(gpu.launches().is_empty());
}

#[test]
fn flag_off_seam_keeps_its_kernels_and_fp32_weight() {
    let gpu = Capture::new(&[
        ("glm_hc_prefill_vec", "glm_hc_decode_partial"),
        ("glm_hc_prefill_vec", "glm_hc_decode_post_partial"),
        ("glm_hc_prefill_vec", "glm_hc_decode_finalize"),
        ("glm_hc_prefill_vec", "glm_hc_decode_post_partial_ilp_bf16"),
        ("glm_hc_prefill_vec", "glm_hc_decode_finalize_ilp_bf16"),
    ]);
    let (fn32, fn16) = (DevicePtr(0x7000), DevicePtr(0x9000));
    for (post, k) in [(false, 1), (true, 2)] {
        let (kernel, weight) = hc_decode_partial(&gpu, "glm5_next", post, fn32, fn16).unwrap();
        assert_eq!((kernel.0, weight), (k, fn32));
    }
    assert_eq!(hc_decode_finalize(&gpu, "glm5_next").unwrap().0, 3);
}
