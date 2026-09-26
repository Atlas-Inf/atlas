// SPDX-License-Identifier: AGPL-3.0-only

//! Byte-identity gate for `gated_delta_rule_wy8`'s `state_is_table`
//! addressing — the K-templated wyN kernels' pointer-table form that lets the
//! cross-sequence batched conv+WY verify arm serve DFlash2's k=8 rows.
//!
//! Same pinning intent as the wy4 batched microtest
//! (`examples/gdn_wy4_batched_microtest.rs`): the contiguous form strides the
//! intermediates by `(b*nv+vh)*hv`, which is the POOL slot stride only for
//! h_state — at batch_size>1 sequence 1's Hi_0 would land on sequence 0's
//! Hi_1. The table form reads each sequence's base from a device pointer
//! table and keeps `inter_stride_floats` between Hi_t within the slot.
//!
//! GPU test: `#[ignore]` per repo convention. Run with
//! ```text
//! cargo test -p spark-model --release --features cuda \
//!   gdn_wy8_table_form -- --ignored --nocapture
//! ```

use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
/// One test case verifies K tokens; intermediates per slot = K-1.
const CONV_DIM: usize = NK * KD * 2 + NV * VD;
const GB_STRIDE: usize = NV * 2;
/// Floats in one sequence's h_state (== the pool's per-slot h stride).
const HV: usize = NV * KD * VD;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> DevicePtr {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1)).unwrap();
    g.copy_h2d(&b, p).unwrap();
    p
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> DevicePtr {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1)).unwrap();
    g.copy_h2d(&b, p).unwrap();
    p
}

fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Device array of pointers, one per sequence — what `state_is_table=1` reads.
fn up_ptr_table(g: &dyn GpuBackend, ptrs: &[DevicePtr]) -> DevicePtr {
    let b: Vec<u8> = ptrs.iter().flat_map(|p| p.0.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1)).unwrap();
    g.copy_h2d(&b, p).unwrap();
    p
}

fn check_wyn_table_form(n: usize, k: usize) {
    let ni = k - 1; // h intermediates per slot
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-27b", "nvfp4")
        .expect("qwen3.8-27b/nvfp4 not in this build");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let fn_name = format!("gated_delta_rule_wy{k}");
    let kernel = g.kernel("gated_delta_rule_wyn", &fn_name).unwrap();
    assert!(kernel.0 != 0, "{fn_name} resolved to handle 0");

    let mut rng = Lcg(0x5eed_8c3a ^ (k as u64) << 32);
    let rows = n * k;
    // Token rows are seq-major [seq0_t0..t{K-1}, seq1_t0.., ...] — the
    // kernel's (b*K+T) indexing.
    let qkv: Vec<bf16> = (0..rows * CONV_DIM)
        .map(|_| bf16::from_f32(rng.r(-1.0, 1.0)))
        .collect();
    let gate: Vec<f32> = (0..rows * GB_STRIDE).map(|_| rng.r(0.90, 0.999)).collect();
    let beta: Vec<f32> = (0..rows * GB_STRIDE).map(|_| rng.r(0.1, 0.9)).collect();
    let h_init: Vec<f32> = (0..n * HV).map(|_| rng.r(-0.5, 0.5)).collect();
    // Sentinel per (slot, token) so a cross-sequence intermediate write is
    // visible even where the kernel would not write.
    let hi_init: Vec<f32> = (0..n * ni * HV)
        .map(|i| -1000.0 - ((i / HV) as f32))
        .collect();

    let d_q = up_bf16(g, &qkv);
    let d_gate = up_f32(g, &gate);
    let d_beta = up_f32(g, &beta);

    let launch = |h: DevicePtr,
                  tok_base: DevicePtr,
                  gate: DevicePtr,
                  beta: DevicePtr,
                  out: DevicePtr,
                  hi_base: DevicePtr,
                  batch: u32,
                  is_table: bool| {
        ops::gdn_decode_wyn(
            g,
            kernel,
            h,
            tok_base,
            tok_base,
            tok_base,
            gate,
            beta,
            out,
            hi_base,
            HV as u32, // inter_stride_floats — intra-slot Hi_t stride
            batch,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            CONV_DIM as u32,
            CONV_DIM as u32,
            GB_STRIDE as u32,
            is_table,
            0,
        )
        .unwrap();
        g.synchronize(0).unwrap();
    };

    // Reference: n sequential single-sequence launches, contiguous addressing.
    let ref_h = up_f32(g, &h_init);
    let ref_hi = up_f32(g, &hi_init);
    let ref_out = g.alloc(rows * NV * VD * 2).unwrap();
    for i in 0..n {
        let i = i as u64;
        launch(
            DevicePtr(ref_h.0 + (i as usize * HV * 4) as u64),
            DevicePtr(d_q.0 + i * (k * CONV_DIM * 2) as u64),
            DevicePtr(d_gate.0 + i * (k * GB_STRIDE * 4) as u64),
            DevicePtr(d_beta.0 + i * (k * GB_STRIDE * 4) as u64),
            DevicePtr(ref_out.0 + i * (k * NV * VD * 2) as u64),
            // pool layout: intermediate (slot, t) at (slot*ni + t) * HV
            DevicePtr(ref_hi.0 + i * (ni * HV * 4) as u64),
            1,
            false,
        );
    }

    // Test: ONE launch, batch_size=n, h table + Hi_0-base table.
    let tst_h = up_f32(g, &h_init);
    let tst_hi = up_f32(g, &hi_init);
    let tst_out = g.alloc(rows * NV * VD * 2).unwrap();
    let h_tbl = up_ptr_table(
        g,
        &(0..n)
            .map(|i| DevicePtr(tst_h.0 + (i * HV * 4) as u64))
            .collect::<Vec<_>>(),
    );
    let hi0_tbl = up_ptr_table(
        g,
        &(0..n)
            .map(|i| DevicePtr(tst_hi.0 + (i * ni * HV * 4) as u64))
            .collect::<Vec<_>>(),
    );
    launch(h_tbl, d_q, d_gate, d_beta, tst_out, hi0_tbl, n as u32, true);

    // h_state, all K-1 intermediates, and output must be BYTE-identical.
    let (a, b) = (down_f32(g, ref_h, n * HV), down_f32(g, tst_h, n * HV));
    if let Some(i) = a
        .iter()
        .zip(&b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        panic!(
            "h_state differs at float {i} (seq {}): ref {} vs batched {}",
            i / HV,
            a[i],
            b[i]
        );
    }
    let (a, b) = (
        down_f32(g, ref_hi, n * ni * HV),
        down_f32(g, tst_hi, n * ni * HV),
    );
    if let Some(i) = a
        .iter()
        .zip(&b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        panic!(
            "INTERMEDIATE differs at float {i} (seq {}, token {}): ref {} vs batched {} \
             — this is the cross-sequence rollback corruption the pointer table fixes",
            i / (ni * HV),
            (i / HV) % ni,
            a[i],
            b[i]
        );
    }
    let mut ob = vec![0u8; rows * NV * VD * 2];
    let mut tb = vec![0u8; rows * NV * VD * 2];
    g.copy_d2h(ref_out, &mut ob).unwrap();
    g.copy_d2h(tst_out, &mut tb).unwrap();
    assert_eq!(ob, tb, "output bytes differ");
    eprintln!("n={n} k={k}: h_state + {ni} intermediates + output all BYTE-IDENTICAL");
}

/// wy8 table form — the DFlash2 γ=8 width.
#[test]
#[ignore]
fn gdn_wy8_table_form_byte_identical() {
    check_wyn_table_form(3, 8);
}

/// wy12 table form — the adaptive-γ γ=12 width (job 260's code/JSON winner).
#[test]
#[ignore]
fn gdn_wy12_table_form_byte_identical() {
    check_wyn_table_form(3, 12);
}
