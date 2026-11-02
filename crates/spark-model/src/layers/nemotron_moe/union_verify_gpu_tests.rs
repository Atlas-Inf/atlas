// SPDX-License-Identifier: AGPL-3.0-only

//! K2 union-grouped expert verify — GPU bit-identity tests (Linux/CUDA).
//!
//! The γ=8 verify MoE runs K sequential `decode_inner` calls; each row's
//! routed experts re-read the same weights when rows share experts (the
//! union over 8×22 routes is ~29% of the 512-expert bank). The grouped
//! kernel `moe_expert_gemv_wide_grouped` loads each matched expert once
//! and iterates the matching rows — these tests prove byte-identity vs
//! the serial per-row kernels at omni dims:
//!   up   : moe_expert_gemv (per row) vs moe_expert_gemv_wide_grouped
//!   down : moe_expert_relu2_down_shared (per row) vs
//!          moe_expert_gemv_wide_grouped(relu2_input=1)
//!
//! Run: `cargo test --release -p spark-model union_verify -- --test-threads 1`

#![cfg(all(test, target_os = "linux"))]

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::ops;

pub(super) fn gpu() -> AtlasCudaBackend {
    AtlasCudaBackend::new(
        0,
        &atlas_kernels::ptx_for_exact_target("nemotron-3.5-super-vl-120b-a12b", "nvfp4")
            .expect("omni target PTX — build with ATLAS_TARGET_MODEL='*'")
            .modules,
    )
    .expect("CUDA backend")
}

pub(super) fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len().max(1)).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

pub(super) fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v).unwrap();
    v
}

/// Deterministic pseudo-random bytes (LCG on an u8 stream; the kernel only
/// needs varied packed nibbles, not a distribution).
pub(super) fn byte_seq(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

pub(super) fn bf16_seq(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            half::bf16::from_f32(u * scale).to_bits()
        })
        .collect()
}

pub(super) fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(super) fn ptr_bytes(ptrs: &[DevicePtr]) -> Vec<u8> {
    ptrs.iter().flat_map(|p| p.0.to_le_bytes()).collect()
}

pub(super) fn f32_seq(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let v = (state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as f32 / u32::MAX as f32;
        out.push((v * 2.0 - 1.0) * scale);
    }
    out
}

pub(super) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Per-expert NVFP4 weight bank: packed u8 [N, K/2], fp8 scales
/// [N, K/16], scalar scale2. Returns (packed_ptrs, scale_ptrs, scale2).
pub(super) fn make_experts(
    g: &AtlasCudaBackend,
    num_experts: usize,
    n: usize,
    k: usize,
) -> (DevicePtr, DevicePtr, DevicePtr) {
    let mut pk = Vec::new();
    let mut sc = Vec::new();
    let mut s2 = Vec::new();
    for e in 0..num_experts {
        pk.push(upload(g, &byte_seq(n * (k / 2), 100 + e as u64)));
        sc.push(upload(g, &byte_seq(n * (k / 16), 900 + e as u64)));
        s2.extend_from_slice(&1.0f32.to_le_bytes());
    }
    (
        upload(g, &ptr_bytes(&pk)),
        upload(g, &ptr_bytes(&sc)),
        upload(g, &s2),
    )
}

struct LayerKernels {
    gemv: spark_runtime::gpu::KernelHandle,
    fused_down: spark_runtime::gpu::KernelHandle,
    grouped: spark_runtime::gpu::KernelHandle,
}

fn kernels(g: &AtlasCudaBackend) -> LayerKernels {
    LayerKernels {
        gemv: g.kernel("moe_expert_gemv", "moe_expert_gemv").unwrap(),
        fused_down: g
            .kernel("moe_relu2_fused", "moe_expert_relu2_down_shared")
            .unwrap(),
        grouped: g
            .kernel(
                "moe_expert_gemv_wide_grouped",
                "moe_expert_gemv_wide_grouped",
            )
            .expect("moe_expert_gemv_wide_grouped must resolve"),
    }
}

/// Serial per-row up + fused relu²+down — the same calls `decode_inner`
/// makes (moe_expert_gemv then moe_expert_relu2_down_shared with the
/// shared slot beyond top_k doing no useful work in this test).
#[allow(clippy::too_many_arguments)]
fn serial_row(
    g: &AtlasCudaBackend,
    k: &LayerKernels,
    latent_in: DevicePtr,
    up_ptrs: (DevicePtr, DevicePtr, DevicePtr),
    down_ptrs: (DevicePtr, DevicePtr, DevicePtr),
    up_out: DevicePtr,
    down_out: DevicePtr,
    indices: DevicePtr,
    shared_up: DevicePtr,
    shared_w: DevicePtr,
    shared_scale: DevicePtr,
    shared_out: DevicePtr,
    topk: u32,
    inter: u32,
    latent: u32,
    shared_inter: u32,
    h: u32,
    stream: u64,
) {
    ops::moe_expert_gemv(
        g, k.gemv, latent_in, up_ptrs.0, up_ptrs.1, up_ptrs.2, up_out, indices, inter, latent,
        topk, 0, 1, stream,
    )
    .unwrap();
    KernelLaunch::new(g, k.fused_down)
        .grid([div_ceil(latent, 8), topk + 1, 1])
        .block([128, 1, 1])
        .shared_mem(shared_inter.max(inter) * 4)
        .arg_ptr(up_out)
        .arg_ptr(down_ptrs.0)
        .arg_ptr(down_ptrs.1)
        .arg_ptr(down_ptrs.2)
        .arg_ptr(down_out)
        .arg_ptr(indices)
        .arg_ptr(shared_up) // shared slot input — row t slice
        .arg_ptr(shared_w) // shared_down.weight
        .arg_ptr(shared_scale)
        .arg_f32(1.0)
        .arg_ptr(shared_out)
        .arg_u32(latent) // N (routed out)
        .arg_u32(inter) // K_routed
        .arg_u32(shared_inter) // K_shared
        .arg_u32(h) // N_shared
        .arg_u32(topk)
        .launch(stream)
        .unwrap();
}

/// One union-verify case at omni dims: `m` rows × `topk` routes over
/// `num_experts` NVFP4 experts. Asserts grouped == serial for the full
/// routed up+down chain.
fn run_union_case(m: usize, num_experts: usize, topk: usize) {
    let g = gpu();
    let kk = kernels(&g);
    let stream = 0u64;
    let inter = 2688u32;
    let latent = 1024u32;
    let shared_inter = 5376u32;
    let h = 4096u32;

    let up_ptrs = make_experts(&g, num_experts, inter as usize, latent as usize);
    let down_ptrs = make_experts(&g, num_experts, latent as usize, inter as usize);

    // Deterministic routing with heavy sharing: slot s of row t picks
    // expert (t*3+s)%num_experts — plenty of overlap across rows.
    let mut idx = Vec::new();
    for t in 0..m {
        for s in 0..topk {
            idx.extend_from_slice(&(((t * 3 + s) % num_experts) as u32).to_le_bytes());
        }
    }
    let indices = upload(&g, &idx);
    let latent_rows = upload(&g, &u16_bytes(&bf16_seq(m * latent as usize, 5, 0.5)));
    let sh_w = upload(&g, &byte_seq(h as usize * shared_inter as usize / 2, 41));
    let sh_s = upload(&g, &byte_seq(h as usize * shared_inter as usize / 16, 42));
    let sh_in = upload(&g, &u16_bytes(&bf16_seq(m * shared_inter as usize, 7, 0.5)));
    let sh_out_a = g.alloc(m * h as usize * 2).unwrap();

    // Serial reference: per row.
    let up_a = g.alloc(m * topk * inter as usize * 2).unwrap();
    let down_a = g.alloc(m * topk * latent as usize * 2).unwrap();
    for t in 0..m {
        serial_row(
            &g,
            &kk,
            latent_rows.offset(t * latent as usize * 2),
            up_ptrs,
            down_ptrs,
            up_a.offset(t * topk * inter as usize * 2),
            down_a.offset(t * topk * latent as usize * 2),
            indices.offset(t * topk * 4),
            sh_in.offset(t * shared_inter as usize * 2),
            sh_w,
            sh_s,
            sh_out_a.offset(t * h as usize * 2),
            topk as u32,
            inter,
            latent,
            shared_inter,
            h,
            stream,
        );
    }
    g.synchronize(stream).unwrap();

    // Grouped UP: grid (ceil(N/32), num_experts) — each expert tile applies
    // to every matched row. input_stride=0 → A row t at tok*K.
    let up_b = g.alloc(m * topk * inter as usize * 2).unwrap();
    ops::moe_expert_gemv_wide_grouped(
        &g,
        kk.grouped,
        latent_rows,
        up_ptrs.0,
        up_ptrs.1,
        up_ptrs.2,
        up_b,
        indices,
        inter,
        latent,
        topk as u32,
        0,
        m as u32,
        num_experts as u32,
        false,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let bytes = m * topk * inter as usize * 2;
    assert_eq!(
        download(&g, up_a, bytes),
        download(&g, up_b, bytes),
        "grouped UP diverges from serial moe_expert_gemv (m={m} experts={num_experts})"
    );

    // Grouped DOWN: route-major input [m, topk, inter], relu² fused,
    // weights read once per matched expert.
    let down_b = g.alloc(m * topk * latent as usize * 2).unwrap();
    ops::moe_expert_gemv_wide_grouped(
        &g,
        kk.grouped,
        up_a, // the serial up output — identical input to both paths
        down_ptrs.0,
        down_ptrs.1,
        down_ptrs.2,
        down_b,
        indices,
        latent,
        inter,
        topk as u32,
        inter,
        m as u32,
        num_experts as u32,
        true,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let bytes = m * topk * latent as usize * 2;
    assert_eq!(
        download(&g, down_a, bytes),
        download(&g, down_b, bytes),
        "grouped relu²+DOWN diverges from serial relu2_down_shared (m={m} experts={num_experts})"
    );
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn union_grouped_matches_serial_m1() {
    run_union_case(1, 64, 22);
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn union_grouped_matches_serial_m8() {
    run_union_case(8, 64, 22);
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn union_grouped_matches_serial_m3() {
    run_union_case(3, 64, 22);
}

/// NVFP4 shared-expert twin: `w4a16_gemv_batch16` must reproduce the
/// serial `w4a16_gemv` (the M=1 kernel `w4a16_decode_gemv` picks when
/// gemv_sw is off) per row — the shared expert's up/down at omni dims.
fn run_w4a16_twin_case(m: usize, n: usize, k: usize) {
    run_w4a16_twin_kernel(m, n, k, "w4a16_gemv_batch16");
}

fn run_w4a16_twin_kernel(m: usize, n: usize, k: usize, twin_name: &str) {
    use crate::weight_map::QuantizedWeight;
    let g = gpu();
    let serial = g.kernel("w4a16_gemv", "w4a16_gemv").unwrap();
    let twin = g
        .kernel("w4a16_gemv", twin_name)
        .unwrap_or_else(|_| panic!("{twin_name} must resolve"));
    let stream = 0u64;

    let a = upload(&g, &u16_bytes(&bf16_seq(m * k, 31, 4.0)));
    let weight = QuantizedWeight {
        weight: upload(&g, &byte_seq(n * k / 2, 7)),
        weight_scale: upload(&g, &byte_seq(n * k / 16, 8)),
        weight_scale_2: 1.0,
        input_scale: upload(&g, &4.0f32.to_le_bytes()),
        weight_scale_2_vec: DevicePtr::NULL,
    };

    let c_seq = g.alloc(m * n * 2).unwrap();
    for t in 0..m {
        ops::w4a16_gemv(
            &g,
            serial,
            a.offset(t * k * 2),
            &weight,
            c_seq.offset(t * n * 2),
            n as u32,
            k as u32,
            stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();

    let c_twin = g.alloc(m * n * 2).unwrap();
    ops::w4a16_gemv_batchm(
        &g, twin, a, &weight, c_twin, m as u32, n as u32, k as u32, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    assert_eq!(
        download(&g, c_seq, m * n * 2),
        download(&g, c_twin, m * n * 2),
        "{twin_name} diverges from serial w4a16_gemv at m={m} n={n} k={k}"
    );
}

/// Attention-wide verify twins at omni dims (QKV/O projections, K=4096):
/// the M=4 (DFlash gamma=3) rows ride `w4a16_gemv_batch4`, M=9 (gamma=8)
/// rides `w4a16_gemv_batch16`; both must be byte-identical to the M=1
/// `w4a16_gemv` decode kernel on this target (ATLAS_W4A16_DP4A=0 pins the
/// dispatch off the int8 arm).
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn w4a16_gemv_batch4_identical_to_serial_attn() {
    for m in [1usize, 4] {
        run_w4a16_twin_kernel(m, 6144, 4096, "w4a16_gemv_batch4");
        run_w4a16_twin_kernel(m, 4096, 1024, "w4a16_gemv_batch4");
    }
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn w4a16_gemv_batch16_identical_to_serial_attn_m9() {
    for m in [9usize, 16] {
        run_w4a16_twin_kernel(m, 6144, 4096, "w4a16_gemv_batch16");
        run_w4a16_twin_kernel(m, 4096, 1024, "w4a16_gemv_batch16");
    }
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn w4a16_batch16_identical_to_serial() {
    // shared-up 5376x4096, shared-down 4096x5376
    run_w4a16_twin_case(1, 5376, 4096);
    run_w4a16_twin_case(2, 4096, 5376);
    run_w4a16_twin_case(8, 5376, 4096);
    run_w4a16_twin_case(8, 4096, 5376);
}

/// Same twin at the omni LM-head shape (131072 x 4096 NVFP4, K-row verify
/// rows M in {1,9,16}) — the chunked batch16 LM-head path relies on this.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn w4a16_batch16_identical_to_serial_lmhead() {
    for m in [1usize, 9, 16] {
        run_w4a16_twin_case(m, 131072, 4096);
    }
}

/// The K2 shared-slot path: the `moe_relu2_down_shared_slot` twin kernel
/// (grid=[ceil(h/8), 1, m]) must equal the per-row fused shared slot
/// byte-for-byte.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn shared_slot_batched_matches_per_row() {
    let m = 8usize;
    let g = gpu();
    let kk = kernels(&g);
    let stream = 0u64;
    let shared_inter = 5376u32;
    let h = 4096u32;
    let topk = 22u32;
    let inter = 2688u32; // smem bound only

    let sh_w = upload(&g, &byte_seq(h as usize * shared_inter as usize / 2, 41));
    let sh_s = upload(&g, &byte_seq(h as usize * shared_inter as usize / 16, 42));
    let sh_in = upload(&g, &u16_bytes(&bf16_seq(m * shared_inter as usize, 7, 0.5)));

    // serial: one fused launch per row; grid.y=topk+1 covers all slots
    // (routed slots read dummy ptrs harmlessly — we only check the shared
    // slot output). This is exactly decode's per-row call shape.
    // Routed slots DO run in this launch (grid.y = topk+1) — they need a
    // valid (single-expert) ptr table so expert_indices=0 resolves real
    // pointers; their output is discarded.
    let latent = 1024u32;
    let rw = upload(&g, &byte_seq(latent as usize * inter as usize / 2, 43));
    let rs = upload(&g, &byte_seq(latent as usize * inter as usize / 16, 44));
    let rt_w = upload(&g, &rw.0.to_le_bytes());
    let rt_s = upload(&g, &rs.0.to_le_bytes());
    let rt_s2 = upload(&g, &1.0f32.to_le_bytes());
    let rup = upload(
        &g,
        &u16_bytes(&bf16_seq(topk as usize * inter as usize, 9, 0.5)),
    );
    let rdummy_c = g.alloc(topk as usize * latent as usize * 2).unwrap();
    let dummy_i = upload(&g, &vec![0u8; topk as usize * 4]);
    let out_a = g.alloc(m * h as usize * 2).unwrap();
    for t in 0..m {
        KernelLaunch::new(&g, kk.fused_down)
            .grid([div_ceil(h, 8), topk + 1, 1])
            .block([128, 1, 1])
            .shared_mem(shared_inter * 4)
            .arg_ptr(rup) // up_out (routed slots)
            .arg_ptr(rt_w) // packed_ptrs
            .arg_ptr(rt_s) // scale_ptrs
            .arg_ptr(rt_s2) // scale2_vals
            .arg_ptr(rdummy_c) // C routed
            .arg_ptr(dummy_i) // expert_indices → all 0
            .arg_ptr(sh_in.offset(t * shared_inter as usize * 2))
            .arg_ptr(sh_w)
            .arg_ptr(sh_s)
            .arg_f32(1.0)
            .arg_ptr(out_a.offset(t * h as usize * 2))
            .arg_u32(latent)
            .arg_u32(inter)
            .arg_u32(shared_inter)
            .arg_u32(h)
            .arg_u32(topk)
            .launch(stream)
            .unwrap();
    }
    g.synchronize(stream).unwrap();

    // batched: the slot twin kernel, one launch, grid.z=m — this is what
    // decode_verify_exact_latent runs for the NVFP4 shared down.
    let slot = g
        .kernel("moe_relu2_down_shared_slot", "moe_relu2_down_shared_slot")
        .expect("moe_relu2_down_shared_slot must resolve");
    let out_b = g.alloc(m * h as usize * 2).unwrap();
    KernelLaunch::new(&g, slot)
        .grid([div_ceil(h, 8), 1, m as u32])
        .block([128, 1, 1])
        .shared_mem(shared_inter * 4)
        .arg_ptr(sh_in)
        .arg_ptr(sh_w)
        .arg_ptr(sh_s)
        .arg_f32(1.0)
        .arg_ptr(out_b)
        .arg_u32(h)
        .arg_u32(shared_inter)
        .launch(stream)
        .unwrap();
    g.synchronize(stream).unwrap();

    assert_eq!(
        download(&g, out_a, m * h as usize * 2),
        download(&g, out_b, m * h as usize * 2),
        "batched shared slot (grid.z=m, slot_base=top_k) diverges"
    );
}

/// The omni union-table GEMV must produce identical bits to the common
/// grouped kernel for both the UP (relu2_input=0) and DOWN
/// (relu2_input=1, input_stride=K) stages, at verify sizes m=4 (df3) and
/// m=9 (df8).
fn run_union_u_case(m: usize) {
    let g = gpu();
    let kk = kernels(&g);
    let match_k = g
        .kernel("moe_union_gemv", "moe_union_match")
        .expect("moe_union_match must resolve on omni");
    let u_k = g
        .kernel("moe_union_gemv", "moe_expert_gemv_wide_grouped_u")
        .expect("moe_expert_gemv_wide_grouped_u must resolve on omni");
    let stream = 0u64;
    let num_experts = 512usize;
    let topk = 22usize;
    let inter = 2688u32;
    let latent = 1024u32;

    let up_ptrs = make_experts(&g, num_experts, inter as usize, latent as usize);
    let down_ptrs = make_experts(&g, num_experts, latent as usize, inter as usize);

    let mut idx = Vec::new();
    for t in 0..m {
        for s in 0..topk {
            idx.extend_from_slice(&(((t * 7 + s * 13) % num_experts) as u32).to_le_bytes());
        }
    }
    let indices = upload(&g, &idx);
    let latent_rows = upload(&g, &u16_bytes(&bf16_seq(m * latent as usize, 5, 0.5)));
    let inter_rows = upload(
        &g,
        &u16_bytes(&bf16_seq(m * topk * inter as usize, 11, 0.5)),
    );

    let u_buf = g.alloc(4 + 4 * (2 * 512 + 2 * 512 * 32)).unwrap();
    let out_ref = g.alloc(m * topk * inter as usize * 2).unwrap();
    let out_u = g.alloc(m * topk * inter as usize * 2).unwrap();

    // UP stage: A row t at tok*K (input_stride=0), C at [t,topk,inter].
    ops::moe_expert_gemv_wide_grouped(
        &g,
        kk.grouped,
        latent_rows,
        up_ptrs.0,
        up_ptrs.1,
        up_ptrs.2,
        out_ref,
        indices,
        inter,
        latent,
        topk as u32,
        0,
        m as u32,
        num_experts as u32,
        false,
        stream,
    )
    .unwrap();
    ops::moe_union_match(&g, match_k, indices, u_buf, m as u32, topk as u32, stream).unwrap();
    ops::moe_expert_gemv_wide_grouped_u(
        &g,
        u_k,
        latent_rows,
        up_ptrs.0,
        up_ptrs.1,
        up_ptrs.2,
        out_u,
        u_buf,
        inter,
        latent,
        topk as u32,
        0,
        m as u32,
        false,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let bytes = m * topk * inter as usize * 2;
    assert_eq!(
        download(&g, out_ref, bytes),
        download(&g, out_u, bytes),
        "union-table UP diverges from grouped (m={m})"
    );

    // DOWN stage: route-major A [t,topk,inter] (input_stride=K), relu²
    // fused, output [t,topk,latent].
    let out_ref2 = g.alloc(m * topk * latent as usize * 2).unwrap();
    let out_u2 = g.alloc(m * topk * latent as usize * 2).unwrap();
    ops::moe_expert_gemv_wide_grouped(
        &g,
        kk.grouped,
        inter_rows,
        down_ptrs.0,
        down_ptrs.1,
        down_ptrs.2,
        out_ref2,
        indices,
        latent,
        inter,
        topk as u32,
        inter,
        m as u32,
        num_experts as u32,
        true,
        stream,
    )
    .unwrap();
    ops::moe_expert_gemv_wide_grouped_u(
        &g,
        u_k,
        inter_rows,
        down_ptrs.0,
        down_ptrs.1,
        down_ptrs.2,
        out_u2,
        u_buf,
        latent,
        inter,
        topk as u32,
        inter,
        m as u32,
        true,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let bytes = m * topk * latent as usize * 2;
    assert_eq!(
        download(&g, out_ref2, bytes),
        download(&g, out_u2, bytes),
        "union-table DOWN diverges from grouped (m={m})"
    );
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn union_table_gemv_matches_grouped_m4() {
    run_union_u_case(4);
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn union_table_gemv_matches_grouped_m9() {
    run_union_u_case(9);
}

/// Reused-scratch prepass race check: `moe_union_match` must produce a
/// fresh table on every call — the previous verify's stale `u_nmatch` must
/// not leak into `u_count`/`u_experts`. Two routings with different
/// counts/expert sets, 200 iterations, GEMV output compared bitwise
/// against a fresh-scratch run each time.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn union_match_reused_scratch_is_fresh() {
    let g = gpu();
    let match_k = g
        .kernel("moe_union_gemv", "moe_union_match")
        .expect("moe_union_match must resolve on omni");
    let u_k = g
        .kernel("moe_union_gemv", "moe_expert_gemv_wide_grouped_u")
        .expect("moe_expert_gemv_wide_grouped_u must resolve on omni");
    let stream = 0u64;
    let num_experts = 512usize;
    let topk = 22usize;
    let m = 9usize;
    let inter = 2688u32;
    let latent = 1024u32;
    let up_ptrs = make_experts(&g, num_experts, inter as usize, latent as usize);
    let latent_rows = upload(&g, &u16_bytes(&bf16_seq(m * latent as usize, 5, 0.5)));

    let u_reuse = g.alloc(4 + 4 * (2 * 512 + 2 * 512 * 32)).unwrap();
    let u_fresh = g.alloc(4 + 4 * (2 * 512 + 2 * 512 * 32)).unwrap();
    let out_a = g.alloc(m * topk * inter as usize * 2).unwrap();
    let out_b = g.alloc(m * topk * inter as usize * 2).unwrap();

    let mut rng: u64 = 0x9E3779B9;
    let rand = |rng: &mut u64| {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    };
    for iter in 0..200u32 {
        // routing A: dense overlap; routing B: disjoint expert set,
        // different unique count. Alternate which runs on the reused
        // scratch second.
        let build = |seed: u64, base: usize| -> Vec<u8> {
            let mut idx = Vec::new();
            let mut r = seed;
            for t in 0..m {
                for s in 0..topk {
                    r ^= r << 13;
                    r ^= r >> 7;
                    idx.extend_from_slice(
                        &(((base + (r as usize % (num_experts / 4)) * 4 + t * 3 + s) % num_experts)
                            as u32)
                            .to_le_bytes(),
                    );
                }
            }
            idx
        };
        let (ia, ib) = (build(rand(&mut rng), 0), build(rand(&mut rng), 1));
        let (first, second) = if iter % 2 == 0 { (ia, ib) } else { (ib, ia) };
        let fa = upload(&g, &first);
        let fb = upload(&g, &second);

        ops::moe_union_match(&g, match_k, fa, u_reuse, m as u32, topk as u32, stream).unwrap();
        ops::moe_union_match(&g, match_k, fb, u_reuse, m as u32, topk as u32, stream).unwrap();
        ops::moe_expert_gemv_wide_grouped_u(
            &g,
            u_k,
            latent_rows,
            up_ptrs.0,
            up_ptrs.1,
            up_ptrs.2,
            out_a,
            u_reuse,
            inter,
            latent,
            topk as u32,
            0,
            m as u32,
            false,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        g.memset(u_fresh, 0, 4 + 4 * (2 * 512 + 2 * 512 * 32))
            .unwrap();
        ops::moe_union_match(&g, match_k, fb, u_fresh, m as u32, topk as u32, stream).unwrap();
        ops::moe_expert_gemv_wide_grouped_u(
            &g,
            u_k,
            latent_rows,
            up_ptrs.0,
            up_ptrs.1,
            up_ptrs.2,
            out_b,
            u_fresh,
            inter,
            latent,
            topk as u32,
            0,
            m as u32,
            false,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        // Live-region compare only: u_nmatch/u_tok/u_slot rows for experts
        // beyond u_count are stale-but-unread by design (the GEMV never
        // touches them) — the table contract covers u_count, u_experts,
        // and each listed expert's match list.
        let read = |g: &AtlasCudaBackend, p: DevicePtr| -> Vec<u32> {
            let b = download(g, p, 4 + 4 * (2 * 512 + 2 * 512 * 32));
            b.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let ra = read(&g, u_reuse);
        let rb = read(&g, u_fresh);
        assert_eq!(ra[0], rb[0], "u_count stale after reuse (iter {iter})");
        let uc = ra[0] as usize;
        for u in 0..uc {
            let (ea, eb) = (ra[1 + u] as usize, rb[1 + u] as usize);
            assert_eq!(ea, eb, "u_experts[{u}] stale after reuse (iter {iter})");
            let (na, nb) = (ra[1 + 512 + ea], rb[1 + 512 + eb]);
            assert_eq!(na, nb, "u_nmatch[{ea}] stale after reuse (iter {iter})");
            for j in 0..na as usize {
                assert_eq!(
                    ra[1 + 2 * 512 + ea * 32 + j],
                    rb[1 + 2 * 512 + eb * 32 + j],
                    "u_tok[{ea}][{j}] stale after reuse (iter {iter})"
                );
                assert_eq!(
                    ra[1 + 2 * 512 + 512 * 32 + ea * 32 + j],
                    rb[1 + 2 * 512 + 512 * 32 + eb * 32 + j],
                    "u_slot[{ea}][{j}] stale after reuse (iter {iter})"
                );
            }
        }
        let bytes = m * topk * inter as usize * 2;
        assert_eq!(
            download(&g, out_a, bytes),
            download(&g, out_b, bytes),
            "GEMV output stale after scratch reuse (iter {iter})"
        );
    }
}

/// Flushed microbench of the E3 pass-pair trio (decode arms).
#[test]
#[ignore = "needs a GPU and a real kernel build"]
#[cfg(atlas_hip)]
fn e3_pair_fused_gemv_bench() {
    let g = gpu();
    let kk = kernels(&g);
    let stream = 0u64;
    let (n, k) = (2688usize, 1024usize); // routed inter, latent
    let num_experts = 512usize;
    let topk = 22u32;
    let (pk, sk, s2) = make_experts(&g, num_experts, n, k);
    let input = upload(&g, &u16_bytes(&bf16_seq(k, 7, 0.5)));
    let idx: Vec<u8> = (0..topk as usize)
        .flat_map(|e| ((e * 23 % num_experts) as u32).to_le_bytes())
        .collect();
    let indices = upload(&g, &idx);
    let out = g.alloc(topk as usize * n * 2).unwrap();
    for _ in 0..5 {
        ops::moe_expert_gemv(
            &g, kk.gemv, input, pk, sk, s2, out, indices, n as u32, k as u32, topk, 0, 1, stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();
    let t0 = std::time::Instant::now();
    let iters = 50;
    for _ in 0..iters {
        ops::moe_expert_gemv(
            &g, kk.gemv, input, pk, sk, s2, out, indices, n as u32, k as u32, topk, 0, 1, stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let bytes = (topk as u64) * (n as u64 * (k as u64 / 2) + n as u64 * (k as u64 / 16));
    eprintln!(
        "moe_expert_gemv E3: {:.1} GB/s ({:.0} us/launch)",
        bytes as f64 * iters as f64 / dt / 1e9,
        dt * 1e6 / iters as f64
    );
}

/// Flushed microbench of the shared-expert down slot kernel (B2 baseline).
#[test]
#[ignore = "needs a GPU and a real kernel build"]
#[cfg(atlas_hip)]
fn shared_down_slot_bench() {
    let g = gpu();
    let slot = g
        .kernel("moe_relu2_down_shared_slot", "moe_relu2_down_shared_slot")
        .unwrap();
    let (h, shared_inter) = (4096u32, 5376u32);
    let sh_up = upload(&g, &u16_bytes(&bf16_seq(shared_inter as usize, 13, 0.5)));
    let sh_s = upload(&g, &byte_seq(h as usize * (shared_inter as usize / 16), 22));
    let sh_out = g.alloc(h as usize * 2).unwrap();
    let stream = 0u64;
    let w_bytes = h as usize * (shared_inter as usize / 2);
    let n_copies = (512 * 1024 * 1024 / w_bytes).max(8);
    let copies: Vec<DevicePtr> = (0..n_copies)
        .map(|_| upload(&g, &byte_seq(w_bytes, 21)))
        .collect();
    let launch = |w: DevicePtr| {
        KernelLaunch::new(&g, slot)
            .grid([div_ceil(h, 8), 1, 1])
            .block([128, 1, 1])
            .shared_mem(shared_inter * 4)
            .arg_ptr(sh_up)
            .arg_ptr(w)
            .arg_ptr(sh_s)
            .arg_f32(1.0)
            .arg_ptr(sh_out)
            .arg_u32(h)
            .arg_u32(shared_inter)
            .launch(stream)
            .unwrap();
    };
    for i in 0..5 {
        launch(copies[i]);
    }
    g.synchronize(stream).unwrap();
    let t0 = std::time::Instant::now();
    let iters = 60;
    for i in 0..iters {
        launch(copies[i % n_copies]);
    }
    g.synchronize(stream).unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let bytes = (w_bytes
        + h as usize * (shared_inter as usize / 16)
        + shared_inter as usize * 2
        + h as usize * 2) as f64;
    eprintln!(
        "shared_down_slot (E3a, N=4096 K=5376): {:.1} GB/s ({:.0} us/launch)",
        bytes * iters as f64 / dt / 1e9,
        dt * 1e6 / iters as f64
    );
}

/// Flushed MoE bench: rotate the 22-expert index set across all 512 experts
/// each iteration (~774 MB bank >> 32 MB MALL). Measures decode moe_expert_gemv
/// (up) and fused relu2_down (routed) at real dims.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
#[cfg(atlas_hip)]
fn moe_gemv_rotated_bench() {
    let g = gpu();
    let kk = kernels(&g);
    let stream = 0u64;
    let (inter, latent) = (2688u32, 1024u32);
    let (h, shared_inter) = (4096u32, 5376u32);
    let num_experts = 512u32;
    let topk = 22u32;
    let m = 8u32;

    // UP experts: n=inter, k=latent — 1.5 MB each, 774 MB bank.
    let up_ptrs = make_experts(&g, num_experts as usize, inter as usize, latent as usize);
    // DOWN experts: n=latent, k=inter.
    let dn_ptrs = make_experts(&g, num_experts as usize, latent as usize, inter as usize);

    // Distinct index sets per iteration: iter i routes to experts
    // {(i*22 + s) % 512} — every launch reads a fresh ~33 MB slice.
    // m*topk indices per set — row t routes to {(i*topk + s + t*17) % 512};
    // every launch touches ~270 MB of fresh expert weights.
    let sets: Vec<DevicePtr> = (0..num_experts / topk)
        .map(|i| {
            let idx: Vec<u8> = (0..m * topk)
                .flat_map(|e| {
                    (((i * topk + e % topk + (e / topk) * 17) % num_experts) as u32).to_le_bytes()
                })
                .collect();
            upload(&g, &idx)
        })
        .collect();

    let input = upload(
        &g,
        &u16_bytes(&bf16_seq(m as usize * latent as usize, 7, 0.5)),
    );
    let up_out = g
        .alloc(m as usize * topk as usize * inter as usize * 2)
        .unwrap();
    let dn_out = g
        .alloc(m as usize * topk as usize * latent as usize * 2)
        .unwrap();
    let sh_up = upload(
        &g,
        &u16_bytes(&bf16_seq(m as usize * shared_inter as usize, 13, 0.5)),
    );
    let sh_w = upload(&g, &byte_seq(h as usize * (shared_inter as usize / 2), 41));
    let sh_s = upload(&g, &byte_seq(h as usize * (shared_inter as usize / 16), 42));
    let sh_out = g.alloc(m as usize * h as usize * 2).unwrap();

    let iters = sets.len();
    // UP (moe_expert_gemv)
    for _ in 0..3 {
        ops::moe_expert_gemv(
            &g, kk.gemv, input, up_ptrs.0, up_ptrs.1, up_ptrs.2, up_out, sets[0], inter, latent,
            topk, 0, m, stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        ops::moe_expert_gemv(
            &g, kk.gemv, input, up_ptrs.0, up_ptrs.1, up_ptrs.2, up_out, sets[i], inter, latent,
            topk, 0, m, stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let up_bytes = m as f64
        * topk as f64
        * (inter as f64 * (latent as f64 / 2.0) + inter as f64 * (latent as f64 / 16.0));
    eprintln!(
        "UP moe_expert_gemv rotated: {:.1} GB/s ({:.0} us/launch)",
        up_bytes * iters as f64 / dt / 1e9,
        dt * 1e6 / iters as f64
    );

    // DOWN (fused relu2_down_shared, routed slots; shared slot = dummy weights)
    let launch_dn = |idx: DevicePtr, tok: u32| {
        KernelLaunch::new(&g, kk.fused_down)
            .grid([div_ceil(latent, 8), topk + 1, 1])
            .block([128, 1, 1])
            .shared_mem(shared_inter.max(inter) * 4)
            .arg_ptr(up_out.offset(tok as usize * topk as usize * inter as usize * 2))
            .arg_ptr(dn_ptrs.0)
            .arg_ptr(dn_ptrs.1)
            .arg_ptr(dn_ptrs.2)
            .arg_ptr(dn_out.offset(tok as usize * topk as usize * latent as usize * 2))
            .arg_ptr(idx.offset(tok as usize * topk as usize * 4))
            .arg_ptr(sh_up.offset(tok as usize * shared_inter as usize * 2))
            .arg_ptr(sh_w)
            .arg_ptr(sh_s)
            .arg_f32(1.0)
            .arg_ptr(sh_out.offset(tok as usize * h as usize * 2))
            .arg_u32(latent)
            .arg_u32(inter)
            .arg_u32(shared_inter)
            .arg_u32(h)
            .arg_u32(topk)
            .launch(stream)
            .unwrap();
    };
    for _ in 0..3 {
        launch_dn(sets[0], 0);
    }
    g.synchronize(stream).unwrap();
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        launch_dn(sets[i], (i % m as usize) as u32);
    }
    g.synchronize(stream).unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let dn_bytes = topk as f64
        * (latent as f64 * (inter as f64 / 2.0) + latent as f64 * (inter as f64 / 16.0))
        + h as f64 * (shared_inter as f64 / 2.0 + shared_inter as f64 / 16.0);
    eprintln!(
        "DOWN relu2_down_shared+slot rotated: {:.1} GB/s ({:.0} us/launch)",
        dn_bytes * iters as f64 / dt / 1e9,
        dt * 1e6 / iters as f64
    );
}
