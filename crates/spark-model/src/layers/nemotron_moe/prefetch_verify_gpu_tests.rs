// SPDX-License-Identifier: AGPL-3.0-only
//! Byte-identity GPU tests for the omni perf kernels:
//!  - `moe_expert_gemv_pf` (software-prefetch) vs `moe_expert_gemv`
//!  - smem-split relu²+down (routed-only @ inter*4 + shared slot kernel
//!    @ shared_inter*4) vs the fused top_k+1 launch @ shared_inter*4.
//!
//! Same math order per element — outputs must be bit-identical.
//! Run: `cargo test --release -p spark-model prefetch_verify -- --test-threads 1`

#![cfg(all(test, target_os = "linux"))]

mod tests {
    use super::super::union_verify_gpu_tests as uv;
    use spark_runtime::cuda_backend::AtlasCudaBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

    fn gpu() -> AtlasCudaBackend {
        AtlasCudaBackend::new(
            0,
            &atlas_kernels::ptx_for_exact_target("nemotron-3.5-super-vl-120b-a12b", "nvfp4")
                .expect("omni target PTX")
                .modules,
        )
        .expect("CUDA backend")
    }
    fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
        let p = g.alloc(bytes.len().max(1)).unwrap();
        g.copy_h2d(bytes, p).unwrap();
        p
    }
    fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        g.copy_d2h(p, &mut v).unwrap();
        v
    }
    /// moe_expert_gemv_pf must reproduce moe_expert_gemv bitwise at omni
    /// routed-UP dims (N=inter, K=h, top_k slots over a small expert bank).
    #[test]
    #[ignore = "needs a GPU and a real kernel build"]
    fn gemv_prefetch_identical_to_serial() {
        let g = gpu();
        let pf = g
            .kernel("moe_expert_gemv_pf", "moe_expert_gemv_pf")
            .expect("moe_expert_gemv_pf resolves on omni");
        let serial = g.kernel("moe_expert_gemv", "moe_expert_gemv").unwrap();
        let (n, k, topk, num_experts) = (2688u32, 4096u32, 22u32, 8usize);
        let a = upload(&g, &uv::u16_bytes(&uv::bf16_seq(k as usize, 3, 2.0)));
        let (pk, sc, s2) = uv::make_experts(&g, num_experts, n as usize, k as usize);
        let idx_bytes: Vec<u8> = (0..topk)
            .map(|e| e % num_experts as u32)
            .flat_map(|e| e.to_le_bytes())
            .collect();
        let idx = upload(&g, &idx_bytes);
        let out_a = g.alloc(topk as usize * n as usize * 2).unwrap();
        let out_b = g.alloc(topk as usize * n as usize * 2).unwrap();
        let stream = 0u64;
        for (kk, out) in [(serial, out_a), (pf, out_b)] {
            crate::layers::ops::moe_expert_gemv(
                &g, kk, a, pk, sc, s2, out, idx, n, k, topk, 0, 1, stream,
            )
            .unwrap();
        }
        g.synchronize(stream).unwrap();
        assert_eq!(
            download(&g, out_a, topk as usize * n as usize * 2),
            download(&g, out_b, topk as usize * n as usize * 2),
            "moe_expert_gemv_pf diverges from moe_expert_gemv"
        );
    }

    /// Routed-only fused launch (smem=inter*4) + moe_relu2_down_shared_slot
    /// (smem=shared_inter*4) must byte-match the fused top_k+1 launch.
    #[test]
    #[ignore = "needs a GPU and a real kernel build"]
    fn relu2_down_smem_split_identical() {
        let g = gpu();
        let fused = g
            .kernel("moe_relu2_fused", "moe_expert_relu2_down_shared")
            .unwrap();
        let slot = g
            .kernel("moe_relu2_down_shared_slot", "moe_relu2_down_shared_slot")
            .unwrap();
        let (h, inter, shared_inter, topk, num_experts) =
            (4096u32, 2688u32, 5376u32, 22u32, 8usize);
        let up_out = upload(
            &g,
            &uv::u16_bytes(&uv::bf16_seq(topk as usize * inter as usize, 11, 1.0)),
        );
        let sh_up = upload(
            &g,
            &uv::u16_bytes(&uv::bf16_seq(shared_inter as usize, 13, 1.0)),
        );
        let (pk, sc, s2) = uv::make_experts(&g, num_experts, h as usize, inter as usize);
        let sh_w = upload(
            &g,
            &uv::byte_seq(h as usize * (shared_inter as usize / 2), 21),
        );
        let sh_s = upload(
            &g,
            &uv::byte_seq(h as usize * (shared_inter as usize / 16), 22),
        );
        let idx_bytes: Vec<u8> = (0..topk)
            .map(|e| e % num_experts as u32)
            .flat_map(|e| e.to_le_bytes())
            .collect();
        let idx = upload(&g, &idx_bytes);
        let dn_a = g.alloc(topk as usize * h as usize * 2).unwrap();
        let dn_b = g.alloc(topk as usize * h as usize * 2).unwrap();
        let sh_a = g.alloc(h as usize * 2).unwrap();
        let sh_b = g.alloc(h as usize * 2).unwrap();
        let stream = 0u64;

        // reference: fused, all slots, full smem
        KernelLaunch::new(&g, fused)
            .grid([div_ceil(h, 8), topk + 1, 1])
            .block([128, 1, 1])
            .shared_mem(shared_inter * 4)
            .arg_ptr(up_out)
            .arg_ptr(pk)
            .arg_ptr(sc)
            .arg_ptr(s2)
            .arg_ptr(dn_a)
            .arg_ptr(idx)
            .arg_ptr(sh_up)
            .arg_ptr(sh_w)
            .arg_ptr(sh_s)
            .arg_f32(1.0)
            .arg_ptr(sh_a)
            .arg_u32(h)
            .arg_u32(inter)
            .arg_u32(shared_inter)
            .arg_u32(h)
            .arg_u32(topk)
            .launch(stream)
            .unwrap();
        // split: routed only at inter*4, then shared slot at shared_inter*4
        KernelLaunch::new(&g, fused)
            .grid([div_ceil(h, 8), topk, 1])
            .block([128, 1, 1])
            .shared_mem(inter * 4)
            .arg_ptr(up_out)
            .arg_ptr(pk)
            .arg_ptr(sc)
            .arg_ptr(s2)
            .arg_ptr(dn_b)
            .arg_ptr(idx)
            .arg_ptr(sh_up)
            .arg_ptr(sh_w)
            .arg_ptr(sh_s)
            .arg_f32(1.0)
            .arg_ptr(sh_b)
            .arg_u32(h)
            .arg_u32(inter)
            .arg_u32(shared_inter)
            .arg_u32(h)
            .arg_u32(topk)
            .launch(stream)
            .unwrap();
        KernelLaunch::new(&g, slot)
            .grid([div_ceil(h, 8), 1, 1])
            .block([128, 1, 1])
            .shared_mem(shared_inter * 4)
            .arg_ptr(sh_up)
            .arg_ptr(sh_w)
            .arg_ptr(sh_s)
            .arg_f32(1.0)
            .arg_ptr(sh_b)
            .arg_u32(h)
            .arg_u32(shared_inter)
            .launch(stream)
            .unwrap();
        g.synchronize(stream).unwrap();
        assert_eq!(
            download(&g, dn_a, topk as usize * h as usize * 2),
            download(&g, dn_b, topk as usize * h as usize * 2),
            "routed down outputs diverge under smem split"
        );
        assert_eq!(
            download(&g, sh_a, h as usize * 2),
            download(&g, sh_b, h as usize * 2),
            "shared down output diverges under smem split"
        );
    }

    /// Grouped GEMM at omni dims: worst-case grid vs exact-tiles grid must
    /// produce byte-identical C with imbalanced routing (one hot expert,
    /// several empty).
    #[test]
    #[ignore = "needs a GPU and a real kernel build"]
    fn grouped_gemm_exact_grid_identical() {
        let g = gpu();
        let relu2 = g
            .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_relu2")
            .expect("grouped relu2 kernel");
        let n_exp = 8usize;
        let (k, inter) = (4096u32, 2688u32); // latent-space omni-ish
        let top_k = 22u32;
        let num_tokens = 4096usize;
        let total_expanded = num_tokens * top_k as usize;
        // routing: expert 3 gets ~half, experts 1/5 empty
        let mut counts = vec![0u32; n_exp];
        for i in 0..total_expanded {
            let e = if i % 2 == 0 { 3 } else { (i * 7) % n_exp };
            counts[e] += 1;
        }
        counts[1] = 0;
        counts[5] = 0;
        // sorted ids + expert_offsets (prefix sums)
        // sorted_token_ids hold the TOKEN index (A row) of each expanded row.
        let mut sorted = Vec::with_capacity(total_expanded);
        let mut offs = vec![0u32; 1];
        for e in 0..n_exp {
            for i in 0..total_expanded {
                let e2 = if i % 2 == 0 { 3 } else { (i * 7) % n_exp };
                if e2 == e {
                    sorted.push((i / top_k as usize) as u32);
                }
            }
            offs.push(*offs.last().unwrap() + counts[e]);
        }
        let offs_bytes: Vec<u8> = offs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sorted_bytes: Vec<u8> = sorted.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sorted_dev = upload(&g, &sorted_bytes);
        let offs_dev = upload(&g, &offs_bytes);

        let a = upload(
            &g,
            &uv::u16_bytes(&uv::bf16_seq(num_tokens * k as usize, 5, 1.0)),
        );
        let (pk, sc, s2) = uv::make_experts(&g, n_exp, inter as usize, k as usize);
        let out_a = g.alloc(total_expanded * inter as usize * 2).unwrap();
        let out_b = g.alloc(total_expanded * inter as usize * 2).unwrap();
        g.memset_async(out_a, 0, total_expanded * inter as usize * 2, 0)
            .unwrap();
        g.memset_async(out_b, 0, total_expanded * inter as usize * 2, 0)
            .unwrap();
        let stream = 0u64;
        let worst = (total_expanded as u32).div_ceil(64).max(1);
        let exact = crate::layers::moe::exact_m_tiles(&offs_bytes, worst);
        for (kk, tiles, out) in [(relu2, worst, out_a), (relu2, exact, out_b)] {
            crate::layers::ops::moe_w4a16_grouped_gemm_ptrtable_64(
                &g,
                kk,
                a,
                pk,
                sc,
                s2,
                out,
                offs_dev,
                sorted_dev,
                n_exp as u32,
                inter,
                k,
                tiles,
                stream,
            )
            .unwrap();
        }
        g.synchronize(stream).unwrap();
        assert!(exact < worst, "exact={exact} worst={worst}");
        assert_eq!(
            download(&g, out_a, total_expanded * inter as usize * 2),
            download(&g, out_b, total_expanded * inter as usize * 2),
            "exact-tiles grid diverges from worst-case grid (tiles {exact} vs {worst})"
        );
    }

    /// The 128-tile ptrtable entry must give the same C whether launched on
    /// the 64-wide grid (_64 wrapper, half its CTAs past N) or the exact
    /// n128 grid — the out-of-range CTAs contribute nothing.
    #[test]
    #[ignore = "needs a GPU and a real kernel build"]
    fn grouped_gemm_n128_grid_64_vs_128_identical() {
        let g = gpu();
        let relu2 = g
            .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_relu2")
            .expect("relu2 entry");
        let plain = g
            .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")
            .expect("plain entry");
        let n_exp = 8usize;
        let (k, inter) = (4096u32, 2688u32);
        let top_k = 22u32;
        let num_tokens = 4096usize;
        let total_expanded = num_tokens * top_k as usize;
        let mut counts = vec![0u32; n_exp];
        for i in 0..total_expanded {
            let e = if i % 2 == 0 { 3 } else { (i * 7) % n_exp };
            counts[e] += 1;
        }
        counts[1] = 0;
        counts[5] = 0;
        let mut sorted = Vec::with_capacity(total_expanded);
        let mut offs = vec![0u32; 1];
        for e in 0..n_exp {
            for i in 0..total_expanded {
                let e2 = if i % 2 == 0 { 3 } else { (i * 7) % n_exp };
                if e2 == e {
                    sorted.push((i / top_k as usize) as u32);
                }
            }
            offs.push(*offs.last().unwrap() + counts[e]);
        }
        let offs_bytes: Vec<u8> = offs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sorted_bytes: Vec<u8> = sorted.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sorted_dev = upload(&g, &sorted_bytes);
        let offs_dev = upload(&g, &offs_bytes);
        let a = upload(
            &g,
            &uv::u16_bytes(&uv::bf16_seq(num_tokens * k as usize, 5, 1.0)),
        );
        let (pk, sc, s2) = uv::make_experts(&g, n_exp, inter as usize, k as usize);
        let bytes_c = total_expanded * inter as usize * 2;
        let out_a = g.alloc(bytes_c).unwrap();
        let out_b = g.alloc(bytes_c).unwrap();
        let stream = 0u64;
        let tiles = (total_expanded as u32).div_ceil(64).max(1);
        for kk in [relu2, plain] {
            g.memset_async(out_a, 0, bytes_c, stream).unwrap();
            g.memset_async(out_b, 0, bytes_c, stream).unwrap();
            crate::layers::ops::moe_w4a16_grouped_gemm_ptrtable_64(
                &g,
                kk,
                a,
                pk,
                sc,
                s2,
                out_a,
                offs_dev,
                sorted_dev,
                n_exp as u32,
                inter,
                k,
                tiles,
                stream,
            )
            .unwrap();
            crate::layers::ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                &g,
                kk,
                a,
                pk,
                sc,
                s2,
                out_b,
                offs_dev,
                sorted_dev,
                n_exp as u32,
                inter,
                k,
                tiles,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            assert_eq!(
                download(&g, out_a, bytes_c),
                download(&g, out_b, bytes_c),
                "64-grid vs n128-grid C differs for kernel {:?}",
                kk
            );
        }
    }

    /// Omni overlay reg kernel vs persistent reference at omni dims: same
    /// inputs -> byte-identical y AND final H.
    #[test]
    #[ignore = "needs a GPU and a real kernel build"]
    fn ssm_prefill_reg_identical_to_persistent() {
        let g = gpu();
        let pers = g
            .kernel("mamba2_ssm", "mamba2_ssm_prefill_persistent")
            .expect("persistent kernel");
        let reg = g
            .kernel("mamba2_ssm_prefill_reg", "mamba2_ssm_prefill_reg")
            .expect("reg kernel");
        for seq in [1usize, 17, 517, 2048] {
            let (h, hd, st, ng) = (128u32, 64u32, 128u32, 8u32);
            let b = 1u32;
            let (xs, bcs, dts, ys) = (h * hd, ng * st, h, h * hd);
            let n_x = seq as u64 * xs as u64;
            let n_bc = seq as u64 * bcs as u64;
            let n_dt = seq as u64 * dts as u64;
            let n_h = (b * h * hd * st) as usize;
            let x = upload(&g, &uv::u16_bytes(&uv::bf16_seq(n_x as usize, 7, 0.5)));
            let bb = upload(&g, &uv::u16_bytes(&uv::bf16_seq(n_bc as usize, 3, 1.0)));
            let cc = upload(&g, &uv::u16_bytes(&uv::bf16_seq(n_bc as usize, 11, 1.0)));
            let dt = upload(&g, &uv::u16_bytes(&uv::bf16_seq(n_dt as usize, 13, 1.0)));
            let al = upload(&g, &uv::f32_bytes(&uv::f32_seq(h as usize, 1, 1.0)));
            let dp = upload(&g, &uv::f32_bytes(&uv::f32_seq(h as usize, 5, 0.5)));
            let db = upload(&g, &uv::f32_bytes(&uv::f32_seq(h as usize, 9, 0.05)));
            let h0: Vec<f32> = (0..n_h)
                .map(|i| ((i * 747796405) % 65536) as f32 / 65536.0 * 0.01)
                .collect();
            let hs_a = upload(&g, &uv::f32_bytes(&h0));
            let hs_b = upload(&g, &uv::f32_bytes(&h0));
            let ya = g.alloc((n_x * 2) as usize).unwrap();
            let yb = g.alloc((n_x * 2) as usize).unwrap();
            let stream = 0u64;
            let args = (
                &g, x, bb, cc, dt, al, dp, db, seq as u32, b, h, hd, st, ng, 1e-9f32, 1e9f32, xs,
                bcs, dts, ys, stream,
            );
            crate::layers::ops::mamba2_ssm_prefill_persistent(
                args.0, pers, hs_a, args.1, args.2, args.3, args.4, args.5, args.6, args.7, ya,
                args.9, args.8, args.10, args.11, args.12, args.13, args.14, args.15, args.16,
                args.17, args.18, args.19, stream,
            )
            .unwrap();
            crate::layers::ops::mamba2_ssm_prefill_reg(
                args.0, reg, hs_b, args.1, args.2, args.3, args.4, args.5, args.6, args.7, yb,
                args.9, args.8, args.10, args.11, args.12, args.13, args.14, args.15, args.16,
                args.17, args.18, args.19, stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            assert_eq!(
                download(&g, ya, (n_x * 2) as usize),
                download(&g, yb, (n_x * 2) as usize),
                "y differs at seq={seq}"
            );
            assert_eq!(
                download(&g, hs_a, n_h * 4),
                download(&g, hs_b, n_h * 4),
                "final H differs at seq={seq}"
            );
        }
    }

    /// E1 host-only grid fix: routed slots emit latent-sized outputs, so the
    /// old max_n grid ran 384 dead x-blocks/slot. Same kernel both launches —
    /// C must be bitwise identical.
    #[test]
    #[ignore = "needs a GPU and a real kernel build"]
    fn relu2_down_routed_grid_128_vs_512_identical() {
        let g = gpu();
        let k = g
            .kernel("moe_relu2_fused", "moe_expert_relu2_down_shared")
            .expect("relu2_down kernel");
        let (topk, h, inter, latent, shared_inter) = (22u32, 4096u32, 2688u32, 1024u32, 5376u32);
        let num_experts = 512usize;
        let up = upload(
            &g,
            &uv::u16_bytes(&uv::bf16_seq(topk as usize * inter as usize, 5, 1.0)),
        );
        let shared_up = upload(
            &g,
            &uv::u16_bytes(&uv::bf16_seq(shared_inter as usize, 9, 1.0)),
        );
        let (pk, sc, s2) = uv::make_experts(&g, num_experts, latent as usize, inter as usize);
        let mut idb = Vec::new();
        for i in 0..topk as usize {
            idb.extend_from_slice(&((i * 23) % num_experts).to_le_bytes());
        }
        let ids = upload(&g, &idb);
        let sh_packed = upload(
            &g,
            &uv::byte_seq(h as usize * (shared_inter as usize / 2), 31),
        );
        let sh_scale = upload(
            &g,
            &uv::byte_seq(h as usize * (shared_inter as usize / 16), 37),
        );
        let n_out = topk as usize * latent as usize;
        let ca = g.alloc(n_out * 2).unwrap();
        let cb = g.alloc(n_out * 2).unwrap();
        let sh_out = g.alloc(h as usize * 2).unwrap();
        for (c, grid_x) in [(ca, div_ceil(h, 8)), (cb, div_ceil(latent, 8))] {
            KernelLaunch::new(&g, k)
                .grid([grid_x, topk, 1])
                .block([128, 1, 1])
                .shared_mem(inter * 4)
                .arg_ptr(up)
                .arg_ptr(pk)
                .arg_ptr(sc)
                .arg_ptr(s2)
                .arg_ptr(c)
                .arg_ptr(ids)
                .arg_ptr(shared_up)
                .arg_ptr(sh_packed)
                .arg_ptr(sh_scale)
                .arg_f32(0.75)
                .arg_ptr(sh_out)
                .arg_u32(latent)
                .arg_u32(inter)
                .arg_u32(shared_inter)
                .arg_u32(h)
                .arg_u32(topk)
                .launch(0)
                .unwrap();
            g.synchronize(0).unwrap();
        }
        assert_eq!(download(&g, ca, n_out * 2), download(&g, cb, n_out * 2));
    }
}
