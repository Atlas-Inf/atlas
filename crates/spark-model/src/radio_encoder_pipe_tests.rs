// SPDX-License-Identifier: AGPL-3.0-only

//! Stage-1 GPU oracle test for `radio_encoder`:
//! patch-embed GEMM + resampled pos-embed + cls/register prefix, against
//! the checkpoint's own embedder output
//! (`n35vl-runs/vision_oracle/{tag}_stages__embedder.npy`).
//! Tolerance: cosine ≥ 0.9999 per row AND max |Δ| ≤ 0.05 (BF16 + host
//! bilinear resample).
#![cfg(all(test, target_os = "linux"))]

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::radio_encoder::*;

const ORACLE: &str = "/home/azeez/n35vl-runs/vision_oracle";

/// GPU tests share one device; parallel `cargo test` threads each build a
/// backend and can interleave launches/uploads, which corrupts oracle
/// comparisons (block0 read a mid-flight buffer, cosine 0.55). Serialize.
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu() -> AtlasCudaBackend {
    AtlasCudaBackend::new(
        0,
        &atlas_kernels::ptx_for_exact_target("nemotron-3.5-super-vl-120b-a12b", "nvfp4")
            .expect("omni target PTX (vision_encoder via the target-local overlay)")
            .modules,
    )
    .expect("CUDA backend")
}

/// Minimal .npy (v1) reader: returns (shape, f32 data).
fn read_npy(path: &str) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = String::from_utf8(b[10..10 + hlen].to_vec()).unwrap();
    let shape: Vec<usize> = hdr
        .split("'shape':")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .split('(')
        .nth(1)
        .unwrap()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data: Vec<f32> = b[10 + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

fn f32_to_bf16(f: f32) -> u16 {
    let x = f.to_bits();
    // round-to-nearest-even
    ((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16
}

fn upload_bf16(g: &dyn GpuBackend, v: &[f32]) -> DevicePtr {
    let b: Vec<u16> = v.iter().map(|&f| f32_to_bf16(f)).collect();
    let bytes = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 2) };
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn download_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    // Sync first: tests launch on the default stream but copy_d2h rides the
    // backend copy stream — under parallel-test GPU load the copy can read
    // a mid-flight buffer (merge-gate flake: block0 cosine 0.5456).
    g.synchronize(g.default_stream()).unwrap();
    let mut v = vec![0u8; n * 2];
    g.copy_d2h(p, &mut v).unwrap();
    v.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn cosine_maxabs(a: &[f32], b: &[f32]) -> (f64, f32) {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut mx = 0f32;
    for i in 0..a.len() {
        dot += (a[i] * b[i]) as f64;
        na += (a[i] * a[i]) as f64;
        nb += (b[i] * b[i]) as f64;
        mx = mx.max((a[i] - b[i]).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), mx)
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn vit32_matches_oracle_square512() {
    vit32_check("square512");
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn vit32_matches_oracle_text640x480() {
    vit32_check("text640x480");
}

fn vit32_check(tag: &str) {
    let _g = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (e_shape, emb) = read_npy(&format!("{ORACLE}/{tag}_stages__embedder.npy"));
    let seq = e_shape[1];
    let (_, want_full) = read_npy(&format!("{ORACLE}/{tag}_stages__vit_features.npy"));
    let g = gpu();
    let hid = 1280usize;
    let inter = 5120usize;
    let heads = 16usize;
    let hd = hid / heads;

    let mut blocks = Vec::new();
    for i in 0..32 {
        let ld = |name: &str| {
            let (_, v) = read_npy(&format!("{ORACLE}/w_b{i}_{name}.npy"));
            upload_bf16(&g, &v)
        };
        blocks.push(RadioBlock {
            norm1_w: ld("norm1_weight"),
            norm1_b: ld("norm1_bias"),
            qkv_w: ld("attn_qkv_weight"),
            qkv_b: ld("attn_qkv_bias"),
            proj_w: ld("attn_proj_weight"),
            proj_b: ld("attn_proj_bias"),
            norm2_w: ld("norm2_weight"),
            norm2_b: ld("norm2_bias"),
            fc1_w: ld("mlp_fc1_weight"),
            fc1_b: ld("mlp_fc1_bias"),
            fc2_w: ld("mlp_fc2_weight"),
            fc2_b: ld("mlp_fc2_bias"),
        });
    }
    let kk = RadioKernels {
        k_norm: g.kernel("vision_encoder", "vision_layer_norm").unwrap(),
        k_gemm_bias: g.kernel("vision_encoder", "vision_gemm_bias").unwrap(),
        k_gemm_pipelined: KernelHandle(0),
        k_add_bias: KernelHandle(0),
        k_add: g.kernel("vision_encoder", "vision_add_inplace").unwrap(),
        k_gelu: g.kernel("vision_encoder", "vision_gelu_erf").unwrap(),
        k_attn: g.kernel("vision_encoder", "vision_attention_rope").unwrap(),
        k_gemm_f32: KernelHandle(0),
        k_softmax: KernelHandle(0),
        k_scatter: KernelHandle(0),
        k_rope_deint: KernelHandle(0),
        k_f32_bf16: KernelHandle(0),
        k_copy: g.kernel("vision_encoder", "vision_bf16_copy").unwrap(),
    };
    let x = upload_bf16(&g, &emb);
    let s_norm = g.alloc(seq * hid * 2).unwrap();
    let s_qkv = g.alloc(seq * 3 * hid * 2).unwrap();
    let s_o = g.alloc(seq * hid * 2).unwrap();
    let s_mlp = g.alloc(seq * inter * 2).unwrap();
    let cos_t = upload_bf16(&g, &vec![1.0f32; seq * hd]);
    let sin_t = upload_bf16(&g, &vec![0.0f32; seq * hd]);

    for (i, blk) in blocks.iter().enumerate() {
        radio_block_forward(
            &g,
            &kk,
            blk,
            x,
            s_norm,
            s_qkv,
            s_o,
            s_mlp,
            cos_t,
            sin_t,
            seq,
            hid,
            heads,
            inter,
            g.default_stream(),
        )
        .unwrap();
        if i % 8 == 0 || i == 31 {
            let g1 = download_f32(&g, x, seq * hid);
            let (_, w) = read_npy(&format!("{ORACLE}/pb_{tag}_block{i}.npy"));
            let (c, m) = cosine_maxabs(&g1, &w);
            eprintln!(
                "block {i}: cosine={c:.6} max|d|={m:.4} absmax={:?}",
                g1.iter().map(|v| v.abs()).fold(0f32, f32::max)
            );
        }
    }
    // strip 10 prefix tokens → features [n_patches, hid]
    let got_full = download_f32(&g, x, seq * hid);
    let got: Vec<f32> = got_full[10 * hid..].to_vec();
    let (c, m) = cosine_maxabs(&got, &want_full);
    eprintln!("vit32 features cosine={c:.6} max|d|={m:.4}");
    // 32 blocks of bf16 attn+GEMM compound: cosine is the gate; absolute
    // drift scales with activation magnitude (absmax ~3.5e3), so bound
    // |d| relative to the oracle range.
    let wmax = want_full.iter().map(|v| v.abs()).fold(0f32, f32::max);
    assert!(c >= 0.999, "cosine {c}");
    assert!(m <= 0.10 * wmax, "max abs {m} vs 10% of {wmax}");
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn final_ln_shuffle_mlp1_match_oracle_square512() {
    pipe_check("square512");
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn final_ln_shuffle_mlp1_match_oracle_text640x480() {
    pipe_check("text640x480");
}

fn pipe_check(tag: &str) {
    let _g = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Block31 output → strip 10 prefix → vision_final_layernorm →
    // pixel-shuffle v2 → mlp1 (RMSNorm → linear1 → relu² → linear2).
    let (b_shape, b31) = read_npy(&format!("{ORACLE}/pb_{tag}_block31.npy"));
    let _seq = b_shape[1];
    let hid = 1280usize;
    let g = gpu();
    let feats: Vec<f32> = b31[10 * hid..].to_vec(); // [n_patches,1280]
    let n_patches = feats.len() / hid;
    let (_, hw) = read_npy(&format!("{ORACLE}/{tag}_stages__patch_hw.npy"));
    let (gh, gw) = (hw[0] as usize, hw[1] as usize);
    assert_eq!(n_patches, gh * gw);

    // ── stage 4: final LN ──
    let lnw = {
        let (_, v) = read_npy(&format!("{ORACLE}/w_final_ln.npy"));
        upload_bf16(&g, &v)
    };
    let lnb = {
        let (_, v) = read_npy(&format!("{ORACLE}/w_final_ln_b.npy"));
        upload_bf16(&g, &v)
    };
    let x = upload_bf16(&g, &feats);
    let k_norm = g.kernel("vision_encoder", "vision_layer_norm").unwrap();
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let stream = g.default_stream();
    KernelLaunch::new(&g, k_norm)
        .grid([n_patches as u32, 1, 1])
        .block([hid.min(1024) as u32, 1, 1])
        .arg_ptr(x)
        .arg_ptr(lnw)
        .arg_ptr(lnb)
        .arg_u32(n_patches as u32)
        .arg_u32(hid as u32)
        .arg_f32(1e-6)
        .launch(stream)
        .unwrap();
    let got_ln = download_f32(&g, x, n_patches * hid);
    let (_, want_ln) = read_npy(&format!("{ORACLE}/{tag}_stages__after_final_ln.npy"));
    let (c, m) = cosine_maxabs(&got_ln, &want_ln);
    eprintln!("final_ln cosine={c:.6} max|d|={m:.4}");
    assert!(c >= 0.9999, "cosine {c}");

    // ── stage 5: pixel-shuffle v2 (host gather via the ported permutation) ──
    // out[n, h/2, w/2, c*4] from [n, w, h, c] layout of [h,w,c] tokens.
    let perm = crate::radio_preprocess::pixel_shuffle_v2_permutation(gh, gw, hid);
    let sh_h = gh / 2;
    let sh_w = gw / 2;
    let sh_c = hid * 4;
    let mut shuffled = vec![0f32; sh_h * sh_w * sh_c];
    for (dst, src) in perm.iter().enumerate() {
        shuffled[dst] = got_ln[*src as usize];
    }
    let (_, want_sh) = read_npy(&format!("{ORACLE}/{tag}_stages__after_pixel_shuffle.npy"));
    let (c, m) = cosine_maxabs(&shuffled, &want_sh);
    eprintln!("shuffle cosine={c:.6} max|d|={m:.4}");
    assert!(c >= 0.9999, "shuffle cosine {c}");

    // ── stage 6: mlp1 = RMSNorm(vanilla w) → l1 → relu² → l2 ──
    let w_nrm = upload_bf16(&g, &{
        let (_, v) = read_npy(&format!("{ORACLE}/w_mlp1_norm.npy"));
        v
    });
    let w_l1 = upload_bf16(&g, &{
        let (_, v) = read_npy(&format!("{ORACLE}/w_mlp1_l1.npy"));
        v
    });
    let w_l2 = upload_bf16(&g, &{
        let (_, v) = read_npy(&format!("{ORACLE}/w_mlp1_l2.npy"));
        v
    });
    let x1 = upload_bf16(&g, &shuffled); // [256, 5120]
    let n_tok = sh_h * sh_w;
    let normed = g.alloc(n_tok * sh_c * 2).unwrap();
    let k_rms_v = g.kernel("vision_encoder", "radio_rms_norm").unwrap();
    KernelLaunch::new(&g, k_rms_v)
        .grid([n_tok as u32, 1, 1])
        .block([sh_c.min(1024) as u32, 1, 1])
        .arg_ptr(x1)
        .arg_ptr(w_nrm)
        .arg_ptr(normed)
        .arg_u32(n_tok as u32)
        .arg_u32(sh_c as u32)
        .arg_f32(1e-5)
        .launch(stream)
        .unwrap();
    let k_gemm = g.kernel("vision_encoder", "vision_gemm_bias").unwrap();
    let k_relu2 = g.kernel("relu2", "relu_squared_inplace").unwrap();
    let hid1 = 20480usize;
    let hid2 = 4096usize;
    let z_bias = upload_bf16(&g, &vec![0f32; hid1]);
    let m1 = g.alloc(n_tok * hid1 * 2).unwrap();
    KernelLaunch::new(&g, k_gemm)
        .grid([div_ceil(hid1 as u32, 32), div_ceil(n_tok as u32, 32), 1])
        .block([32, 32, 1])
        .arg_ptr(normed)
        .arg_ptr(w_l1)
        .arg_ptr(z_bias)
        .arg_ptr(m1)
        .arg_u32(n_tok as u32)
        .arg_u32(hid1 as u32)
        .arg_u32(sh_c as u32)
        .launch(stream)
        .unwrap();
    KernelLaunch::new(&g, k_relu2)
        .grid([div_ceil((n_tok * hid1) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(m1)
        .arg_u32((n_tok * hid1) as u32)
        .launch(stream)
        .unwrap();
    let z_bias2 = upload_bf16(&g, &vec![0f32; hid2]);
    let m2 = g.alloc(n_tok * hid2 * 2).unwrap();
    KernelLaunch::new(&g, k_gemm)
        .grid([div_ceil(hid2 as u32, 32), div_ceil(n_tok as u32, 32), 1])
        .block([32, 32, 1])
        .arg_ptr(m1)
        .arg_ptr(w_l2)
        .arg_ptr(z_bias2)
        .arg_ptr(m2)
        .arg_u32(n_tok as u32)
        .arg_u32(hid2 as u32)
        .arg_u32(hid1 as u32)
        .launch(stream)
        .unwrap();
    let got_m = download_f32(&g, m2, n_tok * hid2);
    let (_, want_m) = read_npy(&format!("{ORACLE}/{tag}_stages__after_mlp1.npy"));
    let (c, m) = cosine_maxabs(&got_m, &want_m);
    eprintln!("mlp1 cosine={c:.6} max|d|={m:.4}");
    assert!(c >= 0.9999, "mlp1 cosine {c}");
}
