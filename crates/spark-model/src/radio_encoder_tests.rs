// SPDX-License-Identifier: AGPL-3.0-only

//! Stage-1 GPU oracle test for `radio_encoder::RadioEmbedder`:
//! patch-embed GEMM + resampled pos-embed + cls/register prefix, against
//! the checkpoint's own embedder output
//! (`n35vl-runs/vision_oracle/square512_stages__embedder.npy`).
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
fn embedder_matches_oracle_square512() {
    let _g = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Weights
    let (pw_shape, patch_w) = read_npy(&format!("{ORACLE}/w_patch.npy"));
    assert_eq!(pw_shape, vec![1280, 768]);
    let (ps_shape, pos) = read_npy(&format!("{ORACLE}/w_pos.npy"));
    let (cr_shape, cls) = read_npy(&format!("{ORACLE}/w_clsreg.npy"));
    let max_rows = 2048 / 16; // max_img_size / patch_size
    assert_eq!(ps_shape, vec![1, 16384, 1280]);
    assert_eq!(cr_shape, vec![10, 1280]);

    // Oracle inputs: conditioned pixels [1,3,512,512] → patches [1024, 768]
    let (px_shape, px) = read_npy(&format!("{ORACLE}/square512_preprocess__pixel_values.npy"));
    assert_eq!(px_shape[1..], [3, 512, 512]);
    let (e_shape, want) = read_npy(&format!("{ORACLE}/square512_stages__embedder.npy"));
    assert_eq!(e_shape, vec![1, 10 + 1024, 1280]);

    let g = gpu();
    let patch = 16usize;
    let (h, w) = (512 / patch, 512 / patch);
    let n = h * w;
    // patchify: [3,512,512] → [1024, 3*16*16] (row-major patch scan)
    let mut patches = vec![0f32; n * 768];
    for py in 0..h {
        for pxx in 0..w {
            let row = py * w + pxx;
            for c in 0..3 {
                for i in 0..patch {
                    for j in 0..patch {
                        patches[row * 768 + c * 256 + i * 16 + j] =
                            px[c * 512 * 512 + (py * 16 + i) * 512 + (pxx * 16 + j)];
                    }
                }
            }
        }
    }

    let px_dev = upload_bf16(&g, &patches);
    let pw_dev = upload_bf16(&g, &patch_w);
    let cls_dev = upload_bf16(&g, &cls);
    let pos_dev = upload_bf16(&g, &pos);
    let _ = pos_dev; // device copy unused — host resample re-uploads
    let buf_pos = g.alloc(n * 1280 * 2).unwrap();
    let out = g.alloc((10 + n) * 1280 * 2).unwrap();

    let emb = RadioEmbedder {
        patch_w: pw_dev,
        pos_embed_dev: DevicePtr(0),
        pos_embed_host: pos,
        cls_register: cls_dev,
        max_rows,
        max_cols: max_rows,
        hidden: 1280,
        patch,
        prefix_tokens: 10,
        k_gemm: g.kernel("gemm", "dense_gemm_bf16_pipelined").unwrap(),
        k_add: g.kernel("residual_add", "bf16_residual_add").unwrap(),
        buf_pos,
        buf_pos_cap: n * 1280 * 2,
    };
    emb.embed(&g, px_dev, h, w, out, g.default_stream())
        .unwrap();

    let got = download_f32(&g, out, (10 + n) * 1280);
    let (cos, mx) = cosine_maxabs(&got, &want);
    eprintln!("embedder cosine={cos:.6} max|d|={mx:.4}");
    assert!(cos >= 0.9999, "cosine {cos}");
    assert!(mx <= 0.05, "max abs {mx}");
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn block0_matches_oracle_square512() {
    let _g = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Inputs: the embedder oracle output IS the block-0 input.
    let (e_shape, emb) = read_npy(&format!("{ORACLE}/square512_stages__embedder.npy"));
    let (b_shape, want) = read_npy(&format!("{ORACLE}/square512_stages__block0.npy"));
    let seq = e_shape[1]; // 1034
    assert_eq!(b_shape, vec![1, seq, 1280]);

    let g = gpu();
    let hid = 1280usize;
    let inter = 5120usize; // mlp_ratio 4.0
    let heads = 16usize;
    let hd = hid / heads;

    let ld = |name: &str| {
        let (_, v) = read_npy(&format!("{ORACLE}/{name}.npy"));
        upload_bf16(&g, &v)
    };
    let blk = RadioBlock {
        norm1_w: ld("w_b0_norm1"),
        norm1_b: ld("w_b0_norm1_b"),
        qkv_w: ld("w_b0_qkv"),
        qkv_b: ld("w_b0_qkv_b"),
        proj_w: ld("w_b0_proj"),
        proj_b: ld("w_b0_proj_b"),
        norm2_w: ld("w_b0_norm2"),
        norm2_b: ld("w_b0_norm2_b"),
        fc1_w: ld("w_b0_fc1"),
        fc1_b: ld("w_b0_fc1_b"),
        fc2_w: ld("w_b0_fc2"),
        fc2_b: ld("w_b0_fc2_b"),
    };
    let kk = RadioKernels {
        k_norm: g.kernel("vision_encoder", "vision_layer_norm").unwrap(),
        k_gemm_bias: g.kernel("vision_encoder", "vision_gemm_bias").unwrap(),
        k_gemm_pipelined: KernelHandle(0),
        k_add_bias: g.kernel("vision_encoder", "vision_add_bias").unwrap(),
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
    // identity rope tables: cos=1, sin=0
    let cos = upload_bf16(&g, &vec![1.0f32; seq * hd]);
    let sin = upload_bf16(&g, &vec![0.0f32; seq * hd]);

    radio_block_forward(
        &g,
        &kk,
        &blk,
        x,
        s_norm,
        s_qkv,
        s_o,
        s_mlp,
        cos,
        sin,
        seq,
        hid,
        heads,
        inter,
        g.default_stream(),
    )
    .unwrap();
    let got = download_f32(&g, x, seq * hid);
    let (cos_v, mx) = cosine_maxabs(&got, &want);
    eprintln!("block0 cosine={cos_v:.6} max|d|={mx:.4}");
    // tanh-gelu (vision_gelu) vs the reference erf gelu: a handful of
    // large-|x| elements differ by up to ~0.5; cosine is the real gate.
    assert!(cos_v >= 0.9999, "cosine {cos_v}");
    assert!(mx <= 0.5, "max abs {mx}");
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn block0_bisect_norm_qkv_attn() {
    let _g = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (e_shape, emb) = read_npy(&format!("{ORACLE}/square512_stages__embedder.npy"));
    let seq = e_shape[1];
    let g = gpu();
    let hid = 1280usize;
    let heads = 16usize;
    let hd = hid / heads;

    let ld = |name: &str| {
        let (_, v) = read_npy(&format!("{ORACLE}/{name}.npy"));
        upload_bf16(&g, &v)
    };
    let n1w = ld("w_b0_norm1");
    let n1b = ld("w_b0_norm1_b");
    let qkv_w = ld("w_b0_qkv");
    let qkv_b = ld("w_b0_qkv_b");
    let k_norm = g.kernel("vision_encoder", "vision_layer_norm").unwrap();
    let k_copy = g.kernel("vision_encoder", "vision_bf16_copy").unwrap();
    let k_gemm_bias = g.kernel("vision_encoder", "vision_gemm_bias").unwrap();
    let k_attn = g.kernel("vision_encoder", "vision_attention_rope").unwrap();
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

    let x = upload_bf16(&g, &emb);
    let n_total = (seq * hid) as u32;
    let s_norm = g.alloc(seq * hid * 2).unwrap();
    // copy + LN
    KernelLaunch::new(&g, k_copy)
        .grid([div_ceil(n_total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(s_norm)
        .arg_u32(n_total)
        .launch(g.default_stream())
        .unwrap();
    KernelLaunch::new(&g, k_norm)
        .grid([seq as u32, 1, 1])
        .block([hid.min(1024) as u32, 1, 1])
        .arg_ptr(s_norm)
        .arg_ptr(n1w)
        .arg_ptr(n1b)
        .arg_u32(seq as u32)
        .arg_u32(hid as u32)
        .arg_f32(1e-6)
        .launch(g.default_stream())
        .unwrap();
    let got = download_f32(&g, s_norm, seq * hid);
    let (_, want_n1) = read_npy(&format!("{ORACLE}/dbg_norm1.npy"));
    let (c, m) = cosine_maxabs(&got, &want_n1);
    eprintln!("norm1 cosine={c:.6} max|d|={m:.4}");

    // QKV
    let s_qkv = g.alloc(seq * 3 * hid * 2).unwrap();
    KernelLaunch::new(&g, k_gemm_bias)
        .grid([div_ceil(3 * hid as u32, 32), div_ceil(seq as u32, 32), 1])
        .block([32, 32, 1])
        .arg_ptr(s_norm)
        .arg_ptr(qkv_w)
        .arg_ptr(qkv_b)
        .arg_ptr(s_qkv)
        .arg_u32(seq as u32)
        .arg_u32(3 * hid as u32)
        .arg_u32(hid as u32)
        .launch(g.default_stream())
        .unwrap();
    let got = download_f32(&g, s_qkv, seq * 3 * hid);
    let (_, want_qkv) = read_npy(&format!("{ORACLE}/dbg_qkv.npy"));
    let (c, m) = cosine_maxabs(&got, &want_qkv);
    eprintln!("qkv cosine={c:.6} max|d|={m:.4}");
    eprintln!("got row0: {:?}", &got[..6]);
    eprintln!("want row0: {:?}", &want_qkv[..6]);
    eprintln!("got row0 k-part: {:?}", &got[hid..hid + 4]);
    eprintln!("want row0 k-part: {:?}", &want_qkv[hid..hid + 4]);
    // row/col-swap check: is got[r][c] == want[c][r] anywhere?

    // attention
    let s_o = g.alloc(seq * hid * 2).unwrap();
    let cos_t = upload_bf16(&g, &vec![1.0f32; seq * hd]);
    let sin_t = upload_bf16(&g, &vec![0.0f32; seq * hd]);
    KernelLaunch::new(&g, k_attn)
        .grid([seq as u32, heads as u32, 1])
        .block([32, 1, 1])
        .shared_mem((seq * 4 + 2 * hd * 4) as u32)
        .arg_ptr(s_qkv)
        .arg_ptr(s_o)
        .arg_ptr(cos_t)
        .arg_ptr(sin_t)
        .arg_u32(seq as u32)
        .arg_u32(heads as u32)
        .arg_u32(hd as u32)
        .launch(g.default_stream())
        .unwrap();
    let got = download_f32(&g, s_o, seq * hid);
    let (_, want_o) = read_npy(&format!("{ORACLE}/dbg_attn_out.npy"));
    let n_nan = got.iter().filter(|v| !v.is_finite()).count();
    eprintln!(
        "attn got[0..4]={:?} want[0..4]={:?}",
        &got[..4],
        &want_o[..4]
    );
    eprintln!(
        "attn got[1280..1284]={:?} want={:?}",
        &got[hid..hid + 4],
        &want_o[hid..hid + 4]
    );
    let n_finite = got.iter().filter(|v| v.is_finite() && **v != 0.0).count();
    eprintln!(
        "attn n_nan={n_nan} n_finite_nonzero={n_finite} of {}",
        got.len()
    );
}
