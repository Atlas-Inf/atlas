// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`; a GB10 and a glm-5.3-flash build with CUTLASS_HOME,
//! so spark-runtime compiles SparkGLM's FlashKDA): the in-tree bridge at TP4's
//! 16 local heads and TP2's 32, over two consecutive 4096-row pieces (the
//! state carries across calls), against
//! SparkGLM 0.2.0's own library byte for byte (when
//! `ATLAS_SPARKGLM_FLASH_LIBRARY` names its `libatlas_mango_flash.so`), the
//! reference recurrence (`kda_recurrent_bf16`) by error (FlashKDA runs the
//! chunked algorithm), and itself over planes (`ATLAS_GLM_KDA_FLASH_PLANES`)
//! byte for byte.
//!
//! ```text
//! ATLAS_SPARKGLM_FLASH_LIBRARY=/opt/atlas/native/libatlas_mango_flash.so \
//!   cargo test --release -p spark-model --lib sparkglm_flash -- --ignored --nocapture
//! ```

use super::*;
use crate::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

const PIECE: usize = 4096;
const D: usize = 128;

fn uniform(seed: &mut u64, lo: f32, hi: f32) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    lo + (hi - lo) * ((*seed >> 40) as f32 / (1u64 << 24) as f32)
}

fn bf16_bytes(n: usize, seed: &mut u64, lo: f32, hi: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            let v = uniform(seed, lo, hi).to_bits();
            (((v + 0x7fff + ((v >> 16) & 1)) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn f32_bytes(n: usize, seed: &mut u64, lo: f32, hi: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| uniform(seed, lo, hi).to_le_bytes())
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(g.default_stream())?;
    let mut v = vec![0u8; bytes];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn as_f32(bytes: &[u8], bf: bool) -> Vec<f32> {
    if bf {
        bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// (max |a - b| / max |b|, mean |a - b| / mean |b|).
fn error(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (mut dmax, mut bmax, mut dsum, mut bsum) = (0f32, 0f32, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        assert!(x.is_finite(), "non-finite FlashKDA value");
        let d = (x - y).abs();
        dmax = dmax.max(d);
        bmax = bmax.max(y.abs());
        dsum += d as f64;
        bsum += y.abs() as f64;
    }
    (dmax / bmax, (dsum / bsum) as f32)
}

/// One head count's random layer over `2 * PIECE` rows, its two state copies
/// and outputs, and the bridge's scratch.
struct Case {
    flash: SparkGlm,
    qkv: DevicePtr,
    ops: Operands,
    state0: Vec<u8>,
    want: (DevicePtr, DevicePtr),
    scratch: [(DevicePtr, usize); 3],
}

impl Case {
    fn new(g: &dyn GpuBackend, heads: usize, seed: u64) -> Result<Self> {
        let (tokens, p) = (2 * PIECE, heads * D);
        let mut seed = seed + heads as u64;
        let qkv = upload(g, &bf16_bytes(tokens * 3 * p, &mut seed, -2.0, 2.0))?;
        let gate = upload(g, &bf16_bytes(tokens * p, &mut seed, -3.0, 1.0))?;
        let beta = upload(g, &bf16_bytes(tokens * heads, &mut seed, -2.0, 2.0))?;
        let a = upload(g, &f32_bytes(heads, &mut seed, 0.0, 2.7))?;
        let bias = upload(g, &f32_bytes(p, &mut seed, -0.5, 0.5))?;
        let state0 = f32_bytes(heads * D * D, &mut seed, -0.1, 0.1);
        let out = tokens * p * 2;
        let ws = bridge::workspace_bytes(PIECE, heads)?;
        let aux = bridge::aux_bytes(PIECE, heads);
        let packed = 3 * bridge::plane_bytes(PIECE, heads);
        Ok(Self {
            flash: SparkGlm {
                heads,
                planes: false,
            },
            qkv,
            ops: [qkv, gate, beta, a, bias, upload(g, &state0)?, g.alloc(out)?],
            want: (upload(g, &state0)?, g.alloc(out)?),
            state0,
            scratch: [
                (g.alloc(packed)?, packed),
                (g.alloc(ws)?, ws),
                (g.alloc(aux)?, aux),
            ],
        })
    }

    fn out_bytes(&self) -> usize {
        2 * PIECE * self.flash.heads * D * 2
    }

    /// Both pieces through `run`, from `ops`' state into `ops`' output.
    fn pieces(&self, ops: Operands, run: impl Fn(Operands) -> Result<()>) -> Result<()> {
        for row in [0, PIECE] {
            run(self.flash.piece(ops, row, 3 * self.flash.heads * D * 2))?;
        }
        Ok(())
    }

    fn in_tree(&self, ops: Operands, stream: u64) -> Result<()> {
        self.pieces(ops, |o| {
            self.flash.run_piece(o, PIECE, self.scratch, stream)
        })
    }

    /// `(output, state)` bytes of the run into `ops` and of `want`.
    fn compare(&self, g: &dyn GpuBackend) -> Result<[(Vec<u8>, Vec<u8>); 2]> {
        let (n, s) = (self.out_bytes(), self.state0.len());
        Ok([
            (read(g, self.ops[6], n)?, read(g, self.ops[5], s)?),
            (read(g, self.want.1, n)?, read(g, self.want.0, s)?),
        ])
    }
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash build with CUTLASS_HOME and ATLAS_SPARKGLM_FLASH_LIBRARY"]
fn sparkglm_flash_matches_sparkglm_library_bytes() -> Result<()> {
    type Run = unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        u64,
        u64,
        u64,
        i32,
        i32,
        f32,
        f32,
        *mut c_void,
    ) -> i32;
    use std::ffi::c_void;
    let path = std::env::var("ATLAS_SPARKGLM_FLASH_LIBRARY")?;
    // SAFETY: the tester names SparkGLM's own pinned bridge.
    let library = unsafe { libloading::Library::new(&path) }?;
    let run = unsafe { *library.get::<Run>(b"atlas_mango_flash_prefill\0")? };
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    for heads in [16usize, 32] {
        let c = Case::new(g, heads, 0x5eed_0000)?;
        c.in_tree(c.ops, stream)?;
        let mut want = c.ops;
        (want[5], want[6]) = c.want;
        let p = |x: DevicePtr| x.0 as *mut c_void;
        let [(packed, pc), (ws, wc), (aux, ac)] = c.scratch;
        c.pieces(want, |[qkv, gate, beta, a, bias, state, out]| {
            // SAFETY: the same live operands and scratch as the in-tree run.
            let status = unsafe {
                run(
                    p(qkv),
                    p(gate),
                    p(beta),
                    p(a),
                    p(bias),
                    p(state),
                    p(out),
                    p(packed),
                    p(ws),
                    p(aux),
                    pc as u64,
                    wc as u64,
                    ac as u64,
                    PIECE as i32,
                    heads as i32,
                    1.0 / (128.0f32).sqrt(),
                    -5.0,
                    stream as *mut c_void,
                )
            };
            ensure!(status == 0, "SparkGLM library failed ({status})");
            Ok(())
        })?;
        let [got, want] = c.compare(g)?;
        ensure!(got.0 == want.0, "H{heads}: output differs from SparkGLM's");
        ensure!(got.1 == want.1, "H{heads}: state differs from SparkGLM's");
        println!("H{heads}: in-tree == SparkGLM 0.2.0 library (2 x {PIECE} rows)");
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash build with CUTLASS_HOME"]
fn sparkglm_flash_heads_match_reference_recurrence() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let reference = g.kernel("kda", "kda_recurrent_bf16")?;
    let stream = g.default_stream();
    let mut errors = Vec::new();
    for heads in [16usize, 32] {
        let c = Case::new(g, heads, 0x5eed_1000)?;
        c.in_tree(c.ops, stream)?;
        let [_, _, beta, a, bias, ..] = c.ops;
        let (state, out) = c.want;
        let rows = (2 * PIECE) as u32;
        let (h, d) = (heads as u32, D as u32);
        ops::kda_recurrent(
            g, reference, c.qkv, c.ops[1], beta, a, bias, state, out, rows, h, d, -5.0, stream,
        )?;
        let [got, want] = c.compare(g)?;
        let out = error(&as_f32(&got.0, true), &as_f32(&want.0, true));
        let state = error(&as_f32(&got.1, false), &as_f32(&want.1, false));
        println!(
            "H{heads}: output err max {:.2e} mean {:.2e}; state err max {:.2e} mean {:.2e}",
            out.0, out.1, state.0, state.1
        );
        errors.push((out, state));
    }
    for &((out_max, out_mean), (state_max, state_mean)) in &errors {
        assert!(
            out_mean < 2e-2 && state_mean < 2e-2 && out_max < 0.1 && state_max < 0.1,
            "FlashKDA strays from the recurrence: {errors:?}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash build with CUTLASS_HOME"]
fn sparkglm_flash_planes_match_packed() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let pack = g.kernel("kda", "kda_pack_qkv")?;
    for heads in [16usize, 32] {
        let c = Case::new(g, heads, 0xf1a5_0000)?;
        let (tokens, p) = (2 * PIECE, heads * D);
        // The packed operand from planes, as the planar convolution leaves them.
        let planes = upload(g, &bf16_bytes(tokens * 3 * p, &mut 7u64, -2.0, 2.0))?;
        ops::kda_pack_qkv(g, pack, planes, c.qkv, tokens as u32, p as u32, stream)?;
        let mut want = c.ops;
        (want[5], want[6]) = c.want;
        c.in_tree(want, stream)?;
        let plane = tokens * p * 2;
        for row in [0, PIECE] {
            let [q, k, v] = [0, 1, 2].map(|i| planes.offset(i * plane + row * p * 2));
            let [_, workspace, aux] = c.scratch;
            c.flash.run_piece_planes(
                [q, k, v],
                c.flash.piece(c.ops, row, 0),
                PIECE,
                [workspace, aux],
                stream,
            )?;
        }
        let [got, want] = c.compare(g)?;
        ensure!(got.0 == want.0, "H{heads}: planar output differs");
        ensure!(got.1 == want.1, "H{heads}: planar state differs");
        println!("H{heads}: planar FlashKDA == packed (2 x {PIECE} rows, output and state)");
    }
    Ok(())
}
