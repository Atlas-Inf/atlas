// SPDX-License-Identifier: AGPL-3.0-only

//! Native-HIP correctness gate for the BF16 paged-prefill kernel
//! `inferspark_prefill_paged` at a chunk-continuation shape.
//!
//! Q/O use `[q_len, nq, head_dim]`; the paged K/V cache uses
//! `[blocks, 16, nkv, head_dim]` with identity `block_table = 0..blocks`.
//! The launch exactly mirrors `ops::prefill_attention_paged`, and the CPU
//! reference computes causal GQA attention in FP32 using absolute query position
//! `q_offset + i`.
//!
//! Usage: cargo run --release -p spark-model \
//!          --example inferspark_attn_paged_bf16_microtest \
//!          --features cuda,gpu-examples -- \
//!          [q_len] [kv_len] [q_offset] [nq] [nkv] [seed]
//! Exit 0 = PASS (cosine >= 0.99), 1 = FAIL.

use std::fmt::Display;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const DEFAULT_Q_LEN: usize = 32;
const DEFAULT_KV_LEN: usize = 2080;
const DEFAULT_Q_OFFSET: usize = 2048;
const DEFAULT_NQ: usize = 2;
const DEFAULT_NKV: usize = 1;
const DEFAULT_SEED: u64 = 0x51A7;
const HEAD_DIM: usize = 256;
const CACHE_BLOCK_SIZE: usize = 16;
const COSINE_GATE: f64 = 0.99;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn parse_arg<T>(args: &[String], index: usize, name: &str, default: T) -> Result<T>
where
    T: FromStr,
    T::Err: Display,
{
    args.get(index).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|error| anyhow::anyhow!("invalid {name} {value:?}: {error}"))
    })
}

fn parse_seed(args: &[String]) -> Result<u64> {
    let Some(value) = args.get(6) else {
        return Ok(DEFAULT_SEED);
    };
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).with_context(|| format!("invalid seed {value:?}"))
    } else {
        value
            .parse()
            .with_context(|| format!("invalid seed {value:?}"))
    }
}

fn bf16_bits_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7FFF + ((bits >> 16) & 1))) >> 16) as u16
}

fn u16s_to_le(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32s_to_le(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn cache_index(
    position: usize,
    kv_head: usize,
    dim: usize,
    block_table: &[i32],
    num_kv_heads: usize,
) -> usize {
    let logical_block = position / CACHE_BLOCK_SIZE;
    let block_offset = position % CACHE_BLOCK_SIZE;
    let physical_block = block_table[logical_block] as usize;
    (((physical_block * CACHE_BLOCK_SIZE + block_offset) * num_kv_heads + kv_head) * HEAD_DIM) + dim
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let q_len = parse_arg(&args, 1, "q_len", DEFAULT_Q_LEN)?;
    let kv_len = parse_arg(&args, 2, "kv_len", DEFAULT_KV_LEN)?;
    let q_offset = parse_arg(&args, 3, "q_offset", DEFAULT_Q_OFFSET)?;
    let nq = parse_arg(&args, 4, "nq", DEFAULT_NQ)?;
    let nkv = parse_arg(&args, 5, "nkv", DEFAULT_NKV)?;
    let seed = parse_seed(&args)?;

    if q_len == 0 || kv_len == 0 || nq == 0 || nkv == 0 {
        bail!("q_len, kv_len, nq, and nkv must all be nonzero");
    }
    if nq % nkv != 0 {
        bail!("nq ({nq}) must be divisible by nkv ({nkv})");
    }
    let query_end = q_offset
        .checked_add(q_len)
        .context("q_offset + q_len overflowed")?;
    if query_end > kv_len {
        bail!(
            "chunk queries must be present in KV: q_offset + q_len ({query_end}) > kv_len ({kv_len})"
        );
    }

    let q_len_u32 = u32::try_from(q_len).context("q_len exceeds u32")?;
    let kv_len_u32 = u32::try_from(kv_len).context("kv_len exceeds u32")?;
    let q_offset_u32 = u32::try_from(q_offset).context("q_offset exceeds u32")?;
    let nq_u32 = u32::try_from(nq).context("nq exceeds u32")?;
    let nkv_u32 = u32::try_from(nkv).context("nkv exceeds u32")?;
    let blocks = kv_len.div_ceil(CACHE_BLOCK_SIZE);
    let block_table: Vec<i32> = (0..blocks)
        .map(|block| i32::try_from(block).context("block count exceeds i32"))
        .collect::<Result<_>>()?;
    let inv_sqrt_d = 1.0f32 / (HEAD_DIM as f32).sqrt();

    println!(
        "=== inferspark_prefill_paged BF16 native-HIP correctness: \
         q_len={q_len} kv_len={kv_len} q_offset={q_offset} nq={nq} nkv={nkv} \
         head_dim={HEAD_DIM} cache_block_size={CACHE_BLOCK_SIZE} blocks={blocks} \
         seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    let q: Vec<u16> = (0..q_len * nq * HEAD_DIM)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let cache_elements = blocks * CACHE_BLOCK_SIZE * nkv * HEAD_DIM;
    let k_cache: Vec<u16> = (0..cache_elements)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let v_cache: Vec<u16> = (0..cache_elements)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let qp = upload(gpu, &u16s_to_le(&q))?;
    let kp = upload(gpu, &u16s_to_le(&k_cache))?;
    let vp = upload(gpu, &u16s_to_le(&v_cache))?;
    let output_init = vec![0x7FC1u16; q_len * nq * HEAD_DIM];
    let op = upload(gpu, &u16s_to_le(&output_init))?;
    let btp = upload(gpu, &i32s_to_le(&block_table))?;

    // Exact production selection, argument order, and native-HIP geometry.
    let use_br64 = q_len >= 256;
    let symbol = if use_br64 {
        "inferspark_prefill_paged_64"
    } else {
        "inferspark_prefill_paged"
    };
    let handle = gpu.kernel("prefill_paged", symbol)?;
    KernelLaunch::new(gpu, handle)
        .grid([nq_u32, div_ceil(q_len_u32, 32), 1])
        .block([if use_br64 { 256 } else { 128 }, 1, 1])
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(op)
        .arg_ptr(btp)
        .arg_u32(q_len_u32)
        .arg_u32(kv_len_u32)
        .arg_u32(q_offset_u32)
        .arg_u32(nq_u32)
        .arg_u32(nkv_u32)
        .arg_u32(HEAD_DIM as u32)
        .arg_u32(CACHE_BLOCK_SIZE as u32)
        .arg_u32(0) // sliding_window = 0 (full attention)
        .arg_u32(1) // causal_mask_enabled
        .arg_f32(inv_sqrt_d)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let mut raw = vec![0u8; q_len * nq * HEAD_DIM * 2];
    gpu.copy_d2h(op, &mut raw)?;
    let output_gpu: Vec<u16> = raw
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();

    // Causal GQA CPU reference in FP32. Query i is at absolute position
    // q_offset+i and therefore attends exactly to keys j <= q_offset+i.
    let gqa = nq / nkv;
    let mut output_cpu = vec![0f32; q_len * nq * HEAD_DIM];
    for head in 0..nq {
        let kv_head = head / gqa;
        for query in 0..q_len {
            let num_keys = (q_offset + query + 1).min(kv_len);
            let mut scores = vec![0f32; num_keys];
            let mut max_score = f32::NEG_INFINITY;
            for (key, score) in scores.iter_mut().enumerate() {
                let mut dot = 0f32;
                for dim in 0..HEAD_DIM {
                    dot += bf16_bits_to_f32(q[(query * nq + head) * HEAD_DIM + dim])
                        * bf16_bits_to_f32(
                            k_cache[cache_index(key, kv_head, dim, &block_table, nkv)],
                        );
                }
                *score = dot * inv_sqrt_d;
                max_score = max_score.max(*score);
            }
            let mut sum = 0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for dim in 0..HEAD_DIM {
                let mut value = 0f32;
                for (key, score) in scores.iter().enumerate() {
                    value += *score / sum
                        * bf16_bits_to_f32(
                            v_cache[cache_index(key, kv_head, dim, &block_table, nkv)],
                        );
                }
                output_cpu[(query * nq + head) * HEAD_DIM + dim] = value;
            }
        }
    }

    let output_f32: Vec<f32> = output_gpu
        .iter()
        .map(|&value| bf16_bits_to_f32(value))
        .collect();
    let non_finite = output_f32.iter().filter(|value| !value.is_finite()).count();
    let zeros = output_f32.iter().filter(|&&value| value == 0.0).count();
    println!(
        "output non_finite={non_finite}/{} zeros={zeros}/{} first8={:?}",
        output_f32.len(),
        output_f32.len(),
        &output_f32[..output_f32.len().min(8)],
    );
    let cosine = cosine_bf16_f32(&output_gpu, &output_cpu);
    println!("cosine(all)={cosine:.6}");
    for ptr in [qp, kp, vp, op, btp] {
        gpu.free(ptr).ok();
    }

    if cosine.is_finite() && cosine >= COSINE_GATE {
        println!("RESULT: PASS (cosine {cosine:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        println!("RESULT: FAIL (cosine {cosine:.6} < {COSINE_GATE})");
        std::process::exit(1);
    }
}

fn cosine_bf16_f32(gpu: &[u16], cpu: &[f32]) -> f64 {
    let (mut dot, mut gpu_norm, mut cpu_norm) = (0f64, 0f64, 0f64);
    for (&gpu_value, &cpu_value) in gpu.iter().zip(cpu) {
        let x = bf16_bits_to_f32(gpu_value) as f64;
        let y = cpu_value as f64;
        dot += x * y;
        gpu_norm += x * x;
        cpu_norm += y * y;
    }
    if gpu_norm == 0.0 || cpu_norm == 0.0 {
        return f64::NAN;
    }
    dot / (gpu_norm.sqrt() * cpu_norm.sqrt())
}
