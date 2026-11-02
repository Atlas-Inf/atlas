// SPDX-License-Identifier: AGPL-3.0-only
//! Host fixtures and independent FP8 packing oracle for the ignored GPU checks.
use anyhow::{Result, ensure};
use half::bf16;

pub(super) const ROWS: usize = 2048;
pub(super) const START: usize = 16384;
pub(super) const END: usize = START + ROWS;
pub(super) const BLOCKS: usize = END / 16 + 3;
pub(super) const SAMPLES: usize = 8;

pub(super) struct Host {
    pub q: Vec<u16>,
    pub kv: Vec<u16>,
    pub table: Vec<i32>,
    pub selected: Vec<i32>,
}

impl Host {
    pub fn new(value: f32) -> Self {
        let mut selected = vec![-1; ROWS * 2051];
        for row in 0..ROWS {
            for col in 0..2048 {
                selected[row * 2051 + col] = col as i32;
            }
            let count = (START + row + 1) % 4;
            for tail in 0..count {
                selected[row * 2051 + 2048 + tail] = (START + row + 1 - count + tail) as i32;
            }
        }
        Self {
            q: vec![0; ROWS * 32 * 512],
            kv: vec![bf16::from_f32(value).to_bits(); BLOCKS * 16 * 512],
            table: (2..END / 16 + 2).rev().map(|v| v as i32).collect(),
            selected,
        }
    }
    pub fn token_mut(&mut self, token: usize) -> &mut [u16] {
        let base = (self.table[token / 16] as usize * 16 + token % 16) * 512;
        &mut self.kv[base..base + 512]
    }
    pub fn inputs(&self) -> [Vec<u8>; 4] {
        [
            u16_bytes(&self.q),
            u16_bytes(&self.kv),
            self.selected.iter().flat_map(|x| x.to_le_bytes()).collect(),
            self.table.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ]
    }
}

pub(super) fn u16_bytes(x: &[u16]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub(super) fn read_u16(x: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(x[offset..offset + 2].try_into().unwrap())
}
pub(super) fn read_f32(x: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(x[offset..offset + 4].try_into().unwrap())
}
pub(super) fn value(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

fn e4m3_values() -> [f32; 127] {
    std::array::from_fn(|code| {
        let exponent = code >> 3;
        let fraction = code & 7;
        if exponent == 0 {
            fraction as f32 / 512.0
        } else {
            (1.0 + fraction as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
        }
    })
}
fn encode_e4m3(x: f32, values: &[f32; 127]) -> u8 {
    let sign = if x.is_sign_negative() { 128 } else { 0 };
    let magnitude = x.abs().min(448.0);
    let upper = values.partition_point(|&v| v < magnitude).min(126);
    if upper == 0 {
        return sign;
    }
    let lower = upper - 1;
    let a = magnitude - values[lower];
    let b = values[upper] - magnitude;
    let code = if a < b || (a == b && lower % 2 == 0) {
        lower
    } else {
        upper
    };
    sign | code as u8
}

pub(super) fn check_prep(host: &Host, qpad: &[u8], packed: &[u8]) -> Result<()> {
    for token_head in 0..ROWS * 32 {
        for dim in 0..576 {
            let expected = if dim < 512 {
                host.q[token_head * 512 + dim]
            } else {
                0
            };
            ensure!(
                read_u16(qpad, (token_head * 576 + dim) * 2) == expected,
                "Q padding mismatch"
            );
        }
    }
    let values = e4m3_values();
    for token in 0..END {
        let physical = (host.table[token / 16] as usize * 16 + token % 16) * 512;
        for group in 0..4 {
            let input = &host.kv[physical + group * 128..physical + (group + 1) * 128];
            let maximum = input
                .iter()
                .map(|&x| value(x).abs())
                .fold(1e-4f32, f32::max);
            let scale = maximum * (1.0f64 / 448.0) as f32;
            ensure!(
                read_f32(packed, token * 656 + 512 + group * 4).to_bits() == scale.to_bits(),
                "KV scale mismatch at {token}/{group}"
            );
            for (dim, &x) in input.iter().enumerate() {
                ensure!(
                    packed[token * 656 + group * 128 + dim]
                        == encode_e4m3(value(x) / scale, &values),
                    "FP8 KV mismatch at {token}/{group}/{dim}"
                );
            }
        }
        ensure!(
            packed[token * 656 + 528..(token + 1) * 656]
                .iter()
                .all(|&x| x == 0),
            "KV padding changed"
        );
    }
    Ok(())
}

pub(super) struct Sample {
    pub row: usize,
    pub head: usize,
    pub output: Vec<u16>,
    pub lse: f32,
}

pub(super) fn sampled_fixture() -> Result<(Host, Vec<Sample>)> {
    use safetensors::{Dtype, SafeTensors};
    use std::io::Read;
    let path = std::env::var_os("ATLAS_GLM_SPARSE_TEST_FIXTURE").ok_or_else(|| {
        anyhow::anyhow!(
            "Set ATLAS_GLM_SPARSE_TEST_FIXTURE to the explicit sampled safetensors fixture"
        )
    })?;
    const LIMIT: u64 = 20 * 1024 * 1024;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(LIMIT + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= LIMIT, "fixture exceeds20MiB");
    let (_, metadata) = SafeTensors::read_metadata(&bytes)?;
    ensure!(
        metadata
            .metadata()
            .as_ref()
            .and_then(|m| m.get("format"))
            .map(String::as_str)
            == Some("atlas-glm-sparse-sampled-v1"),
        "wrong sampled fixture format"
    );
    use sha2::{Digest, Sha256};
    println!(
        "{}",
        serde_json::json!({
            "fixture_sha256": format!("{:x}", Sha256::digest(&bytes)),
            "scope": "original sampled operands and stored scaffold output",
        })
    );
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = |name: &str, dtype: Dtype, shape: &[usize]| -> Result<Vec<u8>> {
        let t = tensors.tensor(name)?;
        ensure!(
            t.dtype() == dtype && t.shape() == shape,
            "wrong fixture dtype/shape: {name}"
        );
        Ok(t.data().to_vec())
    };
    let q = tensor("q", Dtype::BF16, &[SAMPLES, 512])?;
    let kv = tensor("kv", Dtype::BF16, &[SAMPLES, 2051, 512])?;
    let output = tensor("actual_output", Dtype::BF16, &[SAMPLES, 512])?;
    let lse = tensor("actual_lse", Dtype::F32, &[SAMPLES])?;
    let rows = tensor("sample_rows", Dtype::I64, &[SAMPLES])?;
    let heads = tensor("sample_heads", Dtype::I64, &[SAMPLES])?;
    let tails = tensor("tail_counts", Dtype::I64, &[SAMPLES])?;
    ensure!(tensors.names().len() == 7, "unexpected fixture tensors");
    for data in [&q, &kv, &output] {
        ensure!(
            data.chunks_exact(2)
                .all(|x| value(u16::from_le_bytes(x.try_into().unwrap())).is_finite()),
            "nonfinite BF16 fixture"
        );
    }
    let index = |data: &[u8], i: usize| -> Result<usize> {
        Ok(usize::try_from(i64::from_le_bytes(
            data[i * 8..i * 8 + 8].try_into().unwrap(),
        ))?)
    };
    let mut host = Host::new(0.0);
    let mut samples = Vec::new();
    let mut used_rows = std::collections::HashSet::new();
    let mut causal = std::collections::HashMap::<usize, Vec<u16>>::new();
    for sample in 0..SAMPLES {
        let row = index(&rows, sample)?;
        let head = index(&heads, sample)?;
        let count = index(&tails, sample)?;
        ensure!(
            row < ROWS && head < 32 && used_rows.insert(row),
            "invalid/repeated sampled row/head"
        );
        ensure!(
            count == (START + row + 1) % 4,
            "fixture tail count mismatch"
        );
        let actual_lse = read_f32(&lse, sample * 4);
        ensure!(actual_lse.is_finite(), "nonfinite LSE fixture");
        for dim in 0..512 {
            host.q[(row * 32 + head) * 512 + dim] = read_u16(&q, (sample * 512 + dim) * 2);
        }
        for selected in 0..2048 + count {
            let token = if selected < 2048 {
                let token = sample * 2048 + selected;
                host.selected[row * 2051 + selected] = token as i32;
                token
            } else {
                START + row + 1 - count + selected - 2048
            };
            let values: Vec<u16> = (0..512)
                .map(|dim| read_u16(&kv, ((sample * 2051 + selected) * 512 + dim) * 2))
                .collect();
            if selected >= 2048
                && let Some(previous) = causal.insert(token, values.clone())
            {
                ensure!(previous == values, "inconsistent shared causal KV");
            }
            host.token_mut(token).copy_from_slice(&values);
        }
        samples.push(Sample {
            row,
            head,
            output: (0..512)
                .map(|dim| read_u16(&output, (sample * 512 + dim) * 2))
                .collect(),
            lse: actual_lse,
        });
    }
    Ok((host, samples))
}

#[test]
fn scalar_e4m3_reference_rounding_boundaries() {
    let values = e4m3_values();
    for (code, &v) in values.iter().enumerate() {
        assert_eq!(encode_e4m3(v, &values), code as u8);
        assert_eq!(encode_e4m3(-v, &values), code as u8 | 128);
    }
    assert_eq!(encode_e4m3(0.5 / 512.0, &values), 0);
    assert_eq!(encode_e4m3(1.5 / 512.0, &values), 2);
    assert_eq!(encode_e4m3(1.0625, &values), 56);
    assert_eq!(encode_e4m3(1.1875, &values), 58);
    assert_eq!(encode_e4m3(1e9, &values), 126);
}
