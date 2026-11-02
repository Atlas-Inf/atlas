// SPDX-License-Identifier: AGPL-3.0-only

//! The parity, row-invariance and bench checks of `mxfp8_rows_test_gpu.rs`
//! on the real `nvidia/Qwen3.8-Flash-Next-FP4-Local` MXFP8 tensors, each TP2
//! rank's twin cut on the GPU by `weight_map::mxfp8_rows_twin` /
//! `mxfp8_cols_twin` (asserted byte-equal to the host slice first).
//!
//!   ATLAS_FP4L_CKPT=<checkpoint dir> cargo test --release -p spark-model --lib \
//!       mxfp8_real -- --ignored --nocapture --test-threads=1

use super::*;
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;
use crate::weight_map::{RawMxfp8, mxfp8_cols_twin, mxfp8_rows_twin};

/// `(E4M3 [n, k], UE8M0 [n, k/32], n, k)` of checkpoint tensor `prefix`.
pub(crate) fn tensor(dir: &str, prefix: &str) -> Result<(Vec<u8>, Vec<u8>, usize, usize)> {
    let read = |name: &str| read_checkpoint_tensor(std::path::Path::new(dir), name);
    let (wd, ws, w) = read(&format!("{prefix}.weight"))?;
    let (sd, ss, s) = read(&format!("{prefix}.weight_scale"))?;
    let [n, k] = ws[..] else {
        anyhow::bail!("{prefix}: weight shape {ws:?}");
    };
    anyhow::ensure!(
        wd == "F8_E4M3" && sd == "U8" && ss == [n, k / 32],
        "{prefix}: {wd} {ws:?} + {sd} {ss:?} is not MXFP8"
    );
    Ok((w, s, n, k))
}

/// Rows `[lo, hi)` of a `[_, width]` row-major host matrix.
fn rows(m: &[u8], width: usize, lo: usize, hi: usize) -> Vec<u8> {
    m[lo * width..hi * width].to_vec()
}

/// The host-side TP slice: each part's segments cut to `rank`, stacked; or
/// (`cols`) the rank's K columns of one part.
fn host_slice(
    parts: &[(&[u8], &[u8], usize, usize, Vec<usize>)],
    cols: bool,
    rank: usize,
    tp: usize,
) -> (Vec<u8>, Vec<u8>, usize, usize) {
    if cols {
        let (w, s, n, k, _) = &parts[0];
        let (lk, g, lg) = (k / tp, k / 32, k / 32 / tp);
        let wv = (0..*n).flat_map(|r| w[r * k + rank * lk..r * k + (rank + 1) * lk].to_vec());
        let sv = (0..*n).flat_map(|r| s[r * g + rank * lg..r * g + (rank + 1) * lg].to_vec());
        return (wv.collect(), sv.collect(), *n, lk);
    }
    let k = parts[0].3;
    let (mut wv, mut sv, mut n) = (Vec::new(), Vec::new(), 0);
    for (w, s, _, _, segs) in parts {
        let mut base = 0;
        for seg in segs {
            let (lo, hi) = (base + rank * seg / tp, base + (rank + 1) * seg / tp);
            wv.extend(rows(w, k, lo, hi));
            sv.extend(rows(s, k / 32, lo, hi));
            n += hi - lo;
            base += seg;
        }
    }
    (wv, sv, n, k)
}

/// The twin the loader would build for `rank`, read back.
fn device_slice(
    parts: &[(&[u8], &[u8], usize, usize, Vec<usize>)],
    cols: bool,
    rank: usize,
    tp: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let mut raws = Vec::new();
    for (w, s, n, k, _) in parts {
        raws.push(RawMxfp8 {
            data: upload(gpu, w)?,
            scales: upload(gpu, s)?,
            n: *n,
            k: *k,
        });
    }
    let (twin, n, k) = if cols {
        mxfp8_cols_twin(gpu, raws[0], rank, tp, stream)?
    } else {
        let stacked: Vec<(RawMxfp8, &[usize])> = raws
            .iter()
            .zip(parts)
            .map(|(r, p)| (*r, &p.4[..]))
            .collect();
        mxfp8_rows_twin(gpu, &stacked, rank, tp, stream)?
    };
    let (mut w, mut s) = (vec![0u8; n * k], vec![0u8; n * k / 32]);
    gpu.copy_d2h(twin.data, &mut w)?;
    gpu.copy_d2h(twin.scales, &mut s)?;
    for p in raws
        .iter()
        .flat_map(|r| [r.data, r.scales])
        .chain([twin.data, twin.scales])
    {
        gpu.free(p)?;
    }
    Ok((w, s))
}

#[test]
#[ignore = "needs a GB10 GPU, a kernel target with mxfp8_gemv and ATLAS_FP4L_CKPT"]
fn mxfp8_real_tensors_match_bf16_per_tp2_rank() -> Result<()> {
    let dir = std::env::var("ATLAS_FP4L_CKPT").context("ATLAS_FP4L_CKPT=<checkpoint dir>")?;
    let l = |i: usize, rest: &str| format!("model.language_model.layers.{i}.{rest}");
    // GDN layer 0, QSA layer 3, shared expert layer 0. GDN QKVZ stacks
    // in_proj_qkv [Q 2048 | K 2048 | V 6144] and in_proj_z [Z 6144].
    let qkv = tensor(&dir, &l(0, "linear_attn.in_proj_qkv"))?;
    let z = tensor(&dir, &l(0, "linear_attn.in_proj_z"))?;
    let named: Vec<(&str, String, bool)> = vec![
        ("gdn out_proj", l(0, "linear_attn.out_proj"), true),
        ("qsa q_proj", l(3, "self_attn.q_proj"), false),
        ("qsa k_proj", l(3, "self_attn.k_proj"), false),
        ("qsa o_proj", l(3, "self_attn.o_proj"), true),
        (
            "qsa index_qk_proj",
            l(3, "self_attn.indexer.index_qk_proj"),
            false,
        ),
        (
            "shared gate_proj",
            l(0, "mlp.shared_expert.gate_proj"),
            false,
        ),
        (
            "shared down_proj",
            l(0, "mlp.shared_expert.down_proj"),
            false,
        ),
    ];
    let mut cases: Vec<(
        String,
        Vec<(&[u8], &[u8], usize, usize, Vec<usize>)>,
        bool,
        usize,
    )> = Vec::new();
    let qkvz_parts = vec![
        (&qkv.0[..], &qkv.1[..], qkv.2, qkv.3, vec![2048, 2048, 6144]),
        (&z.0[..], &z.1[..], z.2, z.3, vec![6144]),
    ];
    for tp in [1, 2] {
        cases.push((format!("gdn qkvz tp{tp}"), qkvz_parts.clone(), false, tp));
    }
    let loaded: Vec<_> = named
        .iter()
        .map(|(_, p, _)| tensor(&dir, p))
        .collect::<Result<_>>()?;
    for ((name, _, cols), t) in named.iter().zip(&loaded) {
        // The indexer and the shared expert are replicated across ranks.
        let tps: &[usize] = if name.starts_with("qsa i") || name.starts_with("shared") {
            &[1]
        } else {
            &[1, 2]
        };
        for &tp in tps {
            let parts = vec![(&t.0[..], &t.1[..], t.2, t.3, vec![t.2])];
            cases.push((format!("{name} tp{tp}"), parts, *cols, tp));
        }
    }
    let mut rng = Rng(0xfeed_f00d);
    for (name, parts, cols, tp) in &cases {
        for rank in 0..*tp {
            let (w, s, n, k) = host_slice(parts, *cols, rank, *tp);
            let (dw, ds) = device_slice(parts, *cols, rank, *tp)?;
            assert!(dw == w && ds == s, "{name} rank {rank}: twin != host slice");
            let bf16 = dequant(&w, &s, k);
            let label = format!("real {name} r{rank}");
            parity(&mut rng, &label, &w, &s, &bf16, n, k)?;
            if name.starts_with("gdn") && *tp == 2 && rank == 0 {
                bench(&label, &w, &s, &bf16, n, k)?;
            }
        }
    }
    Ok(())
}
