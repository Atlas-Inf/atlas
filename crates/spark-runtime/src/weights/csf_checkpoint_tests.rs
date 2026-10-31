// SPDX-License-Identifier: AGPL-3.0-only

//! Real-checkpoint CSF checks, `#[ignore]`d: they need a local copy of an
//! FP4-Local checkpoint (or a layer dump of one). Both encodings go through
//! the same `CsfAssembler`, so both tests cover either.

use super::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Every CSF scale of a real checkpoint on disk (either encoding) decodes
/// cleanly, timed the way the loaders pay for it (decode only, I/O
/// excluded). Also checks that every U8 `weight_scale` (the MXFP8 backbone's
/// UE8M0 exponents) lies in the range where the MXFP8 -> BF16 expansion is
/// exact (3..=246, see `spark_model::weight_map::mxfp8`).
///
///   ATLAS_CSF_CKPT_DIR=<model dir> cargo test --release -p spark-runtime \
///     --lib csf_whole_checkpoint -- --ignored --nocapture
#[test]
#[ignore]
fn csf_whole_checkpoint() {
    use std::os::unix::fs::FileExt;
    let dir = PathBuf::from(std::env::var("ATLAS_CSF_CKPT_DIR").expect("ATLAS_CSF_CKPT_DIR"));
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let mut shards: Vec<String> = index["weight_map"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    shards.sort();
    shards.dedup();
    let mut asm = CsfAssembler::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let (mut scales, mut decode, mut bytes_out, mut io_bytes) =
        (0usize, Duration::ZERO, 0usize, 0usize);
    let (mut mx_tensors, mut mx_lo, mut mx_hi) = (0usize, u8::MAX, 0u8);
    for shard in &shards {
        let f = std::fs::File::open(dir.join(shard)).unwrap();
        let mut len = [0u8; 8];
        f.read_exact_at(&mut len, 0).unwrap();
        let hlen = u64::from_le_bytes(len) as usize;
        let mut h = vec![0u8; hlen];
        f.read_exact_at(&mut h, 8).unwrap();
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&h).unwrap();
        let read = |info: &serde_json::Value| -> (Vec<u8>, Vec<usize>) {
            let a = info["data_offsets"][0].as_u64().unwrap() as usize;
            let b = info["data_offsets"][1].as_u64().unwrap() as usize;
            let mut buf = vec![0u8; b - a];
            f.read_exact_at(&mut buf, (8 + hlen + a) as u64).unwrap();
            let shape = info["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            (buf, shape)
        };
        for (name, info) in &header {
            if let Some(shape) = info["shape"].as_array() {
                shapes.insert(
                    name.clone(),
                    shape.iter().map(|d| d.as_u64().unwrap() as usize).collect(),
                );
            }
            if csf_part(name).is_some() {
                let (buf, shape) = read(info);
                io_bytes += buf.len();
                let t = Instant::now();
                let done = asm
                    .offer(name, &shape, &buf)
                    .unwrap_or_else(|e| panic!("{e:#}"));
                decode += t.elapsed();
                if let Some(d) = done {
                    scales += 1;
                    bytes_out += d.bytes.len();
                }
            } else if name.ends_with(".weight_scale") && info["dtype"] == "U8" {
                let (buf, _) = read(info);
                mx_tensors += 1;
                mx_lo = mx_lo.min(*buf.iter().min().unwrap());
                mx_hi = mx_hi.max(*buf.iter().max().unwrap());
            }
        }
    }
    // Every decoded scale is the [n, k/16] its packed weight [n, k/2] needs.
    asm.finish(|n| shapes.get(n).cloned()).unwrap();
    eprintln!(
        "{scales} CSF scales: {:.2} GiB read -> {:.2} GiB decoded, decode {:.2} s single-threaded \
         ({:.2} GiB/s); {mx_tensors} U8 weight_scale tensors, exponents {mx_lo}..={mx_hi}",
        io_bytes as f64 / (1u64 << 30) as f64,
        bytes_out as f64 / (1u64 << 30) as f64,
        decode.as_secs_f64(),
        bytes_out as f64 / (1u64 << 30) as f64 / decode.as_secs_f64(),
    );
    assert!(scales > 0, "no CSF scales in {}", dir.display());
    assert!(mx_tensors == 0 || ((3..=246).contains(&mx_lo) && (3..=246).contains(&mx_hi)));
}

/// Every CSF scale of one real layer, decoded here and by the reference,
/// byte-identical: a dump from `scripts/nvfp4_csf_golden.py` (slab, vLLM's
/// reference) or `scripts/nvfp4_csf32_golden.py` (csf_32x2_v1, the LIL
/// FlashInfer reference), `--dump-layer` then `--reference`.
///
///   ATLAS_CSF_LAYER_DUMP=l23.csf cargo test -p spark-runtime --lib \
///     csf_layer_dump_matches_the_reference -- --ignored --nocapture
#[test]
#[ignore]
fn csf_layer_dump_matches_the_reference() {
    let dump = PathBuf::from(std::env::var("ATLAS_CSF_LAYER_DUMP").expect("ATLAS_CSF_LAYER_DUMP"));
    let raw = std::fs::read(&dump).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let index: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let data = &raw[8 + n..];
    let mut want: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&std::fs::read(format!("{}.ref.json", dump.display())).unwrap())
            .unwrap();
    let mut asm = CsfAssembler::new();
    // The dump holds the CSF components only: stand each weight in at the shape
    // its decoded scale implies, so `finish` checks the pairing alone.
    let mut weights: HashMap<String, Vec<usize>> = HashMap::new();
    let mut matched = 0;
    for (name, v) in &index {
        let shape: Vec<usize> = v[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let (off, len) = (
            v[2].as_u64().unwrap() as usize,
            v[3].as_u64().unwrap() as usize,
        );
        if let Some(d) = asm.offer(name, &shape, &data[off..off + len]).unwrap() {
            let expect = want
                .remove(&d.name)
                .unwrap_or_else(|| panic!("{} has no reference", d.name));
            assert_eq!(
                hex(&Sha256::digest(&d.bytes)),
                expect.as_str().unwrap(),
                "{}",
                d.name
            );
            let w = d.name.strip_suffix("_scale").unwrap().to_string();
            weights.insert(w, vec![d.shape[0], d.shape[1] * 8]);
            matched += 1;
        }
    }
    asm.finish(|n| weights.get(n).cloned()).unwrap();
    assert!(
        want.is_empty(),
        "reference decoded {} scales this did not",
        want.len()
    );
    eprintln!("{matched} scales byte-identical to the reference");
    assert!(matched > 0);
}
