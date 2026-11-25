// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use atlas_core::config::parse_config;
use sha2::{Digest, Sha256};

use super::{Pack, Tensor, Vision, check_headers, manifest};

/// deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json, revision 1bae0e10. The
/// NVFP4 and EXL3 packs change only its `quantization_config`.
const PUBLISHED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_data/deepseek_v41_flash_1017_config.json"
));

/// The checkpoint's `vision_config`.
const VISION: Vision = Vision {
    layers: 32,
    width: 1024,
    intermediate: 2816,
    patch: 14,
    downsample: 3,
};

fn published(pack: Pack) -> Vec<Tensor> {
    manifest(&parse_config(PUBLISHED).unwrap(), Some(&VISION), pack).unwrap()
}

fn headers(tensors: &[Tensor]) -> HashMap<String, (String, Vec<usize>)> {
    tensors
        .iter()
        .map(|t| (t.name.clone(), (t.dtype.to_string(), t.shape.clone())))
        .collect()
}

/// Tensor count and SHA-256 of the sorted `name dtype shape` lines of each
/// checkpoint's safetensors headers, read from the Hub on 2026-11-14 with a
/// trellis's last dimension written as 0:
/// deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e10,
/// nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 @ 480550d3,
/// bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw @ b8054f70.
#[test]
fn the_manifest_is_each_published_checkpoint_exactly() {
    for (pack, count, digest) in [
        (
            Pack::Release,
            96_085,
            "493e9f0cfd0c7abbfa781311c0c40120f36e6dcaf4a022035dadff996aa4d6ee",
        ),
        (
            Pack::Nvfp4,
            188_245,
            "8462a14dc2d62bc95bc80f5e65389d56056590aade49824025d29bf6cbf375b1",
        ),
        (
            Pack::Exl3,
            192_496,
            "26783cd1fc5df8598a6570b0a2eb2ac53b786a745feefda21f4ec0dbc789ddcf",
        ),
    ] {
        let mut tensors = published(pack);
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(tensors.len(), count, "{pack:?}");
        let mut hash = Sha256::new();
        for t in &tensors {
            hash.update(format!("{} {} {:?}\n", t.name, t.dtype, t.shape));
        }
        let hash: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hash, digest, "{pack:?}");
    }
}

#[test]
fn a_checkpoint_that_is_its_manifest_passes() {
    for pack in [Pack::Release, Pack::Nvfp4, Pack::Exl3] {
        let expected = published(pack);
        let mut found = headers(&expected);
        // The quantizer picks each trellis's bits per weight.
        for (name, (_, shape)) in found.iter_mut() {
            if name.ends_with(".trellis") {
                *shape.last_mut().unwrap() = 40;
            }
        }
        check_headers(&expected, &found).unwrap();
    }
}

#[test]
fn a_missing_a_changed_and_an_unexpected_tensor_are_each_named() {
    let expected = published(Pack::Release);
    let mut found = headers(&expected);
    found.remove("layers.14.engram.embed.scale");
    // 128-wide scale blocks, as DeepSeek-V4 has them.
    found.get_mut("layers.0.attn.wkv.scale").unwrap().1 = vec![4, 40];
    found.get_mut("layers.3.ffn.experts.7.w1.weight").unwrap().0 = "U8".into();
    found.insert(
        "layers.0.attn.compressor.ape".into(),
        ("F32".into(), vec![2, 512]),
    );

    let error = check_headers(&expected, &found).unwrap_err().to_string();
    for part in [
        "1 missing [layers.14.engram.embed.scale]",
        "2 differ [layers.0.attn.wkv.scale is F8_E8M0 [4, 40], expected F8_E8M0 [16, 160]; ",
        "layers.3.ffn.experts.7.w1.weight is U8 [2304, 2560], expected I8 [2304, 2560]",
        "1 unexpected [layers.0.attn.compressor.ape]",
    ] {
        assert!(error.contains(part), "{part}\n{error}");
    }
}

#[test]
fn one_pack_is_not_another() {
    let release = headers(&published(Pack::Release));
    assert!(check_headers(&published(Pack::Nvfp4), &release).is_err());
    assert!(check_headers(&published(Pack::Exl3), &release).is_err());
}

/// Each pack's real headers, whole: `ATLAS_DSV41_HEADERS` names a directory
/// holding `release.json`, `nvfp4.json` and `exl3.json`, each a map of
/// tensor name to `[dtype, shape]`.
#[test]
fn a_config_whose_per_layer_lists_are_short_is_refused() {
    let mut config = parse_config(PUBLISHED).unwrap();
    config.engram_num_embeddings.pop();
    assert!(manifest(&config, None, Pack::Release).is_err());
    let mut config = parse_config(PUBLISHED).unwrap();
    config.compress_ratios.truncate(2);
    assert!(manifest(&config, None, Pack::Release).is_err());
}

#[test]
#[ignore = "needs ATLAS_DSV41_HEADERS=<directory of header dumps>"]
fn the_published_headers_pass_the_check() {
    let dir = std::env::var("ATLAS_DSV41_HEADERS").expect("ATLAS_DSV41_HEADERS");
    for (pack, file) in [
        (Pack::Release, "release.json"),
        (Pack::Nvfp4, "nvfp4.json"),
        (Pack::Exl3, "exl3.json"),
    ] {
        let text = std::fs::read_to_string(format!("{dir}/{file}")).expect(file);
        let found: HashMap<String, (String, Vec<usize>)> = serde_json::from_str(&text).expect(file);
        check_headers(&published(pack), &found).unwrap_or_else(|e| panic!("{pack:?}: {e}"));
    }
}
