// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// MLA layer 3 builds on both TP2 ranks from FP8_PB_WO q_a/q_b/kv_a/o and
/// BF16 kv_b/indexer (the expansions themselves are checked bit for bit
/// above). TP sharding retires the original BF16 kv_b; expanded cleanup
/// subsequently releases only the FP8 sources and preserves live BF16/indexer data.
#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn mla_layer_3_builds_on_both_tp2_ranks() -> Result<()> {
    let gpu = backend()?;
    let a = |x: &str| l(3, &format!("self_attn.{x}"));
    let fp8 = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"].map(a);
    let mut names = with_parts(&fp8, &["weight", "weight_scale"]);
    for x in [
        "kv_b_proj.weight",
        "q_a_layernorm.weight",
        "kv_a_layernorm.weight",
        "indexer.wq_b.weight",
        "indexer.wk.weight",
        "indexer.weights_proj.weight",
        "indexer.index_kpool_compress_gate",
        "indexer.index_kpool_compress_ape",
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
    ] {
        names.push(a(x));
    }
    names.push(l(3, "input_layernorm.weight"));
    names.push(l(3, "post_attention_layernorm.weight"));
    for rank in 0..2 {
        let config = tp2_rank(rank);
        let (store, host) = upload(gpu, &names)?;
        let kv_b = a("kv_b_proj.weight");
        ensure!(!store.was_reclaimed(&kv_b), "fresh kv_b already retired");
        let lp = config.layer_prefix(3);
        let norm = |x: &str| norm_bf16(&store, &format!("{lp}.{x}"), gpu);
        let _layer = crate::weight_loader::glm5::layers::load_mla_layer(
            &store,
            &lp,
            3,
            0,
            norm("input_layernorm.weight")?,
            norm("post_attention_layernorm.weight")?,
            crate::layers::FfnComponent::None,
            None,
            &config,
            gpu,
            spark_runtime::kv_cache::KvCacheDtype::Bf16,
            false,
            None,
        )?;
        ensure!(
            store.was_reclaimed(&kv_b),
            "TP2 layer construction must retire the original BF16 kv_b"
        );
        let mut live = Vec::new();
        for name in &names {
            if name == &kv_b {
                continue;
            }
            let tensor = store.get(name)?;
            if tensor.dtype != WeightDtype::BF16 && !name.contains(".indexer.") {
                continue;
            }
            ensure!(!store.was_reclaimed(name), "live source retired: {name}");
            let mut bytes = vec![0; tensor.byte_size()];
            gpu.copy_d2h(tensor.ptr, &mut bytes)?;
            ensure!(bytes == host[name].2, "layer changed live source: {name}");
            live.push((name, tensor.ptr, bytes));
        }
        ensure!(live.iter().any(|(name, _, _)| name.contains(".indexer.")));
        let mut released = 0;
        for stem in decoder_expanded_stems(&lp, LayerType::FullAttention, false) {
            released += release_expanded(&store, gpu, &stem, false)?;
        }
        let want: usize = with_parts(&fp8, &["weight", "weight_scale"])
            .iter()
            .map(|n| host[n].2.len())
            .sum();
        ensure!(
            released == want,
            "rank {rank}: released {released} of {want} bytes"
        );
        ensure!(store.was_reclaimed(&kv_b), "kv_b retirement was lost");
        for (name, ptr, before) in live {
            ensure!(
                !store.was_reclaimed(name),
                "expanded cleanup retired {name}"
            );
            let tensor = store.get(name)?;
            ensure!(tensor.ptr == ptr, "expanded cleanup replaced {name}");
            let mut after = vec![0; before.len()];
            gpu.copy_d2h(ptr, &mut after)?;
            ensure!(after == before, "expanded cleanup changed {name}");
        }
        eprintln!("rank {rank}: MLA layer 3 built, {released} FP8 source bytes released");
    }
    Ok(())
}
