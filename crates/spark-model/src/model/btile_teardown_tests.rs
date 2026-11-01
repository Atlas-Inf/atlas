// SPDX-License-Identifier: AGPL-3.0-only
//! Actual model teardown invalidates its real published FFN before owner free.
use super::*;
use crate::{layer::ForwardContext, layers::FfnComponent, traits::Model};

#[test]
fn actual_model_teardown_invalidates_published_readers_and_frees_originals_once() {
    let fixture = crate::layers::moe::btile_model_fixture();
    let mut model = fixture.model;
    model.teardown().unwrap();
    for ptr in fixture.originals {
        assert_eq!(fixture.history.frees(Some(ptr)), 1, "owner {ptr:?}");
    }
    fixture.history.clear();
    let ctx = ForwardContext {
        buffers: &model.buffers,
        gpu: model.gpu.as_ref(),
        config: &model.config,
        dispatch: &model.dispatch,
        derived: &model.derived,
        levers: &model.levers,
        stats: &model.stats,
        ssm_batch: None,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Skip,
    };
    let attention = model.layers[0]
        .as_any_mut()
        .unwrap()
        .downcast_mut::<crate::layers::qwen3_attention::Qwen3AttentionLayer>()
        .unwrap();
    let (_, ffn) = attention.glm_shared_cache_ffn();
    let FfnComponent::Moe(moe) = ffn else {
        panic!("actual published MoE");
    };
    let error = moe
        .forward(ctx.buffers.norm_output(), &ctx, 91)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("construction incomplete or failed"),
        "{error:#}"
    );
    assert!(fixture.history.is_empty(), "stale reader did GPU work");
    model.teardown().unwrap();
    assert_eq!(
        fixture.history.frees(None),
        0,
        "repeat teardown frees again"
    );
}
