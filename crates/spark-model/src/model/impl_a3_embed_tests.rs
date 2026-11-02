// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::GpuBackend;

use crate::model::test_fixture as fixture;

#[test]
fn embedding_rejects_invalid_token_before_device_copy() {
    let f = fixture::Fixture::new(0, true, &[]);
    let output = f.gpu.alloc(256).unwrap();
    for token in [f.model.config.vocab_size as u32, u32::MAX] {
        let copies = f.gpu.d2d_count();
        let err = f.model.embed_ctx(&[], token, output, 0).unwrap_err();
        assert_eq!(f.gpu.d2d_count(), copies, "invalid token reached GPU copy");
        let message = err.to_string();
        assert!(message.contains(&format!("token {token}")), "{message}");
        assert!(message.contains("vocabulary 8"), "{message}");
    }
}

#[test]
fn embedding_copies_last_served_row_without_restricting_model_family() {
    let mut f = fixture::Fixture::new(0, true, &[]);
    let output = f.gpu.alloc(256).unwrap();
    let row: Vec<u8> = (0..256).map(|i| i as u8).collect();
    f.gpu
        .copy_h2d(&row, f.model.embed_tokens.weight.offset(7 * 256))
        .unwrap();
    for model_type in ["glm5_next", "qwen3_next", "gemma4"] {
        f.model.config.model_type = model_type.into();
        f.model.embed_ctx(&[], 7, output, 0).unwrap();
        let mut actual = vec![0; 256];
        f.gpu.copy_d2h(output, &mut actual).unwrap();
        assert_eq!(actual, row);
    }
}

#[test]
fn embedding_fused_rejects_bad_ids_and_short_context_before_upload() {
    let f = fixture::Fixture::new(0, true, &[]);
    let output = f.gpu.alloc(256).unwrap();
    for (tokens, len) in [(vec![8], 1), (vec![0], 2), (vec![u32::MAX], 1)] {
        let events = f.events();
        let launches = f.gpu.launches_snapshot().len();
        assert!(f.model.embed_tokens_fused(&tokens, len, output, 0).is_err());
        assert_eq!(f.events(), events, "invalid input reached GPU upload");
        assert_eq!(f.gpu.launches_snapshot().len(), launches);
    }
}

#[test]
fn embedding_rejects_row_geometry_overflow_before_device_copy() {
    let mut f = fixture::Fixture::new(0, true, &[]);
    let output = f.gpu.alloc(256).unwrap();
    let copies = f.gpu.d2d_count();
    for hidden_size in [0, usize::MAX, usize::MAX / 2] {
        f.model.config.hidden_size = hidden_size;
        assert!(f.model.embed_ctx(&[], 7, output, 0).is_err());
        assert_eq!(f.gpu.d2d_count(), copies);
    }
}
