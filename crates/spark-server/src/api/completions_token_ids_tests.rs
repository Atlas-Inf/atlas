// SPDX-License-Identifier: AGPL-3.0-only

//! `return_token_ids` on `/v1/completions`.

use super::*;

fn response(tokens: Vec<u32>) -> super::super::inference_types::InferenceResponse {
    super::super::inference_types::InferenceResponse {
        output_tokens: tokens,
        finish_reason: "length".into(),
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        logprobs: Vec::new(),
        reasoning_tokens: 0,
        cached_prompt_tokens: 0,
        accepted_prediction_tokens: 0,
        prompt_logprobs: Vec::new(),
        guard_stop: None,
    }
}

#[test]
fn completion_token_ids_preserve_generated_ids_before_text_stripping() {
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "prompt": [41, 42], "echo": true, "return_token_ids": true
    }))
    .expect("request");
    // Text may be echoed and strip a special token; IDs are precisely the
    // generated sequence, including special and byte-piece token IDs.
    let choice = assemble_choice(
        &req,
        3,
        "echoed promptanswer".into(),
        None,
        response(vec![154842, 281, 512]),
    );
    let wire = serde_json::to_value(choice).expect("serialize");
    assert_eq!(wire["token_ids"], serde_json::json!([154842, 281, 512]));
    assert_eq!(wire["index"], 3);
    assert_eq!(wire["text"], "echoed promptanswer");
}

#[test]
fn completion_token_ids_opt_in_preserves_empty_and_omits_default() {
    for requested in [false, true] {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "hello", "return_token_ids": requested
        }))
        .expect("request");
        let choice = assemble_choice(&req, 0, String::new(), None, response(Vec::new()));
        let wire = serde_json::to_value(choice).expect("serialize");
        assert_eq!(wire.get("token_ids").is_some(), requested);
        if requested {
            assert_eq!(wire["token_ids"], serde_json::json!([]));
        }
    }
}
