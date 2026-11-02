// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use serde_json::json;

/// A system + tools preamble long enough that its tokens alone would fill the
/// old first-1024-token window.
fn preamble() -> String {
    "You are an agent. Use the tools. ".repeat(400)
}

fn openai(body: serde_json::Value) -> ChatRequest {
    serde_json::from_value::<crate::openai::ChatCompletionRequest>(body)
        .expect("valid chat request")
        .into()
}

fn chat(messages: serde_json::Value, extra: serde_json::Value) -> ChatRequest {
    let mut body = json!({"model": "m", "messages": messages});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    openai(body)
}

fn convo(first_user: &str) -> serde_json::Value {
    json!([
        {"role": "system", "content": preamble()},
        {"role": "user", "content": first_user},
    ])
}

fn key(req: &ChatRequest) -> (SessionKeySource, u64) {
    resolve_from_request(req).expect("request-derived key")
}

#[test]
fn same_client_id_same_hash() {
    let a = chat(convo("fix the bug"), json!({"prompt_cache_key": "conv-1"}));
    let b = chat(convo("write docs"), json!({"prompt_cache_key": "conv-1"}));
    assert_eq!(key(&a), key(&b));
    assert_eq!(key(&a).0, SessionKeySource::PromptCacheKey);
    assert_eq!(hash_client_id("conv-1"), hash_client_id("conv-1"));
}

#[test]
fn distinct_client_ids_identical_tokens_distinct_hashes() {
    let a = chat(convo("same"), json!({"prompt_cache_key": "conv-1"}));
    let b = chat(convo("same"), json!({"prompt_cache_key": "conv-2"}));
    assert_ne!(key(&a).1, key(&b).1);
}

#[test]
fn client_id_hash_is_fixed_across_processes() {
    // DefaultHasher::new() is SipHash-1-3 with fixed zero keys, so the value is
    // the same in every process. A per-process RandomState would break this pin.
    assert_eq!(hash_client_id("conv-1"), 0x5e76_3b7f_3fdb_2a8a);
}

#[test]
fn header_overrides_body() {
    let mut headers = HeaderMap::new();
    headers.insert("x-session-affinity", "  sess-9 ".parse().unwrap());
    let id = from_headers(&headers).unwrap();
    assert_eq!(id.source, SessionKeySource::Header);
    assert_eq!(id.id, "sess-9");
    let req = with_override(
        chat(convo("x"), json!({"prompt_cache_key": "conv-1"})),
        Some(id),
    );
    assert_eq!(
        key(&req),
        (SessionKeySource::Header, hash_client_id("sess-9"))
    );

    // A blank header is no id: the body key stands.
    let mut blank = HeaderMap::new();
    blank.insert("x-session-id", " ".parse().unwrap());
    let req = with_override(
        chat(convo("x"), json!({"prompt_cache_key": "c"})),
        from_headers(&blank),
    );
    assert_eq!(key(&req).0, SessionKeySource::PromptCacheKey);
}

#[test]
fn openai_metadata_session_id() {
    let req = chat(convo("x"), json!({"metadata": {"session_id": "s-1"}}));
    assert_eq!(
        key(&req),
        (SessionKeySource::Metadata, hash_client_id("s-1"))
    );
}

#[test]
fn user_alone_does_not_merge_a_users_conversations() {
    let a = chat(convo("task one"), json!({"user": "alice"}));
    let b = chat(convo("task two"), json!({"user": "alice"}));
    let c = chat(convo("task one"), json!({"user": "bob"}));
    assert_eq!(key(&a).0, SessionKeySource::User);
    assert_ne!(key(&a).1, key(&b).1);
    assert_ne!(key(&a).1, key(&c).1);
}

#[test]
fn first_user_turn_separates_conversations_sharing_a_preamble() {
    let a = chat(convo("fix the parser"), json!({}));
    let b = chat(convo("fix the lexer"), json!({}));
    assert_eq!(key(&a).0, SessionKeySource::FirstUserTurn);
    assert_ne!(key(&a).1, key(&b).1);
}

#[test]
fn first_user_turn_stable_as_conversation_grows() {
    let turn1 = chat(convo("fix the parser"), json!({}));
    let mut msgs = convo("fix the parser");
    msgs.as_array_mut().unwrap().extend([
        json!({"role": "assistant", "content": "Looking."}),
        json!({"role": "user", "content": "now run the tests"}),
        json!({"role": "assistant", "content": "Done."}),
        json!({"role": "user", "content": "commit it"}),
    ]);
    let turn3 = chat(msgs, json!({}));
    assert_eq!(key(&turn1), key(&turn3));
}

#[test]
fn first_user_turn_respects_part_boundaries() {
    // "ab" + "c" must not equal "a" + "bc" across the system/user split.
    let a = chat(
        json!([{"role": "system", "content": "ab"}, {"role": "user", "content": "c"}]),
        json!({}),
    );
    let b = chat(
        json!([{"role": "system", "content": "a"}, {"role": "user", "content": "bc"}]),
        json!({}),
    );
    assert_ne!(key(&a).1, key(&b).1);
}

#[test]
fn no_user_message_falls_back_to_prefix1024() {
    let req = chat(
        json!([{"role": "system", "content": "only system"}]),
        json!({}),
    );
    assert!(resolve_from_request(&req).is_none());
    let tokens: Vec<u32> = (0..2000).collect();
    let h = chat_session_hash(resolve_from_request(&req), &tokens);
    assert_eq!(h, crate::session_manager::compute_session_hash(&tokens));
    // A request-derived key wins over the token fallback.
    let req = chat(convo("x"), json!({}));
    assert_eq!(
        chat_session_hash(resolve_from_request(&req), &tokens),
        key(&req).1
    );
}
