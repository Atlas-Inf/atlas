// SPDX-License-Identifier: AGPL-3.0-only

//! Chat session key resolution (#209).
//!
//! `session_hash` groups SSM snapshots by conversation for eviction, the tail
//! lease and the tail gate. The old key, a hash of the first 1024 prompt tokens,
//! is one value for every request sharing a tool set: the chat template renders
//! the tool JSON first, so hours of separate agent conversations collapsed into
//! one "session". The key is now picked in priority order:
//!
//! 1. a client conversation id: an HTTP header (`x-session-id`,
//!    `x-session-affinity`, `x-conversation-id`), the Responses `conversation`,
//!    OpenAI `prompt_cache_key`, Anthropic `metadata.user_id` (Claude Code
//!    carries its session uuid there) or OpenAI `metadata.session_id` /
//!    `metadata.conversation_id`;
//! 2. OpenAI `user`, folded with the first user turn: `user` is an end-user
//!    id, so on its own it would merge all of one user's conversations;
//! 3. the messages through the end of the first user turn: the earliest span
//!    that differs between agent conversations sharing a system + tools
//!    preamble, and fixed as the conversation grows;
//! 4. the first-1024-token hash (no user message at all).
//!
//! `previous_response_id` is NOT a key: it names the prior turn, so it changes
//! every turn and would split each conversation into one session per turn.
//!
//! TP=2: the key is computed on the head from request data only, with the
//! fixed-key `DefaultHasher::new()` (no per-process `RandomState`). It is not
//! broadcast; worker-rank sequences keep `session_hash = 0`, exactly as before.

use std::hash::{Hash, Hasher};

use axum::http::HeaderMap;

use crate::ir::{ChatRequest, ContentPart, ImageData, Message, Role};

/// Headers carrying a client conversation id, checked in order.
const SESSION_HEADERS: [&str; 3] = ["x-session-id", "x-session-affinity", "x-conversation-id"];

/// Bytes of first-user-turn content hashed at most. Past this a giant pasted
/// document or inline image adds cost, not distinctness.
const FIRST_TURN_MAX_BYTES: usize = 1 << 20;

/// Where a session key came from (the `source=` of the key log line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKeySource {
    Header,
    Conversation,
    PromptCacheKey,
    Metadata,
    User,
    FirstUserTurn,
    Prefix1024,
}

impl SessionKeySource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Header => "header",
            Self::Conversation => "conversation",
            Self::PromptCacheKey => "prompt_cache_key",
            Self::Metadata => "metadata",
            Self::User => "user",
            Self::FirstUserTurn => "first_user_turn",
            Self::Prefix1024 => "prefix1024",
        }
    }
}

/// A client-provided conversation id, resolved at the API edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientSessionId {
    pub source: SessionKeySource,
    pub id: String,
}

impl ClientSessionId {
    /// `None` for an absent or blank id: an empty string is no conversation.
    pub fn new(source: SessionKeySource, id: Option<&str>) -> Option<Self> {
        let id = id?.trim();
        (!id.is_empty()).then(|| Self {
            source,
            id: id.to_string(),
        })
    }
}

/// The first non-blank session header.
pub fn from_headers(headers: &HeaderMap) -> Option<ClientSessionId> {
    SESSION_HEADERS.iter().find_map(|name| {
        let v = headers.get(*name)?.to_str().ok();
        ClientSessionId::new(SessionKeySource::Header, v)
    })
}

/// The OpenAI chat body's conversation id: `prompt_cache_key`, then
/// `metadata.session_id` / `metadata.conversation_id`, then `user`.
pub fn from_openai_body(
    prompt_cache_key: Option<&str>,
    metadata: Option<&std::collections::HashMap<String, String>>,
    user: Option<&str>,
) -> Option<ClientSessionId> {
    let meta = |k: &str| metadata.and_then(|m| m.get(k)).map(String::as_str);
    ClientSessionId::new(SessionKeySource::PromptCacheKey, prompt_cache_key)
        .or_else(|| ClientSessionId::new(SessionKeySource::Metadata, meta("session_id")))
        .or_else(|| ClientSessionId::new(SessionKeySource::Metadata, meta("conversation_id")))
        .or_else(|| ClientSessionId::new(SessionKeySource::User, user))
}

/// Replace the body-derived id with a stronger edge one (a header, the
/// Responses `conversation`) when there is one.
pub fn with_override(mut req: ChatRequest, id: Option<ClientSessionId>) -> ChatRequest {
    if id.is_some() {
        req.client_session = id;
    }
    req
}

/// `session_hash` 0 means "untracked" downstream, so a real key never is 0.
fn nonzero(h: u64) -> u64 {
    if h == 0 { 1 } else { h }
}

fn tagged_hasher(tag: &[u8]) -> std::collections::hash_map::DefaultHasher {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(tag);
    h.write_u8(0);
    h
}

/// Hash of a client conversation id. The source is not hashed: one id is one
/// conversation whichever channel carried it.
pub fn hash_client_id(id: &str) -> u64 {
    let mut h = tagged_hasher(b"atlas.session.client.v1");
    id.hash(&mut h);
    nonzero(h.finish())
}

/// Hash of every message through the end of the first user message (role and
/// content, length-prefixed), or `None` when there is no user message.
pub fn first_user_turn_hash(messages: &[Message]) -> Option<u64> {
    let end = messages.iter().position(|m| m.role == Role::User)?;
    let mut h = tagged_hasher(b"atlas.session.first_user_turn.v1");
    let mut budget = FIRST_TURN_MAX_BYTES;
    let mut feed = |h: &mut std::collections::hash_map::DefaultHasher, bytes: &[u8]| {
        let take = bytes.len().min(budget);
        budget -= take;
        h.write_usize(take);
        h.write(&bytes[..take]);
    };
    for m in &messages[..=end] {
        feed(&mut h, m.role.as_wire().as_bytes());
        for part in &m.content {
            let (tag, data) = match part {
                ContentPart::Text(t) => (b'T', t.as_str()),
                ContentPart::Image(s) => (b'I', image_data_str(&s.data)),
                ContentPart::Video(s) => (b'V', image_data_str(&s.data)),
            };
            h.write_u8(tag);
            feed(&mut h, data.as_bytes());
        }
    }
    Some(nonzero(h.finish()))
}

fn image_data_str(d: &ImageData) -> &str {
    match d {
        ImageData::Base64(s) | ImageData::Url(s) => s,
    }
}

/// The key from request data alone, or `None` when only the token-prefix
/// fallback is left.
pub fn resolve_from_request(req: &ChatRequest) -> Option<(SessionKeySource, u64)> {
    let first_turn = first_user_turn_hash(&req.messages);
    match &req.client_session {
        Some(c) if c.source == SessionKeySource::User => {
            let mut h = tagged_hasher(b"atlas.session.user.v1");
            c.id.hash(&mut h);
            first_turn.unwrap_or(0).hash(&mut h);
            Some((SessionKeySource::User, nonzero(h.finish())))
        }
        Some(c) => Some((c.source, hash_client_id(&c.id))),
        None => first_turn.map(|h| (SessionKeySource::FirstUserTurn, h)),
    }
}

/// The session hash: `resolved` (from [`resolve_from_request`]), else the
/// first-1024-token hash. Logs the chosen key.
pub fn chat_session_hash(resolved: Option<(SessionKeySource, u64)>, prompt_tokens: &[u32]) -> u64 {
    let (source, hash) = resolved.unwrap_or_else(|| {
        (
            SessionKeySource::Prefix1024,
            crate::session_manager::compute_session_hash(prompt_tokens),
        )
    });
    tracing::info!("session key: source={} hash={hash:#x}", source.as_str());
    hash
}

#[cfg(test)]
#[path = "session_key_tests.rs"]
mod tests;
