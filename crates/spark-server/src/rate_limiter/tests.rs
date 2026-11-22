// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn cfg(rpm: u64, tpm: u64) -> RateLimitConfig {
    RateLimitConfig {
        rpm,
        tpm,
        burst_rpm: rpm.max(1),
        burst_tpm: tpm.max(1),
    }
}

#[test]
fn disabled_limiter_always_allows() {
    let lim = RateLimiter::with_config(cfg(0, 0));
    for _ in 0..100 {
        let d = lim.admit("k", 10_000);
        assert!(d.allowed);
    }
}

#[test]
fn request_bucket_blocks_after_burst() {
    let lim = RateLimiter::with_config(cfg(2, 0));
    assert!(lim.admit("k", 0).allowed);
    assert!(lim.admit("k", 0).allowed);
    let d = lim.admit("k", 0);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Requests)));
    assert!(d.retry_after_secs >= 1);
}

#[test]
fn token_bucket_blocks_on_oversized_request() {
    let lim = RateLimiter::with_config(cfg(0, 1000));
    assert!(lim.admit("k", 600).allowed);
    // 600 consumed, 400 left; next 500 should deny.
    let d = lim.admit("k", 500);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Tokens)));
}

#[test]
fn keys_do_not_interfere() {
    let lim = RateLimiter::with_config(cfg(1, 0));
    assert!(lim.admit("a", 0).allowed);
    assert!(lim.admit("b", 0).allowed);
    assert!(!lim.admit("a", 0).allowed);
    assert!(!lim.admit("b", 0).allowed);
}

#[test]
fn identity_prefers_bearer_over_xff() {
    use axum::http::{HeaderMap, HeaderValue, header};
    let mut h = HeaderMap::new();
    h.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer sk-abc"),
    );
    h.insert("x-forwarded-for", HeaderValue::from_static("1.2.3.4"));
    let id = extract_identity(&h, None);
    assert!(id.starts_with("bearer:"));
}

#[test]
fn identity_falls_back_to_xff_then_peer() {
    use axum::http::{HeaderMap, HeaderValue};
    let mut h = HeaderMap::new();
    h.insert(
        "x-forwarded-for",
        HeaderValue::from_static("9.9.9.9, 1.1.1.1"),
    );
    assert_eq!(extract_identity(&h, None), "xff:9.9.9.9");
    let empty = HeaderMap::new();
    let peer: std::net::SocketAddr = "127.0.0.1:9000".parse().unwrap();
    assert_eq!(extract_identity(&empty, Some(peer)), "peer:127.0.0.1");
}

#[test]
fn token_refund_restores_budget() {
    let lim = RateLimiter::with_config(cfg(0, 1000));
    assert!(lim.admit("k", 900).allowed);
    lim.refund_tokens("k", 500);
    assert!(lim.admit("k", 500).allowed);
}

#[test]
fn identity_map_stops_growing_at_cap_when_every_key_is_active() {
    let lim = RateLimiter::with_config(cfg(60, 0));
    for i in 0..MAX_KEYS {
        assert!(lim.admit(&format!("xff:{i}"), 0).allowed);
    }
    assert_eq!(lim.inner.lock().keys.len(), MAX_KEYS);

    for i in 0..10 {
        let d = lim.admit(&format!("xff:new-{i}"), 0);
        assert!(d.allowed, "a new identity past the cap fails open");
    }
    assert_eq!(lim.inner.lock().keys.len(), MAX_KEYS);
}

#[test]
fn identities_past_the_cap_share_one_budget_without_rescanning() {
    let lim = RateLimiter::with_config(cfg(1, 0));
    for i in 0..MAX_KEYS {
        assert!(lim.admit(&format!("xff:{i}"), 0).allowed);
    }

    assert!(lim.admit("xff:new-a", 0).allowed);
    assert!(!lim.admit("xff:new-a", 0).allowed);
    assert!(!lim.admit("xff:new-b", 0).allowed);

    let map = lim.inner.lock();
    assert_eq!(map.keys.len(), MAX_KEYS);
    assert!(map.next_evictable > Instant::now());
}

fn retry_after_is_wait_until_next_request_fits() {
    let lim = RateLimiter::with_config(cfg(60, 0));
    for _ in 0..60 {
        assert!(lim.admit("k", 0).allowed);
    }
    let d = lim.admit("k", 0);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Requests)));
    assert_eq!(d.retry_after_secs, 1);
    assert_eq!(d.requests.reset_secs, 60);
}

#[test]
fn retry_after_is_wait_until_token_cost_fits() {
    // 6000 TPM refills 100 tokens/s; 200 more tokens fit after 2s.
    let lim = RateLimiter::with_config(cfg(0, 6000));
    assert!(lim.admit("k", 6000).allowed);
    let d = lim.admit("k", 200);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Tokens)));
    assert_eq!(d.retry_after_secs, 2);
    assert_eq!(d.tokens.reset_secs, 60);
}

#[test]
fn retry_after_waits_for_both_buckets_when_both_are_short() {
    // 60 requests x 100 tokens drains both buckets. The 61st request fits
    // the request bucket after 1s, but its 200 tokens need 2s at 100/s.
    let lim = RateLimiter::with_config(cfg(60, 6000));
    for _ in 0..60 {
        assert!(lim.admit("k", 100).allowed);
    }
    let d = lim.admit("k", 200);
    assert!(!d.allowed);
    assert!(matches!(d.denied_by, Some(DenialReason::Requests)));
    assert_eq!(d.retry_after_secs, 2);
}
