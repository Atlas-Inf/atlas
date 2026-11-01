// SPDX-License-Identifier: AGPL-3.0-only
//! CPU policy latency only; not model throughput or streaming latency.
use atlas_tool_guard::Policy;
use std::{hint::black_box, time::Instant};
fn main() {
    let policy = Policy::default();
    for (name, args, iterations) in [
        ("small_allowed", r#"{"city":"Oslo"}"#.to_string(), 100_000),
        (
            "http_allowed",
            r#"{"url":"https://api.example/v1","body":{"message":"hello"}}"#.to_string(),
            100_000,
        ),
        (
            "metadata_blocked",
            r#"{"command":"curl http://169.254.169.254/latest/meta-data/"}"#.to_string(),
            100_000,
        ),
        (
            "8k_allowed",
            serde_json::json!({"content":"a".repeat(8192)}).to_string(),
            10_000,
        ),
        (
            "near_limit_allowed",
            serde_json::json!({"content":"a".repeat(1_000_000)}).to_string(),
            100,
        ),
    ] {
        for _ in 0..100 {
            let _ = black_box(policy.inspect(black_box(&args)));
        }
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(policy.inspect(black_box(&args)));
        }
        println!(
            "{}",
            serde_json::json!({"case":name,"bytes":args.len(),"iterations":iterations,"mean_us":start.elapsed().as_secs_f64()*1e6/iterations as f64})
        );
    }
}
