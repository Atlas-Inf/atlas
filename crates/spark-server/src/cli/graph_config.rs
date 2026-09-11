// SPDX-License-Identifier: AGPL-3.0-only

use super::ServeArgs;

pub(crate) fn parse_bucket_limits(raw: &str, flag: &str) -> Result<Vec<u32>, String> {
    let mut limits = Vec::new();
    for value in raw.split(',') {
        let value = value.trim();
        let parsed = value
            .parse::<u32>()
            .map_err(|_| format!("{flag} contains non-positive integer {value:?}"))?;
        if parsed == 0 {
            return Err(format!("{flag} bucket limits must be greater than zero"));
        }
        if limits.last().is_some_and(|previous| *previous >= parsed) {
            return Err(format!(
                "{flag} bucket limits must be strictly increasing without duplicates"
            ));
        }
        limits.push(parsed);
    }
    if limits.is_empty() {
        return Err(format!("{flag} must contain at least one bucket limit"));
    }
    Ok(limits)
}

pub(crate) fn validate_graph_config(args: &ServeArgs) -> Result<(), String> {
    parse_bucket_limits(&args.cuda_graph_token_buckets, "--cuda-graph-token-buckets")?;
    parse_bucket_limits(
        &args.cuda_graph_request_buckets,
        "--cuda-graph-request-buckets",
    )?;
    if args.cuda_graph_mode != "disabled" && args.cuda_graph_cache_entries == 0 {
        return Err("--cuda-graph-cache-entries must be greater than zero".to_string());
    }
    if args.cuda_graph_mode != "disabled" && args.cuda_graph_cache_mb == 0 {
        return Err("--cuda-graph-cache-mb must be greater than zero".to_string());
    }
    if args.cuda_graph_mode == "disabled" && args.cuda_graph_prewarm_profile.is_some() {
        return Err("--cuda-graph-prewarm-profile requires an enabled graph mode".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_bucket_limits;

    #[test]
    fn bucket_limits_are_positive_and_strictly_increasing() {
        assert_eq!(parse_bucket_limits("1, 4,16", "--x").unwrap(), [1, 4, 16]);
        for invalid in ["", "0", "1,1", "4,2", "1,nope"] {
            assert!(parse_bucket_limits(invalid, "--x").is_err(), "{invalid}");
        }
    }
}
