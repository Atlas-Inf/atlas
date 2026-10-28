// SPDX-License-Identifier: AGPL-3.0-only

//! Deterministic, CPU-only screening of complete tool-call arguments.
//! Ported from Enntity/atlas fa5331b2 et seq. (AGPL-3.0-only).
//! No network, model inference, prompt scanning, or execution. This is a
//! defense-in-depth filter, not a sandbox or an arbitrary-program analyzer.
use serde_json::Value;
use std::collections::HashSet;

mod operation;
#[cfg(test)]
mod regressions;
mod scan;
#[cfg(test)]
mod tests;

/// Maximum serialized arguments per call, and aggregate pending stream bytes.
pub const DEFAULT_MAX_BYTES: usize = 1024 * 1024;
/// Stable wire marker. Details never contain the rejected arguments.
pub const GUARD_STOP: &str = "tool_call_guard";
pub const REFUSAL: &str = "[atlas] A tool call was blocked by the tool-call guard. No blocked call was released. Review the server policy before retrying.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    MetadataAccess,
    BlockedHost,
    CredentialTransfer,
    InvalidArguments,
    ArgumentsTooLarge,
    IncompleteCall,
}
impl BlockReason {
    pub fn code(self) -> &'static str {
        match self {
            Self::MetadataAccess => "metadata_access",
            Self::BlockedHost => "blocked_host",
            Self::CredentialTransfer => "credential_transfer",
            Self::InvalidArguments => "invalid_arguments",
            Self::ArgumentsTooLarge => "arguments_too_large",
            Self::IncompleteCall => "incomplete_call",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Policy {
    pub enabled: bool,
    pub max_bytes: usize,
    allow_hosts: HashSet<String>,
    deny_hosts: HashSet<String>,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_bytes: DEFAULT_MAX_BYTES,
            allow_hosts: HashSet::new(),
            deny_hosts: HashSet::new(),
        }
    }
}
impl Policy {
    /// Exact hosts only: no wildcard domains, URLs, ports, or implicit suffixes.
    /// Explicit denials and metadata rules outrank host exceptions.
    pub fn new(
        enabled: bool,
        allow: &[String],
        deny: &[String],
        max_bytes: usize,
    ) -> Result<Self, String> {
        if max_bytes == 0 || max_bytes > 64 * DEFAULT_MAX_BYTES {
            return Err("tool-guard-max-bytes must be between 1 and 67108864".into());
        }
        let hosts = |items: &[String]| -> Result<HashSet<String>, String> {
            items.iter().map(|s| scan::config_host(s)).collect()
        };
        Ok(Self {
            enabled,
            max_bytes,
            allow_hosts: hosts(allow)?,
            deny_hosts: hosts(deny)?,
        })
    }

    /// Inspect once per complete call; bounded before JSON parsing. JSON escapes
    /// are decoded by serde, and URL parsing uses the standard URL crate.
    pub fn inspect(&self, arguments: &str) -> Result<(), BlockReason> {
        if !self.enabled {
            return Ok(());
        }
        if arguments.len() > self.max_bytes {
            return Err(BlockReason::ArgumentsTooLarge);
        }
        let value: Value =
            serde_json::from_str(arguments).map_err(|_| BlockReason::InvalidArguments)?;
        if !value.is_object() {
            return Err(BlockReason::InvalidArguments);
        }
        self.inspect_operation(&value)
    }

    fn inspect_destination(&self, url: &url::Url) -> Result<bool, BlockReason> {
        let Some(host) = url.host_str() else {
            return Ok(false);
        };
        let normalized = host.trim_end_matches('.').to_ascii_lowercase();
        let host = normalized.as_str();
        if scan::is_metadata(host) {
            return Err(BlockReason::MetadataAccess);
        }
        if self.deny_hosts.contains(host) {
            return Err(BlockReason::BlockedHost);
        }
        if self.allow_hosts.contains(host) {
            return Ok(false);
        }
        if scan::BUILTIN_DENY_HOSTS.contains(&host) {
            return Err(BlockReason::BlockedHost);
        }
        if scan::sensitive_url(url) {
            return Err(BlockReason::CredentialTransfer);
        }
        Ok(true)
    }

    fn inspect_command(&self, command: &str) -> Result<(), BlockReason> {
        self.inspect_command_depth(command, 0)
    }

    fn inspect_command_depth(&self, command: &str, depth: usize) -> Result<(), BlockReason> {
        let decoded = scan::decode_percent(command);
        // Unwrap common literal shell -c/-lc forms, without evaluating code.
        if depth < 3
            && let Some(words) = shlex::split(&decoded)
            && words.first().is_some_and(|w| {
                matches!(w.rsplit('/').next(), Some("sh" | "bash" | "zsh" | "dash"))
            })
            && let Some(flag) = words
                .iter()
                .position(|w| matches!(w.as_str(), "-c" | "-lc"))
            && let Some(script) = words.get(flag + 1)
        {
            self.inspect_command_depth(script, depth + 1)?;
        }
        let lower = decoded.to_ascii_lowercase();
        let mut untrusted = false;
        for url in scan::urls(&lower) {
            untrusted |= self.inspect_destination(&url)?;
        }
        if untrusted
            && ((scan::is_transfer(&decoded) && scan::sensitive_command(&lower))
                || scan::private_key(&lower))
        {
            return Err(BlockReason::CredentialTransfer);
        }
        Ok(())
    }
}
