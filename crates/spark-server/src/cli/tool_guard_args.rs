// SPDX-License-Identifier: AGPL-3.0-only

/// Operator-only configuration. No request-body or model override.
/// Ported from Enntity/atlas fa5331b2 et seq. (AGPL-3.0-only).
#[derive(clap::Args, Debug, Clone, PartialEq)]
pub struct ToolGuardArgs {
    /// Screen complete tool calls before release (off by default here;
    /// SparkQwen ships it on — enable with `--tool-guard on`).
    #[arg(long, default_value = "off", value_parser = ["on", "off"])]
    pub tool_guard: String,
    /// Exact host exception for callback/credential-transfer checks; repeatable.
    /// Does not override metadata protection or an explicit denied host.
    #[arg(long, value_delimiter = ',')]
    pub tool_guard_allow_host: Vec<String>,
    /// Additional exact blocked host; repeatable, comma-separated accepted.
    #[arg(long, value_delimiter = ',')]
    pub tool_guard_deny_host: Vec<String>,
    /// Maximum argument bytes per call and aggregate pending calls.
    #[arg(long, default_value_t = atlas_tool_guard::DEFAULT_MAX_BYTES)]
    pub tool_guard_max_bytes: usize,
}
impl ToolGuardArgs {
    pub fn policy(&self) -> Result<atlas_tool_guard::Policy, String> {
        atlas_tool_guard::Policy::new(
            self.tool_guard == "on",
            &self.tool_guard_allow_host,
            &self.tool_guard_deny_host,
            self.tool_guard_max_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::{Cli, Command};
    use clap::Parser;
    #[test]
    fn tool_guard_default_and_operator_configuration() {
        for (extra, enabled) in [
            (vec![], false),
            (vec!["--tool-guard", "off"], false),
            (vec!["--tool-guard", "on"], true),
        ] {
            let mut args = vec!["spark", "serve", "fixture"];
            args.extend(extra);
            let Command::Serve(s) = Cli::try_parse_from(args).unwrap().command else {
                panic!()
            };
            assert_eq!(s.tool_guard.policy().unwrap().enabled, enabled);
        }
        assert!(
            Cli::try_parse_from(["spark", "serve", "fixture", "--tool-guard", "typo"]).is_err()
        );
        let Command::Serve(s) = Cli::try_parse_from([
            "spark",
            "serve",
            "fixture",
            "--tool-guard-allow-host",
            "https://bad.example/path",
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        assert!(s.tool_guard.policy().is_err());
    }
}
