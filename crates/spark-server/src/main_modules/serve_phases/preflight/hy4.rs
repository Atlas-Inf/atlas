// SPDX-License-Identifier: AGPL-3.0-only

use super::{Result, cli};

pub(crate) fn validate(args: &cli::ServeArgs) -> Result<()> {
    anyhow::ensure!(
        !args.self_speculative && !args.speculative && !args.dflash,
        "HY4 supports only explicit --ngram-speculative prompt lookup, not MTP/DFly/self-speculation"
    );
    anyhow::ensure!(
        !args.ngram_speculative || args.num_drafts.is_some_and(|n| (1..=5).contains(&n)),
        "HY4 --ngram-speculative requires explicit --num-drafts in 1..=5"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(flags: &[&str]) -> cli::ServeArgs {
        let cli::Command::Serve(args) = cli::Cli::try_parse_from(
            ["spark", "serve", "unused"]
                .into_iter()
                .chain(flags.iter().copied()),
        )
        .unwrap()
        .command
        else {
            panic!("serve args")
        };
        args
    }

    #[test]
    fn prompt_lookup_is_opt_in_and_requires_an_explicit_bounded_draft_length() {
        assert!(validate(&args(&[])).is_ok());
        for n in ["1", "5"] {
            assert!(validate(&args(&["--ngram-speculative", "--num-drafts", n])).is_ok());
        }
        assert!(validate(&args(&["--ngram-speculative"])).is_err());
        for n in ["0", "6"] {
            assert!(validate(&args(&["--ngram-speculative", "--num-drafts", n])).is_err());
        }
        for mode in ["--self-speculative", "--speculative", "--dflash"] {
            assert!(validate(&args(&[mode])).is_err());
        }
    }
}
