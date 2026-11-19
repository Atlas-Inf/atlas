// SPDX-License-Identifier: AGPL-3.0-only

use super::{Result, cli};

pub(crate) fn validate(args: &cli::ServeArgs) -> Result<()> {
    anyhow::ensure!(
        !args.dfly_speculative || (args.max_batch_size == 1 && args.max_num_seqs == 1),
        "HY4 DFly requires --max-batch-size=1 and --max-num-seqs=1"
    );
    anyhow::ensure!(
        !args.self_speculative && !args.speculative && !args.dflash,
        "HY4 supports only explicit prompt lookup or DFly, not MTP/self-speculation"
    );
    anyhow::ensure!(
        !args.uses_lookup_verifier() || args.num_drafts.is_some_and(|n| (1..=5).contains(&n)),
        "HY4 speculation requires explicit --num-drafts in 1..=5"
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

    #[test]
    fn dfly_requires_explicit_bounded_count_and_excludes_other_proposers() {
        for n in ["1", "5"] {
            let a = args(&[
                "--dfly-speculative",
                "--num-drafts",
                n,
                "--max-batch-size=1",
                "--max-num-seqs=1",
            ]);
            assert!(validate(&a).is_ok());
            assert!(a.uses_lookup_verifier());
        }
        for n in ["0", "6"] {
            assert!(
                validate(&args(&[
                    "--dfly-speculative",
                    "--num-drafts",
                    n,
                    "--max-batch-size=1",
                    "--max-num-seqs=1"
                ]))
                .is_err()
            );
        }
        assert!(validate(&args(&["--dfly-speculative", "--num-drafts=5"])).is_err());
        assert!(
            validate(&args(&[
                "--dfly-speculative",
                "--num-drafts=5",
                "--max-batch-size=2",
                "--max-num-seqs=1"
            ]))
            .is_err()
        );
        assert!(
            validate(&args(&[
                "--dfly-speculative",
                "--num-drafts=5",
                "--max-batch-size=1",
                "--max-num-seqs=2"
            ]))
            .is_err()
        );
        assert!(
            cli::Cli::try_parse_from(["spark", "serve", "unused", "--dfly-speculative"]).is_err()
        );
        for other in [
            "--speculative",
            "--self-speculative",
            "--ngram-speculative",
            "--dflash",
        ] {
            assert!(
                cli::Cli::try_parse_from([
                    "spark",
                    "serve",
                    "unused",
                    "--dfly-speculative",
                    "--num-drafts",
                    "5",
                    other
                ])
                .is_err()
            );
        }
    }
}
