// SPDX-License-Identifier: AGPL-3.0-only

//! The n-gram lane must feed the accept accounting.
//!
//! **Every speculative lane records its steps into `a.mtp_acct`.** The serial
//! arm calls `record_serial`; the MTP, DFlash and self-spec arms call
//! `record_verify_emitted` with the step's emitted-token count. The n-gram
//! lane did neither, and the failure mode is silent in the worst way: the
//! per-request `Done:` line derives `p1`/`mean_na`/`tok_step` from that
//! counter, so an un-fed lane reports `serial=0.00 mtp=0.00 p1=0.000
//! mean_na=0.000 tok_step=1.000` — which reads exactly like a lane that ran
//! and accepted nothing. `usage.completion_tokens_details.
//! accepted_prediction_tokens` reads the same counter, so it was 0 too.
//!
//! ★ The cost was concrete. On 2026-09-15 the nvidia Flash-Next pack was
//! recorded as "the n-gram lane engages, accepts ~nothing", on the strength of
//! that line, in `docs/porting/QWEN4_EXP_PORT_LOG.md`. The lane's own debug
//! lines in the same job (jobqueue 074) showed `na=1` on 93 of 143 proposals.
//! The zero was the default of a counter nothing wrote, not a measurement.
//!
//! # Scope, deliberately narrow
//!
//! This asserts the DISPATCH SITE only: the `step_ngram(` call in `mod.rs`
//! must be followed by a `record_verify_emitted` in the same statement group.
//! It does not try to audit accounting anywhere else, and it does not assert
//! what the numbers should be — that is the hardware leg's job.

/// The dispatch site's anchor: the one `step_ngram(` call in the scheduler.
const ANCHOR: &str = "step_ngram(";

/// Proof the step's emission was booked.
const NAMED: &str = "record_verify_emitted(";

/// How far past the call the booking may sit. Generous: the call itself spans
/// eight lines of arguments, and a future signature change should not turn
/// this into a false alarm.
const WINDOW: usize = 25;

/// Find the dispatch site and report whether it books its step.
///
/// Returns `(sites, unbooked_line_numbers)`. A site is the `step_ngram(` call;
/// it is unbooked when no `record_verify_emitted` appears within `WINDOW`
/// lines after it.
fn scan(src: &str) -> (usize, Vec<usize>) {
    let lines: Vec<&str> = src.lines().collect();
    let mut sites = 0usize;
    let mut unbooked = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        // Comments describe the rule; only code can break it. The doc comment
        // above this test names `record_verify_emitted`, so an unscoped scan
        // would match its own prose.
        if line.trim_start().starts_with("//") {
            continue;
        }
        if !line.contains(ANCHOR) {
            continue;
        }
        sites += 1;
        let to = (i + WINDOW).min(lines.len() - 1);
        if !lines[i..=to].iter().any(|l| l.contains(NAMED)) {
            unbooked.push(i + 1);
        }
    }
    (sites, unbooked)
}

#[test]
fn the_ngram_dispatch_site_books_its_step() {
    let (sites, unbooked) = scan(include_str!("mod.rs"));

    // Floor: exactly one ngram dispatch site. If a rename or move drops the
    // scan to zero this must fail loudly rather than report a green it never
    // earned.
    assert_eq!(
        sites, 1,
        "expected exactly one `{ANCHOR}` dispatch site, found {sites} — the \
         scan stopped matching. Fix the detection before trusting a green here."
    );

    assert!(
        unbooked.is_empty(),
        "the n-gram dispatch site does not book its step at line(s) {unbooked:?}.\n\
         Without `a.mtp_acct.record_verify_emitted(..)` the lane is invisible to \
         the accept accounting, and the per-request `Done:` line reports \
         `serial=0.00 mtp=0.00 p1=0.000 mean_na=0.000 tok_step=1.000` whatever \
         the lane actually accepted — a vacuous zero that reads as a measured \
         one. `accepted_prediction_tokens` on the usage block is 0 for the same \
         reason. Book the step, or delete this test with the reason."
    );
}

#[test]
fn the_scan_flags_an_unbooked_site_and_clears_a_booked_one() {
    // NEGATIVE half: prove the detector fires. Without this the test above
    // passes for two reasons — the invariant holding, or the scan matching
    // nothing — and only one is good news.
    let unbooked = "\
        if use_ngram_speculative {\n\
        \x20   if let Some(ref mut proposer) = ngram_proposer {\n\
        \x20       step_ngram(&*model, &mut active, &sched, proposer, false, &verify_ctx);\n\
        \x20   }\n\
        }\n";
    let (sites, flagged) = scan(unbooked);
    assert_eq!(sites, 1, "the site should be counted");
    assert_eq!(flagged, vec![3], "an unbooked dispatch must be flagged");

    // POSITIVE half: booking the step clears the finding.
    let booked = unbooked.replace(
        "&verify_ctx);\n",
        "&verify_ctx);\n        let emitted = 1;\n        active[0].mtp_acct.record_verify_emitted(emitted);\n",
    );
    let (sites, flagged) = scan(&booked);
    assert_eq!(sites, 1);
    assert!(flagged.is_empty(), "a booked dispatch must not be flagged");
}
