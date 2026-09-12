// SPDX-License-Identifier: AGPL-3.0-only

//! Test doubles for the runner seam.
//!
//! Public rather than `#[cfg(test)]` because the crate's integration tests
//! live in `tests/`, and because a downstream consumer (spark-server's
//! lifecycle layer) needs the same ability to exercise its own policy without
//! a controller binary on the box.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::{CommandRunner, Invocation, RunnerError, RunnerOutput};

/// Replays canned results in order and records every invocation.
///
/// Deterministic on purpose: a test that scripts three results knows exactly
/// which call got which, so a change in call *order* fails the test rather
/// than silently shifting a scripted outcome onto the wrong operation.
#[derive(Debug, Default)]
pub struct ScriptedRunner {
    scripted: Mutex<VecDeque<Result<RunnerOutput, RunnerError>>>,
    invocations: Mutex<Vec<Invocation>>,
}

impl ScriptedRunner {
    /// An empty script.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a successful process result.
    pub fn then_output(self, output: RunnerOutput) -> Self {
        self.scripted.lock().unwrap().push_back(Ok(output));
        self
    }

    /// Queue a runner-level failure.
    pub fn then_error(self, error: RunnerError) -> Self {
        self.scripted.lock().unwrap().push_back(Err(error));
        self
    }

    /// Every invocation this runner has seen, in order.
    pub fn invocations(&self) -> Vec<Invocation> {
        self.invocations.lock().unwrap().clone()
    }

    /// The most recent invocation, if any.
    pub fn last_invocation(&self) -> Option<Invocation> {
        self.invocations.lock().unwrap().last().cloned()
    }

    /// How many scripted results are still queued.
    pub fn remaining(&self) -> usize {
        self.scripted.lock().unwrap().len()
    }
}

impl CommandRunner for ScriptedRunner {
    fn run(&self, invocation: &Invocation) -> Result<RunnerOutput, RunnerError> {
        self.invocations.lock().unwrap().push(invocation.clone());
        self.scripted
            .lock()
            .unwrap()
            .pop_front()
            .expect("ScriptedRunner was called more times than results were scripted")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invocation() -> Invocation {
        Invocation {
            program: std::ffi::OsString::from("coldsnap"),
            args: vec!["capabilities".to_owned()],
            stdin: None,
            timeout: std::time::Duration::from_secs(1),
            max_stdout_bytes: 1024,
            max_stderr_bytes: 1024,
            environment: Vec::new(),
        }
    }

    #[test]
    fn records_invocations_in_order() {
        let runner = ScriptedRunner::new()
            .then_output(RunnerOutput::exited(0, b"one".to_vec(), Vec::new()))
            .then_output(RunnerOutput::exited(0, b"two".to_vec(), Vec::new()));
        runner.run(&invocation()).unwrap();
        runner.run(&invocation()).unwrap();
        assert_eq!(runner.invocations().len(), 2);
        assert_eq!(runner.remaining(), 0);
    }

    #[test]
    fn replays_a_scripted_error() {
        let runner = ScriptedRunner::new().then_error(RunnerError::Io {
            context: "testing",
            source: std::io::Error::other("boom"),
        });
        assert!(runner.run(&invocation()).is_err());
    }

    #[test]
    fn records_the_invocation_even_when_the_result_is_an_error() {
        let runner =
            ScriptedRunner::new().then_error(RunnerError::Wait(std::io::Error::other("boom")));
        let _ = runner.run(&invocation());
        assert_eq!(runner.last_invocation().unwrap().args, ["capabilities"]);
    }
}
