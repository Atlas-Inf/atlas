// SPDX-License-Identifier: AGPL-3.0-only

//! The real runner: `std::process`, no shell, bounded output.
//!
//! Two details are load-bearing.
//!
//! **No deadlock.** The child is written to and read from on separate threads
//! while the parent waits. A naive write-then-read would deadlock the moment
//! the child filled its stdout pipe before draining stdin — and a `capture`
//! receipt can be small while the adapter's stderr is not.
//!
//! **Kill the group, not the process.** On Unix the child is made a process
//! group leader, so a timeout kills the adapter children it may have spawned
//! too. This is still only *abortive*: if the controller is coordinating
//! manager-side work, killing the CLI does not cancel that work. Hence a
//! timeout is classified unknown, never failed.

use std::io::{Read, Write};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use super::{CommandRunner, Invocation, RunnerError, RunnerOutput};

/// Runs the controller binary as a real child process.
#[derive(Debug, Default, Clone, Copy)]
pub struct StdCommandRunner;

/// Poll interval while waiting for the child. Small enough that short
/// operations are not padded, large enough that a 45-minute capture is free.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

impl CommandRunner for StdCommandRunner {
    fn run(&self, invocation: &Invocation) -> Result<RunnerOutput, RunnerError> {
        let mut command = Command::new(invocation.program.clone());
        command
            .args(&invocation.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in &invocation.environment {
            command.env(key, value);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            // Group leader, so a timeout can signal the whole tree.
            command.process_group(0);
        }

        let mut child = command.spawn().map_err(|source| RunnerError::Spawn {
            program: invocation.program.to_string_lossy().into_owned(),
            source,
        })?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        let stdin_bytes = invocation.stdin.clone();
        let stdin_writer = stdin.map(|mut pipe| {
            std::thread::spawn(move || match stdin_bytes {
                Some(bytes) => match pipe.write_all(&bytes).and_then(|()| pipe.flush()) {
                    Ok(()) => false,
                    // The child exited before reading everything. Its output
                    // and exit status are still evidence, so record and move on.
                    Err(e) => e.kind() == std::io::ErrorKind::BrokenPipe,
                },
                None => false,
            })
        });

        // stdout stops reading at its cap (a receipt that big is a protocol
        // violation, and closing the pipe stops the writer). stderr is
        // *drained* past its cap instead: diagnostics must never be the
        // reason the controller is killed by SIGPIPE.
        let stdout_reader = std::thread::spawn({
            let cap = invocation.max_stdout_bytes;
            move || read_capped(stdout, cap, false)
        });
        let stderr_reader = std::thread::spawn({
            let cap = invocation.max_stderr_bytes;
            move || read_capped(stderr, cap, true)
        });

        // A wait error must not leak the child: kill it before propagating.
        let (exit_code, signal, timed_out) = match wait_with_timeout(&mut child, invocation.timeout)
        {
            Ok(waited) => waited,
            Err(error) => {
                kill_tree(&mut child);
                let _ = child.wait();
                return Err(error);
            }
        };

        let stdin_broken_pipe = stdin_writer
            .map(|handle| handle.join().unwrap_or(false))
            .unwrap_or(false);

        let (stdout, stdout_truncated) = join_reader(stdout_reader, "collecting stdout")?;
        let (stderr, stderr_truncated) = join_reader(stderr_reader, "collecting stderr")?;

        Ok(RunnerOutput {
            exit_code,
            signal,
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
            timed_out,
            stdin_broken_pipe,
        })
    }
}

/// Wait for the child, killing it (and its group) if the budget expires.
///
/// The deadline is re-checked with a second `try_wait` before killing. Without
/// that, a child that exited in the window between the poll and the deadline
/// check would be killed after the fact and reported as a timeout — turning a
/// completed operation into an indeterminate one, which is the expensive
/// direction to be wrong in.
fn wait_with_timeout(
    child: &mut Child,
    timeout: Duration,
) -> Result<(Option<i32>, Option<i32>, bool), RunnerError> {
    let deadline = Instant::now().checked_add(timeout);
    loop {
        match child.try_wait().map_err(RunnerError::Wait)? {
            Some(status) => {
                let (code, signal) = describe(status);
                return Ok((code, signal, false));
            }
            None => {
                let expired = deadline.is_none_or(|at| Instant::now() >= at);
                if !expired {
                    std::thread::sleep(POLL_INTERVAL);
                    continue;
                }
                match child.try_wait().map_err(RunnerError::Wait)? {
                    Some(status) => {
                        let (code, signal) = describe(status);
                        return Ok((code, signal, false));
                    }
                    None => {
                        kill_tree(child);
                        let status = child.wait().map_err(RunnerError::Wait)?;
                        let (code, signal) = describe(status);
                        return Ok((code, signal, true));
                    }
                }
            }
        }
    }
}

/// Signal the child's process group, then the child itself.
fn kill_tree(child: &mut Child) {
    #[cfg(unix)]
    {
        // Guard the sign: `kill(-0)` signals the CALLER's process group, so a
        // zero pid (or one that will not fit an i32) must never reach the call.
        if let Some(pid) = i32::try_from(child.id()).ok().filter(|pid| *pid > 0) {
            // SAFETY: `kill` is async-signal-safe and takes no pointers. A
            // negative pid targets the process group whose id is `pid`; the
            // child was made a group leader at spawn, so this reaches its
            // descendants. A descendant that called `setsid` escapes this —
            // best-effort is the documented limit.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    // Also signal the direct child, in case the group call raced or the
    // platform has no process groups.
    let _ = child.kill();
}

/// Split a status into (exit code, signal).
#[cfg(unix)]
fn describe(status: std::process::ExitStatus) -> (Option<i32>, Option<i32>) {
    use std::os::unix::process::ExitStatusExt as _;
    (status.code(), status.signal())
}

/// Split a status into (exit code, signal).
#[cfg(not(unix))]
fn describe(status: std::process::ExitStatus) -> (Option<i32>, Option<i32>) {
    (status.code(), None)
}

/// Join a reader thread, turning a panic into a runner error rather than
/// propagating it as a panic into the client.
fn join_reader(
    handle: std::thread::JoinHandle<std::io::Result<(Vec<u8>, bool)>>,
    context: &'static str,
) -> Result<(Vec<u8>, bool), RunnerError> {
    match handle.join() {
        Ok(Ok(collected)) => Ok(collected),
        Ok(Err(source)) => Err(RunnerError::Io { context, source }),
        Err(_) => Err(RunnerError::Io {
            context,
            source: std::io::Error::other("reader thread panicked"),
        }),
    }
}

/// Read at most `cap` bytes, reporting whether the cap was reached.
///
/// `drain` decides what happens once the cap is hit. When false, the read end
/// is dropped, which closes the pipe and stops a runaway writer — the right
/// call for a receipt, where exceeding the cap is already a protocol
/// violation. When true, the stream is drained and discarded instead, so a
/// chatty but healthy process is never killed by our own limit.
fn read_capped<R: Read>(
    mut reader: R,
    cap: usize,
    drain: bool,
) -> std::io::Result<(Vec<u8>, bool)> {
    let mut collected = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut truncated = false;
    loop {
        let read = match reader.read(&mut chunk) {
            Ok(0) => return Ok((collected, truncated)),
            Ok(read) => read,
            // A signal during a read is not a failure.
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        // `collected.len() <= cap` holds, so this subtraction cannot go
        // negative; written this way it also cannot overflow.
        let remaining = cap.saturating_sub(collected.len());
        if read > remaining {
            collected.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            if !drain {
                return Ok((collected, true));
            }
            continue;
        }
        collected.extend_from_slice(&chunk[..read]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BinaryLocation;
    use std::ffi::OsString;

    fn invocation(program: &str, args: &[&str], stdin: Option<&[u8]>) -> Invocation {
        Invocation {
            program: OsString::from(program),
            args: args.iter().map(|s| (*s).to_owned()).collect(),
            stdin: stdin.map(<[u8]>::to_vec),
            timeout: Duration::from_secs(10),
            max_stdout_bytes: 64 * 1024,
            max_stderr_bytes: 64 * 1024,
            environment: Vec::new(),
        }
    }

    #[test]
    fn collects_stdout_and_a_zero_exit() {
        let out = StdCommandRunner
            .run(&invocation("/bin/sh", &["-c", "printf hello"], None))
            .unwrap();
        assert_eq!(out.stdout, b"hello");
        assert!(out.exited_zero());
        assert!(!out.timed_out);
    }

    #[test]
    fn reports_a_nonzero_exit_without_erroring() {
        let out = StdCommandRunner
            .run(&invocation("/bin/sh", &["-c", "exit 3"], None))
            .unwrap();
        assert_eq!(out.exit_code, Some(3));
        assert!(!out.exited_zero());
    }

    #[test]
    fn writes_stdin_and_round_trips_it() {
        let out = StdCommandRunner
            .run(&invocation("/bin/sh", &["-c", "cat"], Some(b"payload")))
            .unwrap();
        assert_eq!(out.stdout, b"payload");
    }

    #[test]
    fn a_large_payload_does_not_deadlock() {
        // 1 MiB in and 1 MiB out with small pipes: this is the shape that
        // deadlocks a write-then-read implementation.
        let payload = vec![b'x'; 1024 * 1024];
        let mut inv = invocation("/bin/sh", &["-c", "cat"], Some(&payload));
        inv.max_stdout_bytes = 2 * 1024 * 1024;
        let out = StdCommandRunner.run(&inv).unwrap();
        assert_eq!(out.stdout.len(), payload.len());
        assert!(!out.stdout_truncated);
    }

    #[test]
    fn a_missing_binary_is_a_spawn_error() {
        let err = StdCommandRunner
            .run(&invocation(
                "/nonexistent/coldsnap-does-not-exist",
                &[],
                None,
            ))
            .unwrap_err();
        assert!(matches!(err, RunnerError::Spawn { .. }));
    }

    #[test]
    fn a_timeout_kills_the_child_and_is_reported() {
        let mut inv = invocation("/bin/sh", &["-c", "sleep 30"], None);
        inv.timeout = Duration::from_millis(150);
        let began = Instant::now();
        let out = StdCommandRunner.run(&inv).unwrap();
        assert!(out.timed_out);
        assert!(!out.exited_zero());
        assert!(began.elapsed() < Duration::from_secs(10), "killed promptly");
    }

    #[test]
    fn an_oversized_stream_is_truncated_and_flagged() {
        let mut inv = invocation("/bin/sh", &["-c", "yes abcdefgh"], None);
        inv.max_stdout_bytes = 4096;
        inv.timeout = Duration::from_secs(10);
        let out = StdCommandRunner.run(&inv).unwrap();
        assert!(out.stdout_truncated);
        assert_eq!(out.stdout.len(), 4096);
    }

    #[test]
    fn capped_read_stops_at_the_cap_when_not_draining() {
        let input = std::io::Cursor::new(vec![b'x'; 1000]);
        let (collected, truncated) = read_capped(input, 10, false).unwrap();
        assert_eq!(collected.len(), 10);
        assert!(truncated);
    }

    #[test]
    fn capped_read_drains_the_rest_when_asked_to() {
        // Draining must still return exactly the capped prefix, and must not
        // stop early — that is what keeps a chatty child off SIGPIPE.
        let input = std::io::Cursor::new(vec![b'x'; 1000]);
        let (collected, truncated) = read_capped(input, 10, true).unwrap();
        assert_eq!(collected.len(), 10);
        assert!(truncated);
    }

    #[test]
    fn an_under_cap_stream_is_not_flagged_as_truncated() {
        let input = std::io::Cursor::new(b"short".to_vec());
        let (collected, truncated) = read_capped(input, 10, true).unwrap();
        assert_eq!(collected, b"short");
        assert!(!truncated);
    }

    #[test]
    fn a_zero_cap_collects_nothing_and_flags_it() {
        let input = std::io::Cursor::new(b"anything".to_vec());
        let (collected, truncated) = read_capped(input, 0, true).unwrap();
        assert!(collected.is_empty());
        assert!(truncated);
    }

    #[test]
    fn binary_location_is_used_verbatim() {
        let location = BinaryLocation::Explicit(std::path::PathBuf::from("/bin/echo"));
        assert_eq!(location.program(), std::ffi::OsStr::new("/bin/echo"));
    }
}
