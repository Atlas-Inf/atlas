// SPDX-License-Identifier: AGPL-3.0-only

//! `spark display-carveout`: lend the GB10 display carveout to a server.
//!
//! The firmware reserves 2 GiB for the display (`DISPLAY_FRM`) that Linux
//! never sees and the driver never allocates from on GB10. This launcher
//! exports it as an RM object fd (`rm`), which needs CAP_SYS_ADMIN, then
//! removes CAP_SYS_ADMIN from itself and its bounding set and execs the
//! command with the fd inherited and named in
//! `spark_runtime::gpu::carveout::FD_ENV`. `spark serve` then puts KV pools
//! there (`factory::build::kv_carveout`). Without the fd it serves as before.
//!
//! One process per host may hold the carveout: two would write each other's
//! KV. `--lock` names a file on the host that every launcher shares; the
//! command inherits the held lock, so it lasts exactly as long as the
//! process that maps the memory. If the lock is held, the driver is not a
//! validated release, or the export fails, the command runs without the
//! carveout and the reason is logged.

use std::ffi::CString;
use std::io::Error;
use std::os::fd::IntoRawFd;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::carveout::{FD_ENV, SIZE_ENV};

#[cfg(target_os = "linux")]
mod rm;

/// Driver releases whose RM ABI and carveout handling were checked against
/// their open-gpu-kernel-modules source.
pub(crate) const VALIDATED_DRIVERS: &[&str] = &["580.173.02", "580.178.04"];

#[derive(clap::Args, Debug)]
pub struct DisplayCarveoutArgs {
    /// Lock file shared by every launcher on this host (bind-mount the same
    /// host path into each container).
    #[arg(long)]
    pub lock: PathBuf,
    /// The command to run with the carveout, usually `spark serve ...`.
    #[arg(last = true, required = true)]
    pub command: Vec<String>,
}

/// Exports the carveout if it can, then execs the command. Returns only on
/// failure to exec or to drop CAP_SYS_ADMIN.
pub fn run(args: DisplayCarveoutArgs) -> Result<()> {
    // SAFETY: single-threaded use of the environment just before exec; no
    // other thread reads these variables.
    unsafe {
        std::env::remove_var(FD_ENV);
        std::env::remove_var(SIZE_ENV);
    }
    match lend(&args) {
        Ok((fd, size)) => {
            tracing::info!(
                "display carveout: {} MiB exported as fd {fd}, lock {}",
                size >> 20,
                args.lock.display()
            );
            unsafe {
                std::env::set_var(FD_ENV, fd.to_string());
                std::env::set_var(SIZE_ENV, size.to_string());
            }
        }
        Err(e) => tracing::warn!("display carveout not lent: {e:#}; running without it"),
    }
    drop_sys_admin().context("dropping CAP_SYS_ADMIN before exec")?;
    let argv: Vec<CString> = args
        .command
        .iter()
        .map(|a| CString::new(a.as_str()))
        .collect::<Result<_, _>>()?;
    let mut ptrs: Vec<*const libc::c_char> = argv.iter().map(|a| a.as_ptr()).collect();
    ptrs.push(std::ptr::null());
    unsafe { libc::execvp(ptrs[0], ptrs.as_ptr()) };
    bail!("exec {}: {}", args.command[0], Error::last_os_error())
}

/// Takes the host lock (kept open across exec), checks the driver, exports.
fn lend(args: &DisplayCarveoutArgs) -> Result<(i32, u64)> {
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&args.lock)
        .with_context(|| format!("open {}", args.lock.display()))?
        .into_raw_fd();
    // Inherited by the command, which then holds the lock until it exits.
    unsafe { libc::fcntl(lock, libc::F_SETFD, 0) };
    if unsafe { libc::flock(lock, libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        unsafe { libc::close(lock) };
        bail!(
            "{} is held: another process has the carveout",
            args.lock.display()
        );
    }
    let release = |e: anyhow::Error| {
        unsafe { libc::close(lock) };
        e
    };
    let version =
        std::fs::read_to_string("/proc/driver/nvidia/version").map_err(|e| release(e.into()))?;
    check_driver(&version).map_err(release)?;
    export().map_err(release)
}

fn check_driver(proc_version: &str) -> Result<()> {
    let line = proc_version.lines().next().unwrap_or_default();
    match VALIDATED_DRIVERS
        .iter()
        .find(|v| line.split_whitespace().any(|w| w == **v))
    {
        Some(_) => Ok(()),
        None => bail!("driver {line:?} is not a validated release ({VALIDATED_DRIVERS:?})"),
    }
}

#[cfg(target_os = "linux")]
fn export() -> Result<(i32, u64)> {
    let e = rm::export_display_frm()?;
    tracing::info!("display carveout: DISPLAY_FRM at 0x{:x}", e.base);
    Ok((e.fd, e.size))
}

#[cfg(not(target_os = "linux"))]
fn export() -> Result<(i32, u64)> {
    bail!("the display carveout is a Linux GB10 feature")
}

/// Removes CAP_SYS_ADMIN from the bounding set and this thread's sets, so
/// the command (root in a container) can never create RM objects over
/// physical memory. A no-op where it was never held.
#[cfg(target_os = "linux")]
fn drop_sys_admin() -> Result<()> {
    const CAP_SYS_ADMIN: u32 = 21;
    if unsafe { libc::prctl(libc::PR_CAPBSET_READ, CAP_SYS_ADMIN) } == 1
        && unsafe { libc::prctl(libc::PR_CAPBSET_DROP, CAP_SYS_ADMIN) } != 0
    {
        bail!("PR_CAPBSET_DROP: {}", Error::last_os_error());
    }
    #[repr(C)]
    struct Header {
        version: u32,
        pid: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Sets {
        effective: u32,
        permitted: u32,
        inheritable: u32,
    }
    const LINUX_CAPABILITY_VERSION_3: u32 = 0x2008_0522;
    let mut header = Header {
        version: LINUX_CAPABILITY_VERSION_3,
        pid: 0,
    };
    let mut sets = [Sets::default(); 2];
    if unsafe { libc::syscall(libc::SYS_capget, &mut header, sets.as_mut_ptr()) } != 0 {
        bail!("capget: {}", Error::last_os_error());
    }
    let bit = !(1u32 << CAP_SYS_ADMIN);
    sets[0].effective &= bit;
    sets[0].permitted &= bit;
    sets[0].inheritable &= bit;
    if unsafe { libc::syscall(libc::SYS_capset, &mut header, sets.as_ptr()) } != 0 {
        bail!("capset: {}", Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn drop_sys_admin() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_validated_drivers_pass() {
        let ok =
            "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.173.02  Release Build";
        assert!(check_driver(ok).is_ok());
        let newer =
            "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  590.10.01  Release Build";
        assert!(check_driver(newer).is_err());
        // A validated version only as a substring of another does not pass.
        assert!(check_driver("NVRM version: 580.173.021").is_err());
        assert!(check_driver("").is_err());
    }

    #[test]
    fn the_command_follows_the_separator() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from([
            "spark",
            "display-carveout",
            "--lock",
            "/run/x.lock",
            "--",
            "spark",
            "serve",
            "--port=1",
        ])
        .unwrap();
        let crate::cli::Command::DisplayCarveout(args) = cli.command else {
            panic!("parsed {cli:?}");
        };
        assert_eq!(args.lock, PathBuf::from("/run/x.lock"));
        assert_eq!(args.command, ["spark", "serve", "--port=1"]);
    }

    #[test]
    fn a_held_lock_lends_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("carveout.lock");
        let holder = std::fs::File::create(&path).unwrap();
        use std::os::fd::AsRawFd;
        assert_eq!(unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX) }, 0);
        let args = DisplayCarveoutArgs {
            lock: path,
            command: vec!["true".into()],
        };
        let err = lend(&args).unwrap_err().to_string();
        assert!(err.contains("is held"), "{err}");
    }
}
