// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

pub(super) fn register(gpu: &dyn GpuBackend, comm: Option<&dyn CommBackend>) -> Result<()> {
    install(comm, || gpu.kernel("bf16_add", "bf16_add_inplace"))
}

/// Fails fast when the add kernel is missing: a rank that silently kept NCCL
/// while its peer engaged direct RDMA would join a different collective.
fn install(
    comm: Option<&dyn CommBackend>,
    kernel: impl FnOnce() -> Result<KernelHandle>,
) -> Result<()> {
    if let Some(comm) = comm.filter(|c| c.world_size() > 1) {
        let k = kernel().context("HY4 bf16_add_inplace required for two-rank collectives")?;
        comm.set_add_kernel(k.0);
        tracing::info!("HY4 registered bf16_add_inplace for two-rank collectives");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Comm {
        world: usize,
        kernel: AtomicU64,
    }
    impl CommBackend for Comm {
        fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
            Ok(())
        }
        fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
            Ok(())
        }
        fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
            Ok(())
        }
        fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
            Ok(())
        }
        fn barrier(&self) -> Result<()> {
            Ok(())
        }
        fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
            Ok(())
        }
        fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
            Ok(())
        }
        fn rank(&self) -> usize {
            0
        }
        fn world_size(&self) -> usize {
            self.world
        }
        fn set_add_kernel(&self, k: u64) {
            self.kernel.store(k, Ordering::Relaxed);
        }
    }
    #[test]
    fn registers_for_tp2_but_not_single_rank_or_absent_communicator() {
        let c = Comm {
            world: 2,
            kernel: AtomicU64::new(0),
        };
        install(Some(&c), || Ok(KernelHandle(42))).unwrap();
        assert_eq!(c.kernel.load(Ordering::Relaxed), 42);
        let c = Comm {
            world: 1,
            kernel: AtomicU64::new(0),
        };
        install(Some(&c), || {
            panic!("single rank must not look up add kernel")
        })
        .unwrap();
        install(None, || {
            panic!("absent communicator must not look up add kernel")
        })
        .unwrap();
        assert_eq!(c.kernel.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn missing_kernel_fails_instead_of_diverging_from_peer() {
        let c = Comm {
            world: 2,
            kernel: AtomicU64::new(0),
        };
        assert!(install(Some(&c), || anyhow::bail!("missing kernel")).is_err());
        assert_eq!(c.kernel.load(Ordering::Relaxed), 0);
    }
}
