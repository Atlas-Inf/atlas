// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

pub(super) fn register(gpu: &dyn GpuBackend, comm: Option<&dyn CommBackend>) {
    install(comm, || gpu.kernel("bf16_add", "bf16_add_inplace"));
}

fn install(comm: Option<&dyn CommBackend>, kernel: impl FnOnce() -> Result<KernelHandle>) {
    if let Some(comm) = comm.filter(|c| c.world_size() > 1) {
        match kernel() {
            Ok(k) => {
                comm.set_add_kernel(k.0);
                tracing::info!("HY4 registered bf16_add_inplace for two-rank collectives");
            }
            Err(e) => tracing::warn!("HY4 bf16_add_inplace unavailable; retaining NCCL: {e:#}"),
        }
    }
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
        install(Some(&c), || Ok(KernelHandle(42)));
        assert_eq!(c.kernel.load(Ordering::Relaxed), 42);
        let c = Comm {
            world: 1,
            kernel: AtomicU64::new(0),
        };
        install(Some(&c), || {
            panic!("single rank must not look up add kernel")
        });
        install(None, || {
            panic!("absent communicator must not look up add kernel")
        });
        assert_eq!(c.kernel.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn missing_kernel_retains_nccl_fallback() {
        let c = Comm {
            world: 2,
            kernel: AtomicU64::new(0),
        };
        install(Some(&c), || anyhow::bail!("missing kernel"));
        assert_eq!(c.kernel.load(Ordering::Relaxed), 0);
    }
}
