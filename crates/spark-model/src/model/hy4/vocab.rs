// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::{kernel_args::KernelLaunch, weights::WeightDtype};

pub(super) struct VocabHead {
    pub weight: DevicePtr,
    pub rows: usize,
    pub start: usize,
}

fn partition(vocab: usize, world: usize, rank: usize) -> Result<(usize, usize)> {
    ensure!(
        vocab > 0 && matches!(world, 1 | 2) && rank < world && vocab.is_multiple_of(world),
        "Hy4 vocabulary must divide TP ranks"
    );
    let rows = vocab / world;
    Ok((rank * rows, rows))
}

impl VocabHead {
    pub fn load(
        c: &ModelConfig,
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        memory: &mut Owned<'_>,
    ) -> Result<Self> {
        let (start, rows) = partition(c.vocab_size, c.tp_world_size, c.tp_rank)?;
        let src = store.get("lm_head.weight")?;
        ensure!(
            src.shape == [c.vocab_size, c.hidden_size]
                && matches!(src.dtype, WeightDtype::FP32 | WeightDtype::BF16),
            "Hy4 LM head requires unquantized BF16/FP32 checkpoint weight"
        );
        let elements = rows
            .checked_mul(c.hidden_size)
            .context("Hy4 head size overflow")?;
        ensure!(elements <= u32::MAX as usize, "Hy4 head shard too large");
        let weight = memory.alloc(elements * 2)?;
        let stream = gpu.default_stream();
        KernelLaunch::new(gpu, gpu.kernel("hy4_model", "hy4_head_bf16")?)
            .grid([(elements as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(
                src.ptr
                    .offset(start * c.hidden_size * src.dtype.byte_size()),
            )
            .arg_ptr(weight)
            .arg_u32(elements as u32)
            .arg_u32(u32::from(src.dtype == WeightDtype::FP32))
            .launch(stream)?;
        gpu.synchronize(stream)?;
        if store.get("model.embed_tokens.weight")?.ptr != src.ptr {
            store.reclaim(gpu, "lm_head.weight")?;
        }
        tracing::info!(
            "HY4 BF16 vocab head: rank={} start={start} rows={rows} bytes={}",
            c.tp_rank,
            elements * 2
        );
        Ok(Self {
            weight,
            rows,
            start,
        })
    }

    pub fn gather(&self, s: &Shared, logits: DevicePtr, stream: u64) -> Result<()> {
        if s.config.tp_world_size == 2 {
            let comm = s
                .comm
                .as_deref()
                .context("Hy4 head gather requires communicator")?;
            crate::layers::glm_kv_shard::pair_exchange(
                comm,
                logits.offset(self.start * 4),
                logits.offset((self.rows - self.start) * 4),
                self.rows * 4,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vocab_partition_is_contiguous_and_complete() {
        assert_eq!(partition(120832, 2, 0).unwrap(), (0, 60416));
        assert_eq!(partition(120832, 2, 1).unwrap(), (60416, 60416));
        assert_eq!(partition(17, 1, 0).unwrap(), (0, 17));
    }
    #[test]
    fn invalid_vocab_topology_is_rejected() {
        for args in [(0, 1, 0), (17, 2, 0), (16, 0, 0), (16, 2, 2)] {
            assert!(partition(args.0, args.1, args.2).is_err());
        }
    }
}
