// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::weights::WeightDtype;

struct Loader<'a, 's> {
    memory: Owned<'a>,
    store: &'s WeightStore,
}
impl Loader<'_, '_> {
    fn source(&self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let key = format!("model.dfly_layers.{name}");
        ensure!(
            !self.store.was_reclaimed(&key),
            "DFly {name} already consumed"
        );
        let t = self.store.get(&key)?;
        ensure!(
            t.shape == shape && t.dtype == WeightDtype::BF16,
            "DFly {name}: expected BF16 {shape:?}, got {:?} {:?}",
            t.dtype,
            t.shape
        );
        Ok(t.ptr)
    }
    fn copy(&mut self, src: DevicePtr, bytes: usize) -> Result<DevicePtr> {
        let out = self.memory.alloc(bytes)?;
        self.memory.gpu.copy_d2d(src, out, bytes)?;
        Ok(out)
    }
    fn full(&mut self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let src = self.source(name, shape)?;
        let out = self.copy(src, shape.iter().product::<usize>() * 2)?;
        self.reclaim(name)?;
        Ok(out)
    }
    fn reclaim(&self, name: &str) -> Result<()> {
        self.store
            .reclaim(self.memory.gpu, &format!("model.dfly_layers.{name}"))
    }
    fn rows(
        &mut self,
        name: &str,
        n: usize,
        k: usize,
        rank: usize,
        world: usize,
    ) -> Result<DevicePtr> {
        let src = self.source(name, &[n, k])?;
        let out = self.copy(src.offset(rank * (n / world) * k * 2), n / world * k * 2)?;
        self.reclaim(name)?;
        Ok(out)
    }
    fn columns(
        &mut self,
        name: &str,
        n: usize,
        k: usize,
        rank: usize,
        world: usize,
    ) -> Result<DevicePtr> {
        let src = self.source(name, &[n, k])?;
        let out = self.memory.alloc(n * (k / world) * 2)?;
        let stream = self.memory.gpu.default_stream();
        self.memory.gpu.copy_d2d_2d_async(
            src.offset(rank * (k / world) * 2),
            k * 2,
            out,
            (k / world) * 2,
            (k / world) * 2,
            n,
            stream,
        )?;
        self.memory.gpu.synchronize(stream)?;
        self.reclaim(name)?;
        Ok(out)
    }
}

pub(super) fn load<'a>(
    store: &WeightStore,
    gpu: &'a dyn GpuBackend,
    c: Config,
) -> Result<Dfly<'a>> {
    let ops = ops::Ops::load(gpu, c)?;
    let mut l = Loader {
        memory: Owned::new(gpu),
        store,
    };
    let mut layers = Vec::new();
    let heads = 32 / c.world;
    for i in 0..5 {
        let a = format!("{i}.self_attn.");
        let src = l.source(&format!("{a}kv_b_proj.weight"), &[14336, 512])?;
        let mut kv = vec![0u8; heads * 448 * 512 * 2];
        gpu.copy_d2h(src.offset(c.rank * kv.len()), &mut kv)?;
        l.reclaim(&format!("{a}kv_b_proj.weight"))?;
        let mut key = vec![0u8; heads * 512 * 192 * 2];
        let mut value = Vec::with_capacity(heads * 256 * 512 * 2);
        for h in 0..heads {
            for k in 0..192 {
                for d in 0..512 {
                    let from = (h * 448 * 512 + k * 512 + d) * 2;
                    let to = (h * 512 * 192 + d * 192 + k) * 2;
                    key[to..to + 2].copy_from_slice(&kv[from..from + 2]);
                }
            }
            let start = (h * 448 + 192) * 512 * 2;
            value.extend_from_slice(&kv[start..start + 256 * 512 * 2]);
        }
        layers.push(Layer {
            input_norm: l.full(&format!("{i}.input_layernorm.weight"), &[H])?,
            post_norm: l.full(&format!("{i}.post_attention_layernorm.weight"), &[H])?,
            qa: l.full(&format!("{a}q_a_proj.weight"), &[1024, H])?,
            qan: l.full(&format!("{a}q_a_layernorm.weight"), &[1024])?,
            qb: l.rows(&format!("{a}q_b_proj.weight"), 8192, 1024, c.rank, c.world)?,
            kva: l.full(&format!("{a}kv_a_proj_with_mqa.weight"), &[576, H])?,
            kvn: l.full(&format!("{a}kv_a_layernorm.weight"), &[512])?,
            wk: l.memory.upload(&key)?,
            wv: l.memory.upload(&value)?,
            gate: l.rows(&format!("{a}linear_gate.weight"), 8192, H, c.rank, c.world)?,
            out: l.columns(&format!("{a}o_proj.weight"), H, 8192, c.rank, c.world)?,
            sink: {
                let src = l.source(&format!("{a}learnable_sink_param"), &[32])?;
                let out = l.copy(src.offset(c.rank * heads * 2), heads * 2)?;
                l.reclaim(&format!("{a}learnable_sink_param"))?;
                out
            },
            attn_norm: l.full(&format!("{a}post_proj_norm.weight"), &[H])?,
            mg: l.rows(
                &format!("{i}.mlp.gate_proj.weight"),
                9216,
                H,
                c.rank,
                c.world,
            )?,
            mu: l.rows(&format!("{i}.mlp.up_proj.weight"), 9216, H, c.rank, c.world)?,
            md: l.columns(
                &format!("{i}.mlp.down_proj.weight"),
                H,
                9216,
                c.rank,
                c.world,
            )?,
        });
    }
    let fusion = Fusion {
        en: l.full("mtp_fusion.enorm.weight", &[H])?,
        hn: l.full("mtp_fusion.hnorm.weight", &[H])?,
        projection: l.full("mtp_fusion.eh_proj.weight", &[H, 2 * H])?,
        gate: l.full("mtp_fusion.gate", &[H])?,
    };
    let correction = Correction {
        en: l.full("hidden_correction.embed_norm.weight", &[H])?,
        hn: l.full("hidden_correction.hidden_norm.weight", &[H])?,
        gate: l.full("hidden_correction.gate_proj.weight", &[H, 2 * H])?,
        up: l.full("hidden_correction.up_proj.weight", &[H, 2 * H])?,
        down: l.full("hidden_correction.down_proj.weight", &[H, H])?,
    };
    let final_norm = l.full("final_norm.weight", &[H])?;
    let mut bf = |n| l.memory.alloc(n * 2);
    let scratch = Scratch {
        hidden: bf(ROWS * H)?,
        residual: bf(ROWS * H)?,
        norm: bf(ROWS * H)?,
        qc: bf(ROWS * 1024)?,
        kv: bf(ROWS * 576)?,
        q: bf(ROWS * heads * 256)?,
        absorbed: bf(ROWS * heads * 576)?,
        latent: bf(ROWS * heads * 512)?,
        values: bf(ROWS * heads * 256)?,
        gates: bf(ROWS * heads * 256)?,
        partial: bf(ROWS * H)?,
        mg: bf(ROWS * 9216 / c.world)?,
        mu: bf(ROWS * 9216 / c.world)?,
        ma: bf(ROWS * 9216 / c.world)?,
        concat: bf(2 * H)?,
        cg: bf(H)?,
        cu: bf(H)?,
        ca: bf(H)?,
        delta: bf(H)?,
        corrected: bf(H)?,
        final_hidden: bf(ROWS * H)?,
        embedding: bf(H)?,
        logits: l.memory.alloc(c.vocab * 4)?,
        id: l.memory.alloc(4)?,
        ids: l.memory.alloc(ROWS * 4)?,
    };
    Ok(Dfly {
        config: c,
        layers,
        fusion,
        correction,
        final_norm,
        scratch,
        ops,
        _memory: l.memory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::{gpu::mock::MockGpuBackend, weights::WeightTensor};
    use std::collections::HashMap;

    #[test]
    fn shards_copy_owned_rows_and_columns_and_reject_reusing_consumed_sources() {
        for rank in 0..2 {
            for columns in [false, true] {
                let gpu = MockGpuBackend::new();
                let bytes: Vec<u8> = (0..32u16).flat_map(u16::to_le_bytes).collect();
                let src = gpu.alloc(bytes.len()).unwrap();
                gpu.copy_h2d(&bytes, src).unwrap();
                let store = WeightStore::from_map(HashMap::from([(
                    "model.dfly_layers.test".into(),
                    WeightTensor {
                        ptr: src,
                        shape: vec![4, 8],
                        dtype: WeightDtype::BF16,
                    },
                )]));
                let mut l = Loader {
                    memory: Owned::new(&gpu),
                    store: &store,
                };
                let out = if columns {
                    l.columns("test", 4, 8, rank, 2)
                } else {
                    l.rows("test", 4, 8, rank, 2)
                }
                .unwrap();
                assert!(store.was_reclaimed("model.dfly_layers.test"));
                assert!(l.rows("test", 4, 8, rank, 2).is_err());
                let mut actual = vec![0u8; 32];
                gpu.copy_d2h(out, &mut actual).unwrap();
                let expected = if columns {
                    bytes
                        .chunks_exact(16)
                        .flat_map(|r| r[rank * 8..rank * 8 + 8].iter().copied())
                        .collect()
                } else {
                    bytes[rank * 32..rank * 32 + 32].to_vec()
                };
                assert_eq!(actual, expected);
            }
        }
    }
}
