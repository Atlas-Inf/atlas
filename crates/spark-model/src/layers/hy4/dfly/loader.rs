// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::weights::WeightDtype;

struct Loader<'a, 's> {
    memory: Owned<'a>,
    store: &'s WeightStore,
}
impl Loader<'_, '_> {
    fn source(&self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let t = self.store.get(&format!("model.dfly_layers.{name}"))?;
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
        self.copy(src, shape.iter().product::<usize>() * 2)
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
        self.copy(src.offset(rank * (n / world) * k * 2), n / world * k * 2)
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
        for row in 0..n {
            self.memory.gpu.copy_d2d(
                src.offset((row * k + rank * (k / world)) * 2),
                out.offset(row * (k / world) * 2),
                k / world * 2,
            )?;
        }
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
                l.copy(src.offset(c.rank * heads * 2), heads * 2)?
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
