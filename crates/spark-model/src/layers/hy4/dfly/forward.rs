// SPDX-License-Identifier: AGPL-3.0-only

use super::{ops::Op, *};

impl Dfly<'_> {
    pub(super) fn forward(
        &self,
        input: &Input,
        head: Head,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let c = self.config;
        let s = &self.scratch;
        let o = &self.ops;
        let rows = input.count + 1;
        let heads = 32 / c.world;
        let ids: Vec<u8> = std::iter::once(input.bonus)
            .chain(std::iter::repeat_n(120009, input.count))
            .flat_map(u32::to_le_bytes)
            .collect();
        o.gpu.copy_h2d_async_retained(&ids, s.ids, stream)?;
        o.embed(head.embedding, s.ids, s.hidden, rows, stream)?;
        o.norm(s.hidden, self.fusion.en, s.concat, [1, H, H, H], stream)?;
        o.norm(
            input.target_hidden,
            self.fusion.hn,
            s.concat.offset(H * 2),
            [1, H, H, H],
            stream,
        )?;
        o.linear(
            s.concat,
            self.fusion.projection,
            s.partial,
            [1, H, 2 * H],
            stream,
        )?;
        o.fuse(s.hidden, s.partial, self.fusion.gate, s.hidden, stream)?;
        for (i, b) in self.layers.iter().enumerate() {
            if i > 0 {
                o.element(Op::Add, s.hidden, s.residual, s.hidden, rows * H, stream)?;
            }
            o.gpu
                .copy_d2d_async(s.hidden, s.residual, rows * H * 2, stream)?;
            o.norm(s.hidden, b.input_norm, s.norm, [rows, H, H, H], stream)?;
            o.linear(s.norm, b.qa, s.qc, [rows, 1024, H], stream)?;
            o.norm(s.qc, b.qan, s.qc, [rows, 1024, 1024, 1024], stream)?;
            o.linear(s.qc, b.qb, s.q, [rows, heads * 256, 1024], stream)?;
            o.linear(s.norm, b.kva, s.kv, [rows, 576, H], stream)?;
            o.norm(s.kv, b.kvn, s.kv, [rows, 512, 576, 576], stream)?;
            o.heads(s.q, b.wk, s.absorbed, rows, [512, 192, 256, 576], stream)?;
            let cache = input.caches[i];
            o.attention(
                [
                    s.q,
                    s.kv,
                    s.absorbed,
                    cache.latent,
                    cache.indices,
                    b.sink,
                    s.latent,
                ],
                rows,
                cache.len,
                stream,
            )?;
            o.heads(s.latent, b.wv, s.values, rows, [256, 512, 512, 256], stream)?;
            o.linear(s.norm, b.gate, s.gates, [rows, heads * 256, H], stream)?;
            o.gate(s.values, s.gates, rows * heads * 256, stream)?;
            o.linear(s.values, b.out, s.partial, [rows, H, heads * 256], stream)?;
            o.reduce(s.partial, rows, comm, stream)?;
            o.norm(s.partial, b.attn_norm, s.partial, [rows, H, H, H], stream)?;
            o.element(Op::Add, s.partial, s.residual, s.hidden, rows * H, stream)?;
            o.gpu
                .copy_d2d_async(s.hidden, s.residual, rows * H * 2, stream)?;
            o.norm(s.hidden, b.post_norm, s.norm, [rows, H, H, H], stream)?;
            let size = 9216 / c.world;
            o.linear(s.norm, b.mg, s.mg, [rows, size, H], stream)?;
            o.linear(s.norm, b.mu, s.mu, [rows, size, H], stream)?;
            o.element(Op::Silu, s.mg, s.mu, s.ma, rows * size, stream)?;
            o.linear(s.ma, b.md, s.hidden, [rows, H, size], stream)?;
            o.reduce(s.hidden, rows, comm, stream)?;
        }
        o.element(Op::Add, s.hidden, s.residual, s.hidden, rows * H, stream)?;
        o.norm(
            s.hidden,
            self.final_norm,
            s.final_hidden,
            [rows, H, H, H],
            stream,
        )?;
        let b = &self.correction;
        let mut previous = input.bonus;
        let mut tokens = Vec::with_capacity(input.count);
        for step in 0..input.count {
            o.gpu
                .copy_h2d_async_retained(&previous.to_le_bytes(), s.ids, stream)?;
            o.embed(head.embedding, s.ids, s.embedding, 1, stream)?;
            let hidden = s.final_hidden.offset((step + 1) * H * 2);
            o.norm(hidden, b.hn, s.concat, [1, H, H, H], stream)?;
            o.norm(
                s.embedding,
                b.en,
                s.concat.offset(H * 2),
                [1, H, H, H],
                stream,
            )?;
            o.linear(s.concat, b.gate, s.cg, [1, H, 2 * H], stream)?;
            o.linear(s.concat, b.up, s.cu, [1, H, 2 * H], stream)?;
            o.element(Op::Silu, s.cg, s.cu, s.ca, H, stream)?;
            o.linear(s.ca, b.down, s.delta, [1, H, H], stream)?;
            o.element(Op::Add, hidden, s.delta, s.corrected, H, stream)?;
            previous = o.head(s.corrected, head, s.logits, s.id, comm, stream)?;
            tokens.push(previous);
        }
        Ok(tokens)
    }
}
