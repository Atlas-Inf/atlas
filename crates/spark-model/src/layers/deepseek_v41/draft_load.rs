// SPDX-License-Identifier: AGPL-3.0-only

//! A rank's part of the draft head from a checkpoint (`draft.rs`).

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::super::block::{Block, BlockKernels, Candidates, Handoff, Scratch, Shape, Shared};
use super::super::cache_recipe::WINDOW;
use super::super::draft_head::HostHead;
use super::super::draft_ops::{DraftKernels, Markov};
use super::super::ops::HEAD_DIM;
use super::super::parts::{Held, PARTS};
use super::super::seen::Seen;
use super::super::shape::ladder;
use super::super::weights::{BlockWeights, fp8_held, tensor};
use super::{BLOCK_ROOM, Buffers, Draft, WORD};
use crate::layers::ops::mxfp8_quantize;
use crate::weight_map::Mxfp8Weight;

impl Draft {
    /// This rank's part of the draft head, from `store`. `embed` and `head`
    /// are the model's own.
    pub fn load(
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        embed: DevicePtr,
        head: DevicePtr,
    ) -> Result<Self> {
        let main = Shape::from_config(config)?;
        let shape = Shape {
            experts: match config.dspark_n_routed_experts {
                0 => main.experts,
                experts => experts,
            },
            top_k: match config.dspark_num_experts_per_tok {
                0 => main.top_k,
                top_k => top_k,
            },
            ..main
        };
        let (stages, rows, targets) = (
            config.num_mtp_modules,
            config.dspark_block_size,
            config.dspark_target_layer_ids.len(),
        );
        let (hidden, vocab) = (shape.hidden, config.vocab_size);
        ensure!(
            stages > 0 && (1..=BLOCK_ROOM).contains(&rows) && targets > 0,
            "deepseek_v41: a draft head of {stages} stages, blocks of {rows} and {targets} target layers"
        );
        ensure!(
            vocab.is_multiple_of(PARTS) && hidden.is_multiple_of(PARTS * 32),
            "deepseek_v41: a head of [{vocab}, {hidden}] does not split in {PARTS} parts"
        );
        let held = Held::of_rank(config.ep_rank, config.ep_world_size.max(1))?;
        let stream = gpu.default_stream();
        let kernels = BlockKernels::load(gpu)?;
        let positions = match config.max_seq_len {
            0 => config.max_position_embeddings,
            served => served.min(config.max_position_embeddings),
        };
        let shared = Shared {
            shape,
            held,
            kernels,
            scratch: Arc::new(Scratch::alloc_rows(
                gpu,
                &shape,
                positions + BLOCK_ROOM,
                BLOCK_ROOM,
                held,
            )?),
            handoff: Arc::new(std::sync::Mutex::new(Handoff::default())),
        };
        // A stage attends over a window only, which is not YaRN-scaled.
        let freqs = ladder(gpu, config, shape.rope, 0, config.window_rope_theta)?;
        let blocks = (0..stages)
            .map(|stage| {
                let name = format!("mtp.{stage}");
                let weights = BlockWeights::load_named(
                    store,
                    gpu,
                    &kernels.parts,
                    &shape,
                    &name,
                    0,
                    false,
                    held,
                    false,
                    true,
                    stream,
                )
                .with_context(|| name.clone())?;
                Ok(Block {
                    index: config.num_hidden_layers + stage,
                    ratio: 0,
                    candidates: Candidates::None,
                    freqs,
                    weights,
                    gemm: None,
                    // A draft is checked, not kept: the fast form.
                    tensor_cores: true,
                    shared: shared.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let main_proj = fp8_held(store, gpu, "mtp.0.main_proj", held)?;
        ensure!(
            main_proj.n == hidden && main_proj.k == targets * hidden,
            "mtp.0.main_proj [{}, {}]",
            main_proj.n,
            main_proj.k
        );
        let last = format!("mtp.{}", stages - 1);
        // The Markov head, whole on every rank: a rank reads its part of
        // the vocabulary's rows of `head`, and any token's of `embed`.
        let markov = {
            let rank = config.dspark_markov_rank;
            let table = |name: &str| -> Result<DevicePtr> {
                let name = format!("{last}.markov_head.{name}.weight");
                ensure!(
                    store.get(&name)?.num_elements() == vocab * rank,
                    "{name}: not [{vocab}, {rank}]"
                );
                tensor(store, &name, WeightDtype::BF16)
            };
            ensure!(
                (1..=256).contains(&rank),
                "deepseek_v41: a Markov head of rank {rank}"
            );
            Markov {
                embed: table("embed")?,
                head: table("head")?,
                vocab,
                rank,
            }
        };
        let bf16 = |values: usize| gpu.alloc(values * 2);
        let copies = shape.hc.copies;
        let (_, room, drafts) = Buffers::read_layout(rows, hidden);
        let read = Seen::alloc(gpu, room + drafts)?;
        let buffers = Buffers {
            word: Seen::alloc(gpu, WORD)?,
            means: bf16(WINDOW * targets * hidden)?,
            projected: bf16(WINDOW * hidden)?,
            main: bf16(WINDOW * hidden)?,
            main_normed: bf16(WINDOW * hidden)?,
            keys: bf16(WINDOW * HEAD_DIM)?,
            keys_normed: bf16(WINDOW * HEAD_DIM)?,
            recv: bf16(WINDOW * hidden)?,
            embeds: bf16(rows * hidden)?,
            streams: bf16(rows * copies * hidden)?,
            pre: gpu.alloc(rows * copies * 4)?,
            x: read.at,
            normed: bf16(rows * hidden)?,
            logits: gpu.alloc(held.count * rows * vocab / PARTS * 4)?,
            found: gpu.alloc(rows * PARTS * markov.bytes())?,
            drafts: read.at.offset(room),
            read,
        };
        // The confidence head is the host's, where requests are served.
        let host = match config.ep_rank {
            0 => Some(HostHead::load(
                store,
                gpu,
                &last,
                hidden,
                vocab,
                markov.rank,
            )?),
            _ => None,
        };
        Ok(Self {
            shape,
            held,
            kernels,
            own: DraftKernels::load(gpu)?,
            stages: blocks,
            main_proj,
            main_norm: tensor(store, "mtp.0.main_norm.weight", WeightDtype::BF16)?,
            norm: tensor(store, &format!("{last}.norm.weight"), WeightDtype::BF16)?,
            embed,
            head: {
                let (first, rows) = (held.first * vocab / PARTS, held.count * vocab / PARTS);
                let quantize = gpu.kernel("mxfp8_gemv", "mxfp8_quantize_bf16")?;
                let fp8 = Mxfp8Weight::alloc(gpu, rows, hidden)?;
                let bf16 = head.offset(first * hidden * 2);
                mxfp8_quantize(
                    gpu, quantize, bf16, fp8.data, fp8.scales, rows, hidden, stream,
                )?;
                gpu.synchronize(stream)?;
                fp8
            },
            markov,
            noise: config.dspark_noise_token_id,
            rows,
            targets,
            buffers,
            host,
        })
    }
}
