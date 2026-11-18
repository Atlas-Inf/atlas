// SPDX-License-Identifier: AGPL-3.0-only

//! Real Rust-host DFly checkpoint parity, without loading the target model.
//! Every argument is explicit. A preflight reserve check precedes weight loads.

use anyhow::{Context, Result, ensure};
use atlas_core::scope::ModelResource;
use spark_comm::{CommBackend, nccl_backend::NcclBackend};
use spark_model::layers::hy4::dfly::{Config, Dfly, Head, Input, TargetCache};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::GpuBackend,
    weights::{SafetensorsLoader, WeightLoader},
};
use std::{collections::HashMap, path::Path};

mod io;

fn main() -> Result<()> {
    let mut args = arguments()?;
    let root = args
        .remove("--checkpoint")
        .context("--checkpoint missing")?;
    let fixtures = args.remove("--fixtures").context("--fixtures missing")?;
    let reference = args.remove("--reference").context("--reference missing")?;
    let mode = args.remove("--mode").context("--mode missing")?;
    ensure!(
        mode == "probe" || mode == "parity",
        "--mode must be probe or parity"
    );
    let rank: usize = take(&mut args, "--rank")?;
    let world: usize = take(&mut args, "--world-size")?;
    let master = args.remove("--master").context("--master missing")?;
    let port: u16 = take(&mut args, "--master-port")?;
    let reserve = take::<usize>(&mut args, "--oom-guard-mb")? * 1024 * 1024;
    let budget = take::<usize>(&mut args, "--peak-budget-mb")? * 1024 * 1024;
    ensure!(args.is_empty(), "unknown arguments: {:?}", args.keys());
    let raw = std::fs::read_to_string(Path::new(&root).join("config.json"))?;
    let model = atlas_core::config::parse_config(&raw)?;
    let hy = model.hy4.as_ref().context("HY4 metadata missing")?;
    ensure!(
        model.hidden_size == 3072
            && model.num_attention_heads == 32
            && model.q_lora_rank == 1024
            && model.kv_lora_rank == 512,
        "unexpected HY4 dimensions"
    );
    let config = Config {
        rank,
        world,
        vocab: model.vocab_size,
        eps: model.rms_norm_eps as f32,
        theta: hy.rope_parameters.deepseek_sparse_attention.rope_theta as f32,
    };
    config.validate()?;
    let checkpoint = io::CheckpointIo::open(Path::new(&root))?;
    ensure!(
        budget >= checkpoint.peak_bytes(world, config.vocab)?,
        "configured peak budget is smaller than the checkpoint-derived bound"
    );
    let set = atlas_kernels::ptx_for_config("hy_v4_flash", 3072, &[], None)?
        .context("HY4 kernels missing")?;
    let required = ["hy4_dfly", "hy4_mla", "hy4_model", "bf16_add"];
    let modules: Vec<_> = set
        .modules
        .iter()
        .copied()
        .filter(|(name, _)| required.contains(name))
        .collect();
    ensure!(
        modules.len() == required.len(),
        "DFly kernel module missing"
    );
    println!(
        "DFLY_MODULES loaded={} available={}",
        modules.len(),
        set.modules.len()
    );
    let gpu = AtlasCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    io::guard(g, reserve, budget)?;
    if mode == "probe" {
        return Ok(());
    }
    let mut loader = SafetensorsLoader::new();
    loader.skip_prefixes = checkpoint.auxiliary_skips();
    loader.peak_memory_multiplier = Some(1.);
    let mut store = loader.load(Path::new(&root), g, reserve)?;
    let mut dfly = Dfly::load(&store, g, config)?;
    store.release(g)?;
    io::guard(g, reserve, 0)?;
    let mut buffers = io::Buffers::new(g, reserve);
    let head = Head {
        embedding: checkpoint.rows(
            "model.embed_tokens.weight",
            [config.vocab, 3072],
            0,
            config.vocab,
            &mut buffers,
        )?,
        weight: checkpoint.rows(
            "lm_head.weight",
            [config.vocab, 3072],
            rank * (config.vocab / world),
            config.vocab / world,
            &mut buffers,
        )?,
    };
    println!("DFLY_MEMORY_PHASE before_communicator");
    io::guard(g, reserve, 0)?;
    let comm = if world > 1 {
        let comm = NcclBackend::new(rank, world, &master, port, stream, 1024 * 1024)?;
        comm.set_add_kernel(g.kernel("bf16_add", "bf16_add_inplace")?.0);
        Some(comm)
    } else {
        None
    };
    println!("DFLY_MEMORY_PHASE after_communicator");
    io::guard(g, reserve, 0)?;
    let comm = comm.as_ref().map(|c| c as &dyn CommBackend);
    let reference = io::reference(Path::new(&reference))?;
    ensure!(
        reference.records.len() == 3,
        "expected all three fixed fixtures"
    );
    let mut passed = true;
    for case in reference.records {
        ensure!(
            Path::new(&case.case).components().count() == 1,
            "invalid fixture name"
        );
        let bytes = io::fixture(&Path::new(&fixtures).join(format!("{}.safetensors", case.case)))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let hidden = tensors.tensor("target_hidden")?;
        ensure!(
            hidden.dtype() == safetensors::Dtype::BF16 && hidden.shape() == [1, 3072],
            "invalid fixture hidden"
        );
        let mut caches = Vec::new();
        for i in 0..5 {
            let cache = tensors.tensor(&format!("cache{i}"))?;
            let indices = tensors.tensor(&format!("indices{i}"))?;
            ensure!(
                cache.dtype() == safetensors::Dtype::BF16
                    && cache.shape() == [case.anchor + case.count + 1, 576]
                    && indices.dtype() == safetensors::Dtype::I32
                    && indices.shape() == [2048],
                "invalid cache fixture"
            );
            caches.push(TargetCache {
                latent: buffers.upload(cache.data())?,
                indices: buffers.upload(indices.data())?,
                len: case.anchor,
                capacity: case.anchor + case.count + 1,
            });
        }
        let input = Input {
            target_hidden: buffers.upload(hidden.data())?,
            caches: caches
                .try_into()
                .map_err(|_| anyhow::anyhow!("cache count"))?,
            bonus: case.bonus,
            count: case.count,
        };
        let actual = dfly.draft(&input, head, comm, stream)?;
        for (i, cache) in input.caches.iter().enumerate() {
            let mut prefix = vec![0u8; case.anchor * 576 * 2];
            if !prefix.is_empty() {
                g.copy_d2h(cache.latent, &mut prefix)?;
            }
            ensure!(
                prefix == tensors.tensor(&format!("cache{i}"))?.data()[..prefix.len()],
                "DFly changed accepted target KV history"
            );
        }
        let equal = actual == case.reference;
        passed &= equal;
        println!(
            "{}",
            serde_json::json!({"rank":rank,"world":world,"case":case.case,
            "reference":case.reference,"atlas":actual,"equal":equal,"accepted_kv_unchanged":true})
        );
        io::guard(g, reserve, 0)?;
    }
    ensure!(passed, "DFly RUST OFFLINE PARITY FAILED; do not serve");
    println!("DFly Rust-host TP{world} rank{rank}: 3/3 fixtures and accepted-KV bytes passed");
    Ok(())
}

fn arguments() -> Result<HashMap<String, String>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len().is_multiple_of(2),
        "arguments require explicit values"
    );
    let mut out = HashMap::new();
    for pair in args.chunks_exact(2) {
        ensure!(
            out.insert(pair[0].clone(), pair[1].clone()).is_none(),
            "duplicate argument"
        );
    }
    Ok(out)
}
fn take<T: std::str::FromStr>(args: &mut HashMap<String, String>, key: &str) -> Result<T> {
    args.remove(key)
        .with_context(|| format!("{key} missing"))?
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {key}"))
}
