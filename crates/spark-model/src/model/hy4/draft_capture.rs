// SPDX-License-Identifier: AGPL-3.0-only

use super::Shared;
use crate::layers::hy4::dfly;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static STEPS: AtomicUsize = AtomicUsize::new(0);

fn write_private(path: impl AsRef<std::path::Path>, bytes: &[u8]) -> Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)?.write_all(bytes)?;
    Ok(())
}

pub(super) struct Capture {
    path: PathBuf,
    metadata: Value,
}

fn sizes(width: usize, anchor: usize, count: usize) -> Result<[usize; 3]> {
    ensure!(
        width > 0 && anchor > 0 && (1..=5).contains(&count),
        "Hy4 DFly capture dimensions"
    );
    Ok([width * 2, anchor * 576 * 2, 2048 * 4])
}

pub(super) fn begin(
    shared: &Shared,
    input: &dfly::Input,
    history: &[u32],
    stream: u64,
) -> Result<Option<Capture>> {
    if shared.config.tp_rank != 0 {
        return Ok(None);
    }
    let Some(path) = std::env::var_os("ATLAS_HY4_DFLY_CAPTURE_DIR") else {
        return Ok(None);
    };
    ensure!(!path.is_empty(), "Hy4 DFly capture directory is empty");
    let limit: usize = std::env::var("ATLAS_HY4_DFLY_CAPTURE_MAX_STEPS")
        .context("Hy4 DFly capture requires explicit MAX_STEPS")?
        .parse()?;
    ensure!(limit > 0, "Hy4 DFly capture limit must be positive");
    let step = STEPS.fetch_add(1, Ordering::Relaxed);
    if step >= limit {
        return Ok(None);
    }
    capture(shared, input, history, stream, PathBuf::from(path), step).map(Some)
}

fn capture(
    shared: &Shared,
    input: &dfly::Input,
    history: &[u32],
    stream: u64,
    directory: PathBuf,
    step: usize,
) -> Result<Capture> {
    let anchor = history.len();
    let [hidden_bytes, cache_bytes, index_bytes] =
        sizes(shared.config.hidden_size, anchor, input.count)?;
    ensure!(
        input.caches.iter().all(|cache| cache.len == anchor),
        "Hy4 DFly capture history/cache mismatch"
    );
    let path = directory.join(format!("step-{step:05}"));
    let mut directory = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        directory.mode(0o700);
    }
    directory.recursive(true).create(path.parent().unwrap())?;
    directory.recursive(false).create(&path)?;
    shared.gpu.synchronize(stream)?;
    let write = |name: &str, pointer, len| -> Result<()> {
        let mut bytes = vec![0; len];
        shared.gpu.copy_d2h(pointer, &mut bytes)?;
        write_private(path.join(name), &bytes)?;
        Ok(())
    };
    write("hidden.bf16", input.target_hidden, hidden_bytes)?;
    for (i, cache) in input.caches.iter().enumerate() {
        write(&format!("cache-{i}.bf16"), cache.latent, cache_bytes)?;
        write(&format!("indices-{i}.i32"), cache.indices, index_bytes)?;
    }
    Ok(Capture {
        path,
        metadata: json!({
            "history": history,
            "anchor": anchor,
            "bonus": input.bonus,
            "count": input.count,
            "width": shared.config.hidden_size,
            "world": shared.config.tp_world_size,
            "target_layers": dfly::TARGET_LAYERS,
        }),
    })
}

pub(super) fn teacher_enabled(rank: usize) -> bool {
    rank == 0 && std::env::var_os("ATLAS_HY4_DFLY_TEACHER_DIR").is_some()
}

#[derive(serde::Deserialize)]
struct Plan {
    anchor: usize,
    bonus: u32,
    count: usize,
}

pub(super) fn teacher(
    shared: &Shared,
    hidden: spark_runtime::gpu::DevicePtr,
    caches: [dfly::TargetCache; 5],
    history: &[u32],
    stream: u64,
) -> Result<()> {
    let directory = PathBuf::from(std::env::var_os("ATLAS_HY4_DFLY_TEACHER_DIR").unwrap());
    let plans: Vec<Plan> = serde_json::from_slice(&std::fs::read(directory.join("plans.json"))?)?;
    let Some((step, plan)) = plans
        .iter()
        .enumerate()
        .find(|(_, p)| p.anchor == history.len())
    else {
        return Ok(());
    };
    let input = dfly::Input {
        target_hidden: hidden,
        caches,
        bonus: plan.bonus,
        count: plan.count,
    };
    let mut frame = capture(shared, &input, history, stream, directory, step)?;
    frame.metadata["kind"] = json!("serial_teacher");
    frame.save()
}

impl Capture {
    pub(super) fn finish(mut self, tokens: &[u32]) -> Result<()> {
        ensure!(
            tokens.len() == self.metadata["count"].as_u64().unwrap() as usize,
            "Hy4 DFly capture output count"
        );
        self.metadata["atlas"] = json!(tokens);
        self.save()
    }

    fn save(&self) -> Result<()> {
        write_private(
            self.path.join("metadata.json"),
            &serde_json::to_vec_pretty(&self.metadata)?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_shapes_preserve_full_accepted_prefix_and_all_indices() {
        assert_eq!(sizes(3072, 57, 5).unwrap(), [6144, 65664, 8192]);
        assert_eq!(sizes(32, 1, 1).unwrap(), [64, 1152, 8192]);
    }

    #[test]
    fn capture_rejects_empty_and_invalid_shapes() {
        for (width, anchor, count) in [(0, 57, 5), (3072, 0, 5), (3072, 57, 0), (3072, 57, 6)] {
            assert!(sizes(width, anchor, count).is_err());
        }
    }
}
