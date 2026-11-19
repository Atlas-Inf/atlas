// SPDX-License-Identifier: AGPL-3.0-only

use super::Shared;
use crate::layers::hy4::dfly;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static STEPS: AtomicUsize = AtomicUsize::new(0);

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
    let anchor = history.len();
    let [hidden_bytes, cache_bytes, index_bytes] =
        sizes(shared.config.hidden_size, anchor, input.count)?;
    ensure!(
        input.caches.iter().all(|cache| cache.len == anchor),
        "Hy4 DFly capture history/cache mismatch"
    );
    let path = PathBuf::from(path).join(format!("step-{step:05}"));
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::create_dir(&path)?;
    shared.gpu.synchronize(stream)?;
    let write = |name: &str, pointer, len| -> Result<()> {
        let mut bytes = vec![0; len];
        shared.gpu.copy_d2h(pointer, &mut bytes)?;
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    write("hidden.bf16", input.target_hidden, hidden_bytes)?;
    for (i, cache) in input.caches.iter().enumerate() {
        write(&format!("cache-{i}.bf16"), cache.latent, cache_bytes)?;
        write(&format!("indices-{i}.i32"), cache.indices, index_bytes)?;
    }
    Ok(Some(Capture {
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
    }))
}

impl Capture {
    pub(super) fn finish(mut self, tokens: &[u32]) -> Result<()> {
        ensure!(
            tokens.len() == self.metadata["count"].as_u64().unwrap() as usize,
            "Hy4 DFly capture output count"
        );
        self.metadata["atlas"] = json!(tokens);
        std::fs::write(
            self.path.join("metadata.json"),
            serde_json::to_vec_pretty(&self.metadata)?,
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
