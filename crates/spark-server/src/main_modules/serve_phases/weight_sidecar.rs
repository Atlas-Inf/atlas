// SPDX-License-Identifier: AGPL-3.0-only

//! Weights that live in a checkpoint repo under a file name the safetensors
//! loader does not look for, read without copying or renaming anything:
//!
//! - `--weight-overlay <id>`: an overlay repo's `*.safetensors` (Inkling 2's
//!   Spark NVFP4 overlay ships `spark-nvfp4.safetensors`).
//! - a drafter shipped inside its target's repo: Inkling 2 Mini's DSpark
//!   drafter is `mtp.safetensors`, described by `config.json`'s `mtp_config`,
//!   so `--draft-model thinkingmachines/Inkling-2-Mini` serves it as is.
//!
//! Each file is presented to the loader as `model.safetensors` in a private
//! directory under the temp dir that holds one symlink to it.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// A directory whose only entry is `model.safetensors` -> `file`.
fn single_file_dir(file: &Path, tag: &str) -> Result<PathBuf> {
    let file = file
        .canonicalize()
        .with_context(|| format!("weight file {}", file.display()))?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&file, &mut h);
    let dir = std::env::temp_dir().join(format!(
        "atlas-{tag}-{:016x}-{}",
        std::hash::Hasher::finish(&h),
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let link = dir.join("model.safetensors");
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&file, &link)
        .with_context(|| format!("link {} -> {}", link.display(), file.display()))?;
    #[cfg(not(unix))]
    anyhow::bail!("a weight file outside model.safetensors* needs a Unix host");
    Ok(dir)
}

/// The `*.safetensors` files of an overlay directory, name-sorted.
fn safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("weight overlay {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "weight overlay {} has no .safetensors file",
        dir.display()
    );
    Ok(files)
}

/// Load `--weight-overlay` (when given) into `store`.
pub(crate) fn absorb_weight_overlay(
    args: &crate::cli::ServeArgs,
    store: &mut spark_runtime::weights::WeightStore,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<()> {
    use spark_runtime::weights::WeightLoader;
    let Some(id) = args.weight_overlay.as_deref() else {
        return Ok(());
    };
    let dir = crate::model_resolver::resolve_model_dir(id, args.cache_dir.as_deref())
        .with_context(|| format!("resolve --weight-overlay {id}"))?;
    for file in safetensors_files(&dir)? {
        let mut loader = spark_runtime::weights::SafetensorsLoader::new();
        loader.peak_memory_multiplier = None;
        let overlay = loader
            .load(&single_file_dir(&file, "overlay")?, gpu, 0)
            .with_context(|| format!("load weight overlay {}", file.display()))?;
        tracing::info!(
            "weight overlay {id}: {} tensors from {}",
            overlay.len(),
            file.display()
        );
        store.absorb(overlay)?;
    }
    Ok(())
}

/// A drafter shipped inside its target's checkpoint: the drafter config
/// (`config.json`'s `mtp_config` plus the tokenizer's vocab size) and the
/// directory to load its weights from, when `dir` is such a checkpoint.
pub(crate) fn embedded_drafter(dir: &Path, config_json: &str) -> Result<Option<(String, PathBuf)>> {
    let raw: serde_json::Value =
        serde_json::from_str(config_json).context("Parsing drafter config.json")?;
    let Some(mtp) = raw.get("mtp_config").and_then(|m| m.as_object()) else {
        return Ok(None);
    };
    let weights = dir.join("mtp.safetensors");
    if !weights.exists() {
        return Ok(None);
    }
    let vocab = raw
        .pointer("/text_config/unpadded_vocab_size")
        .and_then(|v| v.as_u64())
        .context("a checkpoint with an mtp_config drafter needs text_config.unpadded_vocab_size")?;
    let mut config = mtp.clone();
    config.insert("vocab_size".into(), vocab.into());
    tracing::info!(
        "DFlash: drafter is the checkpoint's own mtp.safetensors (mtp_config, vocab {vocab})"
    );
    Ok(Some((
        serde_json::Value::Object(config).to_string(),
        single_file_dir(&weights, "drafter")?,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_drafter_reads_mtp_config_and_vocab() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("mtp.safetensors"), b"x").unwrap();
        let cfg = r#"{"mtp_config":{"markov_head_type":"rnn_hidden","block_size":8},
                      "text_config":{"unpadded_vocab_size":201088}}"#;
        let (json, wdir) = embedded_drafter(dir.path(), cfg).unwrap().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["vocab_size"], 201088);
        assert_eq!(v["block_size"], 8);
        assert!(wdir.join("model.safetensors").exists());
    }

    #[test]
    fn plain_drafter_is_not_embedded() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            embedded_drafter(dir.path(), r#"{"hidden_size":8}"#)
                .unwrap()
                .is_none()
        );
        // mtp_config without the weights file is not an embedded drafter either
        let cfg = r#"{"mtp_config":{},"text_config":{"unpadded_vocab_size":4}}"#;
        assert!(embedded_drafter(dir.path(), cfg).unwrap().is_none());
    }
}
