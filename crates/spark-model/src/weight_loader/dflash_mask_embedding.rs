// SPDX-License-Identifier: AGPL-3.0-only

//! Learned DFlash mask-token embedding (`mask_embedding.pt`).
//!
//! MiMo-V2.6-Flash's drafter fills its γ-1 query slots with `mask_token_id`,
//! whose row in the target embedding table is all zeros; the trained vector
//! ships separately as a `torch.save` archive holding one BF16 tensor. A
//! torch archive is an uncompressed ZIP whose tensor bytes live in the
//! `<archive>/data/0` entry, so the raw little-endian BF16 payload is read
//! straight from the ZIP central directory without a pickle parser.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use super::dflash_loader::{DflashConfig, parse_dflash_config};

const EOCD_SIG: u32 = 0x0605_4b50;
const CDIR_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

fn u16_at(b: &[u8], off: usize) -> Result<usize> {
    let s = b.get(off..off + 2).context("zip: truncated")?;
    Ok(u16::from_le_bytes([s[0], s[1]]) as usize)
}

fn u32_at(b: &[u8], off: usize) -> Result<u32> {
    let s = b.get(off..off + 4).context("zip: truncated")?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Raw bytes of the first tensor storage (`*/data/0`) in a `torch.save` ZIP.
pub fn torch_storage0_bytes(archive: &[u8]) -> Result<Vec<u8>> {
    let eocd = (0..archive.len().saturating_sub(21))
        .rev()
        .find(|&i| u32_at(archive, i).ok() == Some(EOCD_SIG))
        .context("zip: no end-of-central-directory record")?;
    let entries = u16_at(archive, eocd + 10)?;
    let mut p = u32_at(archive, eocd + 16)? as usize;
    for _ in 0..entries {
        ensure!(
            u32_at(archive, p)? == CDIR_SIG,
            "zip: bad central directory entry"
        );
        let method = u16_at(archive, p + 10)?;
        let size = u32_at(archive, p + 20)? as usize;
        let (n, e, c) = (
            u16_at(archive, p + 28)?,
            u16_at(archive, p + 30)?,
            u16_at(archive, p + 32)?,
        );
        let local = u32_at(archive, p + 42)? as usize;
        let name = archive
            .get(p + 46..p + 46 + n)
            .context("zip: truncated name")?;
        if name.ends_with(b"/data/0") {
            ensure!(
                method == 0,
                "zip: tensor storage is compressed (method {method})"
            );
            ensure!(
                u32_at(archive, local)? == LOCAL_SIG,
                "zip: bad local header"
            );
            let start = local + 30 + u16_at(archive, local + 26)? + u16_at(archive, local + 28)?;
            let data = archive
                .get(start..start + size)
                .context("zip: truncated data")?;
            return Ok(data.to_vec());
        }
        p += 46 + n + e + c;
    }
    bail!("zip: no `data/0` tensor storage entry")
}

/// `<drafter_dir>/mask_embedding.pt` as raw BF16 bytes, if the file exists.
pub fn load_mask_embedding(drafter_dir: &Path) -> Result<Option<Vec<u8>>> {
    let path = drafter_dir.join("mask_embedding.pt");
    if !path.exists() {
        return Ok(None);
    }
    let archive = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    torch_storage0_bytes(&archive)
        .with_context(|| format!("parse {}", path.display()))
        .map(Some)
}

/// The drafter's `config.json` plus its learned mask embedding, if shipped.
pub fn load_drafter_config(drafter_dir: &Path) -> Result<DflashConfig> {
    let path = drafter_dir.join("config.json");
    let json = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read drafter config.json at {}", path.display()))?;
    let mut config = parse_dflash_config(&json)?;
    config.mask_embedding = load_mask_embedding(drafter_dir)?;
    if let Some(bytes) = &config.mask_embedding {
        ensure!(
            bytes.len() == config.hidden_size * 2,
            "mask_embedding.pt holds {} bytes, expected {} BF16 values",
            bytes.len(),
            config.hidden_size
        );
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::torch_storage0_bytes;

    /// Minimal stored (method 0) ZIP with the given entries, laid out the way
    /// `torch.save` writes them: local headers + data, then the directory.
    fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let (mut out, mut dir) = (Vec::new(), Vec::new());
        for (name, data) in entries {
            let off = out.len() as u32;
            let (n, len) = (name.len() as u16, data.len() as u32);
            out.extend(0x0403_4b50u32.to_le_bytes());
            out.extend([0u8; 14]);
            out.extend(len.to_le_bytes());
            out.extend(len.to_le_bytes());
            out.extend(n.to_le_bytes());
            out.extend(0u16.to_le_bytes());
            out.extend(name.as_bytes());
            out.extend(*data);
            dir.extend(0x0201_4b50u32.to_le_bytes());
            dir.extend([0u8; 16]);
            dir.extend(len.to_le_bytes());
            dir.extend(len.to_le_bytes());
            dir.extend(n.to_le_bytes());
            dir.extend([0u8; 12]);
            dir.extend(off.to_le_bytes());
            dir.extend(name.as_bytes());
        }
        let (cd_off, cd_len) = (out.len() as u32, dir.len() as u32);
        out.extend(dir);
        out.extend(0x0605_4b50u32.to_le_bytes());
        out.extend([0u8; 4]);
        out.extend((entries.len() as u16).to_le_bytes());
        out.extend((entries.len() as u16).to_le_bytes());
        out.extend(cd_len.to_le_bytes());
        out.extend(cd_off.to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out
    }

    #[test]
    fn reads_the_tensor_storage_entry() {
        let zip = stored_zip(&[
            ("mask_embedding/data.pkl", b"pickle"),
            ("mask_embedding/data/0", &[1, 2, 3, 4]),
            ("mask_embedding/version", b"3"),
        ]);
        assert_eq!(torch_storage0_bytes(&zip).unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn missing_storage_is_an_error() {
        let zip = stored_zip(&[("mask_embedding/data.pkl", b"pickle")]);
        assert!(torch_storage0_bytes(&zip).is_err());
    }
}
