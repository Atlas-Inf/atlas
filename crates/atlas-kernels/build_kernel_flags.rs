// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;

pub type KernelFlags = BTreeMap<String, Vec<String>>;

/// Per-source additions are deliberately separate from target-wide flags:
/// sparse attention may use fast math while cache packing must remain precise.
pub fn parse(config: &toml::Value, vendor: &str) -> Result<KernelFlags, String> {
    let key = match vendor {
        "apple" | "metal" => "kernel_metal_flags",
        _ => "kernel_nvcc_flags",
    };
    let Some(value) = config.get("build").and_then(|build| build.get(key)) else {
        return Ok(KernelFlags::new());
    };
    let table = value
        .as_table()
        .ok_or_else(|| format!("build.{key} must be a table"))?;
    let mut parsed = KernelFlags::new();
    for (stem, value) in table {
        if stem.is_empty()
            || !stem
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
        {
            return Err(format!("build.{key}.{stem}: expected a source file stem"));
        }
        let array = value
            .as_array()
            .ok_or_else(|| format!("build.{key}.{stem} must be an array"))?;
        let flags = array
            .iter()
            .map(|flag| {
                flag.as_str()
                    .filter(|s| !s.trim().is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| format!("build.{key}.{stem}: flags must be nonempty strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        parsed.insert(stem.clone(), flags);
    }
    Ok(parsed)
}

/// Preserve compiler argument order: the last conflicting option wins.
pub fn for_source(base: &[String], overrides: &KernelFlags, stem: &str) -> Vec<String> {
    base.iter()
        .chain(overrides.get(stem).into_iter().flatten())
        .cloned()
        .collect()
}
