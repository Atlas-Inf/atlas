// SPDX-License-Identifier: AGPL-3.0-only

use sha2::{Digest, Sha256};
use spark_runtime::weights::WeightStore;

pub(super) fn weight_layout_checksum(store: &WeightStore) -> String {
    let mut names: Vec<&str> = store.names().collect();
    names.sort_unstable();
    let mut digest = Sha256::new();
    for name in names {
        let Ok(tensor) = store.get(name) else {
            continue;
        };
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(format!("{:?}", tensor.dtype).as_bytes());
        for dimension in &tensor.shape {
            digest.update((*dimension as u64).to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}
