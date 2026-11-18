// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};

/// BF16 latent512+rope64 rows, FP8 index128 rows with separate FP32 scales.
/// Page tables are request-local; shared immutable prefix pages may alias.
#[derive(Debug, Clone)]
pub struct PagedLayout {
    pub page_size: usize,
    pub physical_pages: usize,
    pub table: Vec<usize>,
    /// A circular SWA table requires resident logical tags at dispatch.
    pub circular: bool,
}
impl PagedLayout {
    pub fn window_pages(window: usize, page_size: usize) -> Result<usize> {
        ensure!(window > 0 && page_size > 0, "Hy4 window page geometry");
        (window - 1)
            .div_ceil(page_size)
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Hy4 window page overflow"))
    }
    pub fn slot(&self, logical: usize) -> Result<usize> {
        ensure!(
            self.page_size > 0 && self.physical_pages > 0 && !self.table.is_empty(),
            "Hy4 empty page geometry"
        );
        let page = logical / self.page_size;
        let column = if self.circular {
            page % self.table.len()
        } else {
            page
        };
        let physical = *self
            .table
            .get(column)
            .ok_or_else(|| anyhow::anyhow!("Hy4 page table exhausted"))?;
        ensure!(
            physical < self.physical_pages,
            "Hy4 physical page out of bounds"
        );
        physical
            .checked_mul(self.page_size)
            .and_then(|n| n.checked_add(logical % self.page_size))
            .ok_or_else(|| anyhow::anyhow!("Hy4 slot overflow"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TailKey {
    pub value: Vec<f32>,
    pub position: [i64; 3],
}
/// Committed carry only. Clone before speculative append; publish accepted state
/// only. A rejected suffix must never overwrite a committed partial group.
#[derive(Debug, Clone, Default)]
pub struct IndexTail {
    next: usize,
    carry: Vec<TailKey>,
}
impl IndexTail {
    pub fn append(&mut self, start: usize, keys: &[TailKey]) -> Result<Vec<(usize, Vec<TailKey>)>> {
        ensure!(
            start == self.next && start.checked_add(keys.len()).is_some(),
            "Hy4 noncontiguous tail"
        );
        for key in keys {
            ensure!(
                key.value.len() == 128 && key.position.iter().all(|p| *p >= 0),
                "Hy4 tail shape/position"
            );
            super::finite(&key.value)?;
        }
        let mut completed = Vec::new();
        for key in keys {
            self.carry.push(key.clone());
            self.next += 1;
            if self.carry.len() == 4 {
                completed.push((self.next / 4 - 1, std::mem::take(&mut self.carry)));
            }
        }
        Ok(completed)
    }
    pub fn committed_len(&self) -> usize {
        self.next
    }
}
