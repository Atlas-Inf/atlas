// SPDX-License-Identifier: AGPL-3.0-only

use super::{cache::AttentionCache, memory::Owned};
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const TAIL_BYTES: usize = 4 * 128 * 2;
const INDEX_BYTES: usize = 2048 * 4;
const STATE_BYTES: usize = TAIL_BYTES + INDEX_BYTES;

pub(super) struct Prefix<'a> {
    _memory: Owned<'a>,
    states: DevicePtr,
    capacity: usize,
    base: usize,
    rows: usize,
    capturing: bool,
    positions: Vec<Option<[i64; 3]>>,
}

impl<'a> Prefix<'a> {
    fn new(gpu: &'a dyn GpuBackend, rows: usize, indexed: bool) -> Result<Self> {
        let mut memory = Owned::new(gpu);
        let states = if indexed {
            memory.alloc(rows * STATE_BYTES)?
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            _memory: memory,
            states,
            capacity: rows,
            base: 0,
            rows: 0,
            capturing: false,
            positions: Vec::with_capacity(rows),
        })
    }
}

impl AttentionCache<'_, '_> {
    pub(crate) fn begin_prefix(&mut self, rows: usize, stream: u64) -> Result<()> {
        ensure!(
            !self.poisoned
                && rows > 0
                && rows <= crate::layers::hy4::batch::MAX_ROWS
                && rows <= self.layout.table.len() * self.layout.page_size
                && self.stream.is_none_or(|s| s == stream),
            "Hy4 invalid prefix capture"
        );
        if self.prefix.as_ref().is_none_or(|p| p.capacity < rows) {
            self.prefix = Some(Prefix::new(
                self.layer.memory.gpu,
                rows,
                self.pool_bytes > 0,
            )?);
        }
        let p = self.prefix.as_mut().unwrap();
        p.base = self.len;
        p.rows = rows;
        p.capturing = true;
        p.positions.clear();
        Ok(())
    }

    pub(super) fn prefix_capturing(&self) -> bool {
        self.prefix.as_ref().is_some_and(|p| p.capturing)
    }

    pub(super) fn record_prefix(&mut self, stream: u64) -> Result<()> {
        let Some(p) = self.prefix.as_mut() else {
            return Ok(());
        };
        let row = p.positions.len();
        if row == p.rows {
            return Ok(());
        }
        ensure!(
            self.len == p.base + row + 1,
            "Hy4 noncontiguous prefix capture"
        );
        if self.pool_bytes > 0 {
            let gpu = self.layer.memory.gpu;
            gpu.copy_d2d_async(
                self.tail,
                p.states.offset(row * STATE_BYTES),
                TAIL_BYTES,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.scratch.indices,
                p.states.offset(row * STATE_BYTES + TAIL_BYTES),
                INDEX_BYTES,
                stream,
            )?;
        }
        p.positions.push(self.first_group_position);
        Ok(())
    }

    pub(crate) fn prefix_indices(&self, row: usize) -> Result<DevicePtr> {
        let p = self
            .prefix
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Hy4 prefix capture missing"))?;
        ensure!(
            self.pool_bytes > 0 && row < p.positions.len(),
            "Hy4 prefix index row missing"
        );
        Ok(p.states.offset(row * STATE_BYTES + TAIL_BYTES))
    }

    pub(crate) fn commit_prefix(
        &mut self,
        source: &Self,
        accepted: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let p = self
            .prefix
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Hy4 prefix capture missing"))?;
        ensure!(
            !self.poisoned
                && !source.poisoned
                && std::ptr::eq(self.layer, source.layer)
                && self.layout.table == source.layout.table
                && self.layout.page_size == source.layout.page_size
                && self.layout.circular == source.layout.circular
                && self.layout.physical_pages == source.layout.physical_pages
                && self.max_tokens == source.max_tokens
                && self.stream == Some(stream)
                && source.stream == Some(stream)
                && accepted > 0
                && accepted <= rows
                && p.rows == rows
                && p.positions.len() == rows
                && p.base == source.len
                && self.len == source.len + rows,
            "Hy4 invalid exact prefix commit"
        );
        let gpu = self.layer.memory.gpu;
        self.poisoned = true;
        for position in source.len + accepted..self.len {
            let offset = self.layout.slot(position)? * 1152;
            gpu.copy_d2d_async(
                source.latent.offset(offset),
                self.latent.offset(offset),
                1152,
                stream,
            )?;
            if self.pool_bytes > 0 && position % 4 == 3 {
                let group = position / 4;
                let offset = group / 64 * 8448 + group % 64 * 128;
                gpu.copy_d2d_async(
                    source.pool.offset(offset),
                    self.pool.offset(offset),
                    128,
                    stream,
                )?;
                let offset = group / 64 * 8448 + 8192 + group % 64 * 4;
                gpu.copy_d2d_async(
                    source.pool.offset(offset),
                    self.pool.offset(offset),
                    4,
                    stream,
                )?;
            }
        }
        if self.pool_bytes > 0 {
            gpu.copy_d2d_async(
                p.states.offset((accepted - 1) * STATE_BYTES),
                self.tail,
                TAIL_BYTES,
                stream,
            )?;
            gpu.copy_d2d_async(
                p.states.offset((accepted - 1) * STATE_BYTES + TAIL_BYTES),
                self.scratch.indices,
                INDEX_BYTES,
                stream,
            )?;
        }
        self.len = source.len + accepted;
        self.first_group_position = p.positions[accepted - 1];
        if let Some(prefix) = self.prefix.as_mut() {
            prefix.capturing = false;
        }
        self.poisoned = false;
        Ok(())
    }
}
