// SPDX-License-Identifier: AGPL-3.0-only
//! Real model construction and owned byte transport; no numerical emulation.
use crate::model::types::TransformerModel;
use crate::weight_map::DenseWeight;
use anyhow::Result;
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use std::{collections::VecDeque, sync::Arc};

pub(super) const STREAM: u64 = 7;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Event {
    Idle(u64),
    Timed(u64, usize, usize),
    Sync(u64),
    D2h(u64, usize),
    H2d(u64, Vec<u8>),
    Alloc,
    /// Words sent on, and the count received from, the host command channel.
    HostSend(Vec<u32>),
    HostRecv(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Failure {
    Idle,
    Timed,
    Sync,
    D2h,
    Host,
}
#[derive(Default)]
pub(super) struct Record {
    pub events: Mutex<Vec<Event>>,
    pub words: Mutex<VecDeque<u32>>,
    pub failure: Mutex<Option<Failure>>,
    pub timed_failure_at: Mutex<Option<usize>>,
    /// The host command channel's message size; 0 (the default) is no channel.
    pub host_words_max: Mutex<usize>,
}
impl Record {
    fn fail(&self, at: Failure) -> Result<()> {
        anyhow::ensure!(*self.failure.lock() != Some(at), "injected {at:?}");
        Ok(())
    }
}
struct Gpu {
    inner: Arc<MockGpuBackend>,
    record: Arc<Record>,
}
impl GpuBackend for Gpu {
    fn alloc(&self, n: usize) -> Result<DevicePtr> {
        self.record.events.lock().push(Event::Alloc);
        self.inner.alloc(n)
    }
    fn alloc_managed(&self, n: usize) -> Result<DevicePtr> {
        self.alloc(n)
    }
    fn free(&self, p: DevicePtr) -> Result<()> {
        self.inner.free(p)
    }
    fn copy_h2d(&self, bytes: &[u8], p: DevicePtr) -> Result<()> {
        self.record
            .events
            .lock()
            .push(Event::H2d(p.0, bytes.to_vec()));
        self.inner.copy_h2d(bytes, p)
    }
    fn copy_d2h(&self, p: DevicePtr, bytes: &mut [u8]) -> Result<()> {
        self.record.events.lock().push(Event::D2h(p.0, bytes.len()));
        self.record.fail(Failure::D2h)?;
        self.inner.copy_d2h(p, bytes)
    }
    fn copy_d2d(&self, a: DevicePtr, b: DevicePtr, n: usize) -> Result<()> {
        self.inner.copy_d2d(a, b, n)
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.record.events.lock().push(Event::Sync(stream));
        self.record.fail(Failure::Sync)?;
        self.inner.synchronize(stream)
    }
    fn default_stream(&self) -> u64 {
        STREAM
    }
    fn kernel(&self, _: &str, _: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(1))
    }
    fn op_cache(&self) -> &spark_runtime::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn launch(
        &self,
        _: KernelHandle,
        _: [u32; 3],
        _: [u32; 3],
        _: u32,
        _: u64,
        _: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("idle command fixture must not launch numerical kernels")
    }
    fn memset(&self, p: DevicePtr, v: u8, n: usize) -> Result<()> {
        self.inner.memset(p, v, n)
    }
    fn memset_async(&self, p: DevicePtr, v: u8, n: usize, s: u64) -> Result<()> {
        self.inner.memset_async(p, v, n, s)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.inner.sm_count()
    }
}
struct Comm {
    rank: usize,
    inner: Arc<MockGpuBackend>,
    record: Arc<Record>,
}
impl Comm {
    fn transport(&self, ptr: u64, bytes: usize) -> Result<()> {
        if self.rank == 0 {
            return Ok(());
        }
        let mut values = self.record.words.lock();
        let data: Vec<u8> = (0..bytes / 4)
            .flat_map(|_| {
                values
                    .pop_front()
                    .expect("fixture incoming word")
                    .to_le_bytes()
            })
            .collect();
        self.inner.copy_h2d(&data, DevicePtr(ptr))
    }
}
impl CommBackend for Comm {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected all reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected all gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected reduce scatter")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected recv")
    }
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        self.record
            .events
            .lock()
            .push(Event::Timed(ptr, bytes, root));
        let ordinal = self
            .record
            .events
            .lock()
            .iter()
            .filter(|e| matches!(e, Event::Timed(..)))
            .count();
        anyhow::ensure!(
            *self.record.timed_failure_at.lock() != Some(ordinal),
            "injected timed payload {ordinal}"
        );
        self.record.fail(Failure::Timed)?;
        self.transport(ptr, bytes)
    }
    fn receive_idle_command_word(&self, ptr: u64) -> Result<()> {
        self.record.events.lock().push(Event::Idle(ptr));
        self.record.fail(Failure::Idle)?;
        self.transport(ptr, 4)
    }
    fn command_words_max(&self) -> usize {
        *self.record.host_words_max.lock()
    }
    fn send_command_words(&self, words: &[u32]) -> Result<()> {
        let mut events = self.record.events.lock();
        events.push(Event::HostSend(words.to_vec()));
        self.record.fail(Failure::Host)
    }
    fn recv_command_words(&self, words: &mut [u32]) -> Result<()> {
        let mut events = self.record.events.lock();
        events.push(Event::HostRecv(words.len()));
        self.record.fail(Failure::Host)?;
        let mut values = self.record.words.lock();
        for word in words {
            *word = values.pop_front().expect("fixture incoming word");
        }
        Ok(())
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
}

pub(super) struct Fixture {
    pub model: TransformerModel,
    pub record: Arc<Record>,
    pub gpu: Arc<MockGpuBackend>,
}
impl Fixture {
    pub fn new(rank: usize, v2: bool, words: &[u32]) -> Self {
        let record = Arc::new(Record::default());
        let inner = Arc::new(MockGpuBackend::new());
        let gpu = Box::new(Gpu {
            inner: inner.clone(),
            record: record.clone(),
        });
        let mut cfg = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
        cfg.hidden_size = 128;
        cfg.vocab_size = 8;
        cfg.num_hidden_layers = 0;
        cfg.layer_types = vec![];
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = 1;
        cfg.head_dim = 64;
        cfg.intermediate_size = 256;
        cfg.moe_intermediate_size = 256;
        cfg.shared_expert_intermediate_size = 256;
        cfg.num_experts = 2;
        cfg.num_experts_per_tok = 1;
        cfg.linear_num_key_heads = 0;
        cfg.linear_num_value_heads = 0;
        cfg.num_mtp_modules = 0;
        cfg.tp_world_size = 2;
        cfg.ep_world_size = 2;
        cfg.tp_rank = rank;
        cfg.ep_rank = rank;
        cfg.adapter_max_rank = 0;
        let buffers =
            spark_runtime::buffers::BufferArena::new(&cfg, 8, 32, 16, 1, gpu.as_ref()).unwrap();
        let kv = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 64,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            4,
            gpu.as_ref(),
        )
        .unwrap();
        let dense = DenseWeight {
            weight: gpu.alloc(8 * 128 * 2).unwrap(),
        };
        let comm: Arc<dyn CommBackend> = Arc::new(Comm {
            rank,
            inner: inner.clone(),
            record: record.clone(),
        });
        let ssm_pools = crate::model::ssm_pools::SsmPools::new(
            &cfg,
            1,
            false,
            false,
            true,
            false,
            4,
            1,
            gpu.as_ref(),
        )
        .unwrap();
        let mut model = TransformerModel::new(
            cfg,
            dense,
            dense,
            dense,
            None,
            None,
            None,
            vec![],
            buffers,
            kv,
            vec![],
            gpu,
            32,
            1,
            crate::layers::MtpQuantization::Bf16,
            false,
            Box::new(spark_runtime::prefix_cache::NoPrefixCaching),
            8,
            Some(comm),
            false,
            None,
            1,
            16,
            ssm_pools,
        )
        .unwrap();
        model.ep_protocol_v2 = v2;
        record.events.lock().clear();
        record.words.lock().extend(words);
        Self {
            model,
            record,
            gpu: inner,
        }
    }
    pub fn events(&self) -> Vec<Event> {
        self.record.events.lock().clone()
    }
    pub fn word_events(&self, idle: bool) -> Vec<Event> {
        let ptr = self.model.ep_cmd_buf.0;
        vec![
            if idle {
                Event::Idle(ptr)
            } else {
                Event::Timed(ptr, 4, 0)
            },
            Event::Sync(STREAM),
            Event::D2h(ptr, 4),
        ]
    }
}
