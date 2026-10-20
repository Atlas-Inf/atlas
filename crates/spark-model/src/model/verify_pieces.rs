// SPDX-License-Identifier: AGPL-3.0-only

//! Piecewise CUDA-graph capture of the GLM TP2 DFlash verify step
//! (`ATLAS_GLM_VERIFY_GRAPH=1`, default off; design in
//! `docs/glm-verify-graphs.md`).
//!
//! The verify forward cannot be one graph today: the RDMA pair all-reduce
//! refuses a capturing stream (each call hands the proxy thread a host-side
//! job), and the sparse-MLA layers bake the host `seq_len` into their
//! launches. So a run of consecutive KDA layers is captured as a list of
//! graphs split at every collective. During capture a [`Recorder`] stands in
//! for the communicator: it closes the open graph, remembers the call and
//! opens the next graph. Nothing executes while capturing. Replay launches a
//! graph, runs the remembered collective eagerly, launches the next graph and
//! so on: the eager pass's stream order, and the same collective calls in the
//! same order on both ranks whether or not either rank captured.
//!
//! A graph-safe collective removes the split: with `capture_collectives`
//! (the qwen4_exp pieces' `ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES=1`)
//! [`Recorder`]'s `all_reduce_async` first offers the call to the
//! communicator's capturable one-shot, which enqueues it into the open
//! capture; only a refusal splits. The GLM verify cache keeps it off.
//!
//! The cache is generic over its key so the qwen4_exp decode and verify
//! pieces (`model/decode_pieces.rs`) reuse it with a slot-vector key;
//! the GLM verify cache is [`VerifyPieces`], unchanged.

use anyhow::{Result, bail, ensure};
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::gpu::{GpuBackend, GraphHandle};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::time::Instant;

/// `ATLAS_GLM_VERIFY_GRAPH=1`; read once, from the profile both ranks share.
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_VERIFY_GRAPH").as_deref() == Ok("1"))
}

/// A stream-ordered collective, replayed eagerly between two graphs with the
/// arguments it was captured with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommOp {
    AllReduce {
        ptr: u64,
        bytes: usize,
    },
    AllReduceAsync {
        ptr: u64,
        bytes: usize,
        stream: u64,
    },
    PeerExchangeAsync {
        send: u64,
        recv: u64,
        bytes: usize,
        stream: u64,
    },
}

impl CommOp {
    fn run(self, comm: &dyn CommBackend) -> Result<()> {
        match self {
            Self::AllReduce { ptr, bytes } => comm.all_reduce(ptr, bytes),
            Self::AllReduceAsync { ptr, bytes, stream } => {
                comm.all_reduce_async(ptr, bytes, stream)
            }
            Self::PeerExchangeAsync {
                send,
                recv,
                bytes,
                stream,
            } => comm.peer_exchange_async(send, recv, bytes, stream),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Step {
    Graph(GraphHandle),
    Op(CommOp),
}

/// The communicator seen by the layers during one capture pass.
pub(crate) struct Recorder<'a> {
    inner: &'a dyn CommBackend,
    gpu: &'a dyn GpuBackend,
    stream: u64,
    steps: Mutex<Vec<Step>>,
    /// Offer each `all_reduce_async` to the communicator's capturable
    /// one-shot before splitting (see the module docs).
    capture_collectives: bool,
}

impl<'a> Recorder<'a> {
    fn new(inner: &'a dyn CommBackend, gpu: &'a dyn GpuBackend, stream: u64) -> Self {
        Self {
            inner,
            gpu,
            stream,
            steps: Mutex::new(Vec::new()),
            capture_collectives: false,
        }
    }

    /// Close the open graph, remember `op`, open the next graph. An error
    /// leaves the stream out of capture with nothing executed; the caller's
    /// `?` unwinds the capture pass.
    fn split(&self, op: CommOp) -> Result<()> {
        let graph = self.gpu.end_capture(self.stream)?;
        self.steps.lock().extend([Step::Graph(graph), Step::Op(op)]);
        self.gpu.begin_capture(self.stream)
    }

    fn on_stream(&self, stream: u64, op: CommOp) -> Result<()> {
        ensure!(
            stream == self.stream,
            "piecewise verify capture: collective on stream {stream:#x}, capturing {:#x}",
            self.stream
        );
        self.split(op)
    }

    /// Close the last graph and hand back the steps.
    fn finish(self) -> Result<Vec<Step>> {
        let graph = self.gpu.end_capture(self.stream);
        let mut steps = self.steps.into_inner();
        match graph {
            Ok(g) => {
                steps.push(Step::Graph(g));
                Ok(steps)
            }
            Err(e) => {
                destroy(&steps, self.gpu);
                Err(e)
            }
        }
    }

    /// Undo a failed capture pass: leave capture mode, drop the graphs.
    fn abort(self) {
        self.gpu.abort_capture_if_active(self.stream);
        destroy(&self.steps.into_inner(), self.gpu);
    }
}

fn unsupported<T>(what: &str) -> Result<T> {
    bail!("piecewise verify capture does not record {what}")
}

impl CommBackend for Recorder<'_> {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        self.split(CommOp::AllReduce { ptr, bytes })
    }
    /// With `capture_collectives`, the capturable one-shot records the reduce
    /// INTO the open graph. It is the operation the eager `all_reduce_async`
    /// would have enqueued (both try the one-shot first, under the same
    /// size-only eligibility), so replay stays lossless and both ranks
    /// execute the same collective sequence whatever each rank captured. A
    /// refusal (size, channel off) splits as usual.
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        if self.capture_collectives
            && stream == self.stream
            && self.inner.all_reduce_capturable(ptr, bytes, stream)?
        {
            return Ok(());
        }
        self.on_stream(stream, CommOp::AllReduceAsync { ptr, bytes, stream })
    }
    fn peer_exchange_async(&self, send: u64, recv: u64, bytes: usize, stream: u64) -> Result<()> {
        let op = CommOp::PeerExchangeAsync {
            send,
            recv,
            bytes,
            stream,
        };
        self.on_stream(stream, op)
    }
    fn supports_peer_exchange_async(&self) -> bool {
        self.inner.supports_peer_exchange_async()
    }
    fn supports_exchange_async(&self, bytes: usize) -> bool {
        self.inner.supports_exchange_async(bytes)
    }
    fn exchange_async(&self, _: u64, _: u64, _: usize, _: bool, _: u64) -> Result<bool> {
        unsupported("exchange_async")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        unsupported("all_gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        unsupported("reduce_scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        unsupported("broadcast")
    }
    fn receive_idle_command_word(&self, _: u64) -> Result<()> {
        unsupported("receive_idle_command_word")
    }
    fn barrier(&self) -> Result<()> {
        unsupported("barrier")
    }
    fn register_buffer(&self, _: u64, _: usize) -> Result<u64> {
        unsupported("register_buffer")
    }
    fn deregister_buffer(&self, _: u64) -> Result<()> {
        unsupported("deregister_buffer")
    }
    fn symmetric_alloc(&self, _: usize) -> Result<u64> {
        unsupported("symmetric_alloc")
    }
    fn symmetric_free(&self, _: u64) -> Result<()> {
        unsupported("symmetric_free")
    }
    fn set_add_kernel(&self, _: u64) {}
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        unsupported("send_to")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        unsupported("recv_from")
    }
    fn group_start(&self) -> Result<()> {
        unsupported("group_start")
    }
    fn group_end(&self) -> Result<()> {
        unsupported("group_end")
    }
    fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }
    fn attempt_reconnect(&self) -> Result<()> {
        unsupported("attempt_reconnect")
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
}

fn destroy(steps: &[Step], gpu: &dyn GpuBackend) {
    for step in steps {
        if let Step::Graph(g) = *step
            && g.0 != 0
            && let Err(e) = gpu.destroy_graph(g)
        {
            tracing::warn!("piecewise verify graph: destroy: {e:#}");
        }
    }
}

fn graph_count(steps: &[Step]) -> usize {
    steps.iter().filter(|s| matches!(s, Step::Graph(_))).count()
}

fn replay(
    steps: &[Step],
    gpu: &dyn GpuBackend,
    comm: Option<&dyn CommBackend>,
    stream: u64,
) -> Result<()> {
    for step in steps {
        match *step {
            Step::Graph(g) if g.0 != 0 => gpu.launch_graph(g, stream)?,
            Step::Graph(_) => {}
            // A run captured without a communicator recorded no collective.
            Step::Op(op) => op.run(comm.ok_or_else(|| {
                anyhow::anyhow!("piecewise graph replay: a collective without a communicator")
            })?)?,
        }
    }
    Ok(())
}

fn cached<K>(map: &HashMap<K, Entry>) -> usize {
    map.values()
        .map(|e| match e {
            Entry::Captured(steps) => graph_count(steps),
            _ => 0,
        })
        .sum()
}

/// `(slot, verify rows, first layer of the run)`.
pub(crate) type PieceKey = (usize, usize, usize);

enum Entry {
    /// Seen once, eagerly: one-time lazy setup has run before any capture.
    Warm,
    Captured(Vec<Step>),
    /// A capture pass failed; this key stays eager.
    Refused,
}

/// Captured runs of one model, keyed by `K`.
pub(crate) struct Pieces<K> {
    map: Mutex<HashMap<K, Entry>>,
    /// Graph count past which new keys stay eager. A ~14-kernel piece costs
    /// ~0.17-0.25 MiB of unified memory (host RSS ~0.1 MiB of it) on GB10;
    /// C1 at every verify width 2..=8 is 7 x 84 = 588 graphs. Held until a
    /// LoRA clear and outside KV-pool sizing: the default 600 is ~100-150
    /// MiB of the headroom `--gpu-memory-utilization` leaves.
    budget: usize,
    /// Log names: the step (`verify`, `decode`) and the key's fields.
    step: &'static str,
    key_fields: &'static str,
    /// See [`Recorder`]'s `all_reduce_async`.
    capture_collectives: bool,
}

/// The GLM DFlash verify cache.
pub(crate) type VerifyPieces = Pieces<PieceKey>;

impl Default for VerifyPieces {
    /// Budget: `ATLAS_GLM_VERIFY_GRAPH_MAX_GRAPHS`, default 600.
    fn default() -> Self {
        let budget = std::env::var("ATLAS_GLM_VERIFY_GRAPH_MAX_GRAPHS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600);
        Self::with_budget(budget)
    }
}

impl VerifyPieces {
    pub(crate) fn with_budget(budget: usize) -> Self {
        Self::new("verify", "(slot, rows, layer)", budget, false)
    }

    /// [`Pieces::run_with`] with the TP communicator the GLM verify always
    /// has.
    pub(crate) fn run(
        &self,
        key: PieceKey,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
        mut body: impl FnMut(&dyn CommBackend) -> Result<()>,
    ) -> Result<()> {
        self.run_with(key, gpu, Some(comm), stream, |c| match c {
            Some(c) => body(c),
            None => bail!("piecewise verify graph: the communicator went missing"),
        })
    }
}

impl<K: Eq + Hash + Debug> Pieces<K> {
    pub(crate) fn new(
        step: &'static str,
        key_fields: &'static str,
        budget: usize,
        capture_collectives: bool,
    ) -> Self {
        Self {
            map: Mutex::default(),
            budget,
            step,
            key_fields,
            capture_collectives,
        }
    }

    /// Run `body` for `key`: eagerly the first time, captured then replayed
    /// the second, replayed after that. `body` receives the communicator its
    /// layers must use: a [`Recorder`] over `comm` while capturing, `comm`
    /// otherwise, and `None` exactly when `comm` is (a TP1 run has no
    /// collective to split at, so it captures as one graph). A failed
    /// capture pass executes nothing, so the key is refused and `body` runs
    /// eagerly instead. A warm key over the graph budget runs eagerly until
    /// [`Self::clear`] frees room.
    pub(crate) fn run_with(
        &self,
        key: K,
        gpu: &dyn GpuBackend,
        comm: Option<&dyn CommBackend>,
        stream: u64,
        mut body: impl FnMut(Option<&dyn CommBackend>) -> Result<()>,
    ) -> Result<()> {
        {
            let mut map = self.map.lock();
            match map.get(&key) {
                Some(Entry::Captured(steps)) => return replay(steps, gpu, comm, stream),
                Some(Entry::Refused) => {
                    drop(map);
                    return body(comm);
                }
                Some(Entry::Warm) if cached(&map) >= self.budget => {
                    drop(map);
                    return body(comm);
                }
                Some(Entry::Warm) => {}
                None => {
                    map.insert(key, Entry::Warm);
                    drop(map);
                    return body(comm);
                }
            }
        }
        let started = Instant::now();
        let captured = gpu.begin_capture(stream).and_then(|()| match comm {
            Some(comm) => {
                let mut recorder = Recorder::new(comm, gpu, stream);
                recorder.capture_collectives = self.capture_collectives;
                match body(Some(&recorder)) {
                    Ok(()) => recorder.finish(),
                    Err(e) => {
                        recorder.abort();
                        Err(e)
                    }
                }
            }
            None => match body(None) {
                Ok(()) => gpu.end_capture(stream).map(|g| vec![Step::Graph(g)]),
                Err(e) => {
                    gpu.abort_capture_if_active(stream);
                    Err(e)
                }
            },
        });
        let step = self.step;
        let steps = match captured {
            Ok(steps) => steps,
            Err(e) => {
                tracing::warn!("piecewise {step} graph refused for {key:?}, running eager: {e:#}");
                self.map.lock().insert(key, Entry::Refused);
                return body(comm);
            }
        };
        let replayed = replay(&steps, gpu, comm, stream);
        let mut map = self.map.lock();
        let graphs = graph_count(&steps);
        // Logged before the insert takes the key; the replaced entry was
        // `Warm`, so the total is the cache plus this run.
        tracing::info!(
            "piecewise {step} graph {} = {key:?}: {graphs} graphs in {:.1} ms \
             ({} graphs cached)",
            self.key_fields,
            started.elapsed().as_secs_f64() * 1e3,
            cached(&map) + graphs,
        );
        map.insert(key, Entry::Captured(steps));
        replayed
    }

    /// Drop every captured run (LoRA rotation). Freeing a sequence keeps its
    /// slot's runs: everything they bake is a function of the slot alone, so
    /// a later request on the slot replays without a warm-up or capture step.
    pub(crate) fn clear(&self, gpu: &dyn GpuBackend) {
        for (_, entry) in self.map.lock().drain() {
            if let Entry::Captured(steps) = entry {
                destroy(&steps, gpu);
            }
        }
    }
}

#[cfg(test)]
#[path = "verify_pieces_tests.rs"]
mod tests;
