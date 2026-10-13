// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

const S: u64 = 7;

/// The real communicator: logs what actually ran.
#[derive(Default)]
struct Log(Mutex<Vec<String>>);

impl Log {
    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.0.lock())
    }
    fn push(&self, s: String) -> Result<()> {
        self.0.lock().push(s);
        Ok(())
    }
}

impl CommBackend for Log {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        self.push(format!("ar {ptr} {bytes}"))
    }
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        self.push(format!("ara {ptr} {bytes} {stream}"))
    }
    fn peer_exchange_async(&self, send: u64, recv: u64, bytes: usize, stream: u64) -> Result<()> {
        self.push(format!("px {send} {recv} {bytes} {stream}"))
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        Ok(())
    }
    fn broadcast(&self, ptr: u64, _: usize, _: usize) -> Result<()> {
        self.push(format!("bc {ptr}"))
    }
    fn barrier(&self) -> Result<()> {
        Ok(())
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        2
    }
}

/// A two-collective KDA-like run: TP reduce, then the MoE all-reduce.
fn layer(c: &dyn CommBackend) -> Result<()> {
    c.peer_exchange_async(1, 2, 64, S)?;
    c.all_reduce_async(3, 128, S)?;
    c.all_reduce(4, 32)
}

const EAGER: [&str; 3] = ["px 1 2 64 7", "ara 3 128 7", "ar 4 32"];

#[test]
fn warm_then_capture_then_replay_keeps_collective_order() {
    let (gpu, comm, pieces) = (
        MockGpuBackend::new(),
        Log::default(),
        VerifyPieces::default(),
    );
    let mut bodies = 0;
    for _ in 0..4 {
        pieces
            .run((0, 5, 0), &gpu, &comm, S, |c| {
                bodies += 1;
                layer(c)
            })
            .unwrap();
        // Eager, capture (collectives deferred to the replay) and replay all
        // hand the real communicator the same calls in the same order.
        assert_eq!(comm.take(), EAGER);
    }
    assert_eq!(
        bodies, 2,
        "eager warm-up, one capture pass, then replays only"
    );
}

#[test]
fn captured_steps_alternate_graphs_and_collectives() {
    let (gpu, comm) = (MockGpuBackend::new(), Log::default());
    let recorder = Recorder::new(&comm, &gpu, S);
    layer(&recorder).unwrap();
    let steps = recorder.finish().unwrap();
    let kinds: String = steps
        .iter()
        .map(|s| match s {
            Step::Graph(_) => 'g',
            Step::Op(_) => 'c',
        })
        .collect();
    assert_eq!(kinds, "gcgcgcg");
    assert!(comm.take().is_empty(), "nothing executes while capturing");
}

#[test]
fn unrecorded_collective_refuses_the_key_and_runs_eager() {
    let (gpu, comm, pieces) = (
        MockGpuBackend::new(),
        Log::default(),
        VerifyPieces::default(),
    );
    let mut bodies = 0;
    for _ in 0..4 {
        pieces
            .run((0, 3, 4), &gpu, &comm, S, |c| {
                bodies += 1;
                c.all_reduce_async(3, 128, S)?;
                c.broadcast(9, 4, 0)
            })
            .unwrap();
        // The failed capture pass executed nothing; its eager re-run did.
        assert_eq!(comm.take(), ["ara 3 128 7", "bc 9"]);
    }
    assert_eq!(bodies, 5, "warm, failed capture + eager, then eager");
}

#[test]
fn collective_on_another_stream_refuses() {
    let (gpu, comm) = (MockGpuBackend::new(), Log::default());
    let recorder = Recorder::new(&comm, &gpu, S);
    assert!(recorder.all_reduce_async(3, 128, S + 1).is_err());
    assert!(recorder.exchange_async(1, 2, 64, true, S).is_err());
}

#[test]
fn body_error_propagates_without_capture() {
    let (gpu, comm, pieces) = (
        MockGpuBackend::new(),
        Log::default(),
        VerifyPieces::default(),
    );
    let err = pieces.run((0, 2, 0), &gpu, &comm, S, |_| bail!("boom"));
    assert!(err.is_err());
}

#[test]
fn slots_replay_independently_until_cleared() {
    let (gpu, comm, pieces) = (
        MockGpuBackend::new(),
        Log::default(),
        VerifyPieces::default(),
    );
    let bodies = std::cell::Cell::new(0);
    let step = |slot| {
        pieces
            .run((slot, 5, 0), &gpu, &comm, S, |c| {
                bodies.set(bodies.get() + 1);
                layer(c)
            })
            .unwrap()
    };
    for _ in 0..3 {
        step(0);
        step(1);
    }
    assert_eq!(bodies.get(), 4, "each slot: warm-up, capture, replay");
    // A later request on either slot (no free-time invalidation) replays.
    step(0);
    step(1);
    assert_eq!(bodies.get(), 4);
    pieces.clear(&gpu);
    step(1);
    assert_eq!(bodies.get(), 5, "a clear restarts from the warm-up");
    assert_eq!(comm.take().len(), 9 * EAGER.len());
}

#[test]
fn over_budget_keys_stay_eager_until_cleared() {
    let (gpu, comm) = (MockGpuBackend::new(), Log::default());
    let pieces = VerifyPieces::with_budget(4); // one captured `layer` run: 4 graphs
    let bodies = std::cell::Cell::new(0);
    let step = |rows| {
        pieces
            .run((0, rows, 0), &gpu, &comm, S, |c| {
                bodies.set(bodies.get() + 1);
                layer(c)
            })
            .unwrap()
    };
    for _ in 0..3 {
        step(2);
        step(3);
    }
    assert_eq!(
        bodies.get(),
        5,
        "rows=2 captured; rows=3 over budget, eager"
    );
    assert_eq!(comm.take().len(), 6 * EAGER.len());
    pieces.clear(&gpu);
    step(3);
    step(3);
    step(3);
    assert_eq!(
        bodies.get(),
        7,
        "room freed: rows=3 warms, captures, replays"
    );
}
