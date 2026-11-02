// SPDX-License-Identifier: AGPL-3.0-only

//! The command ring with a copy standing in for the NIC: two regions in one
//! process, a proxy thread per rank, real sender and receiver threads.

use super::*;
use std::sync::atomic::AtomicUsize;

/// One rank's zeroed command region (8-byte aligned, as the pinned one is).
struct Region(Vec<u64>);

impl Region {
    fn new() -> Self {
        Self(vec![0; REGION_BYTES / 8])
    }
    fn host(&self) -> usize {
        self.0.as_ptr() as usize
    }
}

/// Copies between the two regions and counts the WRITEs.
struct Copy {
    local: usize,
    peer: usize,
    writes: Arc<AtomicUsize>,
}

impl Wire for Copy {
    fn write(&mut self, src: usize, dst: usize, len: usize) -> Result<()> {
        assert!(src + len <= REGION_BYTES && dst + len <= REGION_BYTES);
        // SAFETY: both spans lie in live regions (asserted) that do not
        // overlap; the ring never sends a span its caller is writing.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.local + src) as *const u8,
                (self.peer + dst) as *mut u8,
                len,
            )
        };
        self.writes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Two connected ranks: their rings, and proxies that run until dropped.
struct Pair {
    rings: [Arc<CmdRing>; 2],
    stop: Arc<AtomicBool>,
    proxies: Vec<std::thread::JoinHandle<()>>,
    _regions: [Region; 2],
}

impl Pair {
    /// `serve` tells whether rank `r`'s proxy runs (a stopped one models a
    /// peer that neither sends nor acknowledges).
    fn new(serve: [bool; 2]) -> Self {
        let regions = [Region::new(), Region::new()];
        let stop = Arc::new(AtomicBool::new(false));
        let mut rings = Vec::new();
        let mut proxies = Vec::new();
        for rank in 0..2 {
            let (ring, mut proxy) = channel(regions[rank].host());
            rings.push(Arc::new(ring));
            let mut wire = Copy {
                local: regions[rank].host(),
                peer: regions[1 - rank].host(),
                writes: Arc::default(),
            };
            let (stop, on) = (stop.clone(), serve[rank]);
            proxies.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    if !on || !proxy.serve(&mut wire).unwrap() {
                        std::thread::yield_now();
                    }
                }
                proxy.close();
            }));
        }
        Self {
            rings: [rings[0].clone(), rings[1].clone()],
            stop,
            proxies,
            _regions: regions,
        }
    }

    fn written(&self, rank: usize) -> u64 {
        self.rings[rank].shared.written.load(Ordering::Acquire)
    }
}

impl Drop for Pair {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for proxy in self.proxies.drain(..) {
            proxy.join().unwrap();
        }
    }
}

fn mix(x: u64) -> u32 {
    (x.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 29) as u32
}

/// Wait (bounded) for `done`, failing the test if it never holds.
fn eventually(what: &str, done: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !done() {
        assert!(t0.elapsed() < Duration::from_secs(20), "{what}");
        std::thread::yield_now();
    }
}

#[test]
fn only_the_value_one_turns_the_ring_on() {
    assert!(parse(Some("1")));
    for off in [None, Some("0"), Some(""), Some("true"), Some("2")] {
        assert!(!parse(off), "{off:?}");
    }
    assert_eq!(wire(false), 0);
    assert_eq!(wire(true), RING_WORDS as u64);
}

#[test]
fn region_words_are_disjoint_and_aligned() {
    let spans = [
        (OUT, RING_BYTES),
        (IN, RING_BYTES),
        (TAIL, 8),
        (ACK, 8),
        (TAIL_SRC, 8),
        (ACK_SRC, 8),
    ];
    for (i, &(off, len)) in spans.iter().enumerate() {
        assert!(off % 64 == 0 && off + len <= REGION_BYTES);
        for &(other, other_len) in &spans[i + 1..] {
            assert!(off + len <= other || other + other_len <= off);
        }
    }
    const { assert!(MAX_WORDS as u64 <= ACK_EVERY && REGION_BYTES.is_multiple_of(64)) };
}

#[test]
fn spans_cover_the_stream_across_the_wrap() {
    let ring = RING_WORDS as u64;
    assert_eq!(spans(0, 5).collect::<Vec<_>>(), [(0, 5)]);
    assert_eq!(
        spans(ring - 2, ring).collect::<Vec<_>>(),
        [(RING_WORDS - 2, 2)]
    );
    assert_eq!(
        spans(ring - 2, ring + 3).collect::<Vec<_>>(),
        [(RING_WORDS - 2, 2), (0, 3)]
    );
    assert_eq!(
        spans(3 * ring + 7, 4 * ring + 7).collect::<Vec<_>>(),
        [(7, RING_WORDS - 7), (0, 7)]
    );
    assert_eq!(spans(9, 9).count(), 0);
}

/// The head's stream of step-shaped messages reaches the worker word for
/// word, in order, over many wraps, whatever sizes the two sides use; the
/// reverse direction carries its own stream at the same time.
#[test]
fn words_arrive_in_order_exactly_once_in_both_directions() {
    const TOTAL: u64 = 40 * RING_WORDS as u64;
    let pair = Pair::new([true, true]);
    std::thread::scope(|s| {
        for rank in 0..2u64 {
            let (tx, rx) = (&pair.rings[rank as usize], &pair.rings[1 - rank as usize]);
            s.spawn(move || {
                let (mut at, mut k) = (0u64, 0u64);
                while at < TOTAL {
                    // slot, command, width, then a token message: 1, 1, 1, n.
                    let n = if k % 4 == 3 {
                        1 + mix(k) as u64 % 64
                    } else {
                        1
                    };
                    let n = n.min(TOTAL - at);
                    let words: Vec<u32> = (at..at + n).map(|i| mix(i << 1 | rank)).collect();
                    tx.send(&words).unwrap();
                    (at, k) = (at + n, k + 1);
                }
            });
            s.spawn(move || {
                let (mut at, mut k) = (0u64, 0u64);
                while at < TOTAL {
                    let n = (1 + mix(!k) as u64 % 64).min(TOTAL - at);
                    let mut words = vec![0u32; n as usize];
                    rx.recv(&mut words).unwrap();
                    for (i, &word) in (at..).zip(&words) {
                        assert_eq!(word, mix(i << 1 | rank), "rank {rank} word {i}");
                    }
                    (at, k) = (at + n, k + 1);
                }
            });
        }
    });
    for rank in 0..2 {
        assert_eq!(pair.written(rank), TOTAL);
    }
}

/// A sender cannot overwrite words the peer has not read: it stops one ring
/// ahead, and goes on once the peer's acknowledgement arrives.
#[test]
fn a_sender_waits_for_the_reader_one_ring_ahead() {
    let pair = Pair::new([true, true]);
    let (head, worker) = (pair.rings[0].clone(), pair.rings[1].clone());
    let messages = RING_WORDS / MAX_WORDS + 2;
    let sender = std::thread::spawn(move || {
        for m in 0..messages {
            let words: Vec<u32> = (0..MAX_WORDS).map(|i| (m * MAX_WORDS + i) as u32).collect();
            head.send(&words).unwrap();
        }
    });
    eventually("the sender fills the ring", || {
        pair.written(0) == RING_WORDS as u64
    });
    // Nothing read, nothing acknowledged: the next message does not go out.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(pair.written(0), RING_WORDS as u64);
    let mut next = 0u32;
    let mut take = |n: usize| {
        let mut words = vec![0u32; n];
        worker.recv(&mut words).unwrap();
        for word in words {
            assert_eq!(word, next);
            next += 1;
        }
    };
    // Below the acknowledgement threshold the sender still waits.
    take(MAX_WORDS);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(pair.written(0), RING_WORDS as u64);
    for _ in 1..messages {
        take(MAX_WORDS);
    }
    sender.join().unwrap();
    assert_eq!(pair.written(0), (messages * MAX_WORDS) as u64);
    assert_eq!(next as usize, messages * MAX_WORDS);
}

/// One message costs the sender's rank two WRITEs (data, tail), and the
/// reader acknowledges once a quarter ring, not per message. Stepped by hand
/// so the counts are exact.
#[test]
fn a_message_is_two_writes_and_acknowledgements_are_batched() {
    let regions = [Region::new(), Region::new()];
    let writes = [0, 1].map(|_| Arc::new(AtomicUsize::new(0)));
    let wire = |rank: usize| Copy {
        local: regions[rank].host(),
        peer: regions[1 - rank].host(),
        writes: writes[rank].clone(),
    };
    let (head, mut head_proxy) = channel(regions[0].host());
    let (worker, mut worker_proxy) = channel(regions[1].host());
    let (mut to_worker, mut to_head) = (wire(0), wire(1));
    let steps = 3 * ACK_EVERY as usize / 4;
    let mut got = [0u32; 4];
    for step in 0..steps as u32 {
        assert!(!head_proxy.serve(&mut to_worker).unwrap());
        head.send(&[step, 1, 2, 3]).unwrap();
        assert!(head_proxy.serve(&mut to_worker).unwrap());
        worker.recv(&mut got).unwrap();
        assert_eq!(got, [step, 1, 2, 3]);
        let due = (4 * (step as u64 + 1)).is_multiple_of(ACK_EVERY);
        assert_eq!(worker_proxy.serve(&mut to_head).unwrap(), due);
    }
    assert_eq!(writes[0].load(Ordering::Relaxed), 2 * steps);
    assert_eq!(writes[1].load(Ordering::Relaxed), 3);
    assert_eq!(
        head.peer_word(ACK).load(Ordering::Acquire),
        4 * steps as u64
    );
}

#[test]
fn messages_outside_one_to_max_words_are_refused() {
    let pair = Pair::new([true, true]);
    let mut big = vec![0u32; MAX_WORDS + 1];
    for ring in &pair.rings {
        assert!(ring.send(&[]).is_err());
        assert!(ring.send(&big).is_err());
        assert!(ring.recv(&mut []).is_err());
        assert!(ring.recv(&mut big).is_err());
    }
    assert_eq!(pair.written(0), 0);
}

/// Once the proxy is gone nothing hangs: a receive with no words pending and
/// a send with no room both fail, while words that already arrived are still
/// delivered.
#[test]
fn a_stopped_proxy_fails_the_waiters_but_delivers_what_arrived() {
    let pair = Pair::new([true, false]);
    pair.rings[0].send(&[7, 8, 9]).unwrap();
    eventually("the words land", || {
        pair.rings[1].peer_word(TAIL).load(Ordering::Acquire) == 3
    });
    let blocked = {
        let worker = pair.rings[1].clone();
        std::thread::spawn(move || {
            let mut first = [0u32; 3];
            worker.recv(&mut first).unwrap();
            (first, worker.recv(&mut [0u32; 1]))
        })
    };
    // The worker's own stream is never delivered or acknowledged: it fills.
    let full = {
        let worker = pair.rings[1].clone();
        std::thread::spawn(move || {
            (0..=RING_WORDS / MAX_WORDS)
                .map(|_| worker.send(&[0; MAX_WORDS]))
                .collect::<Vec<_>>()
        })
    };
    std::thread::sleep(Duration::from_millis(20));
    pair.stop.store(true, Ordering::Release);
    let (first, second) = blocked.join().unwrap();
    assert_eq!(first, [7, 8, 9]);
    let error = second.unwrap_err().to_string();
    assert!(error.contains("the proxy stopped"), "{error}");
    let sends = full.join().unwrap();
    assert!(sends[..RING_WORDS / MAX_WORDS].iter().all(Result::is_ok));
    let error = sends.last().unwrap().as_ref().unwrap_err().to_string();
    assert!(error.contains("the proxy stopped"), "{error}");
}
