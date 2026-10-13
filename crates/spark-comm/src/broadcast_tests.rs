// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use anyhow::Result;
use std::time::Duration;

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Start,
    Launch,
    Synchronize,
    Elapsed,
    Slow,
    Async,
}
struct Recorded {
    elapsed: Duration,
    failure: Option<Event>,
    unhealthy: bool,
    events: Vec<Event>,
}
impl Recorded {
    fn new(elapsed: Duration) -> Self {
        Self {
            elapsed,
            failure: None,
            unhealthy: false,
            events: vec![],
        }
    }
    fn operation(&mut self, event: Event) -> Result<()> {
        let fails = self.failure.as_ref() == Some(&event);
        self.events.push(event);
        anyhow::ensure!(!fails, "injected {:?}", self.failure);
        Ok(())
    }
}
impl Operations for Recorded {
    type Stamp = u64;
    fn start(&mut self) -> u64 {
        self.events.push(Event::Start);
        17
    }
    fn launch(&mut self) -> Result<()> {
        self.operation(Event::Launch)
    }
    fn synchronize(&mut self) -> Result<()> {
        self.operation(Event::Synchronize)
    }
    fn elapsed(&mut self, start: u64) -> Duration {
        assert_eq!(start, 17);
        self.events.push(Event::Elapsed);
        self.elapsed
    }
    fn mark_slow(&mut self, elapsed: Duration) {
        assert_eq!(elapsed, self.elapsed);
        self.events.push(Event::Slow);
        self.unhealthy = true;
    }
    fn check_async_error(&mut self) {
        if self.operation(Event::Async).is_err() {
            self.unhealthy = true;
        }
    }
}

#[test]
fn actual_runner_classifies_only_explicit_idle_duration() {
    for milliseconds in [29_999, 30_000, 376_900] {
        for class in [Classification::TimedPayload, Classification::IdleCommand] {
            let mut ops = Recorded::new(Duration::from_millis(milliseconds));
            run(class, &mut ops).unwrap();
            let slow = class == Classification::TimedPayload && milliseconds >= 30_000;
            assert_eq!(ops.unhealthy, slow, "{class:?} {milliseconds}ms");
            let mut expected = vec![
                Event::Start,
                Event::Launch,
                Event::Synchronize,
                Event::Elapsed,
            ];
            if slow {
                expected.push(Event::Slow);
            }
            expected.push(Event::Async);
            assert_eq!(ops.events, expected);
        }
    }
}

#[test]
fn errors_and_existing_health_are_not_exempted_or_cleared() {
    for class in [Classification::TimedPayload, Classification::IdleCommand] {
        for failure in [Event::Launch, Event::Synchronize, Event::Async] {
            let mut ops = Recorded::new(Duration::from_secs(1));
            let expected = match failure {
                Event::Launch => vec![Event::Start, Event::Launch],
                Event::Synchronize => vec![Event::Start, Event::Launch, Event::Synchronize],
                Event::Async => vec![
                    Event::Start,
                    Event::Launch,
                    Event::Synchronize,
                    Event::Elapsed,
                    Event::Async,
                ],
                _ => unreachable!(),
            };
            let async_failure = failure == Event::Async;
            ops.failure = Some(failure);
            assert_eq!(run(class, &mut ops).is_ok(), async_failure);
            assert_eq!(ops.unhealthy, async_failure);
            assert_eq!(ops.events, expected);
        }
        let mut ops = Recorded::new(Duration::from_secs(1));
        ops.unhealthy = true;
        run(class, &mut ops).unwrap();
        assert!(ops.unhealthy);
    }
}

#[test]
fn idle_boundary_rejects_wrong_receiver_or_word_before_io() {
    for (rank, world, ptr) in [
        (0, 2, 4),
        (1, 1, 4),
        (2, 2, 4),
        (0, 0, 4),
        (1, 2, 0),
        (1, 2, 6),
    ] {
        assert!(validate_idle_receiver(rank, world, ptr).is_err());
    }
    assert!(validate_idle_receiver(1, 2, 4).is_ok());
    assert!(validate_idle_receiver(3, 4, 0x1000).is_ok());
    assert_eq!(COLLECTIVE_TIMEOUT_SECS, 30);
}
