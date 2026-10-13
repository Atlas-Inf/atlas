// SPDX-License-Identifier: AGPL-3.0-only
use crate::CommBackend;
use anyhow::Result;
use std::sync::Mutex;

struct Legacy {
    rank: usize,
    world: usize,
    fail: bool,
    calls: Mutex<Vec<(u64, usize, usize)>>,
}
impl CommBackend for Legacy {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        unreachable!()
    }
    fn barrier(&self) -> Result<()> {
        unreachable!()
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        unreachable!()
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        unreachable!()
    }
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        self.calls.lock().unwrap().push((ptr, bytes, root));
        anyhow::ensure!(!self.fail, "legacy broadcast failed");
        Ok(())
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world
    }
}

#[test]
fn default_idle_entry_uses_legacy_broadcast_once_and_preserves_error() {
    for fail in [false, true] {
        let backend = Legacy {
            rank: 1,
            world: 2,
            fail,
            calls: Mutex::new(vec![]),
        };
        let object: &dyn CommBackend = &backend;
        let result = object.receive_idle_command_word(0x1000);
        assert_eq!(result.is_err(), fail);
        if fail {
            assert_eq!(result.unwrap_err().to_string(), "legacy broadcast failed");
        }
        assert_eq!(*backend.calls.lock().unwrap(), vec![(0x1000, 4, 0)]);
    }
}

#[test]
fn default_idle_entry_invalid_receiver_never_touches_backend() {
    for (rank, world, ptr) in [
        (0, 2, 4),
        (1, 1, 4),
        (2, 2, 4),
        (0, 0, 4),
        (1, 2, 0),
        (1, 2, 6),
    ] {
        let backend = Legacy {
            rank,
            world,
            fail: false,
            calls: Mutex::new(vec![]),
        };
        assert!(backend.receive_idle_command_word(ptr).is_err());
        assert!(backend.calls.lock().unwrap().is_empty());
    }
}
