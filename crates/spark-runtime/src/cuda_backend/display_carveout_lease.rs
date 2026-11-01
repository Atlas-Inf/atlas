// SPDX-License-Identifier: AGPL-3.0-only

//! Partial VMM mappings unwind in reverse acquisition order on failure.

pub(super) struct Lease {
    pub(super) handle: Option<u64>,
    pub(super) address: Option<u64>,
    pub(super) mapped: bool,
    pub(super) size: usize,
    pub(super) unmap: unsafe extern "C" fn(u64, usize) -> i32,
    pub(super) release: unsafe extern "C" fn(u64) -> i32,
    pub(super) free_address: unsafe extern "C" fn(u64, usize) -> i32,
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(address) = self.address {
            if self.mapped {
                unsafe { (self.unmap)(address, self.size) };
            }
            unsafe { (self.free_address)(address, self.size) };
        }
        if let Some(handle) = self.handle {
            unsafe { (self.release)(handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static CALLS: Mutex<Vec<&str>> = Mutex::new(Vec::new());
    unsafe extern "C" fn unmap(_: u64, _: usize) -> i32 {
        CALLS.lock().unwrap().push("unmap");
        0
    }
    unsafe extern "C" fn release(_: u64) -> i32 {
        CALLS.lock().unwrap().push("release");
        0
    }
    unsafe extern "C" fn free_address(_: u64, _: usize) -> i32 {
        CALLS.lock().unwrap().push("free_address");
        0
    }

    #[test]
    fn every_partial_mapping_releases_only_acquired_resources() {
        for (handle, address, mapped, expected) in [
            (Some(1), None, false, vec!["release"]),
            (Some(1), Some(2), false, vec!["free_address", "release"]),
            (
                Some(1),
                Some(2),
                true,
                vec!["unmap", "free_address", "release"],
            ),
            (None, Some(2), true, vec!["unmap", "free_address"]),
            (None, None, false, vec![]),
        ] {
            CALLS.lock().unwrap().clear();
            drop(Lease {
                handle,
                address,
                mapped,
                size: 4096,
                unmap,
                release,
                free_address,
            });
            assert_eq!(*CALLS.lock().unwrap(), expected);
        }
    }
}
