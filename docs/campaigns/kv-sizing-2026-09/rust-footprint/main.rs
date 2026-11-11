// SPDX-License-Identifier: AGPL-3.0-only
// Hardware check of crates/spark-runtime's nvml.rs + own_footprint.rs (copied
// in unchanged): the sample/ledger/since path the CUDA backend runs, against
// real allocations. With NVIDIA_DRIVER_CAPABILITIES=compute the container has
// no libnvidia-ml and the ledger fallback is what runs.
#![allow(dead_code)]
mod own_footprint;
mod backend {
    #[path = "nvml.rs"]
    pub mod nvml;
    pub fn read() -> Option<usize> {
        nvml::process_device_bytes()
    }
}
#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuDeviceGet(dev: *mut i32, ordinal: i32) -> i32;
    fn cuCtxCreate_v2(ctx: *mut u64, flags: u32, dev: i32) -> i32;
    fn cuMemAlloc_v2(p: *mut u64, n: usize) -> i32;
    fn cuMemAllocHost_v2(p: *mut *mut u8, n: usize) -> i32;
    fn cuMemFreeHost(p: *mut u8) -> i32;
    fn cuMemFree_v2(p: u64) -> i32;
    fn cuCtxDestroy_v2(ctx: u64) -> i32;
}
fn sample() -> own_footprint::Sample {
    own_footprint::Sample { driver: backend::read(), host: own_footprint::host_bytes() }
}
fn main() {
    const MIB: usize = 1 << 20;
    let mib = |b: usize| b as f64 / MIB as f64;
    println!("pid {} no context: {:?}", std::process::id(), sample());
    unsafe {
        assert_eq!(cuInit(0), 0);
        let mut dev = 0;
        assert_eq!(cuDeviceGet(&mut dev, 0), 0);
        let mut ctx = 0u64;
        assert_eq!(cuCtxCreate_v2(&mut ctx, 0, dev), 0);
        let base = sample();
        println!("context (baseline): {base:?}");
        let mut ledger = own_footprint::AllocLedger::default();
        // 256 MiB in one block plus 300 x (1 MiB + 1 byte), each of which the
        // driver serves from a 2 MiB chunk; the ledger sees requested bytes.
        for n in std::iter::once(256 * MIB).chain(std::iter::repeat_n(MIB + 1, 300)) {
            let mut p = 0u64;
            assert_eq!(cuMemAlloc_v2(&mut p, n), 0);
            ledger.record(p, n);
        }
        let mut host: *mut u8 = std::ptr::null_mut();
        assert_eq!(cuMemAllocHost_v2(&mut host, 64 * MIB), 0);
        std::ptr::write_bytes(host, 0, 64 * MIB);
        let heap = vec![1u8; 32 * MIB];
        let now = sample();
        let fp = own_footprint::since(base, now, ledger.bytes());
        println!("after allocs: {now:?}");
        println!(
            "ledger {:.2} MiB -> footprint device {:.2} MiB by {} + host {:.2} MiB = {:.2} MiB",
            mib(ledger.bytes()),
            mib(fp.device),
            fp.device_source.label(),
            mib(fp.host),
            mib(fp.total())
        );
        assert!(fp.device >= ledger.bytes());
        assert!(fp.host >= 96 * MIB - MIB);
        match fp.device_source {
            own_footprint::DeviceSource::Driver => assert_eq!(fp.device, (256 + 600) * MIB),
            own_footprint::DeviceSource::Ledger => assert_eq!(fp.device, 556 * MIB + 300),
        }
        std::hint::black_box(&heap);
        assert_eq!(cuMemFreeHost(host), 0);
        for p in ledger.drain() {
            assert_eq!(cuMemFree_v2(p), 0);
        }
        drop(heap);
        let end = sample();
        println!("after frees: {end:?} -> {:?}", own_footprint::since(base, end, ledger.bytes()));
        assert_eq!(cuCtxDestroy_v2(ctx), 0);
    }
    println!("OK");
}
