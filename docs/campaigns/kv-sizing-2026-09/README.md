# KV pool sizing on unified memory (2026-09-30)

What was measured for `crates/spark-runtime/src/own_footprint.rs` and
`crates/spark-model/src/factory/build/kv_budget.rs`, and what still has to be
checked on a two-node pair before those are trusted in production.

## The defect

The KV pool was sized from "free memory at context init minus free memory
now" as Atlas's own footprint. On a GB10 that difference also moves when
another process frees memory during the ~80 s model load. Two TP2 ranks read
100.1 and 97.9 GiB for a process that holds 102.3, sized 123,500 and 143,479
blocks against the usual 102,066, and the host ran out of unified memory on
the second long request.

## Measured (GB10, driver 580.178.04, in a container, `--gpus all`)

The host also served other models, so `MemAvailable` moved by tens to
hundreds of MiB on its own between samples. The process counters did not.

| What the process takes | Driver figure (NVML) | `RssAnon` | `RssShmem` | Source |
|---|---|---|---|---|
| `cuMemAlloc` 1 GiB, untouched or touched | +1024.0 MiB | 0 | 0 | `raw/accounting-probe.settle3.out` A |
| `cuMemAlloc` 2000 x 100 KiB (195.3 MiB) | +200.0 MiB | 0 | 0 | B1 |
| `cuMemAlloc` 300 x (1 MiB + 1 B) | +600.0 MiB | 0 | 0 | B2 |
| `cuMemAlloc` 200 x (3 MiB + 1 B) | +612.5 MiB | 0 | 0 | B3 |
| `cuMemAlloc` 100 x 2 MiB | +200.0 MiB | 0 | 0 | B4 |
| `cuMemAllocHost` 256 MiB | 0 | 0 | +256.0 MiB | C |
| `cuMemHostAlloc` `PORTABLE\|DEVICEMAP` 256 MiB | 0 | 0 | +256.0 MiB | D |
| `cuMemAllocManaged` 256 MiB, touched on device | 0 | 0 | 0 | E |
| heap 256 MiB | 0 | +256.0 MiB | 0 | F |

- The driver carves requests of up to 2 MiB from 2 MiB chunks and rounds
  larger ones up to 64 KiB. Its per-process figure includes that rounding.
- Page-locked host memory is in `RssShmem` only, mapped into the device
  address space or not, so device plus host does not count it twice.
- Managed memory is in none of the three counters. Atlas allocates it only
  as the out-of-memory fallback for weights.
- NVML answered unprivileged inside the container under the in-namespace PID
  (the probe ran as PID 1). `_v2` and `_v3` of
  `nvmlDeviceGetComputeRunningProcesses` returned the same figure.
- With `--gpus all` the library was mounted even with
  `NVIDIA_DRIVER_CAPABILITIES=compute` (an earlier run; its output was not
  kept).
- Driver growth against the drop in `MemAvailable`: 4096.0 against 4101.1 MiB
  (G), and 8192.0 against 8231.0, then 8193.6 two seconds later
  (`raw/scale-probe.out`, cycle 1). In cycle 2 `MemAvailable` fell 670 MiB
  less than the driver figure rose: the first cycle had evicted 7.6 GiB of
  page cache, and the host ended the run with about 1 GiB more available
  than it started with. A shared host's `MemAvailable` is that noisy.
- `rust-footprint/` runs the production `nvml.rs` and `own_footprint.rs`
  against real allocations (`raw/rust-footprint.out`). With the library
  present: ledger 556 MiB, device 856 MiB by driver accounting. With it
  hidden: device 556 MiB by allocation ledger, 300 MiB low.

## Not measured

- Driver growth against the `MemAvailable` drop at 100 GiB, on a quiet host.
- Kernel-side memory that is in neither the driver figure nor RSS.
- The production image and the pair's container flags.
- Anything on AMD unified memory: `own_footprint()` returns `None` there.
- `cuCtxCreate` returned `CUDA_ERROR_OUT_OF_MEMORY` on three of eight tries of
  the Rust harness on the shared host, with 47 GiB available, once with the
  NVML library hidden. Not explained; the Python probes never hit it.

## Pair check

Run on a TP2 pair with the memory watchdog armed. Every line below is logged
by `spark_model::factory::build` on each rank at start.

**A. Quiet start (no load generator, nothing else started or stopped).**

1. Rank 0 and rank 1 each log `KV own-footprint cross-check: free-memory
   measure X GB, tracked Y GB (device D GB by driver accounting + host H GB)`.
   Record X, Y, D, H and the `co-tenants` figure of the line above it.
2. Pass: the text says `by driver accounting` on both ranks, there is no
   `KV own-footprint:` WARN and no `KV pool limited by free memory` WARN,
   and Y is at or below X on both ranks.
3. Rank 0 then logs about 102K blocks at 0.93, as before.
4. Fail, and do not run B: `by allocation ledger` on either rank (fix the
   container so `libnvidia-ml.so.1` is present), or Y below X by more than
   0.5 GiB (the counters miss memory, and B would rest on the floor alone).
5. Y above X on a quiet start means the counters over-count. The pool is
   then smaller on every boot by that amount. It is safe; record the gap and
   decide whether a tolerance is wanted before this replaces production.

**B. Release during the load. Only after A passes on both ranks.**

1. On rank 0's host: `hold-release.py --gib 1 --hold 45`, then start the
   pair as soon as it prints `HOLDING`. It refuses below 16 GiB available
   and frees on its own below 2 GiB.
2. Its `RELEASED` line must fall between rank 0's `GPU 0: ... GB free` line
   and its `KV budget self-relative` line. If not, the release missed the
   load: restart and repeat with a different `--hold`.
3. Expect on rank 0: `baseline-free` about 1 GiB lower than in A, the WARN
   `Atlas allocated Y GB but free memory fell by only X GB — about 1 GB was
   released by something else`, and the same block count as A within 1%.
4. Read the block count before sending any traffic. More than 1% above A's
   means part of the release was sized into the pool and the check failed
   (the floor caps it near 106K blocks, about 1.5 GiB free in steady state):
   send nothing.
5. Restart the pair normally afterwards in every case.
