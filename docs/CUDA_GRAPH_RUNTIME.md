# Atlas CUDA graph runtime

This document describes the unified, phase-aware CUDA graph runtime in
`crates/spark-runtime/src/graph_runtime/` and states what is production-ready,
experimental, and unsupported. It is the operator-facing companion to the
implementation; where the two disagree, the code is the source of truth.

## What it is

Atlas used to keep one bespoke graph cache per path (single-sequence decode,
K=2/3/4 verify, batched verify, DFlash propose, fused decode+verify). The graph
runtime replaces those with a single manager that owns keys, lifetime, cache
bounds, metrics, and fallbacks for every phase.

**Phases.** `Prefill`, `Decode`, `Verify`, `Propose`, `Fused`.

**Modes** (`--cuda-graph-mode`).

| Mode | Meaning |
|---|---|
| `disabled` | No capture; everything eager. |
| `full` | Whole-step capture for decode/verify/fused. Segmented phases (prefill, propose) stay eager. |
| `breakable` | `full` plus explicitly bounded capture segments (e.g. exact-shape prompt-prefill compute). |
| `piecewise` | As `breakable`, for paths that capture pre/post-attention subgraphs separately. |

**Shape buckets.** `--cuda-graph-token-buckets` and
`--cuda-graph-request-buckets` define the (tokens × requests) buckets a graph
may be captured for. A concrete request is classified into the smallest bucket
that contains it; a shape with no bucket falls back to eager with a
`shape_unsupported` reason. Buckets must be positive and strictly increasing.

**Keys.** A graph is identified by phase, mode, segment, speculative algorithm,
environment fingerprint (runtime/model/kernel-build/device/CUDA/driver/memory
layout), a resource generation, and a phase-specific payload (shapes, slots,
depths, layout, or — for `Propose` — an opaque `key_words` vector carrying
model identity such as the DFlash lane/owner). Any change to a fingerprint
dimension invalidates cached graphs and exported manifests.

## Cache and lifetime

- Per-phase entry and byte quotas, plus a process-wide entry/byte cap. Both are
  bounded LRU: a full cache evicts rather than disabling capture.
- Negative caching: a deterministic capture failure can be remembered so the
  same key is not re-attempted.
- Deferred destruction: an evicted graph is retired through a CUDA event and
  destroyed only once the stream has drained and the last lease is dropped.
- Leases: a caller holds a `GraphLease`; the runtime will not destroy a graph
  that is still leased.

CLI: `--cuda-graph-cache-entries`, `--cuda-graph-cache-mb`.

## Capture safety and fault injection

The backend refuses host-blocking operations on the capturing thread while a
capture is active (allocation, free, stream/event creation, synchronization,
blocking copies, dynamic kernel lookup, synchronous memset, graph
launch/destroy). Atlas captures with `CU_STREAM_CAPTURE_MODE_RELAXED`, so other
threads may still operate; the gate is thread-scoped for that reason.

Fault injection for testing (all default off):

| Variable | Injects |
|---|---|
| `ATLAS_GRAPH_FAULT=capture,instantiate,replay,event` | Failure at the named capture points. |

## Metrics

`/metrics` exposes `atlas_cuda_graph_*` counters: captures, recaptures, replays,
capture/replay failures, eager fallbacks (global and per reason), evictions,
launches, graph memory, the active algorithm, and per-phase breakdowns
(`atlas_cuda_graph_phase_*{phase=...}`). A serve with zero captures is not a
graph test; check these counters before believing a graph path ran.

## Export, manifests, and prewarming

`--cuda-graph-export-dir` writes DOT topology, a manifest, and a prewarm profile.
`--cuda-graph-prewarm-profile` recreates the recorded key set at startup. A raw
`cudaGraphExec_t` is process-local, so portable binary serialization is a
deliberate no-go; manifests are invalidated when any fingerprint dimension
changes.

## Compatibility registry

The runtime can declare rules (id, phases, modes, fallback reason, detail) that
force a phase to eager. The prefill decision is routed through the registry so
the declared rules and the actual fallback cannot drift. Rule-driven fallbacks
are recorded with both the reason and the rule id.

## Status

- **Production-ready:** the unified runtime's decode, single-sequence verify,
  batched verify, fused, prefill, and DFlash-propose capture paths, *once a
  hardware run validates them* (see below).
- **Experimental:** the CUDA IF/ELSE and bounded WHILE primitives
  (`kernels/gb10/common/graph_control.cu`); first-class D-Spark (`--dspark`).
- **Unsupported:** confidential computing on DGX Spark/GB10 (see
  `CONFIDENTIAL_COMPUTING.md`); portable binary graph serialization; graph
  capture of layers whose decode is host-stateful (QSA/PLE veto the whole
  model to eager).

## Validation

GPU validation — real capture/replay, the IF/ELSE and WHILE microtests, and the
dense/MoE/hybrid prefill differential matrix — is required before any of the
above is called correct on a given hardware/firmware fingerprint. The
GPU-free gates (format, lints, SPDX, typos, unit tests) do not exercise a
device.
