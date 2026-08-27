# qwen4_exp serving port — running log

Working branch: `feat/qwen4exp-serve` (off `feat/ngram-embed` @ 2bcda1dc).
Goal: bring rsafier's #754 serving work onto this branch, keep this branch's
stronger artifacts, and have `qwen3.8-flash-next` / NVFP4 ready to test the
moment the GB10 is reachable again.

`reiner` is OFF the NetBird tailnet (peer `gx10-e3a3`, 100.105.184.175, Idle,
no WireGuard handshake, 0 B transferred). **Nothing in this log was measured on
hardware here.** Every claim is either (a) quoted from #754/#13 with its
source, or (b) verified by the local gate below.

## The local gate (no GPU, no nvcc)

`scripts/dev/qwen4exp_local_gate.sh` — 13 gates, ~2,800 tests, no CUDA:

```
compile   libs+bins cuda · touched crates all-targets cuda · metal build
tests     atlas-core 158 · spark-server 2306 · spark-storage 87 · spark-runtime 272
repo CI   SPDX headers · kernel shadow structure · 500-LoC cap
lint      fmt · clippy
```

Two env vars are the whole trick: `ATLAS_SKIP_BUILD=1` makes atlas-kernels'
`build.rs` emit a type-checkable stub instead of invoking nvcc, and
`CUDARC_CUDA_VERSION=13000` stops cudarc probing for a toolkit. Nearly all of
this port sits behind `#[cfg(feature = "cuda")]`, so without those the only
thing a laptop can check is the one configuration the code is absent from.

The find worth reusing: test binaries that link `-lcuda` cannot be built on
macOS, but the TESTS are backend-agnostic — only the linking is. Run under
`metal`, 2,665 of them execute here.

Three darwin arms were needed (commit `6d3c6a1d` + follow-ups):
`posix_fallocate`, `posix_fadvise` and `O_DIRECT` have no macOS equivalent.
Production NVMe tiers stay Linux-only; no Linux behaviour changes.

**What the gate cannot say anything about**, printed by the script itself: no
`.cu` is compiled (nvcc is never invoked) and no kernel is run. Every parity
test is `#[ignore]`d and unexecuted.

## Status

| stage | state |
|---|---|
| 0. local gate | **DONE** — 13/13 pass, ~2,800 tests run |
| 1. merge #754 + #746 | **DONE** — 7 conflicts resolved by inspection, 265 files |
| 2. kernel tree integrity | **DONE** — 5 symlinks resolve, no module collision, shadow-structure check green |
| 3. config surface reconciled | **DONE** — one parser, both field families, 158 tests pass |
| 4. clippy + fmt + SPDX | **DONE** — clean |
| 5. oracle gate on the SERVING kernels | **DONE** — mHC (4 entry points, 3 dispatch arms) and the whole PLE chain, no checkpoint needed |
| 6. recipe + serve wiring | **DONE** — vendored recipe (census tests validate it produces a valid serve config), serve script hardened for >8K |
| 7. `--generate N` off-by-one | **DONE** |
| 8. docs + CHANGELOG | **DONE** |
| 9. 500-LoC cap | **DONE** — four pre-existing breaches on this branch fixed |
| 10. run it on the box | **BLOCKED** — `reiner` is off the tailnet |

### Bugs this port work found and fixed

1. **The mHC highway element size.** `hc_streams` was sized per family — f32
   for `hc_mult`, bf16 for `hc_count`. Correct while our own bf16 kernels read
   it; wrong for the kernels that now serve, all of which declare the buffer
   `float*`. Safe today only because the merged parser sets both fields, so the
   f32 arm wins by accident. Read as written the code told the next person to
   restore a branch that would read the wrong half of every value on 48 layers.
2. **The parser skipped `finalize_config`** — an NVFP4 checkpoint parsed as
   unquantized, because ModelOpt writes `quantization_config` beside
   `text_config` and serde on `text_config` alone never sees it.
3. **`ple_layer_ids` accepted 0**, which is not "layer 0" but a malformed
   one-indexed id.
4. **`ngram_dims()` refused the merged config** as a partial LongCat trio. A
   base-form checkpoint now declines that accessor instead, and declaring both
   table sizings is refused.
5. **`--generate N` appended N+1 tokens** (reviewer-reported, reproduced twice
   on a GB10).
6. **Four files over the CI LoC cap**, pre-existing on this branch — #13 is red
   on a gate that runs as its own job.

### Two overlapping mechanisms, checked and documented rather than merged

Our `demand_paged_patterns` SKIPS tensors; #746's deferral skips them AND
records each one's file offset, which is what `NgramRowCache` reads rows
through. The shard loop checks `is_ngram_table` first, so the PLE shards take
the deferred path and our rule never sees them — but if that predicate ever
stopped matching, our rule would quietly take over and the PLE loader would
fail its own "no shard was deferred" check at load. The precedence is now
commented at both ends and pinned by
`name_utils::ngram_table_predicate_matches_the_qwen_ple_shards`.

## Decisions taken (and why)

### Kept from THIS branch
* **`output_gate_type` string, not a bool.** `qwen3_ssm/init.rs` reads the raw
  value and refuses anything but `silu`/`sigmoid`. #754 passes a pre-computed
  `gdn_norm_sigmoid` bool, so an unknown activation silently takes the family
  default. Same swap, stricter failure mode.
* **The sigmoid twins live in `common/rms_norm.cu`** (module `norm`), not in a
  per-target shadow. #754's `gated_norm_sigmoid.cu` was byte-identical to ours
  and its own header warns that a fix to the SiLU originals "should be
  re-derived here". Dropped; every target now inherits the twins.
* **`weight_manifest` + `qwen4exp_preflight`** — 296,142 tensors, 0/0/0 in
  2.81 s and 372 MB RSS. #754 has only `ns_audit.py`, a dev script. Ours runs
  at load time.
* **`atlas_core::qwen4exp_reference`** — the CPU oracle, per-block 1.6e-7 to
  8.0e-7 against HF at real weights, full forward token-identical. This is what
  every GPU kernel is measured against.
* **`Qwen4ExpNgram`** — derives the multipliers/primes/offsets from config and
  asserts equality with the shipped buffers. #754 reads the buffers directly.
  Keeping both means the derivation and the checkpoint check each other.
* **`demand_paged_patterns`** — the never-resident rule, validated on two GB10s
  (78.19 GB pre-flight, ~90.4 GB peak, 47.68 GiB table excluded).
* **`split_ngram_parts`** as the field name — it is the key the checkpoint uses.
* **`finalize_config`** on the qwen4_exp path — #754's parser skipped it, so an
  NVFP4 checkpoint parsed as unquantized.
* **`cap_thinking_at_max_tokens` / `enable_loop_watchdog`** in `[behavior]`.

### Taken from #754
* **Layer placement.** mHC / PLE / QSA hang off the existing `qwen3_ssm` and
  `qwen3_attention` layers rather than a standalone `Qwen4ExpLayer`. That is
  what carries paged attention, CUDA graphs, prefix caching and C>1. Our
  scaffold layer is retired; its kernels and ops wrappers stay as oracles.
* **The four serving kernels**: `hyper_connection.cu` (FP32 highway, fused
  prefill path + 3-launch decode split), `ple.cu` (FP32 end to end),
  `qsa_indexer.cu` (decode + prefill selection), and the five per-file
  symlinks into `qwen3.6-35b-a3b`.
* **`[expected_absent]`** — what takes the fail-closed startup audit from 6
  unresolved kernels to 0. Per-file symlinks replace our whole-target
  `kernel_source` alias.
* **The measured `[behavior]` and sampling presets**, including all three
  sampled-quality fixes (`use_sampling_presets_for_core` in `[behavior]` —
  after a `[sampling.*]` header TOML swallows it as a preset key;
  `min_reasoning_floor_tokens = 0`; `honor_eos_inside_thinking = true`).
* **`norm_topk_prob = true`** — both branches found this independently.
* **`NgramRowCache::open_segmented`** — the 128 PLE shards are NOT contiguous
  (26.4 GB span across a 102.4 GB table), so one base offset reads
  wrong-but-valid rows silently.
* **The parser's stricter surface**: `weight_prefix`, model_type
  normalization, eos_token_id arrays, mandatory `layer_types`.

### Open in BOTH branches
* MTP — refused at pre-flight (issue #753 item I).
* Stacked expert layout — unreached by both published releases.
* Prefix-cache re-ingest for QSA raw indexer keys.
* Thinking-body quality at the card's temp 1.0 after `norm_topk_prob`
  (rsafier's last open thread on #754).

## Traps carried across from #754's comments (each cost a debug cycle there)

1. `hc_expand` must seed at MODEL layer 0, not `attn_layer_idx == 0` — on a
   3:1 interleave those differ by three layers.
2. `hc_head` fires at model layer 47; a `12 == 48` check never fires, and the
   mixer IS the final norm.
3. The GDN gated norm gates with **sigmoid** — wrong activation on 36 of 48
   layers, cos 0.80 before / 0.999990 after.
4. `norm_topk_prob` defaults TRUE in the reference and is OMITTED from the
   checkpoint json; serde's `false` cost 0.33–1.62 nats/token.
5. The terminal dense-BF16 GEMM arm must route to cuBLASLt — prefill
   283 → 747 tok/s.
6. The fused mHC collapse starves at `grid=[1]` on decode — split it:
   4.4 → 16.5 tok/s.
7. All QSA mid-path D2H must use `copy_d2h_on_stream`; `copy_d2h` orders
   against the DEFAULT stream while layers compute on the worker stream, and
   every parity test passed anyway because tests run on the default stream.
8. PLE's host hash must be hoisted out of the CUDA-graph capture region, and
   the batched path must consult the per-layer graph veto too.
9. An always-on debug diag (`diag_norm`: synchronize + copy_d2h, both errors
   swallowed) silently invalidated graph capture. Now behind
   `ATLAS_DIAG_V4_ALL_LAYERS`.
10. The engine's prefill chunk is **8193**, not 8192 — `ATLAS_PLE_MAX_TOKENS`
    must clear it (recipe carries ~9500 for >8K prompts).
11. Raw-row logit dumps are `model.vocab_size()` = 248077 wide, not the
    config's 248320.

## Kill switches / diagnostics available after this merge

`ATLAS_QSA_DISABLE`, `ATLAS_QSA_NO_PREFILL_SELECT`,
`ATLAS_QWEN4EXP_NO_HC_GEMM`, `ATLAS_DEBUG_NO_GRAPH`, `ATLAS_NO_GDN_FLA`,
`ATLAS_GDN_FLASHINFER`, `ATLAS_QWEN4EXP_DUMP`, `ATLAS_QWEN4EXP_PREFILL_PROF`,
`ATLAS_DUMP_HYPER_RMS`, `ATLAS_DUMP_LOGITS_PATH`, `ATLAS_PLE_MAX_TOKENS`,
`ATLAS_DIAG_V4_ALL_LAYERS`, `ATLAS_QWEN4EXP_NO_PLE`.

## First run on the box, when it is back

In order, cheapest first — each one closes a class of risk the local gate
cannot touch.

```sh
# 0. is it reachable at all
netbird status -d | grep -A6 gx10        # want Connected + a recent handshake
ssh reiner 'nvidia-smi -q | grep -i addressing'   # want ATS

# 1. do the kernels COMPILE (nvcc was never invoked locally)
cargo build --release -p spark-model --no-default-features --features cuda

# 2. the serving kernels vs the CPU oracle — no checkpoint, no Python
cargo test --release -p spark-model qwen4exp_oracle -- --ignored --nocapture

# 3. the same kernels vs the real reference module, if the fixture bins exist
#    (hc_golden.npz / ple_golden.npz are committed; qsa_golden.npz is not and
#    must be regenerated from the checkpoint)
ATLAS_HC_TEST_DATA=<dir> cargo test --release -p spark-model hc_lowrank -- --ignored

# 4. the five block microtests (independently reproduced on a second GB10)
cargo run --release -p spark-model --example qwen4exp_grouped_norm_microtest \
      --no-default-features --features cuda

# 5. the checkpoint is described exactly: want 296,142 / 0 / 0 / 0
cargo run --release -p atlas-core --example qwen4exp_preflight -- <ckpt>

# 6. the CPU forward still generates "Paris." from the real weights
cargo run --release -p atlas-core --example qwen4exp_forward -- <ckpt> /fx/prompt.json --generate 8
#    ^ and confirm it now appends 8, not 9

# 7. serve
./serve_qwen4exp_tui.sh                  # raises ATLAS_PLE_MAX_TOKENS itself above 8K
```

If step 7 misbehaves, the kill switches are the bisect: `ATLAS_QSA_DISABLE=1`,
`ATLAS_QSA_NO_PREFILL_SELECT=1`, `ATLAS_QWEN4EXP_NO_HC_GEMM=1`,
`ATLAS_DEBUG_NO_GRAPH=1`, `ATLAS_QWEN4EXP_NO_PLE=1` (output is wrong by
construction under that last one — it exists to bisect the mHC spine).

## Still open, and honest about it

* Nothing here has run on hardware. Every number in this log is either quoted
  from #754/#13 with its source, or produced by the local gate.
* `spark-model`'s lib tests do not COMPILE under `metal` (17 pre-existing
  errors in test code that assumes the cuda backend), so that crate's unit
  tests are type-checked under cuda here but only RUN on the box. Worth fixing
  to widen the gate; out of scope for this port.
* `qsa_golden.npz` is gitignored, so the QSA parity tests need regenerating
  from the checkpoint before they can run. The new oracle gate covers mHC and
  PLE but not QSA — there is no CPU oracle for the indexer yet, and writing one
  is the obvious next contribution to this harness.
* MTP, the stacked expert layout, QSA prefix-cache re-ingest, and thinking-body
  quality at temperature 1.0 are open in both branches.
