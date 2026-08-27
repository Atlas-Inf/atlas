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

`scripts/dev/qwen4exp_local_gate.sh` — 16 gates, 3,468 tests, no CUDA:

```
compile   libs+bins cuda · touched crates all-targets cuda · metal build
tests     atlas-core 164 · spark-server 2307 · spark-model 645
          spark-storage 87 · spark-runtime 272
repo CI   SPDX headers · kernel shadow structure · qwen4_exp kernel NAMES
          500-LoC cap
lint      fmt · clippy
```

`check_qwen4exp_kernel_names.py` is new and is the one class of startup failure
a laptop can rule out: kernel lookups are two STRING literals, so `cargo check`
cannot see a typo — and a name that resolves in ANOTHER model's shadow is worse
than one that does not resolve at all (`hyper_connection` belongs to
DeepSeek-V4's Sinkhorn mHC as well as to this target's low-rank one; the same
name over a different argument list is a segfault or, worse, plausible
numbers). It walks the target's real `.cu` files — through the symlinks into
`qwen3.6-35b-a3b`, plus everything inherited from `common/` — and checks all 37
names the qwen4_exp path asks for, including the three sigmoid twins that are
built by name construction and so are invisible to a grep. Both failure modes
have a proven negative control.

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
| 5b. kernel NAME resolution checked statically | **DONE** — 37/37 resolve; both failure modes have negative controls |
| 5c. QSA oracle + parity gate | **DONE** — the last block with no committed golden now has a CPU oracle |
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

## Audit: every defect in #754's comment thread, verified present here

Not "the merge brought his commits so it must be in there" — each one checked
against the code on this branch. His 16 PR comments report these; the right
column is where the fix lives now.

| # | defect, as he reported it | verified in |
|---|---|---|
| 1 | `hc_expand` seeded the highway THREE LAYERS LATE — `attn_layer_idx` is not the model index, and on a 3:1 interleave `attn_layer_idx == 0` is model layer 3 | `HcWeights::is_first_model_layer`, consumed at 3 sites (`trait_prefill_hc`, `trait_decode_hc`, `trait_decode_multi_seq/hc`) |
| 2 | `hc_head` NEVER FIRED (`12 == 48`) — and the mixer IS the final norm, since the checkpoint ships no `model.norm` | `is_last_model_layer`, set as `idx + 1 == num_hidden_layers` in `weight_loader/qwen4_exp/aux.rs:155` |
| 3 | a SECOND RMS over `hc_pre`'s output, input side | `trait_decode_hc.rs:120` — "No `input_norm`: `hc_norm` inside `hc_pre` is this layer's norm", with the note that the loader's ones-placeholder would NOT make a second pass an identity |
| 4 | `hc_norm` dropped the OFFSET-FROM-1 | `hyper_connection.cu:193` — `x[i] * rms * (1.0f + hc_norm_w[i])`, with "the `1.0f +` is NOT optional" above it |
| 5 | `diag_norm` read the FP32 highway as BF16 and reported NaN on healthy data | a separate `diag_norm_f32`, used at every highway site in `decode_inner.rs` |
| 6 | `ops::dense_gemm` (scalar launcher) handed the PIPELINED kernel → wrong grid → OOB → NaN | the dense arms route through `ops::cublas_bf16_proj` / the cuBLASLt path in `trait_prefill_proj.rs` |
| 7 | defect 3's twin on the FFN side | same site, both `hc_pre_attn` and `hc_pre_ffn` |
| 8 | PLE scratch sized from `max_position_embeddings` (262144 → 8192) because `--max-seq-len` never writes back | `ATLAS_PLE_MAX_TOKENS`, read in `weight_loader/qwen4_exp.rs:211`, and the serve script now raises it above 8K |
| 9 | **the coherence bug**: the GDN gated norm gates with SIGMOID, not SiLU — wrong on 36 of 48 layers, cos 0.80 → 0.999990 | our stricter path: `kernel_select::gated_norm_kernel` reads `output_gate_type` and REFUSES anything but silu/sigmoid |
| 10 | **the sampled-quality root cause**: `norm_topk_prob` defaults TRUE in the reference, is OMITTED from the checkpoint, and serde's default is false — 0.33–1.62 nats/token | pinned true in the parser, with a test asserting the key is absent from the shipped config |
| 11 | runtime BF16→NVFP4 requantization of projections that SHIP BF16 cost 6.04 GB | BF16 by default; `ATLAS_QWEN4EXP_BF16_GDN=0` reverts |
| 12 | the fused mHC collapse starves at `grid=[1]` on decode (2.0 ms × 96 calls) | the three-launch split — `hc_pre_stage` / `hc_pre_down` / `hc_pre_finish`, 5 sites in the kernel, 4 in the wrapper |
| 13 | `qkvz_gemm` was 5.27 s of an 8.4 s TTFT: every quantized arm is `None` on BF16 GDN weights, so dispatch fell to the scalar 16×16 `dense_gemm` | the cuBLASLt arms in `trait_prefill_proj.rs` |
| 14 | PLE's in-capture host work poisoned CUDA-graph capture (pageable H2D from a stack Vec) | `decode_prestage`, with `ple/layer.rs:259` refusing the host path under capture |
| 15 | an ALWAYS-ON debug diag (`diag_norm`: synchronize + `copy_d2h`, both errors swallowed) silently invalidated capture | opt-in behind `ATLAS_DIAG_V4_ALL_LAYERS` |
| 16 | `copy_d2h` orders against the DEFAULT stream while layers compute on the worker stream — the host top-k read half-baked scores, and every parity test passed because tests run on the default stream | **zero** bare `copy_d2h` left in `qsa.rs` / `qsa_select.rs`; 7 uses of `copy_d2h_on_stream` |

Also carried and checked: the three prefix-cache restore sites decline aux-less
slots (`prefix_lookup.rs:208`, `ssm_snapshot.rs:63`) rather than restoring stale
PLE/QSA state; the engine's prefill chunk is **8193**, not 8192; logit-dump rows
are `model.vocab_size()` wide, not the config's; and the vision tower loads via
the qwen35 ViT delegation.

## Kill switches and diagnostics

Enumerated from the code, not from memory — the first pass at this list was
written from #754's comments and MISSED FOUR, which is worse than useless at
3am. `scripts/dev/check_qwen4exp_kernel_names.py`'s sibling discipline applies:
grep the surface, do not recall it.

| var | effect |
|---|---|
| `ATLAS_QSA_DISABLE=1` | detach the indexer entirely (A/B) |
| `ATLAS_QSA_NO_PREFILL_SELECT=1` | keep decode selection, drop stage-2 prefill selection |
| `ATLAS_QSA_MAX_TOKENS` | indexer key-cache ceiling; the guard names it when exceeded |
| `ATLAS_QSA_S2_DIAG=1` | stage-2 prefill-selection diagnostics |
| `ATLAS_QWEN4EXP_NO_HC_GEMM=1` | revert the large-T collapse to the fused FP32 kernel |
| `ATLAS_QWEN4EXP_NO_PLE=1` | DISABLE PLE injection — output is wrong by construction; exists to bisect the mHC spine |
| `ATLAS_QWEN4EXP_BF16_GDN=0` | requantize the GDN projections to NVFP4 (costs 6.04 GB, buys ~1% decode) |
| `ATLAS_QWEN4EXP_DUMP=<dir>` | per-sublayer highway taps, one-shot per file |
| `ATLAS_QWEN4EXP_PREFILL_PROF=1` | per-stage prefill timer |
| `ATLAS_PLE_MAX_TOKENS` | PLE scratch ceiling — must clear 8193 for chunked prompts |
| `ATLAS_PLE_CACHE_SLOTS` | n-gram row-cache slot count |
| `ATLAS_DEBUG_NO_GRAPH=1` | disable CUDA graphs |
| `ATLAS_DIAG_V4_ALL_LAYERS=1` | the mHC diag on every layer (off by default; it used to be always-on and poisoned capture) |
| `ATLAS_DUMP_HYPER_RMS` | per-layer highway RMS trail — the trail that caught `norm_topk_prob` |
| `ATLAS_DUMP_LOGITS_PATH` | raw per-step logit rows at all three sampling entries |
| `ATLAS_NO_GDN_FLA=1` / `ATLAS_GDN_FLASHINFER=1` | GDN recurrence A/B levers |
| `ATLAS_HC_TEST_DATA` / `ATLAS_QSA_TEST_DATA` | fixture dirs for the checkpoint-backed parity tests |

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
* ~~`spark-model`'s lib tests do not compile under `metal`~~ — FIXED. Five GPU
  parity modules were gated on `test` alone, so one unavailable backend took
  the whole crate's unit suite with it. Gated on `all(test, feature = "cuda")`
  and 645 tests now run locally. The `tests/` integration targets
  (`arm2_leg2_decode`, `moe_lora_delta_parity`) still need cuda to link; that
  is pre-existing and outside this port.
* ~~no CPU oracle for QSA~~ — WRITTEN. `qwen4exp_reference::qsa` is
  transcribed from the vendored reference module (not from the kernel) and
  exposes the three per-stage helpers `qsa_select` itself uses, so
  `qwen4exp_oracle_qsa_tests.rs` compares the kernels against the oracle rather
  than a second copy of the formula. Six CPU tests pin the mechanism with no
  GPU at all, including the inertness threshold and the per-head relu.
  `qsa_golden.npz` is still gitignored, so the checkpoint-backed QSA parity
  test needs regenerating on the box; the oracle gate no longer depends on it.
* **A behavioural fact the QSA oracle surfaced**, worth knowing before it looks
  like a bug: when the visible count is a multiple of `compress_ratio` the tail
  is EMPTY, so the current token sits inside a complete block and can be masked
  out if that block loses the ranking. The reference force-includes nothing.
  Pinned by `the_tail_is_visible_and_the_current_token_is_not_force_included`,
  which also asserts the case actually occurs, so a future "helpful" fix cannot
  quietly diverge from the reference on 1 in `ratio` positions.
* MTP, the stacked expert layout, QSA prefix-cache re-ingest, and thinking-body
  quality at temperature 1.0 are open in both branches.
