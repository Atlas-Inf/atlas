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

`check_qwen4exp_kernel_names.py` is the one class of startup failure a laptop can
rule out — and worth reading for how wrong it was at first. Pointing it at the
wider dispatch surface exposed three bugs IN THE CHECKER, all of which made it
UNDER-report the available kernels, i.e. produce a false MISSING for a kernel
that is right there:

1. the name is not always adjacent to `void` —
   `extern "C" __global__ void __launch_bounds__(128, 1)\ngated_delta_rule_decode(`
   and `extern "C" __global__\n__launch_bounds__(128, 3)\nvoid w4a16_gemm_t_m128(`
   both defeat a `void (\w+)\(` regex. The prologue is now scanned. 437 -> 476
   entry points;
2. 21 files name their kernel with `#define KERNEL_NAME x` and `#include` a
   template, so the name never appears in an `extern "C"` line — and the
   template declares both `KERNEL_NAME` and `PAGED_CONCAT(KERNEL_NAME, _64)`.
   476 -> 518;
3. a kernel can be declared through an alias (`#define ATLAS_PREFILL_ENTRY
   inferspark_prefill`), so names resolve through the file's object-like macros.

Only with the available set complete was it worth widening the scope from the
qwen4_exp files to every fail-closed lookup in the init paths of the layers this
model builds. `gpu.kernel` (fail-closed, a miss means the server refuses to
boot) is distinguished from `try_kernel` (0-handle, caller gates, absences are
documented fallbacks in other models' shadows) — conflating them would flag 34
legitimate fallbacks.

This mattered because rsafier took HIS tree from "6 unresolved kernels to 0",
and the merge changed which kernels resolve here: his per-target
`gated_norm_sigmoid.cu` shadow was dropped in favour of ours in `common/`. So
inheriting that result was not safe.

What it does: kernel lookups are two STRING literals, so `cargo check`
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
| 5b. kernel NAME resolution checked statically | **DONE** — all **160 fail-closed lookups** on the layers a qwen4_exp model builds resolve against 518 entry points; 34 `try_kernel` fallbacks correctly not required; both failure modes have negative controls |
| 5c. QSA oracle + parity gate | **DONE** — the last block with no committed golden now has a CPU oracle |
| 6. recipe + serve wiring | **DONE** — vendored recipe (census tests validate it produces a valid serve config), serve script hardened for >8K |
| 7. `--generate N` off-by-one | **DONE** |
| 8. docs + CHANGELOG | **DONE** |
| 9. 500-LoC cap | **DONE** — four pre-existing breaches on this branch fixed |
| 10. run it on the box | **BLOCKED** — `reiner` is off the tailnet |

### What CI caught that no local gate could

The PR (Atlas-Inf #16) runs the real matrix, and it found four things in three
rounds that a mac and a Linux box both miss:

1. **`no_model_shadow_drops_a_common_kernel`** — a repo invariant asserting no
   model shadow drops a kernel its `common/` namesake declares. #13 added four
   kernels to `common/rms_norm.cu`; ten targets carry their own `rms_norm.cu`,
   so all ten "dropped" all four (40 findings). Fixed with a common-level
   `[shadow_exempt]` entry, following the precedent directly above it. **My gate
   missed it because it ran `--lib` everywhere and these invariants live in
   `atlas-kernels`' TEST TARGETS.** That row is now in the gate.
2. **`aux.rs` cannot exist on Windows.** `aux` is a reserved DEVICE name (with
   con, prn, nul, com1-9, lpt1-9) and the reservation holds even with an
   extension, so `git checkout` itself failed — `error: invalid path`, exit 128,
   14 seconds in, before any compiler. Came in with #754. Renamed to
   `attach.rs`; the whole tree swept for the other twelve stems as files AND
   directories; a guard added to the gate.
3. **`ngram_table.rs` had no Windows arm** for its positional read — OUR file.
   `read_exact_at` is Unix-only; `seek_read` is the Windows equivalent and needs
   a short-read loop. Now behind one helper, and `rustup target add
   x86_64-pc-windows-msvc` makes this checkable from a mac, so it is a gate row.
4. **`-INFINITY` in `qwen4exp_attn.cu`** — a `double`, and narrowing it to
   `float` is nvcc diagnostic #221-D, which is an ERROR on the MSVC host and a
   pass on the Linux host. The Linux `nvcc -> PTX` job compiled the file green
   while Windows could not. `-1e30f` is what `argmax_bf16.cu` and
   `inferspark_prefill.cu` already use for a softmax running max.

And one in the gate itself: run without a PATH export, its toolchain fallback
set `CARGO` to an absolute path without putting that bin dir on PATH, so cargo
could not find `rustc` — twelve rows red on a clean tree. A gate that fails for
the wrong reason is the worst kind, so the toolchain is now proved usable before
any row runs, and the fallback exports PATH.

**What CI verified that the local gate cannot:** `nvcc -> PTX (all gb10
targets)` PASSES — every merged `.cu` compiles for every gb10 target. That was
the largest open risk on this branch. `cargo test --workspace` on Linux passes
too.

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

**One command:**

```sh
./scripts/dev/qwen4exp_first_run.sh                       # steps 1-3, no checkpoint
./scripts/dev/qwen4exp_first_run.sh --ckpt /path/to/snap  # steps 1-5
```

Ordered cheapest first, stops at the first real failure, logs each step to
`qwen4exp-first-run/<n>.log`, and prints a summary to paste into the PR. Steps
1-3 need NO checkpoint, which is the point: the 126 GiB download is the slowest
part of a fresh box and three quarters of what can be wrong is provable before
it finishes. It deliberately does not start a server — that wants a human
watching — and prints the command plus the bisect order instead.

"Everything skipped" reports as NOTHING ATTEMPTED rather than as a pass,
because reading the former as the latter is how a plan gets marked done
without evidence.

The same steps, by hand:

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

## Audit round 2: the claims that exist only in COMMIT MESSAGES

The table above covers the 16 defects reported in #754's PR comments. Three of
his last commits describe further bugs only in their commit message bodies, and
they are the concurrency ones — i.e. exactly the "does it SCALE" half. Audited
the same way:

| claim (commit) | verified in |
|---|---|
| **The C=2 row swap** (`a2c960b2`): the scheduler passes stream 0 to `decode_batch` but `decode()`'s kernels run on the BACKEND default stream, so staging the logits rows on the caller's stream ordered the copies against NOTHING — all n copies could execute after the last decode and read the same final row. Measured as a clean two-way swap: stream A emitting stream B's token and vice versa. | `decode_a2.rs:154` — `copy_stream = self.gpu.default_stream()`, with the whole rationale in place, feeding `copy_d2h_on_stream` |
| **The batched graph gate never consulted the per-layer veto** (`08b885fc`): single decode did, the batched path did not, so capture hit PLE's host hash on the first joint step | `decode_graph_unsupported()` — defaulted on `transformer_layer.rs:96`, implemented by the SSM layer (PLE) and the attention layer (QSA), and consulted at BOTH `decode_a.rs:250` (single) and `decode_a2.rs:265` (batched) |
| **QSA ingest continuity** (`08b885fc`): the batched multi-seq path skipped the indexer, so the contiguity guard fired the moment a batch shrank back to one sequence | `multi_seq/mod.rs:246` runs `qsa.decode_select` per sequence, ingest-only below the inert bound |
| **Both batched dense arms must go through cuBLASLt** (`8da3fb22`): the BF16-kept GDN build routed the batched projections onto the terminal scalar `dense_gemm` — the same kernel the prefill saga hit | `ssm_batched.rs:324` (QKVZ) and `:414` (out_proj), both `ops::cublas_bf16_proj_dense`, and the comments carry his measured 381 us / 194 us |

That closes the audit: every defect and fix rsafier reported, in comments or in
commit messages, is present here and pointed at its site.

## The one design claim I re-verified rather than quoted

#754's phase-C comment says the SSM prefill's steps 2-10 "moved verbatim into
`prefill_block` — **verified byte-identical** against `git show HEAD:`". That
claim is load-bearing twice over: if the body drifted, prefill is subtly wrong
on 36 of 48 layers of THIS model, and — because the same body serves every
existing GDN family — on qwen3.5, qwen3.6 and qwen3-next too. Five of the
seven merge conflicts were in these files, so quoting his verification was not
enough.

Checked, on this branch, after the merge:

| | |
|---|---|
| extracted body, code lines (comments/blanks/indent stripped) | 296 |
| appearing VERBATIM in `pr754base`'s `trait_prefill.rs` | **269** |
| not appearing | 27 |

All 27 account for themselves:

* **6** are the new signature — `pub(super) fn prefill_block(`,
  `normed: DevicePtr`, `ssm_layer_idx: usize`, the return type, `Ok(out_proj_buf)`,
  and an `#[allow(unused_variables)]`;
* **4** are locals that used to be in scope from the caller and are now
  re-derived inside. These were the actual risk, and all four are
  CHARACTER-IDENTICAL to the base: `key_dim = nk * kd`,
  `value_dim = nv * vd`, `conv_dim = key_dim * 2 + value_dim`,
  `qkvz_size = ctx.config.ssm_qkvz_size()`;
* the rest are the additive `ATLAS_QWEN4EXP_DUMP` taps (`tap_bf16` / `tap_f32`
  plus their five labels and sizes) and one reworded stage-timer line.

And both entry paths really do share the one body, which is what makes the
"extraction, not duplication" argument hold:

* `trait_prefill.rs:123` — step 1 `rms_norm_residual` → `prefill_block` →
  step 11 `residual_add_rms_norm` → step 12 `forward_prefill` → step 13
  `residual_add`. Unchanged bracketing, so no existing GDN model is on a new
  path.
* `trait_prefill_hc.rs:203` — `prefill_block(hidden, …)`, with steps 1/11/13
  REPLACED by `hc_pre`/`hc_post` rather than wrapped, because under a highway
  the highway is the residual and those three double-count.

So the claim stands, and it stood through the merge.

## The vendored-reference question, resolved with evidence

I flagged this as "needs a maintainer call" and then went and got the facts, so
the call is now cheap:

* **Nothing imports it.** All five golden generators (`hc_golden.py`,
  `ple_golden.py`, `qsa_golden.py`, `forward_ref.py`, `slice_ref.py`) import
  from the INSTALLED `transformers`. `bench/qwen4_exp/ref/` could be deleted
  without breaking a test or a generator. Every other reference to it is a
  comment or a doc.
* **But one citation is by line number.** `ple.cu:7` cites
  `modeling_qwen4_exp.py L1168` for the PLE forward, and that only resolves
  against a pinned copy.
* **And the revision moved under us.** The transcriptions were read against
  transformers 5.8.0.dev0; the generators now run 5.16.1. The copy is the only
  record of the text actually transcribed — which is what an argument about a
  numerical disagreement would need.
* **It is outside the SPDX check.** `.licenserc.yaml` covers `crates/**/*.rs`
  and `kernels/**` only, so nothing there is stamped AGPL and nothing should be.
  No compliance failure; the Apache-2.0 headers are intact.

**Recommendation: keep it, and carry the notice properly** — which is what
`bench/qwen4_exp/ref/LICENSE` now does, following the `bench/ngram_embed/LICENSE`
precedent. Ours is the stronger case: that one covers an independent derivation
with no vendored file, this one covers actual vendored files, so Apache-2.0
applies to them directly. If the directory is ever dropped, the line-number
citations should become revision-pinned URLs in the same change — the notice
says so.

This does not need to block the PR either way.

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

## First run on real hardware (2026-08-27, GB10 `gx10-e3a3`)

The port had a 20/20 static gate, ~3,500 tests, 27 green CI legs and two
completed audit rounds before anything executed. Running it found four
defects in the first ten minutes. Recording them here because the split
between what the gate could and could not reach is the useful part.

Environment: GB10, `sm_121`, CUDA 13.0, 119.6 GB unified
(`memory.total [N/A]`, `Addressing Mode: ATS` — the coherent-memory tell),
`RadixArk/Qwen3.8-Flash-Next-NVFP4` resident at `~/code/radixark-nvfp4`.

### What the static gate DID reach, and got right

| step | result |
|---|---|
| kernels compile (nvcc) | 502/4564 unique invocations, 9.1x dedup, 177 kernels for `qwen3.8-flash-next` |
| kernels vs CPU oracle | **3/3** — mHC highway, PLE tower, QSA indexer, each with its control |
| five block microtests | **5/5** — GDN decode, attention decode, hc expand, hc sandwich, grouped norm |
| checkpoint preflight | **0 missing / 0 unexpected / 0 mismatched** |

So every kernel this model builds is numerically right on the real device,
and every tensor the loader wants exists at the right shape. The defects
were all in COMPOSITION and in reading the checkpoint's own layout.

### Defect 1 — the PLE shards span ten files

    PLE: shard 2 lives in a different file from shard 0; the row cache
    opens ONE backing file

128 shards over 10 `model-plefp8-*.safetensors`, interleaved (0,1 in the
first file; 2 in the fourth). Fixed in `5189a851`: `open_segmented` takes
`(path, offset)` per shard, `row_byte` became `row_loc` returning both.

Unreachable statically: it needs a checkpoint whose converter chose that
layout. Pinned now by `shards_may_span_several_files`.

### Defect 2 — the PLE table is FP8, not BF16

    Prefill chunk layer 1 failed: NgramRowCache: read row 249579227

`row_stride = head_dim * 2` against a `F8_E4M3` table: every row read 320
bytes wide at twice its true offset from the shard base, and off the end
of the file once the local index got large enough.

**This one WAS reachable statically and I missed it.** The loader's own log
line printed `102.4 GB BF16` for a table living inside a 126 GB checkpoint
that also holds 73 GB of other weights. At 1 byte/element it is 51.2 GB =
47.7 GiB, which is the figure this table has been described by throughout.
The arithmetic was on screen and nothing checked it.

Fixed in `884b298f`: stride comes from the shard's dtype, all shards must
agree, BF16 and F8_E4M3 accepted and anything else refused by name.

### Defect 3 — the FP8 dequant scale was never applied

The table ships ONE BF16 scalar (`ngram_embedding.weight_scale`, shape
`[1]`), not the per-row scale file `ScaleCache` was built for.
`set_constant_scale` fills the per-slot array with it so
`batched_embed_fp8` computes `fp8 * scale` per row — correct dequant, no
kernel change, no extra I/O on the fault path.

Worth naming the failure mode: without it the gather returns raw E4M3
magnitudes, the n-gram contribution is off by a constant factor, and the
output stays fluent. Nothing crashes and no log says so. The scale is now
logged at load (`per-tensor scale 0.000199`).

### Defect 4 — the serve script contradicted its own recipe

The vendored recipe has said `kv_cache_dtype: bf16` since it was added and
`the_qwen4_exp_recipe_lands_the_settings_it_documents` asserts it, but
`serve_qwen4exp_tui.sh` never passed the flag. A run started from the
script therefore got the FP8 default, and with it:

    FP8 KV cache selected but the checkpoint ships NO k_scale/v_scale
    ... silently clips BF16 into E4M3 range [-448, 448]

Fixed in `f715376d`. BF16 KV then needs `--gpu-memory-utilization 0.90`
(7.5 GB KV: 20512 blocks x 12 layers — correctly only the 12
full-attention layers, since the 36 GDN layers hold no KV).

### The model answers

Coherence, fresh server, default config: **5/7**. `Paris`; `408`;
`15:40` for "14:05 plus 95 minutes"; recalled `axolotl` across turns;
answered the negation. Decode ~15.4 tok/s, TTFT ~780 ms.

### RESOLVED — GPU prefill disagreed with the CPU reference

**Minimal reproduction, and it is small.** The CPU reference forward
(`examples/qwen4exp_forward`, no GPU kernels at all) is greedy-deterministic
on the real weights:

    [11, 42, 7, 300, 5] -> 5013 ('gt'), 26 (';'), 15 ('0'), 8 (')'), 283 (' =')

Feeding the SAME ids to the served model through `PromptInput::TokenIds`
(`/v1/completions`, `max_tokens: 1`, `temperature: 0`) returns a different top
token, stably (6/6 identical). That is:

  * 5 tokens of prompt, so no chunking,
  * `0 misses` on the PLE row cache, so no NVMe faults and no eviction,
  * prefix caching OFF and `cached_tokens: 0`,
  * ONE token generated, so no decode step at all.

Prefill's first logits row is wrong. Everything below is downstream of that.

In served chat the symptom is that the model reasons COHERENTLY ABOUT AN
INPUT NOBODY SENT: asked for the capital of France it wrote
`'The user has simply sent "FDFDFDFFDFD"'`, and other draws produced Python
source, `责任追究`, `'total_global_expert_capacity'`, Chinese lecture notes on
DNN training. Fluent text from a hidden state unrelated to the prompt.

### Ruled out, each by experiment

| suspect | evidence |
|---|---|
| prefix caching | `Prefix caching: disabled`, `cached 0`, still 0/8 coherent |
| CUDA graphs | `ATLAS_DEBUG_NO_GRAPH=1` is WORSE, not better |
| QSA indexer | `ATLAS_QSA_DISABLE=1` no better; and corruption appears at 60-token prompts, far below the 2048 budget |
| `w4a16_gemm_t` padded stride (the load-time warning) | the kernel does take and use `ldb` |
| row-cache insert bookkeeping | `fetch_into` sets map/slot_row/refbit/pinned correctly |
| FP8 dequant convention | kernel `ngram_fp8_decode(b) * row_scale[slot]` == CPU `fp8_e4m3_to_f32(b) * weight_scale`; `ngram_table.rs:217` already documented the single shared scalar |
| GDN gated norm gate | checkpoint says `output_gate_type: sigmoid`, parser reads it from `text_config`, sigmoid twins resolve |
| `norm_topk_prob` | defaults `true` (`factory.rs:31`), matching the reference, and the checkpoint omits the key |
| PLE row-cache pin lifetime | a REAL bug, fixed (`78976723`) — but the failing runs report `0 misses`, so it was not the active cause here |

### A structural fact that matters

rsafier's fixtures and measurements target the **Inferact** NVFP4 conversion
(`bench/qwen4_exp/hc_golden.py` `DEFAULT_SNAP`). This box has **RadixArk**.
They are different conversions of the same model: RadixArk ships the PLE table
as F8_E4M3 across 10 files with one shared scalar scale, where the code
expected BF16 in one file. Defects 1-3 above are all that difference. So
"coherent output on his branch" is not evidence about this checkpoint — his
branch cannot even load it without the three fixes above.

Whether the residual divergence is also RadixArk-specific is open and is the
first thing worth settling.

### The next instrument, not another hypothesis

`ATLAS_QWEN4EXP_DUMP=<dir>` already taps the `hc_mult`-wide residual highway
per sublayer and writes `L{layer}_{tag}.bin` as raw FP32 — it is how
`83830fe3` localized the previous fault to "layer 0's GDN sublayer". Its
reference side normally comes from Python, and this box has no torch.

The cheap way through: give `examples/qwen4exp_forward` the SAME taps, so the
CPU reference and the GPU write the same files and the diff is Rust vs Rust
with no torch and no fixtures. The prompt is 5 tokens, so every tap is small.
The first tap that disagrees names the sublayer.

### Earlier reading of the same defect (superseded, kept for the record)

Correct answer first, then repetition (`408 408 408`, `Paris. The capital
of France is Paris.`) or runs of `!`. `!` is token id 0, which is what
argmax returns on a degenerate or unwritten logits row.

Ruled out, each by experiment rather than by reading:

| suspect | evidence against |
|---|---|
| prefix caching | identical with `--enable-prefix-caching` off |
| CUDA graphs | `ATLAS_DEBUG_NO_GRAPH=1` made it WORSE (0/7) |
| QSA indexer | `ATLAS_QSA_DISABLE=1` made it worse; 4/7 |
| `w4a16_gemm_t` padded stride (the load-time warning) | the kernel does take and use `ldb` |
| n-gram row cache bookkeeping | suspected and WRONG — `fetch_into` records map/slot_row/refbit/pinned correctly |

Default config is the best of every variant tried, so flag-bisecting was
walking away from the answer.

Two readings of mine that were wrong and changed the diagnosis:
the repetition is WITHIN one response, not across requests; and the
"first request is already broken" probes were not first — the smoke suite
had already run against that server, so the real pattern is cumulative
within a session.

Next, and set up rather than guessed: `PromptInput::TokenIds` feeds the
CPU reference's exact greedy chain (`[11,42,7,300,5] -> 5013, 26, 15, 8,
283`) to the GPU two ways — one token per request (prefill only, decode
never runs) versus all of them in one request (decode). Prefill tracking
the reference while decode diverges localizes the fault to decode without
touching a flag.

### My own measurement errors this session

Two symptoms I reported as engine defects were the harness:

  * **`temperature: 0`.** Pinned for reproducibility. This model's card
    declares thinking `temperature=1.0 top_p=0.95 top_k=20` and MODEL.toml
    sets `use_sampling_presets_for_core = true` precisely so a plain request
    gets those. Greedy decode on a thinking MoE loops on itself, and
    `"408 408 408"` / `"The capital of France is Paris."` repeated are that,
    not a port bug. Fixed: sampling is opt-in via `--temperature`.
  * **`max_tokens: 64`** on a reasoning model, where the think block consumes
    the budget and `content` comes back empty. The needle sweep read as a
    length-dependent failure because of it.

And one clearance stated more broadly than the evidence supported: I checked
that the row cache's INSERT path records its bookkeeping and concluded the
cache was fine, without checking the pin LIFETIME — which was a real bug in
the same file.


## RESOLUTION: the PLE gathered its FP8 table with the BF16 kernel

The bisect that the Rust-vs-Rust taps made possible, on a 5-token prompt:

    L00_in            cosine 1.000000   ratio 1.0000
    L00_post_gdn      cosine 0.999989   ratio 1.0003
    L01_in            cosine 0.000062   ratio 1.25e31   <-- DIVERGES

Layer 0 correct end to end; layer 1's input at 1.25e31. Narrowing with the
GPU's own finer taps put it between `post_moe` (max 0.1985, sane) and the next
layer's input (max 2.18e30), which leaves only the PLE injection — and this
checkpoint puts PLE at model layer 1.

`PleLayer::gather_embed` called `ops::batched_embed` unconditionally: the BF16
gather, two bytes per element, no scale, against an F8_E4M3 table. Every row
came back garbage read a row and a half wide. `batched_embed_fp8` existed and
was wired on the LongCat path in `ngram_embed/embed.rs`; the PLE path never
dispatched to it.

**Why the earlier FP8 work was necessary but not sufficient.** The row stride
(`884b298f`) and the per-tensor scale were both right. The stride fix turned an
OUT-OF-BOUNDS read into an IN-BOUNDS wrong one, which is exactly why the
symptom moved from a hard `NgramRowCache: read row 249579227` to fluent-but-
wrong output. I stopped at the loader instead of following the dtype through to
the kernel choice.

Fixed in `67da29b7`, and made structurally impossible in `f3c2d068`:
`PleLayer::new` now refuses to start when a table's element width and its
gather disagree (`gather_guard::gather_matches_element_size`, pure and
CUDA-free, so the bug that needed a 126 GB checkpoint and a GB10 to reveal is
caught by a test that needs neither).

## Verified on hardware, after the fix

| leg | result |
|---|---|
| coherence | **7/7** — Paris; 408; 15:40 for "14:05 + 95 min"; exact JSON; axolotl recalled; `needle_in_context` 94 (the ~2000-token retrieval that failed every earlier run) |
| BFCL-style tools | **6/6** — native `tool_calls`, enum honoured, numerics typed, parallel calls, irrelevance still declines |
| decode, 1 stream | 14.48 tok/s, TTFT ~780 ms |
| decode, 4 streams | **28.63 tok/s aggregate, 512 tok, zero errors** (~1.98x) |
| trunk vs CPU ref | 36/36 GDN layers; `L01_in` cosine 6.2e-5 -> 0.999527 |
| lib tests | 900 (atlas-core 167, spark-storage 87, spark-model 646) |

## Two more defects the validation legs found

**Tool calling was silently off** (`the BFCL leg`). `tools_active` requires
`state.tool_call_parser.is_some()`, this target set none, so the tool
definitions never reached Jinja and the rendered prompt had no `# Tools` block.
The model said so itself — "I don't have access to ... any tools". Note the
1/6 was a trap: the single pass was the IRRELEVANCE case, where declining is
trivially correct if you were never given a tool. Fixed by
`[behavior] tool_call_parser = "qwen3_coder"`, chosen because this checkpoint's
template instructs the XML syntax
(`<tool_call><function=N><parameter=K>v</parameter></function></tool_call>`),
not hermes JSON.

**Batched decode rejected any batch off a padding boundary** (`the perf leg`).
`decode_batch` rounds the width up with `padded_batch_n` so one captured graph
serves several batch sizes, and pushes dummy states (`ple: None`) for the
padding rows — but the layer got `padded_n` while `host_token_ids` held only
the active tokens, so the PLE loop asserted `host.len() >= padded_n`. Three
live sequences padded to four failed outright: `3 host ids for 4 seqs`, HTTP
500 on every concurrent request. Padding rows now skip the injection; a row
with a LIVE PLE state and no token id still errors, which is the case worth
being loud about.

## KL logit drift, with controls

    control: gpu vs gpu, 3 identical prompts — KL 0.000000000, bitwise identical
    temp 1.0  KL(ref||gpu) 0.621798   top1 AGREE (5013)  rank_of_ref1 1
              top20_overlap 13/20
              top-20 union renormalized:  0.370020
              top-100 union renormalized: 0.326119

The GPU is bitwise deterministic, so the drift is systematic, not jitter. The
card samples `top_k=20`, so full-vocab KL charges the port for ~248k ids that
can never be emitted; restricted to the reachable distribution it is 0.37.

**Not softcapping.** The mechanism exists (`cap * tanh(logits/cap)`) but this
checkpoint sets neither `final_logit_softcapping` nor
`attn_logit_softcapping`, and the reference applies none. The gpu max of
exactly 10.000 is BF16 spacing near 10 (0.0625), not a cap.

**Not MoE routing either** — the hypothesis I expected to confirm. Per-layer
mean cosine change over all 36 GDN layers:

| transition | mean dcos | worst |
|---|---|---|
| `in -> hc_pre_mixed` | -0.003518 | -0.012295 |
| `hc_pre_mixed -> block_out` | +0.001897 | -0.012803 |
| `block_out -> post_gdn` | +0.002447 | -0.018211 |
| `post_gdn -> post_moe` | -0.001313 | -0.005422 |
| `post_moe -> next in` | -0.000758 | -0.005990 |

Net ~-0.00125/layer, which reproduces 0.9994 (L00) to 0.9559 (L35). Flipping
an expert in a top-10-of-512 selection is DISCRETE, so it would show as large
sporadic drops at `post_gdn -> post_moe`; instead that term is a steady
-0.0013, about a third of the mHC collapse term. Expert selection is largely
agreeing.

Caveat worth stating: `hc_pre_mixed` and `block_out` are BF16 taps compared
against an f32 reference, so part of the largest term is the TAP's own
quantization — BF16 relative epsilon is ~0.4%, and 0.0035 of cosine on a
12800-dim vector sits right at that scale.

So the residual is distributed NVFP4/BF16-vs-f32 accumulation with no single
mechanism responsible, which is the expected cost of quantized inference
rather than a port defect — held against coherence 7/7, tools 6/6, and an
identical argmax.

## MLPerf edge-agentic bring-up: two engine characteristics found the hard way

Running the MLCommons Edge Agentic harness (`mlcommons/endpoints`,
`examples/11_Edge_Agentic_Example`) against this target took four attempts.
Three died, and two of the causes are properties of the port worth recording
rather than operator error.

### Shrinking the batch makes it LESS stable

The reference config is single-slot (`parallel slots 1`), so the obvious move is
`--max-num-seqs 1 --max-batch-size 1`. That is what broke it:

| setting | inference reserve | KV | outcome |
|---|---|---|---|
| `--max-batch-size 4` | 6775 MB | 7.5-11.8 GB | stable (coherence 7/7, tools 6/6) |
| `--max-batch-size 1` | **3.6 GB** | 11.8 GB / 32263 blocks | `cuMemAlloc_v2 failed: status 2, requested 8388608 bytes` |

The reserve is computed from batch size. At batch 1 it drops by ~3 GB, KV sizes
itself into the slack — it fills whatever the budget leaves — and prefill then
OOMs on an 8 MB allocation. So the reserve heuristic under-provisions this
architecture's prefill (PLE scratch + GDN chunk + MoE) at batch 1 while KV
happily consumes the difference.

That is arguably an engine defect and not just a footgun, but the sizing
formula is shared across every model, so it wants a maintainer's call rather
than a unilateral change from this branch. Recorded here and on the PR.

### The PLE gather dominates fresh-prompt latency

    PLE gather: 20768 ids, 19769 hits / 999 misses, resolve 3007516us

Three seconds for one gather — ~3 ms per miss, 999 misses out of 20768 ids
(1298 tokens x 16 heads). This is the NVMe demand-paging cost of the 51.2 GB
table, and it is why a cold BFCL sample takes ~13 s while a warm short prompt
takes ~3 s. Two consequences:

  * the row cache's HIT RATE is a first-order latency term, so
    `--enable-prefix-caching` is not incidental on this model;
  * the misses are serial blocking preads under the table mutex (the loader
    already says so), so this is the obvious place to look for prefill wins —
    queue depth, not bandwidth.

### The 32K context is not viable on this config

The agentic-coding performance replay sends turns up to ~25.5K tokens. At
`--max-seq-len 32768 --gpu-memory-utilization 0.92` the server was
OOM-KILLED mid-prefill of a 25,516-token turn — the log ends mid-line with no
panic, no CUDA error and no abort, which is the kernel taking the process
rather than an allocation failing gracefully. GB10 is unified memory, so there
is no device-side failure to catch.

Before it died it recorded `TTFT=112748.5ms` on such a turn, then hit the
request deadline and truncated. So the performance leg needs either a smaller
served context or work on long-prefill throughput; it is not a config tweak.

### A harness-side incompatibility

    apply_chat_template failed for <ckpt> (UndefinedError);
    falling back to whitespace tokenization
    jinja2.exceptions.UndefinedError: 'dict object' has no attribute 'name'

The harness applies this checkpoint's chat template client-side to count
ISL/OSL/TPOT, and the template cannot render the harness's tool-definition
shape. Non-fatal — it degrades only the token metrics — but it would hit
anyone using this checkpoint as a client-side tokenizer with tools.

---

## 2026-09-15 — the nvidia pack, the OOM decomposed, the n-gram lane, and what remains

Box `gx10-e3a3` (GB10, 119.6 GB unified, driver 580.126.09). Tree
`feat/flashnext-nvidia` (PR #35): `review/pr23-fixes` + `9923d498`,
`dbe6f7d2`, `0902edb6`, `9c1ca517`, `45f5b355`. Every number below is an
observation under the fingerprint next to it; run-record ids are
`~/.atlas/runs/<bench>/run-*.json` on the box, and the queue scripts that
produced them are checked in under
`docs/porting/qwen38-nvidia-2026-09-15/jobs/` (on `review/pr21-fixes`), with
the full day's narrative in `docs/porting/QWEN38_NVIDIA_PORT_LOG_2026-09-15.md`.

### The 32K section above is superseded

"The 32K context is not viable on this config" was written against
`--gpu-memory-utilization 0.92` and a 25.5K-token turn that got the process
OOM-killed. It is now understood and no longer true as stated:

* At **util 0.88 / `--max-seq-len 32768` / `--max-prefill-tokens 16384`** the
  nvidia pack ran the full agentic-webserver harness — 50 iterations, 816
  turns, **32 multi-chunk prompts up to 25,191 tokens**, **50/50
  webserver_ok**, 0 CUDA faults (`run-1789474170004681458`). What killed the
  earlier runs was not the 25K turn per se but two host-memory terms, below.
* The perf verdict of that run is FAIL (Σwall 19,351 s > 9,000 s, 23.7 s/turn)
  — that is the serial decode floor of this pack on Atlas, not the memory.

### nvidia/Qwen3.8-Flash-Next-NVFP4 — what loads and what it scores

Deltas vs the RadixArk pack that needed code (config+index audit, then
CPU-verified loader work): the ModelOpt `quantized_layers` MIXED_PRECISION
map (`9923d498`), per-expert FP8-block-scale MTP experts (512×3
`weight + weight_scale_inv`, g=128) routed through
`dequant_fp8_blockscaled_to_bf16 → quantize_to_nvfp4`, and MIXED_PRECISION
detection that excludes `mtp.*` / `*.ple.*` / `*.visual.*` before resolving
the main-model variant (`dbe6f7d2` — nvidia's `mtp.layers.0.mlp.experts`
literally contains `.mlp.experts`, so naive substring matching misroutes the
pack). Key naming, PLE shard set and the expert scale triple are identical to
RadixArk; no loader change there.

| leg | fingerprint | result |
|---|---|---|
| loader smoke | `dbe6f7d2`, 16K, util 0.95, no spec | MTP shard header F8_E4M3 + g128 `weight_scale_inv`; first request 1.5 s |
| MTP-on probe | `9c1ca517`, `--speculative --num-drafts 1`, util 0.93 | healthy in 80 s, 995-token answer; MTP gate arbitrated K=1 |
| **ST-995** (bfcl-subset golden draw, n=995, seed 42, temp 0, thinking ON) | `9c1ca517`, util 0.95 / 16K, no spec | **83.52 overall / 82.45 normalized** (`run-1789433036569430053`) |
| ST-995 again, corrected memory profile | `9c1ca517`, util 0.88 / 32K, `--ngram-speculative` (lane never engaged — see below) | **83.52 / 82.45**, hallucination 85.98 / live 80.00 / non_live 81.35 (`run-1789501991840065669`) |
| agentic-webserver ×50 | `9c1ca517`, util 0.88 / 32K | 50/50 ok, 49/50 directions, 23.7 s/turn (`run-1789474170004681458`) |
| bs4 probe (MinHeap ×N, 256 out) | `9c1ca517`, util 0.88 | C1 17.5–18.5 → **C4 35.2–36.8 agg tok/s**, byte-identical outputs across lanes |
| KL / coherence, serial vs n-gram (5 prompts, temp 0, penalties 0, top-10 logprobs) | `9c1ca517` | coherent, tool call OK, 3/5 byte-identical; lane inert |

Neither ST-995 record gates yet: the BENCH.toml scaffold (`0902edb6`) has no
committed floors by design — these two runs are what the floors get derived
from.

### The OOM, decomposed into two terms

**Term 1 — the pledge is also the host's ceiling.** GB10 memory is unified.
`--gpu-memory-utilization u` reserves `u × 119.6 GB` for the serve, and the KV
pool absorbs whatever is left under the pledge after weights + arena +
reserve (at 0.93: 95.1 GB pre-KV, then **12.2 GB of KV for a bs1 workload
whose 12 full-attention layers need < 1 GB at 32K**). The host keeps
`(1−u) × 119.6 GB` minus co-tenants. Measured: at 0.93 `MemAvailable` fell to
**870 MB** after 25 agentic turns (`run-1789451775373797695` hardware_state);
at 0.95 the OOM killer fired (journalctl, 118/119 GB) after ~33 iterations.
Serve-side accounting from the same boot: weights 73.33 GB on disk → 84.0 GB
resident after load (SGLang reports 83.68 GiB for this pack — parity), + 2.5
GB layer construction, + 6.3 GB buffer arena, + 4.1 GB inference reserve;
Marconi pool 1.8 GB + rollback ring 0.9 GB on top. 0.88 hands ~6 GB back to
the host and is what every leg above ran at.

**Term 2 — the serve's own host memory grows with context.** Under the agentic
run the process's `RssAnon` went **2.0 → 9.8 GB** (30-s trace in the job dir):
~110 MB / 5 min while the transcript was 15–20K tokens deep, ~20 MB / 5 min on
short turns, plateauing near 10 GB. An 8-leg bisect (fresh serve per leg, 6
requests each, `RssAnon` sampled after every request) put it on one
mechanism:

| leg | shape | MB per 1K tokens |
|---|---|---:|
| 51-tok prompt, 321 out | decode-only | ~14 MB / request fixed |
| 27.4K distinct prompts, 16 out, pcache ON | prefill | 6.5 |
| the same 27.4K prompt ×6 (prefix hits) | | 2.3, **flat after the first** |
| 27.4K distinct, pcache OFF | | 5.0 |
| 27.4K distinct, `ATLAS_QSA_DEVICE_TOPK=1` | | 6.7 (≡ ON → not the QSA host top-k) |
| 27.5K prompt + 400 out (the agentic shape), ring ON | | **9.5** |
| same, `ATLAS_SSM_DECODE_RING=0` | | 4.9 |
| same, `MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576` | | 3.5 |

Growth ∝ context depth, halves without the decode-rollback ring, stops on
prefix-cache hits, shrinks under malloc tuning ⇒ **aux-state snapshots**.
`QsaIndexer::snapshot_aux` and `PleLayer::snapshot_aux` serialized their
per-sequence state (QSA raw keys: `ingested × hd × 2` bytes = 256 B/token per
indexer, 12 indexers ≈ 3 KB/token; ≈ 84 MB per 27K prompt) into a **fresh host
`Vec` on every save** — at every boundary token for the decode ring
(`snapshot_boundary_if_ssm`, 620 boundary ids, so nearly every line of code)
and at every Marconi checkpoint / finish-leaf — then dropped the previous one.
The live set was bounded (8 ring slots, 16 Marconi slots) but the multi-MB
alloc/free churn across scheduler threads fragmented glibc arenas (the 2-minute
`smaps` diff showed ~35 new 64 MB arena heaps per 2 min). `45f5b355`
snapshots into caller-owned, capacity-retaining buffers on both paths
(`snapshot_aux_into`, `collect_aux_states_into`; ring entries are taken out of
the map, refilled, re-inserted; Marconi `free()` clears blobs in place instead
of dropping them) and applies restores under the pool lock instead of
`.cloned()`. Bytes on the wire unchanged; 6 unit tests. Verified on the same
bisect legs: ring-on **7.9** vs ring-off **7.8** MB/1K tok (was 9.5 vs 4.9) —
the ring term is gone; the residual matches the retained-by-design 16-slot
Marconi footprint reaching its cap (~1.3 GB at 27K). The end-to-end agentic
rerun on this binary, mem-traced, is queued; until it lands the honest status
is "re-shaped and bounded", not "fixed".

> **SUPERSEDED 2026-09-16.** The rerun landed (job `110-flashnext-nvidia-agentic4`,
> `45f5b355`) and the prediction did not hold: RSS peaked at 9,512 MB against
> the ~3.5 GB predicted and 043's ~9.8 GB. The ring term is gone in the
> bisect, but the end-to-end footprint is NOT bounded, and the ~1.3 GB
> Marconi-cap explanation does not account for the ~8 GB present. Status is
> "one term removed, the dominant term still unidentified". See
> "2026-09-16 — … the memory fix did not bound the curve", below.

Also observed while chasing this: with `RUST_LOG=info` every prefill logs
`PLE gather: N ids, hits/misses` — on the 25K turns the gather was 55,808 ids
with ~half misses, and those misses are serial preads under the table mutex
(the loader's own note). That is the TTFT term for fresh long prompts, not a
memory term.

### The n-gram lane on this pack: engages, accepts ~nothing

> **SUPERSEDED 2026-09-16 — read the next section before using anything
> below.** The `tok_step=1.000 mean_na=0.000` lines this subsection rests on
> are rendered from `a.mtp_acct`, which the n-gram dispatch site never wrote:
> the zeros are an un-fed counter's default, not a measurement. The same
> job's `NGRAM detail` lines show `na=1` on 93 of 143 proposals (77 % warm,
> 8 % cold). And the ST-995 non-engagement is the QSA inert bound (2051),
> not the draft/table. Mechanism and fix: "2026-09-16 — the n-gram lane was
> mis-read …", below.

`--ngram-speculative` (the llama.cpp-style dynamic table + multi-token chains
+ SSD persistence on `review/pr23-fixes`) boots (`N-gram speculative decoding:
ENABLED (K=2/3/4 verify, CPU proposer)`), but:

* KL harness, 5 prompts × 256 tokens: every request `tok_step=1.000
  mean_na=0.000`, ngram wall = serial wall (14.5 vs 14.4 s); 3/5 outputs
  byte-identical to serial.
* Positive control — the PR's own shape (600-word essay, 400 out, temp 0,
  `RUST_LOG=debug`), 2 requests per leg: the proposer **runs** (`NGRAM detail:
  drafts=[…] … na=0` lines: 26 on request 0, 145 on request 1 — the table
  warms) but acceptance is ~0; serial 15.6/16.8 vs ngram 16.0/18.7 tok/s incl.
  TTFT, `tok_step 1.000` both.
* ST-995 under the lane: **0 `NGRAM detail` lines in 995 requests**; score
  identical to the no-spec run (that is expected when nothing engages, not a
  parity proof).

PR #23 measured +36 % (18.7 → 25.5 tok/s, ~93 % acceptance) on the RadixArk
pack. Under this fingerprint on the nvidia pack that does not reproduce. Not
yet determined why; the two things to check first are the draft-vs-target
position alignment for this checkpoint's tokenization (the class of the
off-by-one already fixed once on this lane) and what the dynamic table
actually learns on these prompts. Also observed: the K=2 verify path with
rejected drafts is not output-neutral (2/5 prompts diverge at exact BF16
ties) — same tie-contract class as the DFlash lane on the 27B.

### The external agent report (spark box, ~16 h whiteboard build) vs our records

Their two pinned items are already handled on this line: the 32K request cap
(`qsa.rs` clamps indexer capacity to `--max-seq-len`; `ATLAS_QSA_MAX_TOKENS`
can only raise it) and the prefill-chunk / PLE-scratch coupling (`9c1ca517`
loops the PLE pipeline over `ATLAS_PLE_CHUNK` spans regardless of
`--max-prefill-tokens`). Their unpinned misaligned-address context loss (4× in
35 min of agent traffic, a D2H at layer 0 of a continuation chunk right after
an intermediate SSM checkpoint save) did **not** reproduce in our 819-prefill /
32-continuation-chunk / 1,562-checkpoint agentic run — but our chunk size was
16K (theirs 8K) and our depth 25K (theirs 42.8K), so that is two
observations, not an A/B. The eager + `ATLAS_DEBUG_SYNC_KERNELS=1` method that
named the DFlash fault on the 27B (a Rust backtrace of the launch site) is the
tool to point at it.

### What remains — ordered by what changes the user-visible number

> **Item 1 re-ordered 2026-09-16.** "Diagnose alignment/table first" would
> have spent hardware on a non-cause: the lane accepts (93/143 proposals in
> its own control) and never ran on ST-995 because the QSA inert bound (2051)
> declines it silently on every long prompt. The lane cannot reach agentic or
> long-form decode at all. The lever for the serial floor is the verify
> path's active-QSA support (or MTP), not the n-gram table. See the
> 2026-09-16 section.

1. **Make speculation accept on this pack.** Agentic and long-form decode sit
   at the serial floor (15–18 tok/s, 23.7 s/turn). The n-gram lane is the only
   one that fits at these memory budgets (the BF16 MTP block does not fit at
   0.85; the FP8-expert MTP arm fits but arbitrated to K=1 in the probe) and it
   accepts ~0 here. Diagnose alignment/table first (10-minute leg with
   position-aligned draft-vs-emitted dump), then re-measure the essay control
   and ST-995 with the lane actually firing.
2. **Finish the memory story.** (a) The queued agentic rerun on `45f5b355`
   with the RSS trace — prediction ≲ 3.5 GB flat vs 9.8 GB; (b) move the
   Marconi/ring aux blobs to device (or a preallocated pinned arena) so the
   host footprint stops scaling with slots × depth *and* the per-boundary-token
   `synchronize` + D2H hitch (review note 5) goes away; (c) need-driven KV
   sizing for bs1 profiles so the pledge remainder is not swallowed by a KV
   pool 12× larger than the workload; (d) recipe: util 0.88 / 32K, and
   `ATLAS_SSM_DECODE_RING=0` for agentic serving (the rollback watchdog is the
   feature that false-positives on code).
3. **Gate the nvidia subject.** Derive BENCH.toml floors from the two ST-995
   records + the agentic record, set `status = "measured"`, so
   `--pull-request-gate` can serve it and the recipe
   (`qwen3.8-flash-next-nvfp4-nvidia.yaml`, sparkrun-recipes PR #8) carries the
   profile above instead of the RadixArk one.
4. **Long-prefill TTFT** (unchanged from the gap analysis): `qsa_score_rows`
   is quadratic at 294 GFLOP/s and, with the PLE gather's serial miss preads,
   is why a fresh 25K turn costs 100+ s of TTFT. Queue-depth on the PLE reads
   is the cheap half; the indexer kernel is the real one.
5. **Decode graph capture**: `ATLAS_QSA_DEVICE_TOPK=1` default + the PLE event
   remove the two vetoes; capture has not been re-attempted on this model.
   This is where PR #27's runtime becomes relevant to Flash-Next — not before.
6. **Correctness contracts** still open from the #23 review: kernel
   lane-vs-index argmax merge, residual host last-wins paths, boundary-dense
   snapshot eviction, and the pre-/post-penalty `top_logprobs` inconsistency
   found on the DFlash lane (check the n-gram verify path reports the same
   quantity as serial).
7. **W4A4 activation path** for the routed experts (the pack ships static
   `input_scale`; the loader lands it, the parked `_fp4` MoE kernels do not
   consume it yet) — a throughput lever once 1–2 are done.

## 2026-09-16 — the n-gram lane was mis-read, the memory fix did not bound the curve, and the nvidia subject now gates

Box `gx10-e3a3` (GB10, driver 580.126.09). Tree `feat/flashnext-nvidia`
(PR #35) — the 2026-09-15 section's tree plus `8e4d4691`, `f5f2dafec`,
`f990a691`, `4a48d80f`. Two of those four are corrections to claims in that
section; they are marked as supersede notes below, with the mechanism named,
per the ledger rule.

### SUPERSEDES item 1's premise: the lane accepts; the counter was never fed

The 2026-09-15 section reads "engages, accepts ~nothing" and cites "every
request `tok_step=1.000 mean_na=0.000`" as the evidence. **That line is not a
measurement of the n-gram lane.** It renders `a.mtp_acct`, and the n-gram
dispatch site in `scheduler/mod.rs` never wrote it — the serial arm calls
`record_serial`, the MTP/DFlash/self arms call `record_verify_emitted`, and
the n-gram arm called neither. An un-fed counter reports its default, which
is exactly `serial=0.00 mtp=0.00 p1=0.000 mean_na=0.000 tok_step=1.000`.

The proof is inside the same job. In `074-flashnext-ngram-control`:

* every ngram `Done:` line shows `serial=0.00 mtp=0.00` (`054` shows the
  same), i.e. the accounting recorded *nothing at all* — not a low accept
  rate, zero steps;
* the lane's own `NGRAM detail: drafts=[…] v=[…] na=…` lines, from the same
  serve, show `na=1` on **93 of 143** proposals, and the INFO-level
  `NGRAM K2 ACCEPT … na=1/1` line fires.

Re-read by request: request 0 (cold table) accepted 2 of 25 proposals (8 %);
request 1 (same prompt, warm table) accepted **91 of 118 (77 %)**. Wall for
request 1: ngram 21.41 s vs serial 23.86 s (+11 %); request 0: 25.06 vs
25.67 (+2 %). So the honest summary is "accepts on repetition, ~nothing
cold", which is ordinary n-gram behaviour — not the defect the section
implied, and not PR #23's +36 % either.

Fixed in `f5f2dafec`: the dispatch site books its step
(`record_verify_emitted`), which also repairs
`usage.completion_tokens_details.accepted_prediction_tokens` for this lane.
Deliberately NOT `mtp_accept_debug::record` — that histogram steers
`adaptive_rung`'s MTP dispatch width, so feeding n-gram verifies into it
would steer one lane's regime from the other's statistics. A source-scan
test with a negative control pins the booking.

### The lane never ran on ST-995, and the reason is not the table

The same section attributes the 995-request non-engagement to the draft/table
("the two things to check first are the draft-vs-target position alignment …
and what the dynamic table actually learns"). The dispatch gate says
something simpler and much more consequential:

```
use_ngram_speculative && active.len() == 1 && spec_slots_covered
  && grammar_state.is_none() && temperature == 0.0
  && verify_ctx_limit.is_none_or(|lim| seq.seq_len + 4 <= lim)
```

`verify_context_limit()` is the QSA inert bound — **2051** on this model
(`qsa.rs`: `budget + ratio - 1`) — because the verify paths refuse an ACTIVE
selection past it. So the lane is confined to the first ~2047 tokens of
context, and it declines **silently**: unlike MTP, it has no decline log.

Every BFCL request is admitted well past the bound. `082-flashnext-nvidia-bfcl2`
logs `mtp declined: qsa_active seq_len=2264 num_drafts=1 lim=2051` — the MTP
latch, which runs regardless of lane, and is the only reason this was
findable at all. With `seq_len` above `lim` at admission the n-gram branch can
never be taken, so "0 NGRAM detail lines in 995 requests" is a statement
about the gate, not about the table or an off-by-one.

★ This re-orders item 1. Agentic and long-form decode at 25K context is
**unreachable** by this lane until the verify path can serve an active QSA
selection (or the bound moves); no amount of table/alignment work changes
that. The levers for the serial floor are MTP (FP8-expert arm) and that
verify-path work — not the n-gram table.

A debug `ngram declined:` line now names the reason (same commit).

### SUPERSEDES item 2a's prediction: the aux-buffer fix did not bound RSS end-to-end

The 2026-09-15 section predicted, for the queued agentic rerun on `45f5b355`:
"prediction ≲ 3.5 GB flat vs 9.8 GB". That rerun is job
**110-flashnext-nvidia-agentic4** and it **refutes the prediction**:

| reading | 043 (pre-fix) | 110 (`45f5b355`) |
|---|---:|---:|
| serve `RssAnon`/RSS peak | ~9.8 GB | **9,512 MB** |
| `MemAvailable` minimum | 403 MB | **570 MB** |
| webserver_ok | 50/50 | 50/50 |
| followed_directions | 49/50 | 49/50 |
| Σwall | 19,351 s | 19,150 s |

Same serve profile (util 0.88 / 32K / 16K prefill / prefix caching), same
50-iteration harness, mem-traced at 30 s (`mem-trace.tsv`). The curve is
unchanged within noise. So the bisect result stands as stated — the ring term
is gone (7.9 vs 7.8 MB/1K tok) — but the **end-to-end footprint is not
bounded**, and the "residual matches the retained-by-design 16-slot Marconi
footprint (~1.3 GB at 27K)" explanation does not account for the ~8 GB that
is actually there. The honest status is now "one term removed; the dominant
term is still unidentified", which is a weaker claim than the section's
"re-shaped and bounded".

`45f5b355` is still a correct change — it removes real alloc/free churn — but
it is not the fix for the OOM term 2, and the next step is attribution on
this shape (which mapping/arena the 9.5 GB sits in), not item 2b's
device-side move. Item 2c (need-driven KV) is unaffected: `MemAvailable`
falling to 570 MB is the pledge term, and it is still the reason a bs1
workload needs `--gpu-memory-utilization 0.88`.

### Item 3 done: the nvidia subject gates

`4a48d80f` flips both `kernels/gb10/qwen3.8-flash-next/BENCH.toml` entries to
`status = "measured"`, so `--pull-request-gate` can serve the pack instead of
refusing it.

* **bfcl-subset** — `overall_accuracy` 83.12, `normalized_single_turn_score`
  82.05 (measured 83.52 / 82.45 less the dense sibling's documented ±0.4),
  `samples` pinned at exactly 995. Two records agree to the hundredth
  (`run-1789433036569430053` at util 0.95/16K/no-spec, and
  `run-1789501991840065669` at util 0.88/32K) — and the note states plainly
  that this is reproducibility under two serve profiles, **not** a
  repeat-spread, so the margin is inherited rather than measured here. It
  also records that the second run was SERIAL (the lane declined), so
  "identical score with the lane inert" is not read as a parity proof.
* **agentic-webserver** — `webserver_ok` ≥ 50, `followed_directions` ≥ 49,
  `iterations` pinned at exactly 50, from `run-1789474170004681458`. **No
  wall bound**, deliberately: one record is one tier, and a ceiling derived
  from it is the "ceiling written before calibration" the 3.6 dense entry
  warns about. The measured Σwall is this pack's serial decode floor on
  Atlas, not a defect.

### What job 111 actually did (so its record is not read as a measurement)

`111-flashnext-ple-warm` ran the right leg against a prompt of **45,306
tokens** with `--max-seq-len 32768`, so both requests were rejected
("Prompt too long") and `WARM.tsv` came back as a bare header. The
generator's comment estimated ~24K tokens; the real ratio is ~64 tokens per
paragraph, so 700 paragraphs is 45K. **Nothing about PLE prefetch was
measured in 111.** Replaced by `flashnext-ple-warm2`, which uses 370
paragraphs and fails loudly if `usage.prompt_tokens > 30,000`.

### CI blockers found on this branch (all pre-existing, all now fixed)

`cargo test --workspace` and the file-size job are both red on PR #35 for
reasons unrelated to the nvidia pack:

1. **Two stale gate tests** (`f990a691`). `bench_selfstart_tests` pinned
   `qwen3.8-27b-nvfp4-unsloth` and `bench_variants_tests` pinned
   `unsloth/Qwen3.8-27B-NVFP4`; the 27B entries were relabelled to
   `nvidia/Qwen3.8-27B-NVFP4` / `-agentic` in `fbc3a7259`+`ff3e53c3f` without
   updating them.
2. **The non-CUDA (metal) build** (`8e4d4691`). `2e74cfe2e` left four
   `deny(warnings)` errors in the configuration the local gate's metal row
   checks — the one row that proves the port compiles where its code is
   mostly absent. CI cannot see it: CI checks with `cuda` against stub libs,
   where all four sites are live. The three causes are all "code that only
   means something where `NgramTable::Cached` exists".
3. **The 500-LoC cap** (`.github/workflows/file-size-cap.yml`) is over in
   three files this branch grew: `spec_step.rs` 482→561, `qwen4_exp.rs`
   500→501, `nvfp4_detect.rs` 492→601. Not yet split — no allow-list entry
   was added, since that is the workaround the cap exists to prevent.

### Queued / in flight

* `116-flashnext-ngram-instr` — reruns 074's shape on the accounting fix;
  asserts each ngram `Done:` line reports `mtp>0`, that its `tok_step` agrees
  with the `NGRAM detail` na counts, and that no decline fires on a ~48-token
  prompt.
* `117-flashnext-ngram-longctx` — the one-variable test of the bound above:
  identical prompt text at ~1.2K (under 2051) and ~6.0K (over). Short must
  propose and not decline; long must propose nothing and log a decline.
* `118-flashnext-ple-warm2` — 111's leg with a prompt that fits.
