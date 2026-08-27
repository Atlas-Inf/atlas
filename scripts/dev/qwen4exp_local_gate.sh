#!/usr/bin/env bash
# Local verification gate for the qwen4_exp port — no GPU, no nvcc, no CUDA.
#
# WHY THIS EXISTS. The GB10 is not always reachable (it lives on a mesh VPN),
# and this port touches 99 files across five crates. Without a gate, "it
# compiles on my machine" means the metal build only, which is the one
# configuration the qwen4_exp code is almost entirely absent from — nearly all
# of it sits behind `#[cfg(feature = "cuda")]`.
#
# THE TWO ENV VARS ARE THE WHOLE TRICK:
#
#   ATLAS_SKIP_BUILD=1        atlas-kernels' build.rs writes a type-checkable
#                             stub instead of invoking nvcc, so the CUDA Rust
#                             path compiles with no toolkit present. Nothing is
#                             compiled to PTX, so this gate says NOTHING about
#                             the .cu files themselves.
#   CUDARC_CUDA_VERSION=13000 stops the vendored cudarc's build.rs shelling out
#                             to `nvcc --version` to discover a version.
#
# WHAT IT CANNOT DO, stated so nobody reads a PASS as more than it is:
#   * no kernel is compiled, so a .cu syntax or arity error is invisible here;
#   * no kernel is RUN, so every parity test is `#[ignore]`d and unexecuted;
#   * anything that links `-lcuda` cannot even build a test binary on macOS,
#     which is why the test rows below run under `metal` — the tests
#     themselves are backend-agnostic, the linking is not.
#
# On Linux with a real toolchain, drop both env vars and the `metal` features
# and this is just the ordinary gate.

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

# Prefer an explicit toolchain if the caller has one; otherwise trust PATH.
if [[ -n "${ATLAS_CARGO:-}" ]]; then
  CARGO="$ATLAS_CARGO"
elif command -v cargo >/dev/null 2>&1; then
  CARGO=cargo
else
  TC="$HOME/.rustup/toolchains/1.93.1-aarch64-apple-darwin/bin/cargo"
  [[ -x "$TC" ]] || { echo "no cargo on PATH and no pinned toolchain at $TC" >&2; exit 127; }
  CARGO="$TC"
fi

export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13000}"
export ATLAS_SKIP_BUILD="${ATLAS_SKIP_BUILD:-1}"

FAILED=0
LOGDIR="${TMPDIR:-/tmp}/qwen4exp-gate"
mkdir -p "$LOGDIR"

run() {
  local label="$1"; shift
  printf '%-46s' "$label"
  if "$@" > "$LOGDIR/${label// /_}.log" 2>&1; then
    # Surface the test count when there is one, so a silently-empty run is
    # visible rather than reading as a pass.
    local tally
    tally=$(grep -hoE '[0-9]+ passed' "$LOGDIR/${label// /_}.log" | tail -1)
    echo "PASS ${tally}"
  else
    echo "FAIL  ($(grep -cE '^error' "$LOGDIR/${label// /_}.log") errors) -> $LOGDIR/${label// /_}.log"
    FAILED=$((FAILED + 1))
  fi
}

CUDA=(--no-default-features --features cuda)
METAL=(--no-default-features --features metal)

echo "── compile ──────────────────────────────────────────────────"
run "libs+bins, cuda"        "$CARGO" check --workspace --lib --bins "${CUDA[@]}"
run "touched crates, all targets, cuda" \
    "$CARGO" check -p atlas-core -p spark-model -p spark-runtime -p spark-server \
                   --all-targets "${CUDA[@]}"
run "non-cuda (metal) build" "$CARGO" check -p atlas-core -p spark-model "${METAL[@]}"

echo "── tests that actually run here ─────────────────────────────"
# atlas-core links no CUDA libraries, so its tests run under the cuda feature —
# which matters, because the qwen4_exp parser, manifest and n-gram id core all
# live here.
run "atlas-core lib tests"   "$CARGO" test -q -p atlas-core --lib "${CUDA[@]}"
# The rest link `-lcuda` under the cuda feature and cannot produce a test
# binary on macOS. The tests are backend-agnostic; only the linking is not.
run "spark-server bin tests" "$CARGO" test -q -p spark-server --bins "${METAL[@]}"
# spark-model's own unit tests — the PLE id core, the layer plumbing, the
# weight-map helpers. Its GPU parity modules are gated on the cuda feature
# precisely so this row can exist.
run "spark-model lib tests"  "$CARGO" test -q -p spark-model --lib "${METAL[@]}"
run "spark-storage lib tests" "$CARGO" test -q -p spark-storage --lib "${METAL[@]}"
# `metal_backend::tests::parity_*` needs Metal shaders that ATLAS_SKIP_BUILD did
# not build. Skipped by name rather than by ignoring the whole crate.
run "spark-runtime lib tests" "$CARGO" test -q -p spark-runtime --lib "${METAL[@]}" \
    -- --skip metal_backend

echo "── repo CI gates that need no toolchain ─────────────────────"
# These three run in CI as separate jobs, and none of them needs cargo — so
# there is no excuse for finding out from a red PR. The LoC cap in particular
# is easy to breach by adding a comment.
run "SPDX headers"           python3 scripts/check_spdx.py
run "kernel shadow structure" python3 scripts/check_kernel_shadows.py
# Kernel names are STRING literals, so `cargo check` cannot see a typo — and a
# name that resolves in another model's shadow is worse than one that does not
# resolve at all. Milliseconds, and it rules out a whole class of startup
# failure a laptop otherwise cannot touch.
run "qwen4_exp kernel names"  python3 scripts/dev/check_qwen4exp_kernel_names.py
loc_cap() {
  local over=0
  while IFS= read -r f; do
    local n; n=$(wc -l < "$f")
    if (( n > 500 )) && ! grep -q "\"$f\"" .github/workflows/file-size-cap.yml; then
      echo "OVER $n $f"; over=$((over + 1))
    fi
  done < <(find crates -name '*.rs' -not -name '*.bak' -not -path '*/target/*')
  (( over == 0 ))
}
run "500-LoC cap per .rs"    loc_cap

echo "── lint ─────────────────────────────────────────────────────"
run "fmt"                    "$CARGO" fmt --all -- --check
run "clippy, cuda"           "$CARGO" clippy -p atlas-core -p spark-model -p spark-runtime \
                                      -p spark-server --all-targets "${CUDA[@]}"

echo
if (( FAILED == 0 )); then
  echo "all gates PASS — logs in $LOGDIR"
  echo
  echo "Still unverified by this gate, and only a GB10 can close it:"
  echo "  * the .cu files compile (nvcc was never invoked)"
  echo "  * cargo test -p spark-model --release qwen4exp_oracle -- --ignored"
  echo "  * cargo test -p spark-model --release hc_lowrank      -- --ignored"
  echo "  * cargo run --release -p spark-model --example qwen4exp_grouped_norm_microtest"
  echo "  * cargo run --release -p atlas-core  --example qwen4exp_preflight -- <ckpt>"
  echo "  * ./serve_qwen4exp_tui.sh"
else
  echo "$FAILED gate(s) FAILED — logs in $LOGDIR"
fi
exit $(( FAILED > 0 ))
