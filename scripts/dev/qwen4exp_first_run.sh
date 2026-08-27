#!/usr/bin/env bash
# The qwen4_exp port's first run on a GB10, in one command.
#
# Ordered CHEAPEST FIRST, and each step closes a different class of risk. Steps
# 1-3 need no checkpoint at all, which matters: the 126 GiB download is the
# slowest part of a fresh box, and three quarters of what can be wrong is
# provable before it finishes.
#
#   1  the kernels COMPILE            nvcc, ~10 min, no checkpoint
#   2  kernels vs the CPU oracle      GPU, seconds, no checkpoint, no Python
#   3  the five block microtests      GPU, ~2 min, no checkpoint
#   4  the checkpoint is described    reads headers only, ~3 s   [needs --ckpt]
#   5  the CPU forward still answers  slow but decisive           [needs --ckpt]
#   6  checkpoint-backed parity       [needs --ckpt AND generated fixtures]
#
# It does NOT start a server: that is a long-lived process and wants a human
# watching it. Step 7 prints the command.
#
# Usage:
#   ./scripts/dev/qwen4exp_first_run.sh                       # steps 1-3
#   ./scripts/dev/qwen4exp_first_run.sh --ckpt /path/to/snap  # steps 1-5
#   ./scripts/dev/qwen4exp_first_run.sh --ckpt <p> --from 4   # resume
#
# Everything lands in ./qwen4exp-first-run/<step>.log, and the summary at the
# end is what to paste into the PR.

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

CKPT=""
FROM=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="${2:?--ckpt needs a path}"; shift 2 ;;
    --from) FROM="${2:?--from needs a step number}"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOGDIR="qwen4exp-first-run"
mkdir -p "$LOGDIR"
CUDA=(--no-default-features --features cuda)
declare -a RESULTS=()
FAILED=0
ATTEMPTED=0

note() { printf '\n\033[1m── %s\033[0m\n' "$*"; }
record() { RESULTS+=("$1"); }

# A step that FAILS stops the run: every later step assumes the earlier ones.
# A step that SKIPS does not — a missing checkpoint is not a defect.
step() {
  local n="$1" name="$2"; shift 2
  if (( n < FROM )); then
    record "$(printf '%-2s %-34s SKIP (--from %s)' "$n" "$name" "$FROM")"
    return 0
  fi
  note "step $n — $name"
  ATTEMPTED=$((ATTEMPTED + 1))
  local log="$LOGDIR/$n.log"
  if "$@" > "$log" 2>&1; then
    record "$(printf '%-2s %-34s PASS' "$n" "$name")"
    tail -3 "$log" | sed 's/^/    /'
    return 0
  fi
  record "$(printf '%-2s %-34s FAIL  -> %s' "$n" "$name" "$log")"
  echo "  FAILED. last 25 lines:" >&2
  tail -25 "$log" | sed 's/^/    /' >&2
  FAILED=1
  return 1
}

skip() { record "$(printf '%-2s %-34s SKIP (%s)' "$1" "$2" "$3")"; }

note "environment"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>&1 | sed 's/^/  gpu: /'
  # A GB10 is coherent unified memory; `memory.total [N/A]` and
  # `Addressing Mode: ATS` are the tell, and they are why "offload to host"
  # frees nothing on this box.
  nvidia-smi -q 2>/dev/null | grep -i "addressing mode" | sed 's/^ */  /'
else
  echo "  no nvidia-smi — steps 2, 3, 6 need a GPU and will fail" >&2
fi
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -1 | sed 's/^/  nvcc: /' \
  || echo "  no nvcc — step 1 will fail" >&2
echo "  free: $(df -h . | awk 'NR==2{print $4}') on $(pwd)"

# ── 1. Do the kernels compile? Nothing else can be trusted until they do, and
#       CI's nvcc job covers the same ground on a different host.
step 1 "kernels compile (nvcc)" \
  cargo build --release -p spark-model "${CUDA[@]}" || exit 1

# ── 2. The serving kernels against the in-process CPU oracle. No checkpoint,
#       no Python, no fixture files — this is the cheapest real evidence the
#       mHC highway, the PLE tower and the QSA indexer are numerically right.
step 2 "kernels vs CPU oracle" \
  cargo test --release -p spark-model "${CUDA[@]}" \
    qwen4exp_oracle -- --ignored --nocapture || exit 1

# ── 3. The five block microtests, each with a control that must fail.
#       Independently reproduced on a second GB10 on #13.
step 3 "five block microtests" \
  cargo run --release -p spark-model "${CUDA[@]}" \
    --example qwen4exp_grouped_norm_microtest || exit 1

if [[ -z "$CKPT" ]]; then
  for n in 4 5 6; do
    case $n in
      4) skip 4 "checkpoint preflight" "no --ckpt" ;;
      5) skip 5 "CPU reference forward" "no --ckpt" ;;
      6) skip 6 "checkpoint-backed parity" "no --ckpt" ;;
    esac
  done
else
  [[ -f "$CKPT/config.json" ]] || { echo "no config.json under $CKPT" >&2; exit 2; }

  # ── 4. Every tensor the loader will ask for, by name and shape, against the
  #       checkpoint's own headers. Reads headers only — cheap enough to run
  #       at load time, which is why it exists. Want 0 / 0 / 0.
  step 4 "checkpoint preflight" \
    cargo run --release -p atlas-core --example qwen4exp_preflight -- "$CKPT" || exit 1

  # ── 5. The CPU reference forward on the real weights. Slow (~33 s/token) and
  #       decisive: it answered "Paris." on #13, so a regression here is a
  #       regression in the port and not in the kernels.
  step 5 "CPU reference forward" \
    cargo run --release -p atlas-core --example qwen4exp_forward -- "$CKPT" --generate 4 || exit 1

  # ── 6. The checkpoint-backed parity goldens — STRONGER evidence than step 2,
  #       because their fixtures come from the real reference module rather than
  #       our transcription of it. They need generating first, and
  #       qsa_golden.npz is gitignored.
  if [[ -n "${ATLAS_HC_TEST_DATA:-}" ]]; then
    step 6 "checkpoint-backed parity" \
      cargo test --release -p spark-model "${CUDA[@]}" hc_lowrank -- --ignored --nocapture
  else
    skip 6 "checkpoint-backed parity" "set ATLAS_HC_TEST_DATA; see bench/qwen4_exp/hc_golden.py"
  fi
fi

note "summary"
printf '  %s\n' "${RESULTS[@]}"

note "step 7 — serve (not run here: it is a long-lived process)"
cat <<'EOS'
  ./serve_qwen4exp_tui.sh                     # raises ATLAS_PLE_MAX_TOKENS itself above 8K

  If it misbehaves, bisect with the switches rather than guessing — in order,
  each one removing a mechanism:
    ATLAS_QSA_DISABLE=1            detach the indexer
    ATLAS_QWEN4EXP_NO_HC_GEMM=1    fused collapse instead of the GEMM path
    ATLAS_DEBUG_NO_GRAPH=1         no CUDA graphs
    ATLAS_QWEN4EXP_NO_PLE=1        no PLE — output is WRONG by construction,
                                   this one only isolates the mHC spine
  The full table is in docs/porting/QWEN4_EXP_PORT_LOG.md.
EOS

if (( FAILED )); then
  echo
  echo "at least one step FAILED — logs in $LOGDIR/" >&2
  exit 1
fi
if (( ATTEMPTED == 0 )); then
  # Distinguished from success on purpose: "everything skipped" is not a green
  # run, and reading it as one is how a plan gets marked done without evidence.
  note "NOTHING ATTEMPTED — every step was skipped. Nothing is verified."
  exit 0
fi
note "$ATTEMPTED step(s) attempted, all passed — paste the summary into the PR"
