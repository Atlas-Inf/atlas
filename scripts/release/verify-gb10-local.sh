#!/usr/bin/env bash
# verify-gb10-local.sh — one-command Atlas release verification on a single
# GB10 (the reiner jobs-513/521 pipeline, checked in).
#
#   1. detached worktree at <sha7> (never touches the caller's checkout),
#   2. docker build -f docker/gb10/Dockerfile -t azeezish/atlas-gb10:<sha7>,
#   3. scoped serve matrix: tests/run_all_models.py --roster,
#   4. coherence probe on the Flash-Next recipe serve (--disable-thinking),
#   5. tests/gate_results.py,
#   6. on PASS: docker save | zstd + PRINTED (not run) publish commands.
#
# Usage:
#   scripts/release/verify-gb10-local.sh <sha7> [--skip-build]
#       [--worktree PATH] [--roster tests/rosters/gb10-cached.json]
#       [--out DIR]
#
# Env:
#   ATLAS_HF_CACHE_HEAD — HF checkpoint cache mounted into containers
#                         (default ~/.cache/huggingface).
#   COHERENCE_PORT      — port for the coherence serve (default 8083).
#
# Exit: 0 on PASS, 1 on FAIL, 2 on environment/refusal errors.
# Rootless-docker and the `sudo docker` quirk are handled automatically;
# see docs/releases/README.md §"Verifying on a single GB10".
set -euo pipefail

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ── args ────────────────────────────────────────────────────────────
SHA=""
WT=""
ROSTER="tests/rosters/gb10-cached.json"
OUT=""
SKIP_BUILD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1; shift ;;
        --worktree) WT="$2"; shift 2 ;;
        --roster) ROSTER="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) if [ -z "$SHA" ]; then SHA="$1"; shift; else
               log "FATAL: unknown arg $1"; exit 2; fi ;;
    esac
done
[ -n "$SHA" ] || { log "FATAL: <sha7> required"; sed -n '12,21p' "$0"; exit 2; }

IMAGE="azeezish/atlas-gb10:${SHA}"
HF_CACHE="${ATLAS_HF_CACHE_HEAD:-$HOME/.cache/huggingface}"
PORT="${COHERENCE_PORT:-8083}"
: "${OUT:=$PWD/release-verify-${SHA}}"
mkdir -p "$OUT/matrix" "$OUT/bin"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# ── worktree: default = fresh detached under $TMPDIR ────────────────
if [ -z "$WT" ]; then
    WT="$(mktemp -d "${TMPDIR:-/tmp}/atlas-verify-${SHA}-XXXX")"
    git -C "$REPO_ROOT" worktree add --detach "$WT" "$SHA" || {
        log "FATAL: worktree add at $SHA failed"; exit 2; }
else
    [ -f "$WT/.git" ] || { log "FATAL: $WT is not a worktree"; exit 2; }
    if [ -n "$(git -C "$WT" status --porcelain)" ]; then
        log "FATAL: worktree $WT is not clean — refusing"; exit 2
    fi
fi
ACTUAL="$(git -C "$WT" rev-parse --short=7 HEAD)"
[ "$ACTUAL" = "$SHA" ] || {
    log "FATAL: worktree is at $ACTUAL, expected $SHA — refusing"; exit 2; }
cd "$WT"
echo "${SHA} $(date -u +%Y-%m-%dT%H:%M:%SZ) release-verify-local" \
    > "$OUT/FINGERPRINT.txt"
log "worktree $WT @ $SHA; out=$OUT"

# ── docker access ───────────────────────────────────────────────────
docker ps >/dev/null 2>&1 || { log "FATAL: docker not reachable"; exit 2; }
# run_all_models.py issues `sudo docker` unconditionally — on a rootless
# host real sudo fails, so put a pass-through shim on PATH for that stage.
# Only install the shim when `sudo -n docker` actually fails.
if ! sudo -n docker ps >/dev/null 2>&1; then
    printf '#!/bin/sh\nexec "$@"\n' > "$OUT/bin/sudo"
    chmod +x "$OUT/bin/sudo"
    log "sudo docker unusable -> pass-through shim in $OUT/bin"
fi

# Rootless docker: `--network host` containers join rootlesskit's netns,
# not the host's — run clients inside it via nsenter. Otherwise direct.
DPID="$(pgrep -u "$(id -u)" -x dockerd | head -1 || true)"
if [ -n "$DPID" ]; then
    NS=(nsenter -U --preserve-credentials -n -t "$DPID")
    "${NS[@]}" true || { log "FATAL: cannot enter rootless docker netns"; exit 2; }
    log "rootless dockerd pid=$DPID — clients run via nsenter"
else
    NS=()
fi

# ── stage 1: build ──────────────────────────────────────────────────
if [ "$SKIP_BUILD" = 0 ]; then
    log "=== build $IMAGE ==="
    docker build -f docker/gb10/Dockerfile \
        --build-arg "ATLAS_GIT_SHA=${SHA}" -t "$IMAGE" "$WT" || {
            log "FAIL: docker build"; exit 1; }
else
    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
        log "FATAL: --skip-build but $IMAGE not present"; exit 2; }
    log "=== build skipped: reusing $IMAGE ==="
fi

# ── stage 2: scoped serve matrix ────────────────────────────────────
log "=== serve matrix: run_all_models.py --roster $ROSTER ==="
rm -rf tests/all_models_results
MATRIX=0
"${NS[@]}" env PATH="$OUT/bin:$PATH" \
    ATLAS_IMAGE="$IMAGE" \
    ATLAS_HEAD_IP=127.0.0.1 \
    ATLAS_HF_CACHE_HEAD="$HF_CACHE" \
    python3 tests/run_all_models.py --roster "$ROSTER" || MATRIX=1
cp -r tests/all_models_results "$OUT/matrix/" 2>/dev/null || true

# ── stage 2b: coherence on the Flash-Next recipe, reasoning OFF ─────
# The probe's Edge Cases read message.content after max_tokens 10-20;
# a thinking serve spends that budget inside <think> (job 513's 3
# failures were exactly this artifact). Serve reasoning-off — the
# MLPerf leg config; every other flag mirrors the shipped recipe.
log "=== coherence probe on flash-next (--disable-thinking) ==="
docker rm -f atlas-verify-coherence >/dev/null 2>&1 || true
docker run -d --name atlas-verify-coherence --gpus all --ipc=host \
    --network host -v "$HF_CACHE":/root/.cache/huggingface \
    "$IMAGE" serve nvidia/Qwen3.8-Flash-Next-NVFP4 \
    --model-name qwen4exp --bind 0.0.0.0 --port "$PORT" --no-tui \
    --kernel-target qwen3.8-flash-next \
    --max-seq-len 32768 --max-prefill-tokens 8192 --kv-cache-dtype bf16 \
    --gpu-memory-utilization 0.90 --max-num-seqs 1 --max-batch-size 1 \
    --speculative --num-drafts 1 \
    --enable-prefix-caching true --fast-load-prefetch-shards \
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
    --disable-thinking
COH=0; COHUP=0
for _ in $(seq 1 180); do
    if "${NS[@]}" curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        COHUP=1; break
    fi
    docker ps -q -f name=atlas-verify-coherence 2>/dev/null | grep -q . || {
        log "coherence serve died"
        docker logs atlas-verify-coherence 2>&1 | tail -20; break; }
    sleep 5
done
if [ "$COHUP" = 1 ]; then
    "${NS[@]}" python3 scripts/test_coherence.py \
        --url "http://127.0.0.1:$PORT" || COH=1
else
    COH=1
fi
docker rm -f atlas-verify-coherence >/dev/null 2>&1 || true

# ── stage 2c: gate ──────────────────────────────────────────────────
log "=== gate_results.py ==="
GATE=0
python3 tests/gate_results.py || GATE=$?

VERDICT=FAIL
if [ "$GATE" -eq 0 ] && [ "$COH" -eq 0 ] && [ "$MATRIX" -eq 0 ]; then
    VERDICT=PASS
fi
echo "VERDICT=$VERDICT matrix=$MATRIX gate=$GATE coherence=$COH" \
    | tee "$OUT/RESULTS.txt"

# ── stage 3: save + print publish commands ──────────────────────────
if [ "$VERDICT" = "PASS" ]; then
    log "=== image save ==="
    docker save "$IMAGE" | zstd -o "$OUT/atlas-gb10-${SHA}.tar.zst"
    {
      echo "PASS $SHA — publish commands for the maintainer (NOT run):"
      echo "  docker tag $IMAGE azeezish/atlas-gb10:latest"
      echo "  docker tag $IMAGE ghcr.io/atlas-inf/atlas-gb10:${SHA}"
      echo "  docker tag $IMAGE ghcr.io/atlas-inf/atlas-gb10:latest"
      echo "  docker push $IMAGE && docker push azeezish/atlas-gb10:latest"
      echo "  docker push ghcr.io/atlas-inf/atlas-gb10:${SHA} && docker push ghcr.io/atlas-inf/atlas-gb10:latest"
      echo "  (offline image: $OUT/atlas-gb10-${SHA}.tar.zst)"
    } | tee -a "$OUT/RESULTS.txt"
    exit 0
fi
echo "FAIL $SHA — no image saved, nothing tagged for publish" \
    | tee -a "$OUT/RESULTS.txt"
exit 1
