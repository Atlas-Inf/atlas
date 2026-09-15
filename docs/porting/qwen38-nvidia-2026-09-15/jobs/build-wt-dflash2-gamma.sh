#!/usr/bin/env bash
# JOB (no-gpu): build-wt-dflash2-gamma — prepare the serve binary the overnight
# DFlash2 legs run on, WITHOUT touching ~/code/wt-dflash2 (its binary backs
# jobs that may still be queued/running — the 007/010 exit-127 lesson).
#
# Creates/refreshes worktree ~/code/wt-dflash2-gamma on Atlas-Inf/atlas
# review/pr21-fixes at the sha given in GAMMA_SHA (the commit that resolves
# --dflash-gamma from MODEL.toml), builds spark-server release, and asserts
# the built binary's `spark serve --help` no longer shows a γ default of 16.
# Serialized through the queue so the cargo build never competes with a GPU
# job for host memory (unified-memory box).
set -u
source /home/azeez/code/build_env.sh 2>/dev/null || true
export OUT="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

: "${GAMMA_SHA:?GAMMA_SHA must be set via qctl sub --env GAMMA_SHA=<sha>}"
REPO=/home/azeez/code/wt-dflash2
WT=/home/azeez/code/wt-dflash2-gamma

log "== fetch + worktree at $GAMMA_SHA =="
git -C "$REPO" fetch origin review/pr21-fixes 2>&1 | tail -3
git -C "$REPO" cat-file -e "$GAMMA_SHA^{commit}" || { log "FAIL: $GAMMA_SHA not present after fetch"; exit 5; }
if [ -d "$WT" ]; then
    git -C "$WT" checkout --detach "$GAMMA_SHA" 2>&1 | tail -2
else
    git -C "$REPO" worktree add --detach "$WT" "$GAMMA_SHA" 2>&1 | tail -2
fi
git -C "$WT" log --oneline -1 | tee "$OUT/FINGERPRINT.txt"

log "== cargo build --release -p spark-server =="
( cd "$WT" && cargo build --release -p spark-server ) > "$OUT/build.log" 2>&1 \
    || { log "FAIL: build"; tail -40 "$OUT/build.log"; exit 6; }
tail -3 "$OUT/build.log"

log "== assert: --dflash-gamma has no clap default anymore =="
"$WT/target/release/spark" serve --help 2>&1 | grep -A6 -- "--dflash-gamma" | tee "$OUT/help-gamma.txt"
if grep -q "default: 16" "$OUT/help-gamma.txt"; then
    log "FAIL: binary still carries the γ=16 clap default — wrong sha built?"
    exit 9
fi
log "JOB DONE — $WT/target/release/spark ready"
