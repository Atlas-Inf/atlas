#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: Gufo (gufo-org/gufo, the Strix Halo engine) on Qwen3.8-Flash-Next, out of the box, on the same
# protocol Atlas is measured with (scripts/devlab/gufo_bench.py: pp ~2048 / tg128 at depth 0/8192/16384, C 1/4/8).
#
#   scripts/devlab/gufo_flashnext.sh [KEY=VAL ...]       KEYS: CONTEXT (default 32768), SESSIONS (default 8)
#
# Gufo has no NVFP4 path: it serves GGUF only, and its Flash-Next guide pins Unsloth UD-Q4_K_XL (4 shards) + the shared
# MTP Q8_0 at revision 38bb39ee97821de2c9009abb7e93950eec396e66. Built with Gufo's own nix flake (gufo.cachix.org gfx1151
# binaries, ROCm 10.1 userspace) inside a nixos/nix container, so the runner's ROCm is untouched. Arms: AR and MTP.
set -uo pipefail
for kv in "$@"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
CONTEXT=${CONTEXT:-32768}; SESSIONS=${SESSIONS:-8}
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
HERE=$(cd "$(dirname "$0")" && pwd)
# Models land in the runner's HF cache ($HF_HOME; /mnt/scratch/cache itself is not writable), shared with llamacpp_gguf.sh.
REV=38bb39ee97821de2c9009abb7e93950eec396e66

# ---- build Gufo (nix in a container) -----------------------------------------------------------------------
docker pull -q nixos/nix:2.30.2 >/dev/null 2>&1 || { log "docker pull failed"; exit 1; }
docker volume create gufo-nix >/dev/null
docker run --rm -v gufo-nix:/nix nixos/nix:2.30.2 sh -c '
  nix --extra-experimental-features "nix-command flakes" build "github:gufo-org/gufo" --accept-flake-config \
    --option build-users-group "" --option sandbox false \
    --option extra-substituters https://gufo.cachix.org \
    --option extra-trusted-public-keys gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE= \
    --print-out-paths --no-link 2>&1 | tail -3' > "$OUT/nix-build.log" 2>&1
P=$(grep -aE "^/nix/store/.*gufo" "$OUT/nix-build.log" | tail -1)
[ -n "$P" ] || { log "GUFO BUILD FAILED: $(tail -3 "$OUT/nix-build.log")"; exit 1; }
log "gufo store path: $P"

# ---- model (README-pinned revision, through the runner's HF mirror) ----------------------------------------
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub
t0=$(date +%s)
python3 -c "import hf_transfer" 2>/dev/null || python3 -m pip install --user -q --break-system-packages hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
M=$(python3 -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download("unsloth/Qwen3.8-Flash-Next-GGUF", revision=sys.argv[1], allow_patterns=["UD-Q4_K_XL/*", "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"]))' "$REV" 2>"$OUT/download.log") \
  || { log "MODEL DOWNLOAD FAILED: $(tail -2 "$OUT/download.log")"; exit 1; }
TGT=$(ls "$M"/UD-Q4_K_XL/*.gguf | sort | head -1); MTP="$M/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
log "model ready in $(( $(date +%s) - t0 ))s: target $(du -shL "$M/UD-Q4_K_XL" | cut -f1) ($(basename "$TGT")), mtp $(du -shL "$MTP" 2>/dev/null | cut -f1)"
GR=$(getent group render | cut -d: -f3); GV=$(getent group video | cut -d: -f3)
docker run --rm -v gufo-nix:/nix --device /dev/kfd --device /dev/dri --group-add "$GV" --group-add "$GR" nixos/nix:2.30.2 \
  "$P/bin/gufo" diagnose > "$OUT/diagnose.log" 2>&1; tail -12 "$OUT/diagnose.log" | sed 's/^/  diagnose: /' | tee -a "$RES"

# ---- serve + bench, AR and MTP -----------------------------------------------------------------------------
PORT=18083
for arm in ar mtp; do
  extra=""; [ $arm = mtp ] && extra="--speculative mtp --mtp-model $MTP"
  docker rm -f gufo-fn >/dev/null 2>&1
  docker run -d --name gufo-fn --network host --device /dev/kfd --device /dev/dri --group-add "$GV" --group-add "$GR" \
    --security-opt seccomp=unconfined -v gufo-nix:/nix -v "${HF_HOME:-$HOME/.cache/huggingface}":"${HF_HOME:-$HOME/.cache/huggingface}":ro nixos/nix:2.30.2 \
    "$P/bin/gufo" serve --host 127.0.0.1 --port $PORT llm --think off --model "$TGT" \
    --sessions "$SESSIONS" --context "$CONTEXT" $extra > /dev/null
  ok=0
  for _ in $(seq 1 240); do
    curl -s -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' 2>/dev/null | grep -q choices && { ok=1; break; }
    docker ps -q -f name=gufo-fn | grep -q . || break; sleep 5
  done
  docker logs gufo-fn > "$OUT/serve-$arm.log" 2>&1
  if [ $ok != 1 ]; then log "$arm SERVE FAILED:"; grep -aiE "error|fail|memory|alloc" "$OUT/serve-$arm.log" | tail -6 | tee -a "$RES"; docker rm -f gufo-fn >/dev/null 2>&1; continue; fi
  log "$arm serving"
  python3 "$HERE/gufo_bench.py" $PORT "gufo_${arm}" "$OUT/gufo-$arm.json" --depths 0,8192,16384 --cs 1,4,8 --kinds mixed 2>&1 \
    | grep -aE "^\s+\[" | tee -a "$RES"
  docker logs gufo-fn > "$OUT/serve-$arm.log" 2>&1; docker rm -f gufo-fn >/dev/null 2>&1
done
log "DONE"
