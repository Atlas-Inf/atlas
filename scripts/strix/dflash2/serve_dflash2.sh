#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Serve Qwen3.8-27B-NVFP4 + the DFlash2 drafter on AzeezStrix (gfx1151)
# with the memory recipe that clears the ~55.5 GB allocatable ceiling on
# this unified-memory part (reported GTT pool is 60 GB; the last ~4.4 GB
# is unusable — every late startup alloc dies there regardless of size).
#
# Requires the acc-pool prime + startup balloons + contiguous Marconi
# snapshot blobs (perf/dflash2-ctx-carry >= 80cd40471). On that stack:
#   GPU_UTIL=0.84 fits budget 50.4 GB + ~4.6 GB untracked < 55.6 GB wall;
#   12 Marconi slots @ 512-block interval => anchors every 8192 tokens;
#   KV lands ~4.5 GB / ~73K tokens (measured 2026-09-24, varies with
#   pre-KV drift). Warm prefix hit on a 14.5K prompt: TTFT ~0.5 s.
#
# NOTE: GPU_UTIL=0.84 is for standalone serving. When a benchmark harness
# is co-resident on this 61 GB host (the agentic leg), run the harness
# OFF-BOX instead (HOST=0.0.0.0 + remote endpoint) — at 0.84 with a
# co-resident harness the host OOM-killer SIGKILLed spark mid-run
# (2026-09-24T05:08Z kern.log).
#
#   usage: PORT=8097 bash scripts/strix/dflash2/serve_dflash2.sh
# Every knob is env-overridable: DRAFTER GAMMA GPU_UTIL MAX_SEQ_LEN PORT
# HOST SSM_SLOTS SSM_CKPT_INTERVAL MAX_PREFILL_TOKENS ...
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root — scripts/strix/dflash2/

PORT="${PORT:-8097}"
HOST="${HOST:-127.0.0.1}"
MODEL="${MODEL:-nvidia/Qwen3.8-27B-NVFP4}"
DRAFTER="${DRAFTER:-/home/azeez/.models/dflash2}"   # incoai/Qwen3.8-27B-DFlash2

export DFLASH=1 DRAFT_MODEL="$DRAFTER" DFLASH_GAMMA="${GAMMA:-8}"
export PORT HOST MAX_SEQ_LEN=26624 GPU_UTIL="${GPU_UTIL:-0.84}"
export SSM_SLOTS=12 SSM_CKPT_INTERVAL=512 MAX_PREFILL_TOKENS=1024
export ATLAS_DFLASH_STEP_TIMING=1 ATLAS_MTP_ACCEPT_DEBUG=1
export ATLAS_NO_MTP_DRAFTER_CONTEXT=1 ATLAS_DFLASH_OPTION_B=1
export ATLAS_DFLASH_CTX_WINDOW=12288
export HF_HUB_OFFLINE=1

exec ./serve-amd.sh "$MODEL" \
    --enable-prefix-caching --dflash-window-size 2048 \
    --disable-thinking --request-timeout 900 --vision-max-pixels 65536 "$@"
