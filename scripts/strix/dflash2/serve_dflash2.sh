#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# serve_dflash2.sh — one-line DFlash2 serve on AzeezStrix (gfx1151, ROCm).
# Thin wrapper over ./serve-amd.sh with the certified DFlash γ=8 Option-B
# recipe baked in (the same profile dflash_overnight.sh leg 1 runs under):
# util 0.80 (0.88 OOMs once the ~5 GB drafter allocates), seq 8192, prefill
# 2048, kv bf16, lm_head nvfp4, batch=1, ssm-slots 0, thinking off.
#
#   usage: ./scripts/strix/dflash2/serve_dflash2.sh
# Every knob is env-overridable: DRAFT_MODEL DFLASH_GAMMA GPU_UTIL
# MAX_SEQ_LEN PORT HOST MODEL SSM_SLOTS SSM_CKPT_INTERVAL LM_HEAD ...
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root — scripts/strix/dflash2/

env HF_HUB_OFFLINE=1 RUST_LOG=info ATLAS_MTP_ACCEPT_DEBUG=1 \
    DFLASH=1 \
    DRAFT_MODEL="${DRAFT_MODEL:-$HOME/.models/dflash2}" \
    DFLASH_GAMMA="${DFLASH_GAMMA:-8}" \
    GPU_UTIL="${GPU_UTIL:-0.80}" \
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}" \
    PORT="${PORT:-8093}" \
    HOST="${HOST:-127.0.0.1}" \
    MODEL_NAME="${MODEL_NAME:-nvidia/Qwen3.8-27B-NVFP4}" \
    SSM_SLOTS="${SSM_SLOTS:-0}" \
    ./serve-amd.sh "${MODEL:-nvidia/Qwen3.8-27B-NVFP4}" \
      --disable-thinking --request-timeout 900 "$@"
