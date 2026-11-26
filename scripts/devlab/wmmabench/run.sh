#!/usr/bin/env bash
# devlab-only: compile + run the wmma rate microbench on gfx1151.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
HIPCC=$(ls -d /opt/rocm*/bin/hipcc 2>/dev/null | head -1 || command -v hipcc)
echo "hipcc: $HIPCC"; "$HIPCC" --version | head -2
"$HIPCC" --offload-arch=gfx1151 -O3 -x hip "$HERE/wmma_rate.cu" -o /tmp/wmma_rate \
  && /tmp/wmma_rate | tee "$DEVLAB_OUT/wmma_rate.txt"
