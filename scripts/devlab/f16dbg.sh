#!/usr/bin/env bash
# devlab-only: compile + run the f16 tile differential test on gfx1151.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
HIPCC=$(ls -d /opt/rocm*/bin/hipcc 2>/dev/null | head -1 || command -v hipcc)
echo "hipcc: $HIPCC"; "$HIPCC" --version | head -2
"$HIPCC" --offload-arch=gfx1151 -O2 -x hip "$HERE/f16dbg.cu" -o /tmp/f16dbg \
  && /tmp/f16dbg | tee "$DEVLAB_OUT/f16dbg.txt"
