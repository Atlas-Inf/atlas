# SPDX-License-Identifier: AGPL-3.0-only
# Hardware check of spark-runtime's own-footprint path, without a tree build.
# Run in any image with rustc, the repo root mounted read-only at /repo:
#   docker run --rm --gpus all [-e HIDE_NVML=1] -v "$PWD":/repo:ro \
#     --entrypoint bash <image> /repo/docs/campaigns/kv-sizing-2026-09/rust-footprint/run.sh
set -e
here=/repo/docs/campaigns/kv-sizing-2026-09/rust-footprint
work=/tmp/kvs-footprint-src
mkdir -p $work/backend
cp $here/main.rs $here/shim.rs /repo/crates/spark-runtime/src/own_footprint.rs $work/
cp /repo/crates/spark-runtime/src/cuda_backend/nvml.rs $work/backend/
cd $work
rustc --version
rustc --edition 2024 --crate-type rlib --crate-name libc shim.rs -o /tmp/liblibc.rlib
rustc --edition 2024 -O main.rs --extern libc=/tmp/liblibc.rlib -L /usr/local/cuda/lib64/stubs -L /usr/lib/aarch64-linux-gnu -o /tmp/kvs-footprint
if [ "$HIDE_NVML" = 1 ]; then
  # A container without the library: a file that is not a shared object
  # shadows it on the search path, so dlopen fails as it would if absent.
  mkdir -p /tmp/nonvml && echo "not a library" > /tmp/nonvml/libnvidia-ml.so.1
  export LD_LIBRARY_PATH=/tmp/nonvml
  echo "libnvidia-ml.so.1 hidden"
fi
grep -E 'MemFree|MemAvailable' /proc/meminfo | tr '\n' ' '; echo
# cuCtxCreate intermittently returned CUDA_ERROR_OUT_OF_MEMORY (2) on the
# shared host this was run on; nothing is held when it does, so retry.
for attempt in 1 2 3 4 5 6; do
  /tmp/kvs-footprint && exit 0
  echo "attempt $attempt failed; retrying in 5 s"; sleep 5
done
exit 1
