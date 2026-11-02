#!/usr/bin/env bash
# Build Atlas for AMD GPUs. Verified on gfx1151 / Strix Halo, ROCm 7.13/10, native
# Ubuntu. See docs/porting/amd-strix-halo-scale.md.
#
# Two backends, selected with ATLAS_TARGET_HW:
#
#   strix-hip  (default)  native HIP via hipcc. Needs ROCm and cargo, nothing
#                         else — atlas-kernels builds its own libcuda/libcudart/
#                         libcublasLt shims from crates/atlas-kernels/hip/.
#   strix                 SCALE (scale-lang.com) recompiles the unmodified CUDA.
#                         Needs a SCALE install at $SCALE_HOME.
#
# ATLAS_TARGET_MODEL selects which kernel targets are embedded. The default '*'
# builds every target under kernels/$ATLAS_TARGET_HW/ into ONE binary that can
# serve any of them — Qwen3.8-27B reuses Qwen3.6-27B's kernel tree via
# `kernel_source`, so this costs far less than it sounds (measured: 98 unique
# nvcc invocations for 283 requested, 2.9x dedup). Set it to a single target
# name (e.g. qwen3.8-27b) for a smaller binary.
set -euo pipefail
cd "$(dirname "$0")"
ROCM_HOME="${ATLAS_ROCM_HOME:-/opt/rocm}"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"

export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-strix-hip}"
export ATLAS_TARGET_MODEL="${ATLAS_TARGET_MODEL:-*}"
export ATLAS_TARGET_QUANT="${ATLAS_TARGET_QUANT:-nvfp4}"
export CUDARC_CUDA_VERSION=12080
# Strix Halo is a single-APU laptop part; the RDMA verbs shim is irrelevant and
# this opt-out removes the libibverbs-dev prerequisite the old recipe carried.
export ATLAS_NO_RDMA="${ATLAS_NO_RDMA:-1}"

case "$ATLAS_TARGET_HW" in
  strix-hip)
    # Prefer an existing hipcc on PATH; fall back to $ROCM_HOME/bin.
    if [ -z "${ATLAS_HIPCC:-}" ]; then
      ATLAS_HIPCC="$(command -v hipcc || true)"
      [ -n "$ATLAS_HIPCC" ] || ATLAS_HIPCC="$ROCM_HOME/bin/hipcc"
    fi
    export ATLAS_HIPCC
    if [ ! -x "$ATLAS_HIPCC" ]; then
      echo "hipcc not found at $ATLAS_HIPCC — install ROCm HIP dev tools:" >&2
      echo "  sudo apt install rocm-hip-dev        # or amdgpu-install --usecase=hip" >&2
      echo "  (override with ATLAS_HIPCC=/path/to/hipcc if ROCm lives elsewhere)" >&2
      exit 2
    fi
    # hipcc must be able to find hip_runtime.h on its own. On partial installs
    # (hipcc present, no dev headers) every kernel compile dies with
    # "'hip/hip_runtime.h' file not found" — catch it here, not 120 kernels in.
    HIP_INC_DIR="$(dirname "$(dirname "$(readlink -f "$ATLAS_HIPCC")")")/include"
    if [ ! -f "$HIP_INC_DIR/hip/hip_runtime.h" ] && [ ! -f "$ROCM_HOME/include/hip/hip_runtime.h" ]; then
      echo "hip_runtime.h not found beside $ATLAS_HIPCC or under $ROCM_HOME/include." >&2
      echo "The HIP runtime headers are missing — install them:" >&2
      echo "  sudo apt install rocm-hip-dev libamdhip64-dev" >&2
      exit 2
    fi
    export ATLAS_HIP_COMPAT_INCLUDE="$PWD/crates/atlas-kernels/hip/compat"
    export PATH="$ROCM_HOME/bin:$PATH"
    # Deliberately NO `RUSTFLAGS=-L .../hip-port/link`. atlas-kernels/build.rs
    # compiles the three HIP shims into OUT_DIR and puts that first on the link
    # path; pointing -L at a hand-built shim directory shadows them with an
    # older libcuda.so and the link fails on cuStreamQuery /
    # cuMemHostGetDevicePointer_v2 / the 11 cublasLt* symbols.
    echo "hipcc -> $ATLAS_HIPCC  (native HIP, $ATLAS_TARGET_HW/$ATLAS_TARGET_MODEL/$ATLAS_TARGET_QUANT)"
    ;;
  strix)
    : "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
    export SCALE_HOME
    export CUDA_PATH="$SCALE_HOME/targets/gfx1151"
    export CUDA_HOME="$CUDA_PATH"
    export PATH="$SCALE_HOME/targets/gfx1151/bin:$ROCM_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_HOME/lib:$SCALE_HOME/targets/gfx1151/lib:${LD_LIBRARY_PATH:-}"
    echo "nvcc -> $(command -v nvcc)  (SCALE, $ATLAS_TARGET_HW/$ATLAS_TARGET_MODEL/$ATLAS_TARGET_QUANT)"
    ;;
  *) echo "ATLAS_TARGET_HW must be strix-hip or strix (got: $ATLAS_TARGET_HW)" >&2; exit 2 ;;
esac

# rustup installs cargo to ~/.cargo/bin, which a non-interactive shell (ssh
# "cmd", CI, systemd) does not get from the login profile.
command -v cargo >/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null; then
  echo "cargo not found — install Rust via rustup (distro rustc is too old;" >&2
  echo "rust-toolchain.toml pins $(grep -m1 'channel' rust-toolchain.toml | tr -d 'channel =" ')):" >&2
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" >&2
  exit 2
fi

rm -rf "$TARGET_DIR"/release/build/atlas-kernels-* "$TARGET_DIR"/release/build/spark-storage-*
cargo build --release -p spark-server --no-default-features --features cuda
