#!/usr/bin/env bash
# serve-nemotron-strix-linux — single release serve entry for Nemotron 3.5 Super VL on a
# Strix Halo (gfx1151) under native Linux + ROCm. Linux twin of serve-nemotron-gb10.sh and
# serve-nemotron-amd.ps1; same interface (--spec off|g3|g8, --check).
#
# Requires: BIOS iGPU allocation 96 GB (amdgpu reports >= 90 GiB VRAM), /dev/kfd present
# (on Secure Boot machines the amdgpu-dkms MOK must be enrolled), ROCm with hipcc, rustup cargo.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/target/release/spark"
MODEL=${MODEL:-$HOME/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4}
DRAFTER=${DRAFTER:-$HOME/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr}
NAME=nemotron-3.5-super-vl
# Linux output matches the Windows build byte-for-byte on the goldens, so both share them.
GOLDEN=${GOLDEN:-nemotron-amd}
export PATH="$HOME/.cargo/bin:/opt/rocm/bin:$PATH"

SPEC=g8 PORT=8391 CHECK=0 ALLOW_DIRTY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --spec) SPEC=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --check) CHECK=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "$SPEC" in off|g3|g8) ;; *) echo "--spec must be off|g3|g8" >&2; exit 2 ;; esac

# Pre-flight: GPU visible, 96 GB carve-out, weights present.
[ -e /dev/kfd ] || { echo "refuse: /dev/kfd missing — amdgpu not loaded (Secure Boot: enroll the amdgpu-dkms MOK)" >&2; exit 3; }
VRAM_GIB=0
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
  [ -r "$f" ] && VRAM_GIB=$(( $(cat "$f") / 1073741824 ))
done
[ "$VRAM_GIB" -ge 90 ] || { echo "refuse: amdgpu VRAM ${VRAM_GIB} GiB < 90 — set the BIOS iGPU allocation to 96 GB" >&2; exit 3; }
[ -f "$MODEL/config.json" ] || { echo "refuse: model not found at $MODEL" >&2; exit 3; }
if [ "$SPEC" != off ] && [ ! -f "$DRAFTER/config.json" ]; then
  echo "refuse: drafter not found at $DRAFTER" >&2; exit 3
fi

HEAD=$(git -C "$REPO" rev-parse HEAD)
TAG=$(git -C "$REPO" describe --tags --exact-match HEAD 2>/dev/null || echo untagged)
if [ "$ALLOW_DIRTY" -eq 0 ] && ! git -C "$REPO" diff --quiet HEAD -- 2>/dev/null; then
  echo "refuse: dirty tracked files in $REPO (pass --allow-dirty to override)" >&2
  exit 3
fi
# Rebuild only when a build input changed (docs/goldens-only commits do not).
BUILD_HEAD=$(git -C "$REPO" log -1 --format=%H -- . ':(exclude)scripts/golden' ':(exclude)*.md' ':(exclude)docs')
if [ ! -x "$BIN" ] || [ ! -f "$REPO/.built-$BUILD_HEAD" ]; then
  rm -f "$REPO"/.built-*
  echo "=== build $REPO ($BUILD_HEAD, strix-hip nemotron) ===" >&2
  ( cd "$REPO" && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} \
      ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL=nemotron-3.5-super-vl-120b-a12b \
      ./build-amd.sh ) || { echo "BUILD FAILED" >&2; exit 6; }
  touch "$REPO/.built-$BUILD_HEAD"
fi
BINSHA=$(sha256sum "$BIN" | awk '{print $1}')
# The binary links the HIP shims (libcuda/libcudart/libcublasLt) that atlas-kernels builds
# into its OUT_DIR, with no rpath: point the loader at the newest one, then ROCm's runtime.
SHIM_DIR=$(ls -dt "$REPO"/target/release/build/atlas-kernels-*/out 2>/dev/null \
  | while read -r d; do [ -f "$d/libcuda.so" ] && { echo "$d"; break; }; done)
[ -n "$SHIM_DIR" ] || { echo "refuse: no HIP shim libcuda.so under target/release/build — rebuild" >&2; exit 6; }
export LD_LIBRARY_PATH="$SHIM_DIR:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

serve_args=(
  serve "$MODEL" --model-name "$NAME"
  --bind 127.0.0.1 --port "$PORT" --no-tui
  --max-seq-len 32768 --max-prefill-tokens 4096 --kv-cache-dtype bf16
  --gpu-memory-utilization 0.85 --max-num-seqs 1 --max-batch-size 1
  --ssm-cache-slots 8 --enable-prefix-caching true --request-timeout 0
  --disable-thinking
)
case "$SPEC" in
  g3|g8) serve_args+=(--dflash --draft-model "$DRAFTER" --dflash-gamma "${SPEC#g}") ;;
esac

banner() {
  echo "serve-nemotron-strix-linux tag=$TAG sha=$HEAD binsha=$BINSHA spec=$SPEC port=$PORT vram_gib=$VRAM_GIB model=$MODEL drafter=$DRAFTER"
}
banner | tee /dev/stderr
cd "$REPO"  # jinja-templates/ resolve repo-relative

# Same runtime env as the verified winbox profile.
run_serve() {
  exec env RUST_LOG=info ATLAS_MTP_ACCEPT_DEBUG=1 ATLAS_MEM_PROFILE=1 \
    ATLAS_KV_EXTERNAL_RESERVE_GB=0 ATLAS_OOM_WATCHDOG_MB=400 \
    ATLAS_W4A16_DP4A=0 ATLAS_NO_MAMBA_VERIFY_FUSED=1 ATLAS_SSM_DECODE_RING=0 \
    "$BIN" "${serve_args[@]}"
}

if [ "$CHECK" -eq 0 ]; then
  run_serve
fi

LOG=$(mktemp /tmp/n35vl-serve-check.XXXXXX.log)
( run_serve ) > "$LOG" 2>&1 &
SPID=$!
cleanup() { kill -INT "$SPID" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done; kill -9 "$SPID" 2>/dev/null; }
trap cleanup EXIT
ok=0
for _ in $(seq 1 360); do
  if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK.\"}],\"max_tokens\":2,\"temperature\":0}" \
      | grep -q '"choices"' && { ok=1; break; }
  fi
  kill -0 "$SPID" 2>/dev/null || { echo "serve died; log:" >&2; tail -20 "$LOG" >&2; exit 4; }
  sleep 5
done
[ "$ok" -eq 1 ] || { echo "serve timeout; log:" >&2; tail -20 "$LOG" >&2; exit 5; }

python3 - "$REPO" "$GOLDEN" "$PORT" "$NAME" <<'PY'
import json, sys, glob, urllib.request, os
repo, golden, port, model = sys.argv[1:5]
pd = f"{repo}/scripts/golden/nemotron-gb10/prompts"
gd = f"{repo}/scripts/golden/{golden}/atlas"
fail = 0
for pf in sorted(glob.glob(pd + "/*.json")):
    name = os.path.basename(pf)[:-5]
    body = json.load(open(pf))
    body["model"] = model
    body["temperature"] = 0
    r = urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}),
        timeout=600)
    resp = json.load(r)
    msg = resp["choices"][0]["message"]
    got = {"content": msg.get("content"), "tool_calls": msg.get("tool_calls")}
    gf = f"{gd}/{name}.json"
    if not os.path.exists(gf):
        print(f"{name}: FAIL no golden {gf}"); fail = 1; continue
    want = json.load(open(gf))
    if got == want:
        print(f"{name}: PASS"); continue
    fail = 1
    a = json.dumps(got, sort_keys=True); b = json.dumps(want, sort_keys=True)
    i = next((j for j, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
    print(f"{name}: FAIL first-diff-char {i}")
    print(f"  got : {a[max(0,i-80):i+80]}")
    print(f"  want: {b[max(0,i-80):i+80]}")
sys.exit(fail)
PY
exit $?
