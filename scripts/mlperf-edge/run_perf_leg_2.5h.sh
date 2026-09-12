#!/bin/bash
# Runner for the MLPerf Edge Agentic Performance Leg (2.5h) on Qwen3.8-27B DFlash2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="${HARNESS_DIR:-$HOME/endpoints-mlperf}"
[ -d "$HARNESS_DIR" ] || HARNESS_DIR="$HOME/endpoints"
[ -d "$HARNESS_DIR" ] || { echo "HARNESS_DIR not found: $HARNESS_DIR" >&2; exit 1; }

PORT="${PORT:-8899}"
ENDPOINT="http://127.0.0.1:${PORT}"
MODEL="unsloth/Qwen3.8-27B-NVFP4"
TAG="dflash2_perf25h"
TS=$(date +%Y%m%d_%H%M%S)
RD="results/${TAG}_${TS}"

echo "=== MLPerf Edge Agentic Performance Leg (2.5h) ==="
echo "Target Endpoint: $ENDPOINT"
echo "Model: $MODEL"
echo "Harness: $HARNESS_DIR"

# Verify endpoint is live
curl -sf -m5 "${ENDPOINT}/v1/models" >/dev/null || {
  echo "Error: Endpoint ${ENDPOINT} is not reachable" >&2
  exit 1
}

cd "$HARNESS_DIR"
CFG="$(mktemp -t "${TAG}_XXXX.yaml")"
python3 - "$RD" "$CFG" "${SCRIPT_DIR}/perf_leg_2.5h_template.yaml" "$PORT" <<'PY'
import sys, yaml
rd, cfg, base, port = sys.argv[1:5]
c = yaml.safe_load(open(base))
c["report_dir"] = rd
c.setdefault("endpoint_config", {})["endpoints"] = [f"http://127.0.0.1:{port}"]
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
print(f"Generated config -> {cfg} (report_dir={rd})")
PY

echo "Starting inference-endpoint benchmark in perf mode..."
./.venv/bin/inference-endpoint benchmark from-config -c "$CFG" --mode perf -v

echo "=== Performance Leg Complete ==="
f=$(find "$RD" -name result_summary.json 2>/dev/null | head -1)
if [ -n "$f" ]; then
  python3 -c "
import json
r = json.load(open('$f'))
print('Duration (s):', round(r.get('duration_ns', 0)/1e9, 1))
print('Throughput (tok/s):', round(r.get('tps', 0), 2))
print('Median TTFT (ms):', round(r.get('ttft', {}).get('median', 0)/1e6, 1))
print('Median TPOT (ms):', round(r.get('tpot', {}).get('median', 0)/1e6, 2))
print('Samples Completed:', r.get('n_samples_completed', 0))
"
fi
