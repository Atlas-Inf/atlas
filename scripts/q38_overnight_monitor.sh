#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Read-only status snapshot for the Linux and Windows Strix performance legs.
set -u
printf '=== %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

printf '[strix] processes:\n'
pgrep -af '^bash scripts/q38_replay1007.sh|inference-endpoint|spark serve' || echo 'none'
LATEST_LOCAL=$(ls -t "$HOME"/q38-*-agentic*.log "$HOME"/q38-replay*.log 2>/dev/null | head -1)
if [[ -n "${LATEST_LOCAL:-}" ]]; then
  echo "[strix] latest log: $LATEST_LOCAL"
  tail -5 "$LATEST_LOCAL"
fi
TEMP=$(amd-smi metric --temperature 2>/dev/null | LC_ALL=C sed -n 's/.*EDGE: *//p' | head -1)
echo "[strix] GPU temp: ${TEMP:-unknown}"

WIN_STATUS=$(ssh -o ConnectTimeout=10 -o BatchMode=yes winbox \
  'powershell -NoProfile -NonInteractive -Command "@(Get-Process spark -ErrorAction SilentlyContinue).Count"' 2>/dev/null)
if [[ -z "$WIN_STATUS" ]]; then
  echo '[winbox] unreachable'
elif [[ "$WIN_STATUS" == 0 ]]; then
  echo '[winbox] no spark process'
else
  echo "[winbox] spark processes: $WIN_STATUS"
  ssh -o ConnectTimeout=10 -o BatchMode=yes winbox \
    'powershell -NoProfile -NonInteractive -Command "$f=Get-ChildItem C:\Users\azeez\q38-agentic-perf-*-serve.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if($f){Write-Output $f.FullName; Get-Content $f.FullName -Tail 5}"' 2>/dev/null
fi
printf '%s\n' '---'
