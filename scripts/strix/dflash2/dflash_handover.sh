#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Watch the running dflash_overnight chain; the moment leg 1 (ST-995) has
# finished and the chain logs its LEG2 banner, stop the chain (ST-996 is
# dropped to fit the merge window) and start the agentic-only leg instead.
# Leg-1 artifacts (record copy, accept summary) are written BEFORE the LEG2
# banner, so nothing from ST-995 is lost.
set -uo pipefail
CHAIN_OUT=/home/azeez/dp4a-ab/out/dflash-overnight-20260915T1706Z
AB=/home/azeez/dp4a-ab
LOG=$AB/out/dflash-handover.log
say(){ echo "=== $(date -u +%FT%TZ) $* ===" | tee -a "$LOG"; }

say "handover armed; waiting for LEG2 banner in $CHAIN_OUT/chain.log"
while ! grep -q "LEG2 ST-996" "$CHAIN_OUT/chain.log" 2>/dev/null; do
  pgrep -f "dflash_overnight.sh" >/dev/null || { say "chain exited before LEG2 — nothing to hand over"; break; }
  sleep 10
done
say "LEG2 banner seen (or chain gone) — stopping the chain and its serve"
pkill -f "dflash_overnight.sh" 2>/dev/null; sleep 2
pkill -f '[s]park[_a-zA-Z0-9.-]* serve' 2>/dev/null
for _ in $(seq 1 15); do pgrep -f '[s]park[_a-zA-Z0-9.-]* serve' >/dev/null || break; sleep 2; done
pkill -9 -f '[s]park[_a-zA-Z0-9.-]* serve' 2>/dev/null; sleep 5
pkill -f "inference-endpoint" 2>/dev/null
say "ST-995 result lines:"; grep -E "Overall accuracy|Normalized single-turn|ST-995 record" "$CHAIN_OUT/chain.log" | tee -a "$LOG"
say "starting agentic-only leg"
cd "$AB" && GAMMA=8 bash ./dflash_agentic_only.sh >> "$AB/out/dflash-agentic-launch.out" 2>&1
say "agentic-only leg finished"
