#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Hold a bounded amount of host memory, then release it: a stand-in co-tenant
for checking Atlas KV-pool sizing on a unified-memory host (no GPU use, no load
generator).

  release-during-load (the 2026-09-30 defect):
      hold-release.py --gib 1 --hold 45      # start BEFORE the server
  allocate-during-load (the delta over-measures, the pool shrinks):
      hold-release.py --gib 1 --delay 30 --hold 120   # start WITH the server

Timeline: wait --delay seconds, allocate and touch --gib GiB of anonymous
memory, print HOLDING, keep it for --hold seconds (or until SIGUSR1), free it,
print RELEASED, exit.

Safe by construction: at most 8 GiB; refuses to allocate unless MemAvailable
would stay above --floor-gib (default 16); frees early if MemAvailable drops
under --panic-gib (default 2) while holding; SIGINT/SIGTERM free and exit; the
kernel frees everything if the process dies.
"""
import argparse
import mmap
import os
import signal
import sys
import time

GIB = 1 << 30
MAX_GIB = 8.0


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise SystemExit("no MemAvailable in /proc/meminfo (Linux only)")


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg} (MemAvailable {mem_available() / GIB:.2f} GiB)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=1.0, help="GiB to hold (max 8)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before allocating")
    ap.add_argument("--hold", type=float, default=45.0, help="seconds to hold before releasing (max 900)")
    ap.add_argument("--floor-gib", type=float, default=16.0, help="refuse unless this much stays available")
    ap.add_argument("--panic-gib", type=float, default=2.0, help="release early below this MemAvailable")
    a = ap.parse_args()
    if not 0 < a.gib <= MAX_GIB:
        raise SystemExit(f"--gib must be in (0, {MAX_GIB}]")
    if not 0 <= a.hold <= 900 or not 0 <= a.delay <= 900:
        raise SystemExit("--hold and --delay must be in [0, 900] seconds")
    size = int(a.gib * GIB)

    stop = {"why": None}
    for sig, why in ((signal.SIGUSR1, "SIGUSR1"), (signal.SIGINT, "SIGINT"), (signal.SIGTERM, "SIGTERM")):
        signal.signal(sig, lambda *_args, why=why: stop.update(why=why))

    def wait(seconds: float, watch_panic: bool) -> None:
        deadline = time.monotonic() + seconds
        while stop["why"] is None and time.monotonic() < deadline:
            if watch_panic and mem_available() < a.panic_gib * GIB:
                stop["why"] = f"MemAvailable under {a.panic_gib} GiB"
                return
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    say(f"pid {os.getpid()}: will hold {a.gib} GiB after {a.delay}s for {a.hold}s; `kill -USR1 {os.getpid()}` releases now")
    wait(a.delay, watch_panic=False)
    if stop["why"] in ("SIGINT", "SIGTERM"):
        say(f"stopped before allocating ({stop['why']})")
        return 1
    stop["why"] = None
    if mem_available() - size < a.floor_gib * GIB:
        say(f"REFUSED: holding {a.gib} GiB would leave under {a.floor_gib} GiB available")
        return 2

    block = mmap.mmap(-1, size)  # anonymous, private; freed by close() or exit
    try:
        chunk = b"\x01" * (64 << 20)
        written = 0
        while written < size and stop["why"] is None:
            if mem_available() < a.panic_gib * GIB:
                stop["why"] = f"MemAvailable under {a.panic_gib} GiB"
                break
            n = min(len(chunk), size - written)
            block.write(chunk[:n])  # touch every page so the memory is really resident
            written += n
        if stop["why"] is None:
            say(f"HOLDING {written / GIB:.2f} GiB")
            wait(a.hold, watch_panic=True)
    finally:
        block.close()
    say(f"RELEASED ({stop['why'] or 'hold elapsed'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
