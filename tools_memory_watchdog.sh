#!/usr/bin/env bash
# Kill a runaway run before it takes the host down with it.
#
# On a 64 GB box an omnidreams run that oversubscribes memory does not OOM
# cleanly -- it thrashes swap until the machine stops answering SSH, and the
# OOM killer arrives far too late to help. Watch MemAvailable and pull the
# plug while there is still enough headroom to do so.
#
#   ./tools_memory_watchdog.sh                 # kill flashdreams-run under 1 GiB
#   THRESHOLD_GB=4 ./tools_memory_watchdog.sh  # more headroom
#   ./tools_memory_watchdog.sh 'interactive-drive|flashdreams-run'
#
# Exits 1 when it kills something, 0 when the watched process is gone.
set -u

PATTERN=${1:-flashdreams-run}
THRESHOLD_GB=${THRESHOLD_GB:-1}
INTERVAL=${INTERVAL:-5}

avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }

# `pgrep -f` also matches this script and whatever shell launched it, because
# PATTERN sits on their command lines too. Anything mentioning this script by
# name is us; a real run never does. Without this the watchdog waits on itself
# and never sees the run finish.
_SELF_NAME=$(basename "$0")
target_pids() {
  local pid cmd
  for pid in $(pgrep -f "$PATTERN" 2>/dev/null); do
    # Unreadable means the pid died between pgrep and here -- typically the
    # subshell of this very $(...), which inherits our command line and so
    # matches PATTERN. Either way it is not a run worth watching.
    cmd=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
    [ -n "$cmd" ] || continue
    case "$cmd" in *"$_SELF_NAME"*) continue ;; esac
    echo "$pid"
  done
}

printf '%s watching "%s", killing under %s GiB\n' \
  "$(date +%H:%M:%S)" "$PATTERN" "$THRESHOLD_GB" >&2

# Wait for the run to appear so the watchdog can be started first.
while [ -z "$(target_pids)" ]; do
  sleep "$INTERVAL"
done

while :; do
  A=$(avail_gb)
  echo "$(date +%H:%M:%S) avail=${A}G"

  if [ "$A" -lt "$THRESHOLD_GB" ]; then
    echo "$(date +%H:%M:%S) KILLING: ${A}G available, below ${THRESHOLD_GB}G" >&2
    # shellcheck disable=SC2046  # word splitting is the point
    kill -9 $(target_pids) 2>/dev/null
    exit 1
  fi

  if [ -z "$(target_pids)" ]; then
    echo "$(date +%H:%M:%S) run finished, ${A}G available" >&2
    exit 0
  fi

  sleep "$INTERVAL"
done
