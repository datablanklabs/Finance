#!/bin/sh
# Wrapper for one scheduled trading cycle. Intended to be the only thing the
# scheduler (launchd) invokes.
#
# Invoking run_cycle.py directly from a scheduler is broken in three ways
# that all fail silently at 10:30 in the morning:
#
#   1. A scheduler's PATH is minimal, where `python3` is macOS's system Python
#      -- which does not have pandas, openbb or mcp installed. Confirmed:
#      "ModuleNotFoundError: No module named 'mcp'". The interpreter must be
#      named absolutely.
#   2. launchd does not inherit your shell's cwd, and run_cycle.py resolves
#      state/, audit/ and .cache/ relative to it. Without a cd it writes a
#      second, empty set of state files in the wrong place -- including
#      peak_equity, whose whole job is to survive restarts.
#   3. OpenBB reads credentials from ~/.openbb_platform/user_settings.json,
#      so HOME must be set or the FRED key silently disappears and the VIX
#      overlay no-ops (see MacroRegimeFilter's max_age_days).
#
# Usage:  ./scheduled_cycle.sh         # dry run (default, sends nothing)
#         ./scheduled_cycle.sh --live  # actually submits orders
set -u

PROJECT="/Users/socrates/Applications/AI-trader"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
HOME="${HOME:-/Users/socrates}"
export HOME

LOG_DIR="$PROJECT/logs"
LOG="$LOG_DIR/cycle-$(date +%Y-%m).log"        # one file per month
mkdir -p "$LOG_DIR"

cd "$PROJECT" || { echo "cannot cd to $PROJECT" >&2; exit 3; }

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "===== $(stamp) starting: run_cycle.py $* =====" >> "$LOG"

"$PYTHON" run_cycle.py "$@" >> "$LOG" 2>&1
code=$?

echo "----- $(stamp) exit $code -----" >> "$LOG"

# Exit 2 is not a normal failure. run_cycle.py returns it when an in-flight
# order could not be accounted for, and its own docstring says to halt and
# investigate *before the next cycle*. A scheduler that simply runs again
# tomorrow is the one thing that must not happen: the position of record is
# unknown, and planning on top of an unknown book is how a reconciliation
# problem becomes a trading one.
#
# Raising the kill switch is the codebase's own mechanism for this
# (ExecutionPolicy.kill_switch_path). Every later cycle then aborts at
# preflight with "kill switch present" until a human looks and removes the
# file, which is exactly the intended behaviour -- loud, blocking, and
# requiring a decision rather than a retry.
if [ "$code" -eq 2 ]; then
    echo "HALT: unresolved in-flight order. Raising the kill switch so the" >> "$LOG"
    echo "      next scheduled cycle cannot trade until this is resolved." >> "$LOG"
    echo "      Investigate, then: rm $PROJECT/KILL" >> "$LOG"
    date > "$PROJECT/KILL"
fi

exit "$code"
