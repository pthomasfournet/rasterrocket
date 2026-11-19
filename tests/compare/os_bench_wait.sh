#!/usr/bin/env bash
# Linux-side waiter for a Windows run of tests/compare/os_bench.ps1: polls the
# run's result directory over ssh until done.flag or failed.flag appears.
#
# Usage:
#   tests/compare/os_bench_wait.sh HOST DIR [TIMEOUT_S]
#
#   HOST       ssh destination of the Windows machine, whose shell runs
#              `powershell` (Windows PowerShell 5.1).
#   DIR        the run's -Out directory as an absolute Windows path, e.g.
#              C:\Users\NAME\rr-bench\smoke (no single quote).
#   TIMEOUT_S  seconds to wait in all (default 14400).
#
# Every 60 s one ssh call tests DIR\failed.flag, then DIR\done.flag.  Exit
# status: 0 when done.flag exists; 2 when failed.flag exists, after printing
# it; 124 when TIMEOUT_S passes first; 2 on bad usage.  An ssh call that
# fails (the machine unreachable, or the call itself past 120 s) is reported
# on stderr and the polling goes on until the deadline.
set -Eeuo pipefail

poll_s=60
ssh_limit_s=120

usage() {
    sed -n '/^# Usage:/,/^# Every 60 s/p' "$0" | sed '$d; s/^# \{0,1\}//' >&2
    exit 2
}

(($# == 2 || $# == 3)) || usage
host=$1
dir=${2%\\}
timeout_s=${3:-14400}
[[ $timeout_s =~ ^[1-9][0-9]*$ ]] || { echo "TIMEOUT_S must be a positive integer, got: $timeout_s" >&2; exit 2; }
[[ $dir =~ ^[A-Za-z]:\\ ]] || { echo "DIR must be an absolute Windows path, got: $dir" >&2; exit 2; }
[[ $dir != *"'"* && $dir != *'"'* ]] || { echo "DIR may not hold quotes: $dir" >&2; exit 2; }

# remote PS: run the PowerShell command PS on HOST; ssh's status is the
# command's exit code (the trailing exit passes it through the login shell).
remote() {
    timeout "$ssh_limit_s" ssh -o BatchMode=yes -o ConnectTimeout=30 "$host" \
        "powershell -NoProfile -Command \"$1\"; exit \$LASTEXITCODE"
}

test_flags="if (Test-Path -LiteralPath '$dir\\failed.flag') { exit 2 } elseif (Test-Path -LiteralPath '$dir\\done.flag') { exit 0 } else { exit 1 }"
deadline=$((SECONDS + timeout_s))
while :; do
    rc=0
    remote "$test_flags" 2> /dev/null || rc=$?
    case $rc in
        0)
            printf 'os_bench_wait.sh: %s\\done.flag found\n' "$dir"
            exit 0
            ;;
        2)
            printf 'os_bench_wait.sh: %s\\failed.flag found:\n' "$dir" >&2
            remote "Get-Content -LiteralPath '$dir\\failed.flag'" >&2 || true
            exit 2
            ;;
        1) ;;
        *) echo "os_bench_wait.sh: ssh $host exited $rc; polling on" >&2 ;;
    esac
    left=$((deadline - SECONDS))
    if ((left <= 0)); then
        echo "os_bench_wait.sh: no flag in $dir after $timeout_s s" >&2
        exit 124
    fi
    sleep $((left < poll_s ? left : poll_s))
done
