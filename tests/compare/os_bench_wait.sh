#!/usr/bin/env bash
# Linux-side waiter for a run of tests/compare/os_bench.ps1 started with
# -Launch: polls the run's result directory and its scheduled task over ssh
# until the run ends.
#
# Usage:
#   tests/compare/os_bench_wait.sh HOST DIR [TIMEOUT_S]
#
#   HOST       ssh destination of the Windows machine, whose shell runs
#              `powershell` (Windows PowerShell 5.1).
#   DIR        the run's -Out directory as an absolute Windows path, e.g.
#              C:\Users\NAME\rr-bench\smoke (no quote, $ or backtick).
#   TIMEOUT_S  seconds to wait in all (default 14400).
#
# Every 60 s one ssh call tests DIR\failed.flag, then DIR\done.flag, then the
# state of the task rr-os-bench.  Exit status:
#   0    done.flag exists;
#   2    failed.flag exists (printed), or bad usage;
#   3    DIR does not exist on HOST;
#   4    neither flag exists 5 s after the task was found not running or not
#        registered (its state and last result are printed): the run ended
#        without a flag, or never started;
#   124  TIMEOUT_S passed first.
# An ssh call that fails (the machine unreachable, or the call itself past
# 120 s) is reported on stderr and the polling goes on until the deadline.
set -Eeuo pipefail

poll_s=60
ssh_limit_s=120
task=rr-os-bench

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
[[ $dir != *[\'\"\$\`]* ]] || { echo "DIR may not hold quotes, \$ or backticks: $dir" >&2; exit 2; }

# The remote check, as one PowerShell script; DIR sits in single quotes,
# where PowerShell expands nothing.
check=$(
    cat << EOF
\$d = '$dir'
if (-not (Test-Path -LiteralPath \$d -PathType Container)) { exit 3 }
function Test-Flag {
    if (Test-Path -LiteralPath "\$d\\failed.flag") { Get-Content -LiteralPath "\$d\\failed.flag"; exit 2 }
    if (Test-Path -LiteralPath "\$d\\done.flag") { exit 0 }
}
Test-Flag
\$t = Get-ScheduledTask -TaskName '$task' -ErrorAction SilentlyContinue
if (\$t -and @('Running', 'Queued') -contains [string]\$t.State) { exit 1 }
Start-Sleep -Seconds 5
Test-Flag
if (\$t) {
    \$i = \$t | Get-ScheduledTaskInfo
    'task $task is ' + \$t.State + ', last result 0x' + ('{0:X}' -f \$i.LastTaskResult)
} else {
    'task $task is not registered'
}
exit 4
EOF
)
# -EncodedCommand takes UTF-16LE in base64, so no quoting reaches either shell.
encoded=$(printf '%s' "$check" | iconv -f UTF-8 -t UTF-16LE | base64 -w 0)

deadline=$((SECONDS + timeout_s))
while :; do
    rc=0
    said=$(timeout "$ssh_limit_s" ssh -o BatchMode=yes -o ConnectTimeout=30 "$host" \
        "powershell -NoProfile -NonInteractive -EncodedCommand $encoded; exit \$LASTEXITCODE" 2> /dev/null) || rc=$?
    case $rc in
        0)
            printf 'os_bench_wait.sh: %s\\done.flag found\n' "$dir"
            exit 0
            ;;
        1) ;;
        2)
            printf 'os_bench_wait.sh: %s\\failed.flag found:\n%s\n' "$dir" "$said" >&2
            exit 2
            ;;
        3)
            printf 'os_bench_wait.sh: %s does not exist on %s\n' "$dir" "$host" >&2
            exit 3
            ;;
        4)
            printf 'os_bench_wait.sh: no flag in %s and the run is over: %s\n' "$dir" "$said" >&2
            exit 4
            ;;
        *) echo "os_bench_wait.sh: ssh $host exited $rc; polling on" >&2 ;;
    esac
    left=$((deadline - SECONDS))
    if ((left <= 0)); then
        echo "os_bench_wait.sh: no flag in $dir after $timeout_s s" >&2
        exit 124
    fi
    sleep $((left < poll_s ? left : poll_s))
done
