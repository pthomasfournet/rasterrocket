#!/usr/bin/env bash
# ab_render.sh — byte comparison of two rrocket builds (or two backends) over PDFs.
#
# Usage:
#   ab_render.sh BEFORE AFTER OUTDIR [--] PDF...
#
# OUTDIR is created if missing; the renders go to a fresh temporary
# directory inside it, removed on exit, so nothing else in OUTDIR is touched.
#
# Each PDF is rendered by both binaries at 50 dpi, first three pages, in five
# output modes: RGB PNG, JPEG, gray PNG, mono PNG and gray PGM.  Every output
# file of BEFORE is compared byte for byte with the same-named file of AFTER.
#
# Environment:
#   BEFORE_FLAGS  extra arguments for BEFORE only (word-split)
#   AFTER_FLAGS   extra arguments for AFTER only, e.g. "--backend vulkan"
#   AB_TIMEOUT    per-render limit in seconds (default 300)
#
# Output: one "DIFF <pdf> <mode> <file>" line per differing file, one
# "COUNT <pdf> <mode> <n> vs <m>" line when the two produced different page
# counts, one "EXIT <pdf> <mode> <s> vs <t>" line when their exit statuses
# differ, one "FAILED <pdf> <mode> <s>" line when both failed alike, then
# "<N> outputs compared, <M> differ", where N counts the file pairs present
# on both sides and M the DIFF, COUNT and EXIT lines.
#
# Exit status: 0 when nothing differs and at least one output was compared;
# 1 on any difference or when nothing was compared; 2 on bad usage.
# Requires bash >= 4.4 (macOS: Homebrew bash); `timeout` or `gtimeout` is
# used when present.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=ab_lib.sh
source "$(dirname "$0")/ab_lib.sh"

(($# >= 4)) || ab_usage
before=$1 after=$2 out=$3
shift 3
[[ ${1-} == -- ]] && shift
(($# > 0)) || ab_usage_error "no PDF given"
ab_require_bins "$before" "$after"

ab_flags
ab_limiter "${AB_TIMEOUT:-300}"

mkdir -p "$out"
work=$(mktemp -d "$out/ab.XXXXXX")
trap 'rm -rf "$work"' EXIT

# Runs one render into DEST and prints its exit status.
render() {
    local bin=$1 dest=$2 status=0
    shift 2
    "${limiter[@]}" "$bin" --no-ram -r 50 -l 3 "$@" "$dest/p" </dev/null >/dev/null 2>&1 || status=$?
    echo "$status"
}

modes=("--png" "--jpeg" "--gray --png" "--mono --png" "--gray")
compared=0 differ=0
for pdf in "$@"; do
    for mode in "${modes[@]}"; do
        read -r -a mode_flags <<<"$mode"
        rm -rf "$work/b" "$work/a"
        mkdir -p "$work/b" "$work/a"
        b_status=$(render "$before" "$work/b" "${before_flags[@]}" "${mode_flags[@]}" "$pdf")
        a_status=$(render "$after" "$work/a" "${after_flags[@]}" "${mode_flags[@]}" "$pdf")
        if [[ $b_status != "$a_status" ]]; then
            differ=$((differ + 1))
            echo "EXIT $pdf $mode $b_status vs $a_status"
        elif [[ $b_status != 0 ]]; then
            echo "FAILED $pdf $mode $b_status"
        fi
        shopt -s nullglob
        b_files=("$work"/b/*) a_files=("$work"/a/*)
        shopt -u nullglob
        for file in "${b_files[@]}"; do
            [[ -e $work/a/${file##*/} ]] || continue
            compared=$((compared + 1))
            if ! cmp -s "$file" "$work/a/${file##*/}"; then
                differ=$((differ + 1))
                echo "DIFF $pdf $mode ${file##*/}"
            fi
        done
        if ((${#b_files[@]} != ${#a_files[@]})); then
            differ=$((differ + 1))
            echo "COUNT $pdf $mode ${#b_files[@]} vs ${#a_files[@]}"
        fi
    done
done
echo "$compared outputs compared, $differ differ"
if ((compared == 0)); then
    echo "ab_render.sh: no output was compared" >&2
    exit 1
fi
((differ == 0))
