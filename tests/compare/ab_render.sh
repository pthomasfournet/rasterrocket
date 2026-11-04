#!/usr/bin/env bash
# ab_render.sh — byte comparison of two rrocket builds (or two backends) over PDFs.
#
# Usage:
#   ab_render.sh BEFORE AFTER OUTDIR [--] PDF...
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
# counts, then "<N> outputs compared, <M> differ", where N counts the file
# pairs present on both sides and M the DIFF and COUNT lines.  A render that
# fails or times out is not an error by itself: it is compared through its
# output count, so a failure in only one binary shows up as COUNT.
#
# Exit status: 0 when nothing differs, 1 on any DIFF or COUNT, 2 on bad usage.
# Requires bash >= 4 (macOS: Homebrew bash); `timeout` or `gtimeout` is used
# when present.
set -euo pipefail

if (($# < 4)); then
    sed -n '2,25p' "$0" >&2
    exit 2
fi
before=$1 after=$2 out=$3
shift 3
[[ ${1-} == -- ]] && shift
(($# > 0)) || { echo "ab_render.sh: no PDF given" >&2; exit 2; }
for bin in "$before" "$after"; do
    [[ -x $bin ]] || { echo "ab_render.sh: not executable: $bin" >&2; exit 2; }
done

read -r -a before_flags <<<"${BEFORE_FLAGS-}"
read -r -a after_flags <<<"${AFTER_FLAGS-}"
limit=${AB_TIMEOUT:-300}
if command -v timeout >/dev/null; then
    limiter=(timeout "$limit")
elif command -v gtimeout >/dev/null; then
    limiter=(gtimeout "$limit")
else
    limiter=()
fi

# Runs one render; a non-zero exit is recorded through the output count.
render() {
    local bin=$1 dest=$2
    shift 2
    "${limiter[@]}" "$bin" --no-ram -r 50 -l 3 "$@" "$dest/p" >/dev/null 2>&1 || true
}

modes=("--png" "--jpeg" "--gray --png" "--mono --png" "--gray")
compared=0 differ=0
for pdf in "$@"; do
    for mode in "${modes[@]}"; do
        read -r -a mode_flags <<<"$mode"
        rm -rf "$out"
        mkdir -p "$out/b" "$out/a"
        render "$before" "$out/b" "${before_flags[@]}" "${mode_flags[@]}" "$pdf"
        render "$after" "$out/a" "${after_flags[@]}" "${mode_flags[@]}" "$pdf"
        shopt -s nullglob
        b_files=("$out"/b/*) a_files=("$out"/a/*)
        shopt -u nullglob
        for file in "${b_files[@]}"; do
            [[ -e $out/a/${file##*/} ]] || continue
            compared=$((compared + 1))
            if ! cmp -s "$file" "$out/a/${file##*/}"; then
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
((differ == 0))
