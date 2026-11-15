#!/usr/bin/env bash
# trace_blame.sh — attribute the pixel differences between two rrocket builds
# on one page to the paint operations that cover them.
#
# Usage:
#   trace_blame.sh BEFORE AFTER PDF PAGE [DPI]
#
# Renders PAGE of PDF with both binaries as PPM at DPI (default 50, the
# resolution of ab_render.sh), records the page's paint trace at the same
# resolution with the `paint_trace` dev binary (built from this checkout:
# cargo build --release -p bench --bin paint_trace), and prints the
# operations that cover the differing pixels, most pixels first.
#
# Environment:
#   PAINT_TRACE  the paint_trace binary (default target/release/paint_trace)
#
# Exit status: 0 when the renders are identical, 1 when they differ, 2 on
# bad usage or a failed render.
set -euo pipefail

usage() {
    sed -n '3,6p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

(($# == 4 || $# == 5)) || usage
before=$1 after=$2 pdf=$3 page=$4 dpi=${5:-50}
root=$(cd "$(dirname "$0")/../.." && pwd)
tool=${PAINT_TRACE:-$root/target/release/paint_trace}
for bin in "$before" "$after" "$tool"; do
    [[ -x $bin ]] || { echo "trace_blame.sh: $bin is not an executable" >&2; exit 2; }
done
[[ $page =~ ^[1-9][0-9]*$ ]] || { echo "trace_blame.sh: PAGE must be a page number" >&2; exit 2; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Renders PAGE with binary $1 into directory $2 and prints the one PPM written.
render() {
    mkdir -p "$2"
    "$1" --no-ram -r "$dpi" -f "$page" -l "$page" "$pdf" "$2/p" </dev/null >/dev/null 2>&1 ||
        { echo "trace_blame.sh: $1 failed to render $pdf page $page" >&2; exit 2; }
    shopt -s nullglob
    local files=("$2"/*.ppm)
    shopt -u nullglob
    ((${#files[@]} == 1)) || { echo "trace_blame.sh: $1 wrote ${#files[@]} PPM files" >&2; exit 2; }
    echo "${files[0]}"
}

b=$(render "$before" "$work/before")
a=$(render "$after" "$work/after")
"$tool" dump "$pdf" -p "$page" -r "$dpi" >"$work/page.trace"
status=0
"$tool" blame "$work/page.trace" "$b" "$a" || status=$?
exit "$status"
