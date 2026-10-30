#!/usr/bin/env bash
# Per-page render latency, rasterrocket against pdfium, measured like for like:
# both in process, one open document per file, pages rendered sequentially on
# one pinned core, grayscale output.
#
# Usage: tests/compare/latency.sh [-r DPI] [-c CPU] PDF...
#
# Writes rr.tsv and pf.tsv (pdf, page, open_us, page_us) to $OUT
# (default: a fresh temporary directory) and prints a summary over the pages
# both renderers produced.  pdfium runs from the virtualenv described in
# compare.sh ($PDFIUM_VENV, default tests/compare/.venv).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
dpi=50
cpu=2
while getopts "r:c:" opt; do
    case $opt in
        r) dpi=$OPTARG ;;
        c) cpu=$OPTARG ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || { echo "usage: $0 [-r DPI] [-c CPU] PDF..." >&2; exit 2; }

venv=${PDFIUM_VENV:-$here/.venv}
out=${OUT:-$(mktemp -d)}

(cd "$root" && cargo build -q --release -p rasterrocket --example page_latency)
RAYON_NUM_THREADS=1 taskset -c "$cpu" \
    "$root/target/release/examples/page_latency" "$dpi" "$@" > "$out/rr.tsv" 2>/dev/null
taskset -c "$cpu" "$venv/bin/python" -I "$here/pdfium_latency.py" "$dpi" "$@" \
    > "$out/pf.tsv" 2>/dev/null
"$venv/bin/python" -I "$here/latency_summary.py" "$out/rr.tsv" "$out/pf.tsv"
echo "tables: $out"
