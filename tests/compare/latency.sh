#!/usr/bin/env bash
# Per-page render latency, rasterrocket against pdfium, measured like for like:
# both in process, one open document per file, pages rendered sequentially on
# one pinned core, grayscale output.
#
# Usage: tests/compare/latency.sh [-r DPI] [-c CPU] PDF...
#
# Writes rr.tsv and pf.tsv (pdf, page, open_us, page_us) to $OUT
# (default: a fresh temporary directory) and prints a summary over the pages
# both renderers produced.  The documents and pages a renderer skipped are
# listed in rr.log and pf.log beside the tables.  pdfium runs from the
# virtualenv described in compare.sh ($PDFIUM_VENV, default
# tests/compare/.venv).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
dpi=50
cpu=2
usage() { echo "usage: $0 [-r DPI] [-c CPU] PDF..." >&2; exit 2; }
while getopts "r:c:" opt; do
    case $opt in
        r) dpi=$OPTARG ;;
        c) cpu=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -gt 0 ]] || usage
[[ "$dpi" =~ ^[1-9][0-9]*$ ]] || { echo "-r DPI must be a positive integer, got: $dpi" >&2; exit 2; }
[[ "$cpu" =~ ^[0-9]+$ ]] || { echo "-c CPU must be a CPU index, got: $cpu" >&2; exit 2; }

venv=${PDFIUM_VENV:-$here/.venv}
[[ -x "$venv/bin/python" ]] \
    || { echo "no virtualenv at $venv: create it (see compare.sh -h) or set PDFIUM_VENV" >&2; exit 2; }
out=${OUT:-$(mktemp -d)}
mkdir -p "$out"

# run NAME CMD...: CMD's table to $out/NAME.tsv and its diagnostics to
# $out/NAME.log.  A failing CMD prints the log and stops the script.
run() {
    local name=$1
    shift
    if ! "$@" > "$out/$name.tsv" 2> "$out/$name.log"; then
        echo "$name: the latency run failed:" >&2
        cat "$out/$name.log" >&2
        exit 1
    fi
    if [[ -s "$out/$name.log" ]]; then
        echo "$name: $(grep -c . "$out/$name.log") diagnostic line(s) in $out/$name.log" >&2
    fi
}

(cd "$root" && cargo build -q --release -p rasterrocket --example page_latency)
run rr env RAYON_NUM_THREADS=1 taskset -c "$cpu" \
    "$root/target/release/examples/page_latency" "$dpi" "$@"
run pf taskset -c "$cpu" "$venv/bin/python" -I "$here/pdfium_latency.py" "$dpi" "$@"
"$venv/bin/python" -I "$here/latency_summary.py" "$out/rr.tsv" "$out/pf.tsv"
echo "tables: $out"
