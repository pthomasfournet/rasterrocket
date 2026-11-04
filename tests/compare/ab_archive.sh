#!/usr/bin/env bash
# ab_archive.sh — page-level byte comparison of two rrocket builds over a corpus.
#
# Usage:
#   ab_archive.sh BEFORE AFTER OUTPREFIX [LIST]
#
# Every PDF named in LIST (one path per line) is rendered by both binaries at
# 50 dpi gray, first four pages, four threads.  Without LIST the corpus is
# the repository fixtures plus a fixed 150-file sample of the archive at
# $AB_ARCHIVE (default ~/mss-content/archives); the archive part is skipped
# when that directory does not exist.  The sample is reproducible: `shuf`
# draws from a constant random source.
#
# Environment:
#   BEFORE_FLAGS, AFTER_FLAGS  extra arguments for one binary only (word-split)
#   AB_TIMEOUT                 per-render limit in seconds (default 120)
#   AB_ARCHIVE                 archive root for the default sample
#
# Files written: OUTPREFIX.list (the corpus), OUTPREFIX.diffs (one line per
# difference) and the scratch directory OUTPREFIX.  Differences are
# "PAGE <pdf> <page>" (pixels differ), "COUNT <pdf>" (page counts differ) and
# "EXIT <pdf> <before> <after>" (exit statuses differ).
#
# Exit status: 0 when nothing differs, 1 otherwise, 2 on bad usage.
# Requires bash >= 4; `timeout` or `gtimeout` is used when present.
set -euo pipefail

if (($# < 3 || $# > 4)); then
    sed -n '2,25p' "$0" >&2
    exit 2
fi
before=$1 after=$2 prefix=$3 list=${4-}
for bin in "$before" "$after"; do
    [[ -x $bin ]] || { echo "ab_archive.sh: not executable: $bin" >&2; exit 2; }
done
repo=$(cd "$(dirname "$0")/../.." && pwd)
archive=${AB_ARCHIVE:-$HOME/mss-content/archives}

if [[ -n $list ]]; then
    [[ -r $list ]] || { echo "ab_archive.sh: cannot read list: $list" >&2; exit 2; }
    cp "$list" "$prefix.list"
else
    {
        ls "$repo"/tests/fixtures/*.pdf
        if [[ -d $archive ]]; then
            find "$archive" -iname '*.pdf' | sort | shuf -n 150 --random-source=<(yes)
        fi
    } >"$prefix.list"
fi

read -r -a before_flags <<<"${BEFORE_FLAGS-}"
read -r -a after_flags <<<"${AFTER_FLAGS-}"
limit=${AB_TIMEOUT:-120}
if command -v timeout >/dev/null; then
    limiter=(timeout "$limit")
elif command -v gtimeout >/dev/null; then
    limiter=(gtimeout "$limit")
else
    limiter=()
fi

# Renders into $prefix/<tag>-NNN.pgm and stores the exit status in $prefix/<tag>.status.
render() {
    local tag=$1 bin=$2 status=0
    shift 2
    "${limiter[@]}" "$bin" --no-ram -r 50 --gray -l 4 --threads 4 "$@" "$pdf" "$prefix/$tag" \
        >"$prefix/$tag.err" 2>&1 || status=$?
    echo "$status" >"$prefix/$tag.status"
}

diffs=$prefix.diffs
: >"$diffs"
scanned=0
while IFS= read -r pdf; do
    [[ -n $pdf ]] || continue
    scanned=$((scanned + 1))
    rm -rf "$prefix"
    mkdir -p "$prefix"
    render B "$before" "${before_flags[@]}"
    render A "$after" "${after_flags[@]}"
    shopt -s nullglob
    b_pages=("$prefix"/B-*.pgm) a_pages=("$prefix"/A-*.pgm)
    shopt -u nullglob
    for page in "${b_pages[@]}"; do
        cmp -s "$page" "${page/\/B-//A-}" || echo "PAGE $pdf ${page##*-}" >>"$diffs"
    done
    ((${#b_pages[@]} == ${#a_pages[@]})) || echo "COUNT $pdf" >>"$diffs"
    b_status=$(<"$prefix/B.status") a_status=$(<"$prefix/A.status")
    [[ $b_status == "$a_status" ]] || echo "EXIT $pdf $b_status $a_status" >>"$diffs"
done <"$prefix.list"

count=$(wc -l <"$diffs" | tr -d ' ')
echo "scanned $scanned files; $count differences"
head -20 "$diffs"
((count == 0))
