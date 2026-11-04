#!/usr/bin/env bash
# ab_archive.sh — page-level byte comparison of two rrocket builds over a corpus.
#
# Usage:
#   ab_archive.sh BEFORE AFTER OUTPREFIX [LIST]
#
# Every PDF named in LIST (one path per line) is rendered by both binaries at
# 50 dpi gray, first four pages, four threads.  Without LIST the corpus is
# the repository fixtures plus a fixed 150-file sample of the archive at
# $AB_ARCHIVE when that names a directory, and none otherwise.  The sample is reproducible: the
# paths are sorted in byte order and `shuf` draws from a constant random
# source.
#
# Environment:
#   BEFORE_FLAGS, AFTER_FLAGS  extra arguments for one binary only (word-split)
#   AB_TIMEOUT                 per-render limit in seconds (default 120)
#   AB_ARCHIVE                 archive root for the default sample
#
# Files written: OUTPREFIX.list (the corpus), OUTPREFIX.diffs (one line
# per difference) and OUTPREFIX.errors (the output of every render that
# exited non-zero, each under a "== <side> <status> <pdf>" line); renders go
# to a temporary directory beside them, removed on exit.  Differences are
# "PAGE <pdf> <page>" (pixels differ), "EXIT <pdf> <before> <after>" (exit
# statuses differ) and "COUNT <pdf>" (page counts differ though the exit
# statuses agree; an EXIT line already covers the pages a failure leaves out).
# Only pages present on both sides are compared.  A file both builds
# fail on with the same status prints "FAILED <pdf> <status>: <first line of
# AFTER's output>" on standard output.  The summary line counts the page
# pairs present on both sides and compared, and the differences.
#
# Exit status: 0 when nothing differs and at least one page pair was
# compared; 1 on any difference or when nothing was compared; 2 on bad usage.
# Requires bash >= 4.4, and `shuf` or `gshuf` for the archive sample;
# `timeout` or `gtimeout` is used when present.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=ab_lib.sh
source "$(dirname "$0")/ab_lib.sh"

(($# >= 3 && $# <= 4)) || ab_usage
before=$1 after=$2 prefix=$3 list=${4-}
ab_require_bins "$before" "$after"
repo=$(cd "$(dirname "$0")/../.." && pwd)
archive=${AB_ARCHIVE:-}

if [[ -n $list ]]; then
    [[ -r $list ]] || ab_usage_error "cannot read list: $list"
    [[ $list -ef $prefix.list ]] || cp "$list" "$prefix.list"
else
    {
        printf '%s\n' "$repo"/tests/fixtures/*.pdf
        if [[ -n $archive && -d $archive ]]; then
            shuffle=$(command -v shuf || command -v gshuf) ||
                ab_usage_error "the archive sample needs shuf or gshuf"
            # Byte-order sort and NUL separators: the same input order, and
            # so the same sample, in every locale.  LIST holds one path per
            # line, so paths containing a newline are left out.
            find "$archive" -iname '*.pdf' ! -path $'*\n*' -print0 | LC_ALL=C sort -z |
                "$shuffle" -z -n 150 --random-source=<(yes) | tr '\0' '\n'
        fi
    } >"$prefix.list"
fi

ab_flags
ab_limiter "${AB_TIMEOUT:-120}"

work=$(mktemp -d "$prefix.work.XXXXXX")
trap 'rm -rf "$work"' EXIT

# render TAG BIN PDF [FLAG...] renders PDF into $work/<TAG>-NNN.pgm and
# stores the exit status in $work/<TAG>.status; a non-zero exit appends the
# render's output to $errors.
render() {
    local tag=$1 bin=$2 pdf=$3 status=0
    shift 3
    "${limiter[@]}" "$bin" --no-ram -r 50 --gray -l 4 --threads 4 "$@" "$pdf" "$work/$tag" \
        </dev/null >"$work/$tag.err" 2>&1 || status=$?
    echo "$status" >"$work/$tag.status"
    if ((status != 0)); then
        echo "== $tag $status $pdf" >>"$errors"
        cat "$work/$tag.err" >>"$errors"
    fi
}

diffs=$prefix.diffs errors=$prefix.errors
: >"$diffs"
: >"$errors"
scanned=0 compared=0
while IFS= read -r pdf; do
    [[ -n $pdf ]] || continue
    scanned=$((scanned + 1))
    rm -f "$work"/*
    render B "$before" "$pdf" "${before_flags[@]}"
    render A "$after" "$pdf" "${after_flags[@]}"
    shopt -s nullglob
    b_pages=("$work"/B-*.pgm) a_pages=("$work"/A-*.pgm)
    shopt -u nullglob
    for page in "${b_pages[@]}"; do
        [[ -e $work/A-${page##*/B-} ]] || continue
        compared=$((compared + 1))
        cmp -s "$page" "$work/A-${page##*/B-}" || echo "PAGE $pdf ${page##*-}" >>"$diffs"
    done
    b_status=$(<"$work/B.status") a_status=$(<"$work/A.status")
    if [[ $b_status != "$a_status" ]]; then
        echo "EXIT $pdf $b_status $a_status" >>"$diffs"
    else
        ((${#b_pages[@]} == ${#a_pages[@]})) || echo "COUNT $pdf" >>"$diffs"
        [[ $b_status == 0 ]] || echo "FAILED $pdf $b_status: $(head -n 1 "$work/A.err")"
    fi
done <"$prefix.list"

count=$(wc -l <"$diffs" | tr -d ' ')
echo "scanned $scanned files, $compared pages compared; $count differences"
head -20 "$diffs"
if ((compared == 0)); then
    echo "ab_archive.sh: no page was compared; see $errors" >&2
    exit 1
fi
((count == 0))
