#!/usr/bin/env bash
# compare.sh — pixel-accurate comparison of rrocket against reference renderers.
#
# Usage:
#   compare.sh [OPTIONS] <PDF>
#   compare.sh [OPTIONS] <DIR>      every *.pdf under DIR, one summary row each
#
# Options:
#   -r DPI          Resolution in DPI (default: 150)
#   -f PAGE         First page, 1-based (default: 1)
#   -l PAGE         Last page, 1-based (default: all)
#   -t THRESHOLD    Max allowed RMSE per page, 0-255 scale (default: 2.0)
#   -k TILE_DELTA   Max allowed tile delta per page, 0-255 scale (default: 12)
#   -R RENDERER     Reference renderer: pdftoppm (default), pdfium or oxide
#   -v              Verbose: print per-page stats even when passing
#   -o DIR          Write diff images to DIR (default: discarded)
#   -n              Dry run: print commands without executing
#   -s              Print one tab-separated summary row instead of the report
#
# A page fails when its RMSE exceeds -t, when its tile delta exceeds -k, or
# when the two renders differ in size by more than 1 px in either dimension
# (reported as SIZE: a page-box or rotation disagreement).  Renders within
# 1 px of each other are cropped to their common size before scoring.
#
# The tile delta (scored by score_pair.sh) is the largest difference in mean
# luminance between the two renders over 32 x 32 px tiles.  Anti-aliasing
# and hinting differences between renderers move coverage within a tile but
# hardly change its mean (text pages score 4-6 against pdftoppm at 150 dpi),
# while content present in one render and missing from the other changes it
# wholesale (one dropped word scores ~26) and barely moves the page-wide RMSE.
#
# Summary row (-s, and each row of directory mode), tab-separated:
#   status pages failed size_mismatches avg_rmse max_rmse max_tile_delta
#   rrocket_stderr_lines pdf
#
# Exit code: 0 = all pages within threshold; 1 = failures or errors.
# Dry-run always exits 0.
#
# Several independent references are supported so that a disagreement can be
# localised: when the references agree and rrocket differs, the defect is
# ours. `pdftoppm` is invoked with -cropbox because rrocket renders the
# CropBox; pdftoppm defaults to the MediaBox, and without the flag any page
# whose CropBox is smaller than its MediaBox diffs on geometry alone.
# pdfium renders the CropBox by default. It runs from a virtualenv holding
# pypdfium2 and pillow, by default tests/compare/.venv:
#   uv venv tests/compare/.venv
#   uv pip install --python tests/compare/.venv/bin/python pypdfium2 pillow
# Set PDFIUM_VENV to use another one.
# The oxide reference is an independent Rust implementation; set
# PDF_OXIDE_BIN to override discovery.
#
# Requires: pdftoppm (or the pdfium venv, or pdf-oxide), rrocket (release
#           build), ImageMagick (compare, identify, convert), bc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RASTER_BIN="${REPO_ROOT}/target/release/rrocket"
OXIDE_BIN="${PDF_OXIDE_BIN:-pdf-oxide}"
PDFIUM_VENV="${PDFIUM_VENV:-${SCRIPT_DIR}/.venv}"
PDFIUM_BIN="${PDFIUM_VENV}/bin/pypdfium2"
PDFIUM_PY="${PDFIUM_VENV}/bin/python"

# ── defaults ──────────────────────────────────────────────────────────────────
DPI=150
FIRST=1
LAST=""
THRESHOLD="2.0"
TILE_LIMIT="12"
REFERENCE="pdftoppm"
VERBOSE=false
OUT_DIR=""
DRY_RUN=false
SUMMARY=false

# The header comment block, from the line after the shebang to the first
# line that is not a comment; exits with status $1 (default 0).
usage() {
    awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"
    exit "${1:-0}"
}

die() { echo "Error: $*" >&2; exit 1; }

is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_pos_number()   { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

while getopts ":r:f:l:t:k:R:vo:nsh" opt; do
    case $opt in
        r) DPI="$OPTARG" ;;
        f) FIRST="$OPTARG" ;;
        l) LAST="$OPTARG" ;;
        t) THRESHOLD="$OPTARG" ;;
        k) TILE_LIMIT="$OPTARG" ;;
        R) REFERENCE="$OPTARG" ;;
        v) VERBOSE=true ;;
        o) OUT_DIR="$OPTARG" ;;
        n) DRY_RUN=true ;;
        s) SUMMARY=true ;;
        h) usage ;;
        :) die "option -$OPTARG requires an argument." ;;
       \?) die "unknown option -$OPTARG." ;;
    esac
done
shift $((OPTIND - 1))

# ── input validation ──────────────────────────────────────────────────────────
[[ $# -lt 1 ]] && { echo "Error: PDF argument required." >&2; usage 1 >&2; }
PDF="$1"

is_positive_int "$DPI"   || die "-r DPI must be a positive integer, got: $DPI"
is_positive_int "$FIRST" || die "-f PAGE must be a positive integer, got: $FIRST"
[[ -z "$LAST" ]] || is_positive_int "$LAST" || die "-l PAGE must be a positive integer, got: $LAST"
is_pos_number "$THRESHOLD" || die "-t THRESHOLD must be a non-negative number, got: $THRESHOLD"
is_pos_number "$TILE_LIMIT" || die "-k TILE_DELTA must be a non-negative number, got: $TILE_LIMIT"
[[ -n "$LAST" && "$LAST" -lt "$FIRST" ]] && die "-l ($LAST) must be >= -f ($FIRST)"
case "$REFERENCE" in
    pdftoppm|pdfium|oxide) ;;
    *) die "-R must be pdftoppm, pdfium or oxide, got: $REFERENCE" ;;
esac

# ── directory mode: one summary row per PDF ───────────────────────────────────
if [[ -d "$PDF" ]]; then
    file_opts=(-s -r "$DPI" -f "$FIRST" -t "$THRESHOLD" -k "$TILE_LIMIT" -R "$REFERENCE")
    [[ -n "$LAST" ]] && file_opts+=(-l "$LAST")
    mapfile -d '' pdfs < <(find "$PDF" -type f -iname '*.pdf' -print0 | sort -z)
    [[ ${#pdfs[@]} -gt 0 ]] || die "no PDF files under $PDF"
    $DRY_RUN && { printf '%s\n' "${pdfs[@]}"; exit 0; }
    printf "status\tpages\tfailed\tsize\tavg_rmse\tmax_rmse\ttile\tstderr\tpdf\n"
    any_fail=0
    for pdf in "${pdfs[@]}"; do
        opts=("${file_opts[@]}")
        # Diff images of each PDF go to their own subdirectory.
        [[ -n "$OUT_DIR" ]] && opts+=(-o "${OUT_DIR}/$(basename "$pdf" .pdf)")
        row="$("$0" "${opts[@]}" "$pdf" 2>/dev/null)" || any_fail=1
        # A run that stopped before scoring prints no row.
        if [[ -n "$row" ]]; then
            printf '%s\n' "$row"
        else
            printf "ERROR\t-\t-\t-\t-\t-\t-\t-\t%s\n" "$pdf"
        fi
    done
    exit "$any_fail"
fi

[[ -f "$PDF" ]] || die "file not found: $PDF"

# ── build argument lists early (dry-run needs them) ───────────────────────────
# -cropbox: rrocket renders the CropBox, pdftoppm defaults to the MediaBox.
ref_args=(-cropbox -r "$DPI" -f "$FIRST")
[[ -n "$LAST" ]] && ref_args+=(-l "$LAST")
# pdf-oxide takes an inclusive page range rather than -f/-l, and rejects an
# open-ended "N-". The end page is filled in from the document's page count
# at render time; omitting --pages renders every page.
oxide_args=(render --dpi "$DPI" --format png)
# pypdfium2 takes a scale in pixels per point and, like pdf-oxide, an
# inclusive page range; the X channel is dropped so the PNG is plain RGB.
pdfium_args=(render --scale "$(bc -l <<< "${DPI} / 72")" --format png --no-x-channel --prefix page_)
new_args=(-r "$DPI" -f "$FIRST")
[[ -n "$LAST" ]] && new_args+=(-l "$LAST")

# ── dry-run ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "DRY RUN — commands that would be executed:"
    echo ""
    if [[ "$REFERENCE" == oxide ]]; then
        echo "  [render ref]  $OXIDE_BIN ${oxide_args[*]} -o <OUT>/ref $PDF"
    elif [[ "$REFERENCE" == pdfium ]]; then
        echo "  [render ref]  $PDFIUM_BIN ${pdfium_args[*]} --pages ${FIRST}-${LAST:-<N>} -o <OUT>/ref $PDF"
    else
        echo "  [render ref]  pdftoppm ${ref_args[*]} $PDF <OUT>/ref/page"
    fi
    echo "  [render new]  $RASTER_BIN ${new_args[*]} $PDF <OUT>/new/page"
    echo "  [per page]    compare -metric RMSE <OUT>/ref/page-NNN.ppm \\"
    echo "                        <OUT>/new/page-NNN.ppm <DIFF>/page-NNN-diff.png"
    echo "  [per page]    convert <ref> <new> -colorspace gray -filter box -resize <tiles>! \\"
    echo "                        -compose difference -composite   (tile delta)"
    echo ""
    echo "  Settings: DPI=${DPI}  first=${FIRST}  last=${LAST:-all}  threshold=${THRESHOLD}/255  tile=${TILE_LIMIT}/255  ref=${REFERENCE}"
    [[ -n "$OUT_DIR" ]] && echo "  Diff dir: ${OUT_DIR}"
    exit 0
fi

# ── dependency check ──────────────────────────────────────────────────────────
for cmd in compare identify convert bc; do
    command -v "$cmd" >/dev/null 2>&1 || die "$cmd not found in PATH"
done
if [[ "$REFERENCE" == oxide ]]; then
    command -v "$OXIDE_BIN" >/dev/null 2>&1 \
        || die "$OXIDE_BIN not found in PATH — set PDF_OXIDE_BIN to its location"
elif [[ "$REFERENCE" == pdfium ]]; then
    [[ -x "$PDFIUM_BIN" ]] \
        || die "pypdfium2 not found at $PDFIUM_BIN — create the venv (see -h) or set PDFIUM_VENV"
else
    command -v pdftoppm >/dev/null 2>&1 || die "pdftoppm not found in PATH"
fi
[[ -x "$RASTER_BIN" ]] || {
    echo "Error: rrocket not built at $RASTER_BIN" >&2
    echo "  Run: cargo build --release -p rasterrocket-cli" >&2
    exit 1
}

# ── temp workspace ────────────────────────────────────────────────────────────
WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

REF_DIR="${WORK_DIR}/ref"
NEW_DIR="${WORK_DIR}/new"
DIFF_DIR="${OUT_DIR:-${WORK_DIR}/diff}"
mkdir -p "$REF_DIR" "$NEW_DIR" "$DIFF_DIR"

# ── render: reference ─────────────────────────────────────────────────────────
if [[ "$REFERENCE" == oxide ]]; then
    # pdf-oxide names outputs "<stem>_<n>.png" with n 1-based over the whole
    # document, and needs an explicit end page. Render into a staging dir and
    # normalise to the page-NNN.ppm layout the diff loop below expects.
    OX_RAW="${WORK_DIR}/ox_raw"
    mkdir -p "$OX_RAW"
    if [[ -n "$LAST" ]]; then
        oxide_args+=(--pages "${FIRST}-${LAST}")
    elif [[ "$FIRST" -ne 1 ]]; then
        ox_count="$("$OXIDE_BIN" info --json "$PDF" 2>/dev/null \
            | grep -Eo '"page_count"[[:space:]]*:[[:space:]]*[0-9]+' \
            | grep -Eo '[0-9]+$')"
        [[ -n "$ox_count" ]] || die "could not determine page count from $OXIDE_BIN info"
        oxide_args+=(--pages "${FIRST}-${ox_count}")
    fi

    if ! "$OXIDE_BIN" "${oxide_args[@]}" -o "$OX_RAW" "$PDF" \
            >"${WORK_DIR}/ref.log" 2>&1; then
        echo "Error: $OXIDE_BIN failed — see log:" >&2
        cat "${WORK_DIR}/ref.log" >&2
        exit 1
    fi
    # The CLI reports some failures on stdout with exit status 0.
    if grep -qi '^Error:' "${WORK_DIR}/ref.log"; then
        echo "Error: $OXIDE_BIN reported an error:" >&2
        cat "${WORK_DIR}/ref.log" >&2
        exit 1
    fi

    # The page index is zero-padded to the width of the document's page count,
    # so the width is not fixed. Converting is deferred until after rrocket has
    # run, where the width is read back off its own output filenames.
    RAW_REF_DIR="$OX_RAW"
elif [[ "$REFERENCE" == pdfium ]]; then
    # pypdfium2 names outputs "page_<n>.png", n zero-padded like pdf-oxide's.
    PF_RAW="${WORK_DIR}/pf_raw"
    mkdir -p "$PF_RAW"
    pf_last="$LAST"
    if [[ -z "$pf_last" ]]; then
        pf_last="$("$PDFIUM_PY" -c 'import sys, pypdfium2; print(len(pypdfium2.PdfDocument(sys.argv[1])))' "$PDF" 2>/dev/null)" \
            || die "could not determine page count with pypdfium2"
    fi
    if ! "$PDFIUM_BIN" "${pdfium_args[@]}" --pages "${FIRST}-${pf_last}" -o "$PF_RAW" "$PDF" \
            >"${WORK_DIR}/ref.log" 2>&1; then
        echo "Error: pypdfium2 failed — see log:" >&2
        cat "${WORK_DIR}/ref.log" >&2
        exit 1
    fi
    RAW_REF_DIR="$PF_RAW"
elif ! pdftoppm "${ref_args[@]}" "$PDF" "${REF_DIR}/page" 2>"${WORK_DIR}/ref.log"; then
    echo "Error: pdftoppm failed — see log:" >&2
    cat "${WORK_DIR}/ref.log" >&2
    exit 1
fi

# ── render: rrocket ────────────────────────────────────────────────────────
if ! "$RASTER_BIN" "${new_args[@]}" "$PDF" "${NEW_DIR}/page" 2>"${WORK_DIR}/new.log"; then
    echo "Error: rrocket failed — see log:" >&2
    cat "${WORK_DIR}/new.log" >&2
    exit 1
fi

# ── normalise a PNG reference onto rrocket's filenames ────────────────────────
if [[ -n "${RAW_REF_DIR:-}" ]]; then
    shopt -s nullglob
    new_names=("${NEW_DIR}"/page-*.ppm)
    shopt -u nullglob
    [[ ${#new_names[@]} -gt 0 ]] || die "rrocket produced no output pages for ${PDF}"
    # Width of the zero-padded index rrocket chose, e.g. "page-001.ppm" -> 3.
    pad_sample="$(basename "${new_names[0]}" .ppm)"
    pad_width="${#pad_sample}"
    pad_width=$((pad_width - 5))   # strip the "page-" prefix

    shopt -s nullglob
    for png in "${RAW_REF_DIR}"/*.png; do
        idx="$(basename "$png" .png)"
        idx="${idx##*_}"
        [[ "$idx" =~ ^[0-9]+$ ]] || die "unexpected $REFERENCE output name: $png"
        # Strip leading zeros so the value is not read as octal.
        idx=$((10#$idx))
        convert "$png" "$(printf '%s/page-%0*d.ppm' "$REF_DIR" "$pad_width" "$idx")" \
            || die "convert failed on $REFERENCE output: $png"
    done
    shopt -u nullglob
fi

# ── compare page by page ──────────────────────────────────────────────────────
# Per-page lines; suppressed in summary-row mode.
# shellcheck disable=SC2059  # forwards the caller's format string
say() { $SUMMARY || printf "$@"; }

fail=0
pass=0
size_mismatch=0
total_rmse="0"
max_rmse="0"
worst_page=""
max_tile="0"
worst_tile_page=""

shopt -s nullglob
ref_pages=("${REF_DIR}"/page-*.ppm)
shopt -u nullglob

if [[ ${#ref_pages[@]} -eq 0 ]]; then
    echo "Error: reference renderer (${REFERENCE}) produced no output pages for ${PDF}" >&2
    if [[ -s "${WORK_DIR}/ref.log" ]]; then
        echo "  reference stderr:" >&2
        cat "${WORK_DIR}/ref.log" >&2
    fi
    exit 1
fi

for ref_file in "${ref_pages[@]}"; do
    page_tag="$(basename "$ref_file" .ppm)"   # e.g. "page-001"
    new_file="${NEW_DIR}/${page_tag}.ppm"
    diff_file="${DIFF_DIR}/${page_tag}-diff.png"

    if [[ ! -f "$new_file" ]]; then
        say "MISSING  %s — rrocket produced no output\n" "$page_tag"
        fail=$((fail + 1))
        continue
    fi

    read -r kind a b c d < <("${SCRIPT_DIR}/score_pair.sh" "$ref_file" "$new_file" "$diff_file") \
        || die "scoring failed for $page_tag"
    if [[ "$kind" == SIZE ]]; then
        say "SIZE     %-12s  ref=%dx%d  rrocket=%dx%d\n" "$page_tag" "$a" "$b" "$c" "$d"
        size_mismatch=$((size_mismatch + 1))
        fail=$((fail + 1))
        continue
    fi
    rmse_255="$a"
    tile_255="$b"

    total_rmse="$(bc -l <<< "scale=4; ${total_rmse} + ${rmse_255}")"

    rmse_over="$(bc -l <<< "${rmse_255} > ${THRESHOLD}")"
    tile_over="$(bc -l <<< "${tile_255} > ${TILE_LIMIT}")"
    if [[ "$rmse_over" -eq 1 || "$tile_over" -eq 1 ]]; then
        say "FAIL     %-12s  RMSE=%8s/255  tile=%8s/255  (limits %s, %s)  diff→%s\n" \
            "$page_tag" "$rmse_255" "$tile_255" "$THRESHOLD" "$TILE_LIMIT" "$diff_file"
        fail=$((fail + 1))
    else
        if $VERBOSE; then
            say "OK       %-12s  RMSE=%8s/255  tile=%8s/255\n" "$page_tag" "$rmse_255" "$tile_255"
        fi
        pass=$((pass + 1))
    fi

    if [[ "$(bc -l <<< "${rmse_255} > ${max_rmse}")" -eq 1 ]]; then
        max_rmse="$rmse_255"
        worst_page="$page_tag"
    fi
    if [[ "$(bc -l <<< "${tile_255} > ${max_tile}")" -eq 1 ]]; then
        max_tile="$tile_255"
        worst_tile_page="$page_tag"
    fi
done

total=$((pass + fail))
scored=$((pass + fail - size_mismatch))
avg_rmse="0"
[[ $scored -gt 0 ]] && avg_rmse="$(bc -l <<< "scale=4; ${total_rmse} / ${scored}")"
# Lines rrocket wrote to stderr: warnings about content it skipped or
# could not decode.
stderr_lines="$(grep -c . "${WORK_DIR}/new.log" || true)"

if $SUMMARY; then
    status=PASS
    [[ $fail -eq 0 ]] || status=FAIL
    printf "%s\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%d\t%s\n" "$status" "$total" "$fail" \
        "$size_mismatch" "$avg_rmse" "$max_rmse" "$max_tile" "$stderr_lines" "$PDF"
    [[ $fail -eq 0 ]] || exit 1
    exit 0
fi

echo ""
echo "── Summary ──────────────────────────────────────────────────────────────"
printf "  PDF:        %s\n" "$(basename "$PDF")"
printf "  Reference:  %s\n" "$REFERENCE"
printf "  DPI:        %s\n" "$DPI"
printf "  Pages:      %d total, %d passed, %d failed (%d size mismatches)\n" \
    "$total" "$pass" "$fail" "$size_mismatch"
printf "  Avg RMSE:   %.4f / 255\n" "$avg_rmse"
printf "  Max RMSE:   %.4f / 255  (%s)\n" "$max_rmse" "${worst_page:-none}"
printf "  Max tile:   %.4f / 255  (%s)\n" "$max_tile" "${worst_tile_page:-none}"
printf "  Thresholds: RMSE %.1f / 255, tile %.1f / 255\n" "$THRESHOLD" "$TILE_LIMIT"
printf "  rrocket stderr: %d lines\n" "$stderr_lines"
[[ -n "$OUT_DIR" ]] && printf "  Diff dir:   %s\n" "$DIFF_DIR"
echo "─────────────────────────────────────────────────────────────────────────"

if [[ $fail -eq 0 ]]; then
    echo "PASSED"
else
    echo "FAILED"
    exit 1
fi
