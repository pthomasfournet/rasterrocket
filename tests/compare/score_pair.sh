#!/usr/bin/env bash
# score_pair.sh — score one rendered page against a reference rendering.
#
# Usage:
#   score_pair.sh REF NEW [DIFF]
#
# Prints one line:
#   SIZE <ref_w> <ref_h> <new_w> <new_h>   sizes differ by more than 1 px
#   SCORE <rmse> <tile_delta>              both on a 0-255 scale
#
# Renders within 1 px of each other are cropped to their common area before
# scoring.  RMSE is ImageMagick's over all channels.  The tile delta is the
# largest difference in mean luminance over 32 x 32 px tiles: anti-aliasing
# and hinting differences move coverage within a tile but hardly change its
# mean, while content present in only one render changes it wholesale.
# DIFF, when given, receives ImageMagick's difference image.
#
# Exit code: 0 = scored (or SIZE); 2 = an image could not be read.
#
# Requires: ImageMagick (identify, convert, compare)

set -euo pipefail

TILE_PX=32

[[ $# -ge 2 ]] || { echo "usage: score_pair.sh REF NEW [DIFF]" >&2; exit 2; }
REF="$1"
NEW="$2"
DIFF="${3:-null:}"

fail() { echo "score_pair: $*" >&2; exit 2; }

read -r ref_w ref_h < <(identify -format "%w %h\n" "${REF}[0]" 2>/dev/null) \
    || fail "cannot read $REF"
read -r new_w new_h < <(identify -format "%w %h\n" "${NEW}[0]" 2>/dev/null) \
    || fail "cannot read $NEW"

dw=$((ref_w - new_w))
dh=$((ref_h - new_h))
if (( dw < -1 || dw > 1 || dh < -1 || dh > 1 )); then
    echo "SIZE $ref_w $ref_h $new_w $new_h"
    exit 0
fi

w=$(( ref_w < new_w ? ref_w : new_w ))
h=$(( ref_h < new_h ? ref_h : new_h ))
# Both images flattened onto white and cropped to the common area, as
# 8-bit RGB, so an alpha channel or a palette in a PNG reference does not
# enter the scores.
pair=(
    \( "${REF}[0]" -background white -alpha remove -crop "${w}x${h}+0+0" +repage -type TrueColor \)
    \( "${NEW}[0]" -background white -alpha remove -crop "${w}x${h}+0+0" +repage -type TrueColor \)
)

# compare exits 1 when the images differ (expected); 2 means a real error.
rmse_line="$(compare -metric RMSE "${pair[@]}" "$DIFF" 2>&1)" \
    || { ec=$?; [[ $ec -eq 1 ]] || fail "compare failed ($ec): $rmse_line"; }
# Output format: "<absolute> (<normalized>)", e.g. "3.14 (0.0123)".
rmse_norm="$(printf '%s' "$rmse_line" | grep -Eo '\([0-9.e+-]+\)' | tr -d '()')" \
    || fail "cannot parse RMSE from: $rmse_line"

tiles_x=$(( (w + TILE_PX - 1) / TILE_PX ))
tiles_y=$(( (h + TILE_PX - 1) / TILE_PX ))
tile_norm="$(convert "${pair[@]}" -colorspace gray -filter box \
    -resize "${tiles_x}x${tiles_y}!" -compose difference -composite \
    -format "%[fx:maxima]" info:)" || fail "tile delta failed"

awk -v r="$rmse_norm" -v t="$tile_norm" 'BEGIN { printf "SCORE %.4f %.4f\n", r * 255, t * 255 }'
