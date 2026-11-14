#!/usr/bin/env bash
# PGO + BOLT release build of the rrocket CLI.
#
# Usage: ./scripts/release_pgo_build.sh [TRAINING.pdf ...]
#
# Prereqs:
#   - rustup component add llvm-tools-preview      (for `llvm-profdata`)
#   - llvm-bolt and merge-fdata for the optional BOLT step: $LLVM_BOLT and
#     $MERGE_FDATA, else `llvm-bolt` and `merge-fdata` on PATH.  Ubuntu
#     ships both in `bolt-N` (under /usr/lib/llvm-N/bin; the package named
#     `bolt` is the Thunderbolt daemon), or download a build from
#     https://github.com/llvm/llvm-project/releases.  The BOLT profile comes
#     from instrumentation, so neither perf nor LBR is needed.
#   - python3, when the synthetic training set is used
#
# Training input: the PDFs given as arguments; else
# tests/fixtures/corpus-04-ebook-mixed.pdf when present; else the synthetic
# documents of tests/bench/instructions.py, generated from files committed
# to the repository (hinted TrueType text, vector paths with dashes and
# shadings, a transparency group, DCT, JPX and soft-masked Flate images,
# CMYK fills).  The first 10 pages of each are rendered twice, RGB at
# 150 dpi and gray at 300 dpi.
#
# PGO_REQUIRE_BOLT=1 makes a missing llvm-bolt or a failed BOLT step an
# error; otherwise either leaves the PGO-only binary.
#
# Codegen flags: $RUSTFLAGS when set, else `-C target-cpu=native` (the
# `.cargo/config.toml` default, which any RUSTFLAGS value replaces).  The
# PGO flags are appended to them.
#
# The BOLT step instruments the PGO binary, replays the training render on
# it and reorders the binary from that profile.  BOLT needs the symbol table
# and the static relocations, so that build keeps both (`strip = "none"`,
# `--emit-relocs`) and the result is stripped afterwards.
#
# Output: target/release/rrocket (PGO-optimised; BOLT applied on top if available).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The training documents given as arguments, relative ones resolved against
# the caller's directory before the script changes to the repository root.
ARG_PDFS=()
for arg in "$@"; do
    [[ "$arg" == /* ]] || arg="$PWD/$arg"
    ARG_PDFS+=("$arg")
done
cd "$REPO_ROOT"

BINARY="./target/release/rrocket"
BASE_RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

# --- Locate llvm-profdata. -----------------------------------------------------
# rustup ships it under the toolchain sysroot when `llvm-tools-preview` is added.
SYSROOT="$(rustc --print sysroot)"
HOST_TRIPLE="$(rustc -vV | sed -n 's|host: ||p')"
PROFDATA="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/llvm-profdata"
if [[ ! -x "$PROFDATA" ]]; then
    if command -v llvm-profdata >/dev/null 2>&1; then
        PROFDATA="$(command -v llvm-profdata)"
    else
        echo "error: llvm-profdata not found." >&2
        echo "       Install with: rustup component add llvm-tools-preview" >&2
        exit 1
    fi
fi

# --- Scratch dirs cleaned on exit. --------------------------------------------
PROFDIR="$(mktemp -d -t pdf_raster_pgo.XXXXXX)"
TRAIN_OUTDIR="$(mktemp -d -t pdf_raster_train.XXXXXX)"
TRAIN_INDIR="$(mktemp -d -t pdf_raster_train_in.XXXXXX)"
cleanup() {
    rm -rf "$PROFDIR" "$TRAIN_OUTDIR" "$TRAIN_INDIR"
}
trap cleanup EXIT

# --- Training documents. -------------------------------------------------------
REQUIRE_BOLT="${PGO_REQUIRE_BOLT:-0}"
CORPUS_PDF="tests/fixtures/corpus-04-ebook-mixed.pdf"
if [[ ${#ARG_PDFS[@]} -gt 0 ]]; then
    TRAINING_PDFS=("${ARG_PDFS[@]}")
elif [[ -f "$CORPUS_PDF" ]]; then
    TRAINING_PDFS=("$CORPUS_PDF")
else
    echo "==> Generating the synthetic training set"
    python3 -I - "$REPO_ROOT/tests/bench" "$TRAIN_INDIR" <<'PY'
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from instructions import synthetic

for path in synthetic(pathlib.Path(sys.argv[2])):
    print(f"         {path.name}")
PY
    TRAINING_PDFS=("$TRAIN_INDIR"/*.pdf)
fi
for pdf in "${TRAINING_PDFS[@]}"; do
    if [[ ! -f "$pdf" ]]; then
        echo "error: training document missing: $pdf" >&2
        exit 1
    fi
done

# train BIN: render the training pages with BIN, discarding the output.
# Fails when BIN fails on a document or writes no page.
train() {
    local pdf i=0
    rm -rf "${TRAIN_OUTDIR:?}"/*
    for pdf in "${TRAINING_PDFS[@]}"; do
        i=$((i + 1))
        if ! "$1" "$pdf" "$TRAIN_OUTDIR/$i-rgb" -f 1 -l 10 >/dev/null \
            || ! "$1" "$pdf" "$TRAIN_OUTDIR/$i-gray" -f 1 -l 10 --gray -r 300 >/dev/null; then
            echo "error: the training run of $1 on $pdf failed." >&2
            return 1
        fi
    done
    local pages
    pages="$(find "$TRAIN_OUTDIR" -type f | wc -l)"
    if [[ "$pages" -eq 0 ]]; then
        echo "error: the training run of $1 wrote no pages." >&2
        return 1
    fi
    echo "         rendered $pages page(s)"
}

LLVM_BOLT="${LLVM_BOLT:-llvm-bolt}"
MERGE_FDATA="${MERGE_FDATA:-merge-fdata}"
if command -v "$LLVM_BOLT" >/dev/null 2>&1 && command -v "$MERGE_FDATA" >/dev/null 2>&1; then
    BOLT=true
    # BOLT rewrites relocation-mode binaries only, and refuses stripped ones.
    PGO_RUSTFLAGS_EXTRA="-C link-arg=-Wl,--emit-relocs"
    export CARGO_PROFILE_RELEASE_STRIP=none
elif [[ "$REQUIRE_BOLT" == 1 ]]; then
    echo "error: PGO_REQUIRE_BOLT=1 and $LLVM_BOLT or $MERGE_FDATA is not found." >&2
    exit 1
else
    BOLT=false
    PGO_RUSTFLAGS_EXTRA=""
fi

echo "==> [1/4] Instrumented build (profile-generate)"
echo "         PROFDIR=$PROFDIR"
RUSTFLAGS="$BASE_RUSTFLAGS -Cprofile-generate=$PROFDIR" cargo build --release -p rasterrocket-cli

echo "==> [2/4] Profile training: ${#TRAINING_PDFS[@]} document(s)"
train "$BINARY" || exit 1
RAW_PROFILES="$(find "$PROFDIR" -name '*.profraw' | wc -l)"
if [[ "$RAW_PROFILES" -eq 0 ]]; then
    echo "error: no .profraw files produced — training run did not write profile data." >&2
    exit 1
fi
echo "         collected $RAW_PROFILES .profraw file(s)"

echo "==> [3/4] Merging PGO profile data ($PROFDIR -> merged.profdata)"
"$PROFDATA" merge -o "$PROFDIR/merged.profdata" "$PROFDIR"

echo "==> [4/4] PGO-guided rebuild (profile-use)"
RUSTFLAGS="$BASE_RUSTFLAGS -Cprofile-use=$PROFDIR/merged.profdata -Cllvm-args=-pgo-warn-missing-function $PGO_RUSTFLAGS_EXTRA" \
    cargo build --release -p rasterrocket-cli

# bolt_optimise: instrument $BINARY, train the instrumented copy and write
# the reordered binary to $BOLT_OUT.  Each training process writes its own
# profile (`--instrumentation-file-append-pid`); merge-fdata combines them.
BOLT_INSTRUMENTED="$PROFDIR/rrocket.bolt-instrumented"
BOLT_OUT="$PROFDIR/rrocket.bolt"
BOLT_PROFDIR="$PROFDIR/bolt"
BOLT_FDATA="$PROFDIR/rrocket.fdata"
bolt_optimise() {
    mkdir -p "$BOLT_PROFDIR"
    "$LLVM_BOLT" "$BINARY" -instrument -o "$BOLT_INSTRUMENTED" \
        --instrumentation-file="$BOLT_PROFDIR/rrocket" \
        --instrumentation-file-append-pid || return 1
    train "$BOLT_INSTRUMENTED" || return 1
    local profiles=("$BOLT_PROFDIR"/*)
    if [[ ! -e "${profiles[0]}" ]]; then
        echo "error: the instrumented training wrote no BOLT profile." >&2
        return 1
    fi
    echo "         collected ${#profiles[@]} BOLT profile(s)"
    "$MERGE_FDATA" "${profiles[@]}" >"$BOLT_FDATA" || return 1
    "$LLVM_BOLT" "$BINARY" -o "$BOLT_OUT" -data="$BOLT_FDATA" \
        -reorder-blocks=ext-tsp \
        -reorder-functions=hfsort \
        -split-functions \
        -split-all-cold \
        -dyno-stats
}

if $BOLT; then
    echo "==> [bonus] BOLT optimisation ($LLVM_BOLT)"
    if bolt_optimise; then
        cp "$BOLT_OUT" "$BINARY"
        echo "    BOLT applied."
    elif [[ "$REQUIRE_BOLT" == 1 ]]; then
        echo "error: BOLT failed and PGO_REQUIRE_BOLT=1." >&2
        exit 1
    else
        echo "    BOLT failed; keeping the PGO-only binary." >&2
    fi
    # The release profile strips symbols; the BOLT build kept them.
    if command -v strip >/dev/null 2>&1; then
        strip "$BINARY"
    else
        echo "    strip not on PATH; $BINARY keeps its symbol table." >&2
    fi
else
    echo "==> [bonus] $LLVM_BOLT or $MERGE_FDATA not found — skipping BOLT."
    echo "         Install: apt install bolt-N   (Ubuntu; or set LLVM_BOLT and MERGE_FDATA)"
fi

echo "==> Smoke render with the final binary"
train "$BINARY" || exit 1

echo
echo "==> Done. Binary at $BINARY"
ls -la "$BINARY"
