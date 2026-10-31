#!/usr/bin/env bash
# PGO + BOLT release build of the rrocket CLI.
#
# Usage: ./scripts/release_pgo_build.sh
#
# Prereqs:
#   - rustup component add llvm-tools-preview      (for `llvm-profdata`)
#   - llvm-bolt on PATH for the optional BOLT step
#       apt install bolt          # Ubuntu 24.04+
#       or download a build from  https://github.com/llvm/llvm-project/releases
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

# --- Verify training fixture is present. --------------------------------------
TRAINING_PDF="tests/fixtures/corpus-04-ebook-mixed.pdf"
if [[ ! -f "$TRAINING_PDF" ]]; then
    echo "error: training fixture missing: $TRAINING_PDF" >&2
    exit 1
fi

# --- Scratch dirs cleaned on exit. --------------------------------------------
PROFDIR="$(mktemp -d -t pdf_raster_pgo.XXXXXX)"
TRAIN_OUTDIR="$(mktemp -d -t pdf_raster_train.XXXXXX)"
cleanup() {
    rm -rf "$PROFDIR" "$TRAIN_OUTDIR"
}
trap cleanup EXIT

# train BIN: render the training pages with BIN, discarding the output.
# Fails when BIN fails or writes no page.
train() {
    rm -rf "${TRAIN_OUTDIR:?}"/*
    if ! "$1" "$TRAINING_PDF" "$TRAIN_OUTDIR/page" -f 1 -l 10 >/dev/null; then
        echo "error: the training run of $1 failed." >&2
        return 1
    fi
    local pages
    pages="$(find "$TRAIN_OUTDIR" -type f | wc -l)"
    if [[ "$pages" -eq 0 ]]; then
        echo "error: the training run of $1 wrote no pages." >&2
        return 1
    fi
    echo "         rendered $pages page(s)"
}

if command -v llvm-bolt >/dev/null 2>&1; then
    BOLT=true
    # BOLT rewrites relocation-mode binaries only, and refuses stripped ones.
    PGO_RUSTFLAGS_EXTRA="-C link-arg=-Wl,--emit-relocs"
    export CARGO_PROFILE_RELEASE_STRIP=none
else
    BOLT=false
    PGO_RUSTFLAGS_EXTRA=""
fi

echo "==> [1/4] Instrumented build (profile-generate)"
echo "         PROFDIR=$PROFDIR"
RUSTFLAGS="$BASE_RUSTFLAGS -Cprofile-generate=$PROFDIR" cargo build --release -p rasterrocket-cli

echo "==> [2/4] Profile training: rendering 10 pages of $TRAINING_PDF"
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

if $BOLT; then
    echo "==> [bonus] BOLT optimisation"
    BOLT_INSTRUMENTED="$PROFDIR/rrocket.bolt-instrumented"
    BOLT_OUT="$PROFDIR/rrocket.bolt"
    BOLT_FDATA="$PROFDIR/rrocket.fdata"
    if llvm-bolt "$BINARY" -instrument -o "$BOLT_INSTRUMENTED" \
            --instrumentation-file="$BOLT_FDATA" \
        && train "$BOLT_INSTRUMENTED" \
        && llvm-bolt "$BINARY" -o "$BOLT_OUT" -data="$BOLT_FDATA" \
            -reorder-blocks=ext-tsp \
            -reorder-functions=hfsort \
            -split-functions \
            -split-all-cold \
            -dyno-stats; then
        cp "$BOLT_OUT" "$BINARY"
        echo "    BOLT applied."
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
    echo "==> [bonus] BOLT not on PATH — skipping."
    echo "         Install: apt install bolt   (Ubuntu 24.04+)"
fi

echo
echo "==> Done. Binary at $BINARY"
ls -la "$BINARY"
