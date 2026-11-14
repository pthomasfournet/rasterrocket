#!/usr/bin/env bash
# Delete the test and binary executables Cargo wrote under each profile
# directory, so a CI build cache saved afterwards holds no executable that
# relinks in seconds.
#
# Usage: ci_prune_test_executables.sh [--keep NAME]... PROFILE_DIR...
#
# PROFILE_DIR is a Cargo profile directory such as target/debug or
# target/release.  Cargo writes each unit's outputs to
# PROFILE_DIR/build/<package>/<hash>/out; the executables there are
# deleted, except the build-script executables (`build_script_build*`),
# whose absence would rerun every build script on a cache hit, and those
# named by --keep (without the `.exe` suffix).
#
# On Windows ($RUNNER_OS = Windows) an executable is a `.exe`, deleted with
# the `.pdb` debug information beside it; elsewhere it is a file without an
# extension and with the owner's execute bit.
#
# Exit code: 0 = pruned; 1 = usage error or a PROFILE_DIR that does not exist.

set -euo pipefail
# `${keep[@]+"${keep[@]}"}` expands an empty array under `set -u` in bash
# 3.2 (macOS), where a bare "${keep[@]}" is an unbound-variable error.

die() { echo "ci_prune_test_executables: $*" >&2; exit 1; }

keep=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)
            [[ $# -ge 2 ]] || die "--keep requires a name"
            keep+=("$2")
            shift 2 ;;
        --) shift; break ;;
        -*) die "unknown option $1" ;;
        *) break ;;
    esac
done
[[ $# -gt 0 ]] || die "usage: ci_prune_test_executables.sh [--keep NAME]... PROFILE_DIR..."

if [[ "${RUNNER_OS:-}" == Windows ]]; then
    predicate=(\( -name '*.exe' -o -name '*.pdb' \))
    for name in ${keep[@]+"${keep[@]}"}; do
        predicate+=(! -name "${name}.exe" ! -name "${name}.pdb")
    done
else
    predicate=(-perm -u+x ! -name '*.*')
    for name in ${keep[@]+"${keep[@]}"}; do
        predicate+=(! -name "$name")
    done
fi

for dir in "$@"; do
    [[ -d "$dir" ]] || die "not a directory: $dir"
    find "$dir" -path '*/build/*/out/*' -type f "${predicate[@]}" \
        ! -name 'build_script_build*' -delete
done
