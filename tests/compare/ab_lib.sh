# ab_lib.sh — definitions shared by ab_render.sh and ab_archive.sh.
# Sourced, not run; messages carry the sourcing script's name.
# shellcheck shell=bash

# Print the sourcing script's header comment (every line above its
# `set -euo` line) to standard error and exit 2.
ab_usage() {
    sed -n '/^set -euo/q; 2,$p' "$0" >&2
    exit 2
}

# Print "<script>: MESSAGE" to standard error and exit 2.
ab_usage_error() {
    echo "${0##*/}: $*" >&2
    exit 2
}

# Exit 2 unless every argument is an executable file.
ab_require_bins() {
    local bin
    for bin; do
        [[ -x $bin ]] || ab_usage_error "not executable: $bin"
    done
}

# Set the arrays before_flags and after_flags to BEFORE_FLAGS and
# AFTER_FLAGS split into words.
ab_flags() {
    # shellcheck disable=SC2034  # read by the sourcing script
    read -r -a before_flags <<<"${BEFORE_FLAGS-}"
    # shellcheck disable=SC2034
    read -r -a after_flags <<<"${AFTER_FLAGS-}"
}

# Set the array limiter to a command prefix that stops a render after $1
# seconds: `timeout`, else `gtimeout`, else empty when neither is installed.
ab_limiter() {
    if command -v timeout >/dev/null; then
        limiter=(timeout "$1")
    elif command -v gtimeout >/dev/null; then
        limiter=(gtimeout "$1")
    else
        # shellcheck disable=SC2034
        limiter=()
    fi
}
