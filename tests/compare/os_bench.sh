#!/usr/bin/env bash
# Linux driver of the cross-OS benchmark: builds the binaries, records the
# machine state, and runs the rows of tests/compare/os_bench_rows.tsv through
# the os_bench harness under fixed CPU affinity.
#
# Usage:
#   tests/compare/os_bench.sh [-g GROUP]... [--only ROW] [--reps K] [--smoke]
#       [--ram-root DIR] [--gpu-features LIST] [--dry-run] OUT_DIR
#
#   -g GROUP          run the rows of GROUP (library, startup, disk, ram, avx2,
#                     gpu); repeatable; default every group.
#   --only ROW        run ROW alone (overrides -g).
#   --reps K          repetitions per row (default 3).
#   --smoke           one document per row, one repetition, and none of the
#                     machine-state checks (governor, transparent huge pages,
#                     clean tree).  The RUSTFLAGS check stays; a RAM root that
#                     fails its check skips the ram rows, with a note on stderr.
#   --ram-root DIR    root of the ram rows' --ram-path directories (default
#                     /mnt/rr-bench-ram).  It must itself be a tmpfs
#                     mountpoint of at most 4 GiB, writable, holding nothing
#                     but this driver's rr-os-bench directory; otherwise a run
#                     with ram rows refuses to start.
#   --gpu-features L  features of the gpu build (default
#                     nvjpeg,gpu-icc,gpu-deskew,cache,vulkan).  A list without
#                     a CUDA feature skips the --backend cuda row, one without
#                     vulkan the --backend vulkan rows.
#   --dry-run         print every command (builds, affinity prefixes, harness
#                     calls) and exit 0; nothing is built, run or written.
#
# Rows whose os column is windows are never run here.  The S-version row runs
# first in every run, whatever the selection, since the summary needs it.
# Rows of the avx2 group run twice: on HEAD, and as ROW-pre on a scratch
# worktree (target/os-bench/pre-avx2) with commits a5cfd530 and 205be502
# reverted; a revert that does not apply cleanly stops the run.  A worktree
# left by an earlier run at the same HEAD, unmodified, is reused.  Rows of the
# gpu group run on a build with --gpu-features in target/os-bench/gpu
# (CUDA_ARCH=sm_75); the --backend cuda row also needs a GPU driver whose
# nvidia-smi version is 580 or later.  Each skipped gpu row adds a
# `skipped ROW: reason` line to OUT_DIR/skipped.txt.
#
# Affinity: a 1-thread row runs under `taskset -c 2`, an N-thread row under
# `taskset -c 0-(N-1)`, both with RAYON_NUM_THREADS=N.  The start-up rows
# (S-version, S-page, G-cuda-start, G-vulkan-start) are 1-thread rows: the
# process, with every runtime and driver thread it starts, shares CPU 2, so
# a G row less S-page is the backend's initialisation on one core.  Disk
# rows write under OUT_DIR/scratch (--no-ram and a path prefix, counted
# with --expect-pages); ram rows write to --ram-path under the RAM root and
# spill to an empty OUT_DIR/scratch directory.  After the rows, every row
# `os_bench noisy` lists is run again with 5 repetitions into rerun.tsv,
# which supersedes its samples.
#
# Prerequisites: cargo and rustc of the repository toolchain, git, taskset,
# findmnt, and the fixtures named in the rows file under tests/fixtures/
# (gitignored).  RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS and
# CARGO_BUILD_RUSTFLAGS must be unset, so that the repository's
# -C target-cpu=native applies.  A full run also requires the performance
# governor on every CPU, transparent huge pages `madvise` or `never`
# (recorded as env key thp), and a working tree equal to HEAD.  Each failed
# check prints its remedy; the system changes need the owner's go-ahead:
#   sudo cpupower frequency-set -g performance
#   echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
#   sudo mkdir -p /mnt/rr-bench-ram &&
#     sudo mount -t tmpfs -o size=4G,mode=1777 tmpfs /mnt/rr-bench-ram
# Where builds share a memory budget with other work, run the cargo builds
# first under that budget's wrapper; the driver's own builds are then no-ops.
#
# OUT_DIR must be absent or empty.  It receives:
#   env.tsv      key<TAB>value machine state, from a fixed key set (priority
#                records the driver's niceness, which the rows inherit)
#   results.tsv  one harness sample per line
#   rerun.tsv    5-repetition re-runs of noisy rows (when any)
#   pages.tsv    per-page render times of the rows that keep them
#   build.log    output of every build step
#   skipped.txt  rows not run and why (when any)
#   done.flag    touched last, on success
#   failed.flag  instead, on any error or on SIGINT/TERM/HUP once OUT_DIR is
#                taken (a usage error or a non-empty OUT_DIR exits 2 without
#                it), holding the first failing step and command
# SIGINT, SIGTERM and SIGHUP sent to the script alone are handled once the
# running child (a build or a harness call) exits; Ctrl-C and a signal to the
# whole process group stop the child as well, so they take effect at once.
# Paths in build.log and failed.flag may name the home directory: neither is
# published.  env.tsv, results.tsv, rerun.tsv, pages.tsv and skipped.txt hold
# no host or user name.
set -Eeuo pipefail
shopt -s inherit_errexit

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
rows_file=$here/os_bench_rows.tsv

ram_gib_limit=4
ram_remedy='with the owner'"'"'s go-ahead: sudo mkdir -p /mnt/rr-bench-ram && sudo mount -t tmpfs -o size=4G,mode=1777 tmpfs /mnt/rr-bench-ram'
thp_file=/sys/kernel/mm/transparent_hugepage/enabled
thp_remedy="with the owner's go-ahead: echo madvise | sudo tee $thp_file"
avx2_commits=(a5cfd530 205be502)
cuda_features=(nvjpeg nvjpeg2k gpu-icc gpu-deskew cache)
cuda_driver_floor=580

usage() {
    sed -n '/^# Usage:/,/^# Rows whose/p' "$0" | sed '$d; s/^# \{0,1\}//' >&2
    exit "${1:-2}"
}

groups=()
only=
reps=3
smoke=0
dry=0
ram_root=/mnt/rr-bench-ram
gpu_features=nvjpeg,gpu-icc,gpu-deskew,cache,vulkan
out=
while (($#)); do
    case $1 in
        -g) (($# >= 2)) || usage; groups+=("$2"); shift 2 ;;
        --only) (($# >= 2)) || usage; only=$2; shift 2 ;;
        --reps) (($# >= 2)) || usage; reps=$2; shift 2 ;;
        --ram-root) (($# >= 2)) || usage; ram_root=$2; shift 2 ;;
        --gpu-features) (($# >= 2)) || usage; gpu_features=$2; shift 2 ;;
        --smoke) smoke=1; shift ;;
        --dry-run) dry=1; shift ;;
        -h | --help) usage 0 ;;
        -*) echo "unknown flag: $1" >&2; usage ;;
        *) [[ -z $out ]] || usage; out=$1; shift ;;
    esac
done
[[ -n $out ]] || usage
[[ $reps =~ ^[1-9][0-9]*$ ]] || { echo "--reps must be a positive integer, got: $reps" >&2; exit 2; }
if ((smoke)); then reps=1; fi
out=$(realpath -m -- "$out")
ram_root=$(realpath -m -- "$ram_root")

head_rr=$root/target/release/rrocket
head_bench=$root/target/release/os_bench
pre_dir=$root/target/os-bench/pre-avx2
pre_target=$root/target/os-bench/pre-avx2-target
pre_bench=$pre_target/release/os_bench
pre_stamp=$root/target/os-bench/pre-avx2.tree
gpu_target=$root/target/os-bench/gpu
gpu_rr=$gpu_target/release/rrocket
scratch=$out/scratch
ram_dir=$ram_root/rr-os-bench

# ---- failure reporting ------------------------------------------------------

step="argument checks"
flag_ok=0   # failed.flag may be written once OUT_DIR is the run's own.

# fail MESSAGE: report MESSAGE on stderr and in failed.flag, then exit 1.
# Paths in MESSAGE are shown relative to the repository and the home
# directory.  The first report is kept: a failure inside a command
# substitution has already written the specific message when the enclosing
# command fails in turn.
fail() {
    local msg=${1//"$root/"/}
    if [[ -n ${HOME:-} && $HOME != / ]]; then
        msg=${msg//"$HOME"/\~}
    fi
    if ((flag_ok)) && [[ -e $out/failed.flag ]]; then
        exit 1
    fi
    echo "os_bench.sh: $msg" >&2
    if ((flag_ok)); then
        printf '%s\n' "$msg" > "$out/failed.flag"
    fi
    exit 1
}

# die MESSAGE [REMEDY]: a check failed; the remedy is printed with it.
die() {
    fail "$step: $1${2:+ (remedy: $2)}"
}

on_err() {
    local rc=$1 line=$2 cmd=$3
    trap - ERR
    if [[ $step == build* && -f $out/build.log ]]; then
        tail -n 20 "$out/build.log" >&2 || true
    fi
    fail "$step: line $line: '$cmd' exited $rc"
}

cleanup() {
    if [[ -n ${ram_made:-} ]]; then
        rm -rf -- "$ram_dir"
        ram_made=
    fi
}
trap cleanup EXIT

if ((!dry)); then
    if [[ -e $out ]] && [[ -n $(ls -A -- "$out") ]]; then
        echo "os_bench.sh: $out is not empty; give a new directory" >&2
        exit 2
    fi
    mkdir -p -- "$out"
    flag_ok=1
fi
trap 'on_err $? $LINENO "$BASH_COMMAND"' ERR
for sig in INT TERM HUP; do
    # shellcheck disable=SC2064 # the signal name is bound now, by design.
    trap "fail \"\$step: interrupted by SIG$sig\"" "$sig"
done
cd "$root"

# ---- the rows file ------------------------------------------------------------

r_group=() r_row=() r_os=() r_kind=() r_threads=() r_dpi=() r_sink=() r_inv=() r_extra=() r_docs=()
step="reading $rows_file"
while IFS= read -r line || [[ -n $line ]]; do
    [[ -z $line || $line == \#* ]] && continue
    tabs=${line//[^$'\t']/}
    ((${#tabs} == 9)) || die "row line has $((${#tabs} + 1)) fields, not 10: $line"
    IFS=$'\t' read -r g r o k t d s i e docs <<< "$line"
    for v in g r o k t d s i e docs; do
        [[ -n ${!v} ]] || die "empty field in: $line"
        if [[ ${!v} == - ]]; then printf -v "$v" ''; fi
    done
    [[ $o =~ ^(both|linux|windows)$ ]] || die "$r: os must be both, linux or windows"
    [[ $k =~ ^(library|deskew|process)$ ]] || die "$r: kind must be library, deskew or process"
    [[ $s =~ ^(disk|ram|none)$ ]] || die "$r: sink must be disk, ram or none"
    [[ $t =~ ^[1-9][0-9]*$ && $d =~ ^[1-9][0-9]*$ ]] || die "$r: threads and dpi must be positive integers"
    r_group+=("$g") r_row+=("$r") r_os+=("$o") r_kind+=("$k") r_threads+=("$t")
    r_dpi+=("$d") r_sink+=("$s") r_inv+=("$i") r_extra+=("$e") r_docs+=("$docs")
done < "$rows_file"

# row_index NAME: the index of row NAME in the rows file.
row_index() {
    local n
    for n in "${!r_row[@]}"; do
        if [[ ${r_row[n]} == "$1" ]]; then echo "$n"; return 0; fi
    done
    die "no row named $1 in $rows_file"
}

step="row selection"
selected=()
if [[ -n $only ]]; then
    n=$(row_index "$only")
    [[ ${r_os[n]} != windows ]] || die "$only runs only on Windows"
    selected=("$n")
else
    if ((${#groups[@]} == 0)); then
        mapfile -t groups < <(printf '%s\n' "${r_group[@]}" | awk '!seen[$0]++')
    fi
    for g in "${groups[@]}"; do
        printf '%s\n' "${r_group[@]}" | grep -qxF -- "$g" || die "no group named $g"
    done
    for n in "${!r_row[@]}"; do
        [[ ${r_os[n]} == windows ]] && continue
        for g in "${groups[@]}"; do
            if [[ ${r_group[n]} == "$g" ]]; then selected+=("$n"); fi
        done
    done
fi
startup_row=$(row_index S-version)

# ---- gpu rows that cannot run here ----------------------------------------------

has_feature() {
    local f feats
    IFS=, read -ra feats <<< "$gpu_features"
    for f in "${feats[@]}"; do
        if [[ $f == "$1" ]]; then return 0; fi
    done
    return 1
}

# skip_reason N: why row N cannot run on this host, or nothing.
skip_reason() {
    local extra=" ${r_extra[$1]} " f major
    if [[ $extra == *" --backend cuda "* ]]; then
        for f in "${cuda_features[@]}"; do
            has_feature "$f" && break
            f=
        done
        [[ -n $f ]] || { echo "--gpu-features $gpu_features holds no CUDA feature"; return 0; }
        major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | awk -F. 'NR == 1 {print $1}' || true)
        [[ $major =~ ^[0-9]+$ ]] || { echo "nvidia-smi reports no driver version"; return 0; }
        ((major >= cuda_driver_floor)) || { echo "GPU driver $major is older than $cuda_driver_floor, which CUDA 13 PTX needs"; return 0; }
    fi
    if [[ $extra == *" --backend vulkan "* ]] && ! has_feature vulkan; then
        echo "--gpu-features $gpu_features lacks vulkan"
    fi
    return 0
}

run_rows=()
skips=()
for n in "${selected[@]}"; do
    [[ $n == "$startup_row" ]] && continue
    reason=$(skip_reason "$n")
    if [[ -n $reason ]]; then
        skips+=("skipped ${r_row[n]}: $reason")
    else
        run_rows+=("$n")
    fi
done

# ram_root_problem: why the RAM root cannot hold the ram rows, or nothing.  It
# must be a tmpfs mountpoint of at most the limit, writable, and hold nothing
# but this driver's own directory.
ram_root_problem() {
    local mount fstype size entry others
    if ! mount=$(findmnt -bno FSTYPE,SIZE "$ram_root" 2> /dev/null); then
        echo "$ram_root is not a mountpoint"
        return 0
    fi
    read -r fstype size <<< "$mount"
    if [[ $fstype != tmpfs ]]; then
        echo "$ram_root is $fstype, not tmpfs"
    elif ! [[ $size =~ ^[0-9]+$ ]] || ((size > ram_gib_limit * 1024 ** 3)); then
        echo "$ram_root holds ${size} bytes, more than $ram_gib_limit GiB"
    elif ! [[ -w $ram_root ]]; then
        echo "$ram_root is not writable"
    else
        others=()
        shopt -s nullglob dotglob
        for entry in "$ram_root"/*; do
            if [[ ${entry##*/} != "${ram_dir##*/}" ]]; then others+=("${entry##*/}"); fi
        done
        shopt -u nullglob dotglob
        if ((${#others[@]})); then
            echo "$ram_root holds other entries: ${others[*]:0:5}"
        fi
    fi
    return 0
}

ram_rows=0
for n in "${run_rows[@]}"; do
    if [[ ${r_sink[n]} == ram ]]; then ram_rows=1; fi
done
if ((ram_rows && !dry)); then
    step="RAM root check"
    reason=$(ram_root_problem)
    if [[ -n $reason && $smoke == 1 ]]; then
        echo "os_bench.sh: --smoke skips the ram rows: $reason" >&2
        kept=()
        for n in "${run_rows[@]}"; do
            if [[ ${r_sink[n]} != ram ]]; then kept+=("$n"); fi
        done
        run_rows=("${kept[@]}")
    elif [[ -n $reason ]]; then
        die "$reason" "$ram_remedy"
    fi
fi

need_pre=0 need_gpu=0 need_ram=0
for n in "${run_rows[@]}"; do
    if [[ ${r_group[n]} == avx2 ]]; then need_pre=1; fi
    if [[ ${r_group[n]} == gpu ]]; then need_gpu=1; fi
    if [[ ${r_sink[n]} == ram ]]; then need_ram=1; fi
done

# ---- checks -----------------------------------------------------------------------

# thp_mode: the bracketed transparent-huge-page mode, or nothing.
thp_mode() {
    sed -n 's/.*\[\([a-z]*\)\].*/\1/p' "$thp_file" 2> /dev/null || true
}

step="fixture check"
for n in "$startup_row" "${run_rows[@]}"; do
    for doc in ${r_docs[n]}; do
        [[ -f tests/fixtures/${doc%:*} ]] || die "tests/fixtures/${doc%:*} is missing (it is gitignored: copy it into this clone)"
    done
done

if ((!dry)); then
    step="build environment check"
    for v in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS; do
        [[ -z ${!v:-} ]] || die "$v is set and would replace the repository's -C target-cpu=native" "unset $v"
    done
fi

if ((!smoke && !dry)); then
    step="machine-state check"
    thp=$(thp_mode)
    [[ $thp != always ]] || die "transparent huge pages are 'always'; large pages must be off on both sides" "$thp_remedy"
    shopt -s nullglob
    govs=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor)
    shopt -u nullglob
    ((${#govs[@]})) || die "no scaling_governor files: the power policy cannot be confirmed"
    bad=$(grep -LxF performance "${govs[@]}" || true)
    [[ -z $bad ]] || die "governor is not performance on: $(echo "$bad" | tr '\n' ' ')" "sudo cpupower frequency-set -g performance"
    git diff --quiet HEAD -- || die "the working tree differs from HEAD, whose commit env.tsv records" "commit or stash the changes"
fi

# ---- command execution --------------------------------------------------------------

# show CMD...: the command as one shell-quoted line.
show() {
    local q
    printf -v q '%q ' "$@"
    echo "${q% }"
}

# build DIR CMD...: CMD in DIR, its output appended to build.log.
build() {
    local dir=$1
    shift
    if ((dry)); then
        echo "(cd $(show "$dir") && $(show "$@")) >> $(show "$out/build.log") 2>&1"
    else
        echo "== $(show "$@")  [in .${dir#"$root"}]" >> "$out/build.log"
        (cd "$dir" && "$@") >> "$out/build.log" 2>&1
    fi
}

# cpus THREADS: the affinity list of a THREADS-worker row.
cpus() {
    if (($1 == 1)); then echo 2; else echo "0-$(($1 - 1))"; fi
}

# harness DEST THREADS BENCH ARGS...: one pinned harness call, stdout to DEST.
harness() {
    local dest=$1 threads=$2
    shift 2
    local cmd=(taskset -c "$(cpus "$threads")" env "RAYON_NUM_THREADS=$threads" "$@")
    if ((dry)); then
        echo "$(show "${cmd[@]}") >> $(show "$dest")"
    else
        "${cmd[@]}" >> "$dest"
    fi
}

# run_row N DEST REPS VARIANT [DOC]: row N on build VARIANT (head|pre), its
# samples appended to DEST; with DOC, only the document of that file name.
run_row() {
    local n=$1 dest=$2 row_reps=$3 variant=$4 only_doc=${5:-}
    local label=${r_row[n]} bench=$head_bench rr=$head_rr
    if [[ $variant == pre ]]; then label+=-pre; bench=$pre_bench; fi
    if [[ ${r_group[n]} == gpu ]]; then rr=$gpu_rr; fi
    step="row $label"
    local docs=() doc
    for doc in ${r_docs[n]}; do
        if [[ -n $only_doc && ${doc%:*} != "$only_doc" ]]; then continue; fi
        docs+=("tests/fixtures/$doc")
        if ((smoke)); then break; fi
    done
    [[ -z ${r_docs[n]} || ${#docs[@]} -gt 0 ]] || die "$label has no document $only_doc"
    local args=() extra=() w
    read -ra extra <<< "${r_extra[n]}"
    case ${r_kind[n]} in
        library | deskew)
            args=(library --group "${r_group[n]}" --row "$label" --dpi "${r_dpi[n]}"
                --threads "${r_threads[n]}" --reps "$row_reps")
            if [[ ${r_kind[n]} == deskew ]]; then args+=(--deskew); fi
            for w in "${extra[@]}"; do
                if [[ $w == --page-times ]]; then
                    # Re-runs keep the first run's per-page times.
                    if [[ $dest == "$out/results.tsv" ]]; then
                        args+=(--page-times "$out/pages.tsv")
                    fi
                else
                    args+=("$w")
                fi
            done
            args+=("${docs[@]}")
            ;;
        process)
            args=(process --group "${r_group[n]}" --row "$label" --rrocket "$rr"
                --dpi "${r_dpi[n]}" --threads "${r_threads[n]}" --reps "$row_reps")
            if [[ -n ${r_inv[n]} ]]; then args+=(--invocations "${r_inv[n]}"); fi
            case ${r_sink[n]} in
                disk) args+=(--disk "$scratch/$label" --expect-pages) ;;
                ram) args+=(--ram "$ram_dir" --spill "$scratch/spill-$label") ;;
                none) args+=(--none) ;;
            esac
            args+=("${docs[@]}")
            if ((${#extra[@]})); then args+=(-- "${extra[@]}"); fi
            ;;
    esac
    harness "$dest" "${r_threads[n]}" "$bench" "${args[@]}"
    if ((!dry)); then
        rm -rf -- "$scratch"
    fi
}

# ---- builds -----------------------------------------------------------------------------

step="build (HEAD)"
build "$root" cargo build --release -p rasterrocket-cli
build "$root" cargo build --release -p bench --bin os_bench

# pre_current: the pre-AVX2 worktree exists as this run would create it: at
# HEAD, nothing unstaged, and its index the tree recorded when the revert was
# applied.  Reusing it keeps the build's fingerprints valid.
pre_current() {
    [[ -d $pre_dir && -f $pre_stamp ]] || return 1
    [[ $(git -C "$pre_dir" rev-parse HEAD 2> /dev/null) == "$(git rev-parse HEAD)" ]] || return 1
    git -C "$pre_dir" diff --quiet 2> /dev/null || return 1
    [[ $(git -C "$pre_dir" write-tree 2> /dev/null) == "$(cat -- "$pre_stamp")" ]]
}

if ((need_pre)); then
    step="build (pre-avx2 worktree)"
    if ((dry)); then
        show git worktree add --detach "$pre_dir" HEAD
        show git -C "$pre_dir" revert --no-commit "${avx2_commits[@]}"
    else
        if pre_current; then
            echo "== reusing ${pre_dir#"$root"/} (HEAD with ${avx2_commits[*]} reverted)" >> "$out/build.log"
        else
            git worktree prune
            rm -f -- "$pre_stamp"
            if [[ -e $pre_dir ]]; then
                git worktree remove --force "$pre_dir" \
                    || die "$pre_dir exists and is not a removable worktree"
            fi
            git worktree add --detach "$pre_dir" HEAD >> "$out/build.log" 2>&1
            git -C "$pre_dir" revert --no-commit "${avx2_commits[@]}" >> "$out/build.log" 2>&1 \
                || die "reverting ${avx2_commits[*]} at HEAD does not apply cleanly; the pre-AVX2 rows cannot run"
            git -C "$pre_dir" write-tree > "$pre_stamp"
        fi
    fi
    build "$pre_dir" cargo build --release -p bench --bin os_bench --target-dir "$pre_target"
fi

if ((need_gpu)); then
    step="build (gpu features)"
    build "$root" env CUDA_ARCH=sm_75 cargo build --release -p rasterrocket-cli \
        --features "$gpu_features" --target-dir "$gpu_target"
fi

# ---- environment ------------------------------------------------------------------------

# probe CMD...: CMD's output, or nothing when it fails or is absent.
probe() {
    "$@" 2> /dev/null || true
}

# env_key KEY VALUE: one env.tsv line; an empty VALUE omits the key.
env_key() {
    local v=${2//[$'\t\n\r']/ }
    [[ -z $v ]] || printf '%s\t%s\n' "$1" "$v" >> "$out/env.tsv"
}

if ((!dry)); then
    step="environment capture"
    env_key os linux
    env_key os_build "$(probe sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '"')"
    env_key kernel "$(uname -r)"
    env_key commit "$(git rev-parse HEAD)"
    env_key rustc "$(rustc -V)"
    env_key profile release
    env_key cpu_model "$(awk -F': *' '/^model name/ {print $2; exit}' /proc/cpuinfo 2> /dev/null || true)"
    env_key logical_cpus "$(probe nproc --all)"
    env_key ram_gib "$(awk '/^MemTotal:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo 2> /dev/null || true)"
    env_key thp "$(thp_mode)"
    env_key governor "$(probe cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor | sort -u | paste -sd, -)"
    env_key priority "nice $(nice)"
    env_key gpu_driver "$(probe nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | paste -sd, -)"
    if ((need_gpu)) && [[ -z $(skip_reason "$(row_index G-cuda-start)") ]]; then
        env_key cuda "$(probe nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    fi
    env_key fs "$(probe findmnt -no FSTYPE -T "$out")"
    if ((need_ram)); then
        env_key ram_root_fs "$(probe findmnt -no FSTYPE,SIZE -T "$ram_root" | tr -s ' ')"
    fi
    env_key date "$(date -u +%F)"
    : > "$out/results.tsv"
    : > "$out/pages.tsv"
    if ((${#skips[@]})); then
        printf '%s\n' "${skips[@]}" > "$out/skipped.txt"
    fi
    if ((need_ram)); then
        rm -rf -- "$ram_dir"
        mkdir -- "$ram_dir"
        ram_made=1
    fi
else
    for s in "${skips[@]}"; do echo "# $s"; done
fi

# ---- rows ---------------------------------------------------------------------------------

run_row "$startup_row" "$out/results.tsv" "$reps" head
for n in "${run_rows[@]}"; do
    run_row "$n" "$out/results.tsv" "$reps" head
    if [[ ${r_group[n]} == avx2 ]]; then
        run_row "$n" "$out/results.tsv" "$reps" pre
    fi
done

step="noise pass"
if ((dry)); then
    echo "$(show "$head_bench" noisy "$out")  # each ROW<TAB>DOC listed: that row and document again, --reps 5, >> $(show "$out/rerun.tsv")"
else
    noisy=$("$head_bench" noisy "$out")
    while IFS=$'\t' read -r label doc; do
        [[ -n $label ]] || continue
        variant='head'
        if [[ $label == *-pre ]]; then
            variant=pre
            label=${label%-pre}
        fi
        run_row "$(row_index "$label")" "$out/rerun.tsv" 5 "$variant" "$doc"
    done <<< "$noisy"
fi

step="finish"
if ((!dry)); then
    cleanup
    touch "$out/done.flag"
fi
