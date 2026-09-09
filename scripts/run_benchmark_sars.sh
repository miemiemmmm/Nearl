#!/bin/bash
# Run the SARS-CoV-2 feature benchmark for a SINGLE git hash/tag and append the
# result to the CSV.
#
# Usage:
#   run_benchmark_sars.sh <hash-or-tag> <label>
#   run_benchmark_sars.sh WORKING <label>
#
#   <hash-or-tag>  A git commit hash (full or short) or a tag/branch name to
#                  benchmark. The repo is CLONED at this ref into a temp dir,
#                  so the current checkout is never touched.
#   WORKING        Special ref: benchmark the CURRENT working tree in place
#                  (the repo this script lives in), including any uncommitted
#                  local changes. No clone is made; the local src/all_actions.so
#                  is used as-is. The CSV's git_ref column is written as
#                  "<label>" (no hash, since the working tree has none).
#   <label>        A human-readable label for this ref (e.g. "v0.1.0",
#                  "parallel-featurizer"). It is combined with the hash into
#                  the CSV's git_ref column as "<hash> <label>".
#
# Example:
#   run_benchmark_sars.sh 4b630a7fe9ea8b9dd9f1187c43fba2ec45da32fb v0.1.0
#   run_benchmark_sars.sh WORKING "current (working tree)"
#
# The CSV's git_ref column is written as "<hash> <label>" (e.g.
# "4b630a7fe v0.1.0") so the notebook legend can show f'{hash} {label}'. For
# WORKING mode it is just "<label>".
#
# IMPORTANT (stale site-packages pitfall): the venv has nearl installed as a
# REAL COPY in site-packages. Instead of reinstalling per ref (which would race
# across concurrent slurm jobs sharing the venv), each clone builds its own
# CUDA extension (src/all_actions.so), copies it into the clone's nearl/
# package, and the benchmark runs with PYTHONPATH pointing at the clone root so
# `import nearl` resolves to the clone's code. We verify nearl.__file__ points
# into the clone before benchmarking.
set -u

# Resolve paths relative to this script so it can be run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEARL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$NEARL_DIR/../nearl_env/bin/activate"
REPO_URL="git@github.com:miemiemmmm/Nearl.git"
SARS_TRAJLIST="/capstor/scratch/cscs/course_00567/data/SARS-CoV-2/trajlist.txt"
OUT_DIR="/tmp/sars_bench"
CSV="$NEARL_DIR/results/sars_benchmark_walltime.csv"
CLONE_ROOT="${CLONE_ROOT:-/tmp/nearl_bench}"

FEATURES=(mass mo pdf mass_prot mo_prot pdf_prot)

# ---------------------------------------------------------------------------
# benchmark_dataset.py parameters. These are written to the CSV so every row
# records exactly how the benchmark was run. The values below are the ones
# passed on the command line; any parameter not listed here uses the script's
# argparse default (see scripts/benchmark_dataset.py).
# ---------------------------------------------------------------------------
TRAJLIST_FORMAT="paired"
DIMENSION=32
CUTOFF=2.5
SIGMA=1.5
LENGTH=24
# Parameters left at their argparse defaults (recorded in the CSV for clarity):
DATABASE_DIR=""            # -m/--database_dir
WINDOWSIZE=20              # -w/--windowsize
FOCUS_MASK=""              # --focus_mask
H5PREFIX="Output"          # --h5prefix
BASELINE_MAP="data/PDBBind_general_v2020.csv"  # --baseline_map
TASK_NR=1                  # --task_nr
TASK_INDEX=0               # --task_index
PRODUCER_THREADS="${PRODUCER_THREADS:-2}"   # --producer_threads (env-overridable)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hash-or-tag> <label>" >&2
    exit 1
fi
REF="$1"
LABEL="$2"

# Sanitize the label for use in a directory name (keep it readable but safe).
SAFE_LABEL="$(echo "$LABEL" | tr -c 'A-Za-z0-9._-' '_')"
CLONE_DIR="$CLONE_ROOT/${SAFE_LABEL}"

source "$VENV" || exit 1

# ---------------------------------------------------------------------------
# 1) Resolve the code to benchmark.
#
# WORKING mode uses the current working tree in place (no clone), so uncommitted
# local changes are benchmarked. Otherwise the repo is cloned at the requested
# ref into a temp dir.
# ---------------------------------------------------------------------------
if [ "$REF" = "WORKING" ]; then
    echo "=============================================="
    echo "WORKING TREE ($LABEL) -> $NEARL_DIR (in place)"
    echo "=============================================="
    CLONE_DIR="$NEARL_DIR"
    HASH=""
    # The working tree must have the CUDA extension built already.
    if [ ! -f "$CLONE_DIR/src/all_actions.so" ]; then
        echo "ERROR: src/all_actions.so not found in working tree; build it first (make all_actions in src/)" >&2
        exit 1
    fi
    if [ ! -f "$CLONE_DIR/nearl/all_actions.so" ]; then
        echo "  copying src/all_actions.so -> nearl/all_actions.so"
        cp "$CLONE_DIR/src/all_actions.so" "$CLONE_DIR/nearl/all_actions.so"
    fi
else
    echo "=============================================="
    echo "CLONE $REF ($LABEL) -> $CLONE_DIR"
    echo "=============================================="
    rm -rf "$CLONE_DIR"
    mkdir -p "$CLONE_ROOT"
    # Fetch all branches (not just the default branch) so that refs living on
    # non-default branches are reachable for checkout. Also fetch PR head refs
    # (refs/pull/*/head) so that commits merged into branches that were later
    # deleted from the remote (e.g. PR20's commit on parallel-featurizer) are
    # still reachable by hash.
    git clone --no-checkout --no-single-branch "$REPO_URL" "$CLONE_DIR" \
        || { echo "CLONE FAILED"; exit 1; }
    cd "$CLONE_DIR" || exit 1
    git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*' >/dev/null 2>&1 \
        || true
    git checkout "$REF" 2>/dev/null || { echo "CHECKOUT FAILED for $REF"; exit 1; }
    HASH="$(git rev-parse --short HEAD)"
    echo "Checked out $REF at hash $HASH"

    # -----------------------------------------------------------------------
    # 1b) Ensure scripts/benchmark_dataset.py exists in the clone.
    #
    # The benchmark script was introduced in PR #14, so refs that predate it
    # (e.g. the v0.1.0 tag, or any PR merged before #14) do not contain it. The
    # script is API-compatible with those older refs (verified: TrajectoryLoader
    # signature is identical, Featurizer ignores unknown parms like
    # producer_threads, and the feature constructors accept the same kwargs), so
    # we simply copy the current version into the clone when the ref lacks it.
    # This lets ANY ref be benchmarked without modifying git history.
    # -----------------------------------------------------------------------
    if [ ! -f "$CLONE_DIR/scripts/benchmark_dataset.py" ]; then
        echo "  scripts/benchmark_dataset.py missing in $REF; copying current version in"
        cp "$NEARL_DIR/scripts/benchmark_dataset.py" "$CLONE_DIR/scripts/benchmark_dataset.py"
    fi

    # -----------------------------------------------------------------------
    # 2) Build the CUDA extension from the clone's sources and place the .so
    #    where `import nearl` will find it (nearl/all_actions.so in the clone).
    # -----------------------------------------------------------------------
    echo "  Building CUDA extension ..."
    ( cd src && make clean >/dev/null 2>&1; make all_actions >/dev/null 2>&1 ) \
        || { echo "BUILD FAILED"; exit 1; }
    if [ ! -f src/all_actions.so ]; then
        echo "ERROR: src/all_actions.so was not produced" >&2
        exit 1
    fi
    cp src/all_actions.so nearl/all_actions.so
fi

# ---------------------------------------------------------------------------
# 3) Verify `import nearl` resolves to the code being benchmarked (the clone,
#    or the working tree in WORKING mode) — not the stale site-packages copy.
# ---------------------------------------------------------------------------
NEARL_FILE="$(PYTHONPATH="$CLONE_DIR:${PYTHONPATH:-}" python -c 'import nearl; print(nearl.__file__)')"
case "$NEARL_FILE" in
    "$CLONE_DIR"/*) echo "  nearl resolves to $NEARL_FILE" ;;
    *) echo "ERROR: nearl resolves to $NEARL_FILE, not $CLONE_DIR" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 3b) Detect driver capabilities.
#
# The benchmark driver (scripts/benchmark_dataset.py) gained two features in
# PR #24: the --producer_threads CLI arg and the GPU_BUSY_SECONDS output line.
# Older refs (v0.1.0, PR14/17/20/21) predate one or both. Probe the driver in
# the code being benchmarked and adapt:
#   - only pass --producer_threads if the driver accepts it
#   - only require gpu_busy if the driver emits GPU_BUSY_SECONDS
# Older refs are single-producer (pre-PR24), so when the driver lacks
# --producer_threads we record producer_threads=1 in the CSV for a fair
# comparison against the producer-worker refs.
# ---------------------------------------------------------------------------
DRIVER="$CLONE_DIR/scripts/benchmark_dataset.py"
if grep -q -- '--producer_threads' "$DRIVER"; then
    DRIVER_HAS_PRODUCER_THREADS=1
else
    DRIVER_HAS_PRODUCER_THREADS=0
fi
if grep -q 'GPU_BUSY_SECONDS' "$DRIVER"; then
    DRIVER_HAS_GPU_BUSY=1
else
    DRIVER_HAS_GPU_BUSY=0
fi
if [ "$DRIVER_HAS_PRODUCER_THREADS" -eq 1 ]; then
    CSV_PRODUCER_THREADS="$PRODUCER_THREADS"
else
    CSV_PRODUCER_THREADS=1
fi

# ---------------------------------------------------------------------------
# 4) Append to the CSV (create header only if the file does not exist yet, so
#    repeated invocations APPEND).
#
# The header check-and-write is protected by an flock because the slurm wrapper
# (scripts/slurm/benchmark_sars.slurm) launches one job per ref concurrently.
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$CSV")" "$OUT_DIR"
(
    flock 9
    if [ ! -s "$CSV" ]; then
        echo "hash,label,feature_type,wall_time_s,gpu_busy_s,trajlist,trajlist_format,database_dir,output_dir,dimension,length,cutoff,sigma,windowsize,focus_mask,h5prefix,baseline_map,task_nr,task_index,producer_threads" > "$CSV"
    fi
) 9>"$CSV.lock"

# The CSV stores the git hash and the human-readable label in SEPARATE columns
# (hash, label) so the notebook can combine them into a "ref" key for plotting.
# The hash is emitted separately in the hash column; the label column keeps
# only the human-readable label.
GIT_REF="$LABEL"

# ---------------------------------------------------------------------------
# 5) Run the 6-feature suite in parallel, one process per local GPU.
# ---------------------------------------------------------------------------
run_feature_suite() {
    # $1 = label for the CSV (git_ref column)
    local label="$1"
    local -a pids=()
    local slot=0
    for feat in "${FEATURES[@]}"; do
        local gpu=$((slot % 4))
        mkdir -p "$OUT_DIR/gpu$slot"
        rm -f "$OUT_DIR/gpu$slot/Output0.h5"
        (
            # The benchmark prints "BENCHMARK_RUN_SECONDS=<t>" and
            # "GPU_BUSY_SECONDS=<t>" on stdout after feat.run() completes; this
            # measures only the featurization (the CUDA kernels synchronize
            # internally, so all GPU work is included) and excludes interpreter
            # startup and trajectory-list setup. GPU_BUSY_SECONDS is the
            # accumulated dispatch-to-collection window, i.e. how much of the
            # wall time the GPU was actually busy.
            outfile=$(mktemp)
            local producer_args=()
            if [ "$DRIVER_HAS_PRODUCER_THREADS" -eq 1 ]; then
                producer_args=(--producer_threads "$PRODUCER_THREADS")
            fi
            CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$CLONE_DIR:${PYTHONPATH:-}" python scripts/benchmark_dataset.py \
                -f "$SARS_TRAJLIST" --trajlist_format "$TRAJLIST_FORMAT" -o "$OUT_DIR/gpu$slot" -t "$feat" \
                -d "$DIMENSION" -c "$CUTOFF" -s "$SIGMA" -l "$LENGTH" \
                "${producer_args[@]}" >"$outfile" 2>&1
            rc=$?
            wall=$(grep -oE 'BENCHMARK_RUN_SECONDS=[0-9]+\.[0-9]+' "$outfile" | cut -d= -f2 | tail -1)
            gpu_busy=$(grep -oE 'GPU_BUSY_SECONDS=[0-9]+\.[0-9]+' "$outfile" | cut -d= -f2 | tail -1)
            # Reliability guard: only append a row when the benchmark actually
            # succeeded AND produced parseable timings. Otherwise we would
            # silently write a row of empty/invalid values that corrupts the
            # CSV and any downstream analysis.
            if [ "$rc" -ne 0 ] || [ -z "$wall" ] || { [ "$DRIVER_HAS_GPU_BUSY" -eq 1 ] && [ -z "$gpu_busy" ]; }; then
                echo "  $label / $feat : FAILED (rc=$rc, wall='$wall', gpu_busy='$gpu_busy')" >&2
                rm -f "$outfile"
                exit 1
            fi
            rm -f "$outfile"
            # Append one row with the git hash, label, feature, wall time, GPU
            # busy time, and every benchmark_dataset.py parameter so the CSV is
            # self-describing.
            echo "$HASH,$label,$feat,$wall,$gpu_busy,$SARS_TRAJLIST,$TRAJLIST_FORMAT,$DATABASE_DIR,$OUT_DIR/gpu$slot,$DIMENSION,$LENGTH,$CUTOFF,$SIGMA,$WINDOWSIZE,$FOCUS_MASK,$H5PREFIX,$BASELINE_MAP,$TASK_NR,$TASK_INDEX,$CSV_PRODUCER_THREADS" >> "$CSV"
            if [ "$DRIVER_HAS_GPU_BUSY" -eq 1 ]; then
                echo "  $label / $feat : ${wall}s (gpu ${gpu_busy}s)"
            else
                echo "  $label / $feat : ${wall}s"
            fi
        ) &
        pids+=("$!")
        slot=$((slot + 1))
        # Once 4 are in flight (one per GPU), wait for them before launching
        # the next batch.
        if [ "${#pids[@]}" -ge 4 ]; then
            wait "${pids[@]}" || return 1
            pids=()
        fi
    done
    if [ "${#pids[@]}" -gt 0 ]; then
        wait "${pids[@]}" || return 1
    fi
}

run_feature_suite "$GIT_REF" || exit 1

echo "DONE. Appended $GIT_REF to $CSV"
