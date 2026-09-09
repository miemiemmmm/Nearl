#!/bin/bash
# Launch one slurm job per (hash, label) pair, each running the single-hash
# SARS benchmark (scripts/run_benchmark_sars.sh) and appending to the CSV.
#
# The hash -> label map is defined below as an associative array. Each entry
# submits an independent slurm job (one per hash), so all hashes are
# benchmarked concurrently on separate nodes. Each job clones the repo at its
# hash into its own temp dir and builds its own CUDA extension, so the jobs do
# not interfere with each other or with the current checkout.
#
# Usage:
#   benchmark_sars.slurm
#
# The CSV (results/sars_benchmark_walltime.csv) is shared: each job appends its
# own rows. Because each job writes its own (git_ref, feature) rows and the
# git_ref column includes the unique hash, concurrent appends never collide.
set -u

# Resolve paths relative to this script so it can be run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEARL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_SCRIPT="$NEARL_DIR/scripts/run_benchmark_sars.sh"
LOG_DIR="$NEARL_DIR/results/slurm_logs"

# Slurm parameters (adjust account/partition/time to your allocation).
ACCOUNT="csstaff"
PARTITION="normal"
TIME="01:00:00"
GPUS_PER_NODE=4

# ---------------------------------------------------------------------------
# Hash -> label map. Each key is a git hash (full or short) or tag; each value
# is the human-readable label used in the CSV's git_ref column and the legend.
#
# This map is built from the GitHub PR list
# (https://github.com/miemiemmmm/Nearl/pulls), keeping only PRs that are
# compatible with scripts/benchmark_dataset.py. That script was introduced in
# PR #14, so only refs that are descendants of #14 (i.e. whose tree contains
# scripts/benchmark_dataset.py) can run the benchmark. For merged PRs the key
# is the merge_commit_sha; for open PRs it would be the head sha (none of the
# currently open PRs are compatible, as they branched before #14).
#
#   #14 b530384552  Parallel featurizer            (introduces the benchmark)
#   #17 ae932e2e40  Create slots for GPU/Pinned Host buffers  (== current main)
#   #20 c66fd0bfdb  Add the GIL fix to parallel featurizer (merged into
#                   parallel-featurizer branch, not main)
#
# v0.1.0 is kept as a release-point baseline (tag 4b630a7fe9ea...).
# ---------------------------------------------------------------------------
declare -A HASH_LABEL=(
    [4b630a7fe9]="v0.1.0"
    [b530384552]="PR14 parallel-featurizer"
    [ae932e2e40]="PR17 gpu-slots (main)"
    [c66fd0bfdb]="PR20 gil-fix"
    # WORKING benchmarks the current working tree (including uncommitted
    # changes, e.g. the GPU busy-time refactoring) in place.
    [WORKING]="current (working tree)"
    # [<hash-or-tag>]="<label>"
)

if [ "${#HASH_LABEL[@]}" -eq 0 ]; then
    echo "ERROR: HASH_LABEL map is empty; add at least one hash->label entry." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

for hash in "${!HASH_LABEL[@]}"; do
    label="${HASH_LABEL[$hash]}"
    # Sanitize the label for use in the slurm job name and log filename.
    safe_label="$(echo "$label" | tr -c 'A-Za-z0-9._-' '_')"
    job_name="nearl_bench_${safe_label}"
    log_file="$LOG_DIR/${safe_label}_%j.out"

    echo "Submitting job for $hash ($label) ..."
    # --uenv-passthrough=use: this launcher may itself be run from inside a uenv
    # session (sbatch refuses to submit otherwise). The job needs the loaded
    # uenv (nearl-dev) because it provides nvcc/CUDA, which is required to build
    # the CUDA extension (src/all_actions.so). The benchmark still uses its own
    # venv (nearl_env) for Python.
    sbatch \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --nodes=1 \
        --ntasks=1 \
        --gpus="$GPUS_PER_NODE" \
        --uenv-passthrough=use \
        --job-name="$job_name" \
        --output="$log_file" \
        --wrap="bash '$BENCH_SCRIPT' '$hash' '$label'"
done

echo "Submitted ${#HASH_LABEL[@]} slurm job(s). Logs in $LOG_DIR"
