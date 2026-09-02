#!/bin/bash
# format_cpp.sh — format the C++/CUDA sources with clang-format
# Usage: ./format_cpp.sh [update|check]
# Default: check
#
# NOTE: src/ has not been reformatted yet, so `check` currently reports every
# file. Run `update` once to adopt the style before wiring this into CI.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-check}"
# Major version only: clang-format's output changes between major releases,
# but not within one.
REQUIRED_CLANG_FORMAT_MAJOR="23"

if [[ "${MODE}" != "update" && "${MODE}" != "check" ]]; then
    echo "Usage: $0 [update|check]" >&2
    exit 2
fi

if ! command -v clang-format >/dev/null 2>&1; then
    echo "clang-format is not installed." >&2
    echo "Install it with: micromamba install -c conda-forge clang-format=${REQUIRED_CLANG_FORMAT_MAJOR}" >&2
    exit 2
fi

CLANG_FORMAT_VERSION="$(clang-format --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)"
CLANG_FORMAT_MAJOR="${CLANG_FORMAT_VERSION%%.*}"

if [[ "${CLANG_FORMAT_MAJOR}" != "${REQUIRED_CLANG_FORMAT_MAJOR}" ]]; then
    echo "clang-format ${REQUIRED_CLANG_FORMAT_MAJOR}.x is required; found ${CLANG_FORMAT_VERSION}." >&2
    exit 2
fi

mapfile -d '' CPP_FILES < <(
    find "${REPO_ROOT}" \
        \( -name ".git" -o -name "build" -o -name "dist" \) -prune -o \
        -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) -print0
)

if (( ${#CPP_FILES[@]} == 0 )); then
    echo "No C++/CUDA files found in ${REPO_ROOT}."
    exit 0
fi

if [[ "${MODE}" == "check" ]]; then
    echo "Checking formatting..."
    clang-format --dry-run --Werror "${CPP_FILES[@]}"
    exit "$?"
fi

echo "Formatting..."
clang-format -i "${CPP_FILES[@]}"
echo "Formatted ${#CPP_FILES[@]} files."
