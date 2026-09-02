#!/bin/bash
# format_python.sh — lint and format Python scripts with Ruff
# Usage: ./format_python.sh [update|check]
# Default: check

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-check}"
REQUIRED_RUFF_VERSION="0.16.5"

if [[ "${MODE}" != "update" && "${MODE}" != "check" ]]; then
    echo "Usage: $0 [update|check]" >&2
    exit 2
fi

if ! command -v ruff >/dev/null 2>&1; then
    echo "Ruff is not installed." >&2
    echo "Install it with: python -m pip install ruff==${REQUIRED_RUFF_VERSION}" >&2
    exit 2
fi

RUFF_VERSION="$(ruff --version | awk '{print $2}')"

if [[ "${RUFF_VERSION}" != "${REQUIRED_RUFF_VERSION}" ]]; then
    echo "Ruff ${REQUIRED_RUFF_VERSION} is required; found ${RUFF_VERSION}." >&2
    exit 2
fi

mapfile -d '' PYTHON_FILES < <(
    find "${REPO_ROOT}" \
        \( -name ".git" -o -name "__pycache__" -o -name "*.egg-info" -o -name "build" -o -name "dist" -o -name ".ruff_cache" -o -name ".pytest_cache" \) -prune -o \
        -type f -name "*.py" -print0
)

if (( ${#PYTHON_FILES[@]} == 0 )); then
    echo "No Python files found in ${REPO_ROOT}."
    exit 0
fi

if [[ "${MODE}" == "check" ]]; then
    status=0

    echo "Checking lint rules..."
    ruff check "${PYTHON_FILES[@]}" || status=1

    echo "Checking formatting..."
    ruff format --check "${PYTHON_FILES[@]}" || status=1

    exit "${status}"
fi

echo "Applying lint fixes..."
ruff check --fix "${PYTHON_FILES[@]}"

echo "Formatting..."
ruff format "${PYTHON_FILES[@]}"