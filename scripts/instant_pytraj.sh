#!/bin/bash -e
# Installs pytraj (github.com/Amber-MD/pytraj) from source into the active
# Python environment. PyPI's pytraj==2.0.5 (2020) doesn't build against
# modern Python/numpy/cython; pytraj's git master tracks cpptraj's current
# API instead (the two commits below are a pair tested together).
#
# This script is only kept before the official release of the pytraj 3 series.
#
# Usage:
#   micromamba activate pytraj_env
#   bash instant_pytraj.sh

PREFIX="${CONDA_PREFIX:?Activate the conda/micromamba env to install pytraj into first}"

# Assumes netcdf4, openblas, zlib, bzip2, make, and a C/C++/Fortran
# toolchain (gcc, g++, gfortran) are already in $PREFIX.
micromamba install -y -c conda-forge cython numpy setuptools

# cpptraj's ./configure hardcodes -L$PREFIX/lib64 regardless of whether
# conda installed to lib/ or lib64/ (usually lib/) -- mirror one into the
# other so the checks find them.
mkdir -p "$PREFIX/lib64"
for f in "$PREFIX"/lib/*; do
  [ -e "$f" ] && ln -sf "$f" "$PREFIX/lib64/$(basename "$f")"
done

# OpenBLAS doesn't ship the libblas.so/liblapack.so names configure looks
# for -- alias them (same soname suffix, e.g. .so.0).
for lib in "$PREFIX"/lib/libopenblas.so*; do
  [ -e "$lib" ] || continue
  suffix=${lib##*/libopenblas.so}
  ln -sf "$(basename "$lib")" "$PREFIX/lib/libblas.so$suffix"
  ln -sf "$(basename "$lib")" "$PREFIX/lib/liblapack.so$suffix"
done

export LD_LIBRARY_PATH="$PREFIX/lib64:$PREFIX/lib:$LD_LIBRARY_PATH"

# --- cpptraj --- built in-place under $PREFIX, not a temp dir: pytraj's
# compiled extension bakes in an rpath to this exact location.
rm -rf "$PREFIX/opt/cpptraj"
git clone --filter=tree:0 https://github.com/Amber-MD/cpptraj "$PREFIX/opt/cpptraj"
cd "$PREFIX/opt/cpptraj"
git checkout 19a0bb7fd63396bcf274df34bf51f55e9f5db671
# Auto-answer the interactive FFTW prompt (unused here; no --with-fftw3).
yes n | bash configure -shared -openmp -noarpack \
    --with-netcdf="$PREFIX" --with-blas="$PREFIX" \
    --with-zlib="$PREFIX" --with-bzlib="$PREFIX" \
    gnu
make libcpptraj -j"$(nproc)"
export CPPTRAJHOME="$PREFIX/opt/cpptraj"

# Step2: Install pytraj
# Built in a temp dir; only its source tree needs to survive. Captured
# before entering $BUILD.
START_DIR="$PWD"
BUILD=$(mktemp -d)
cd "$BUILD"
git clone --filter=tree:0 https://github.com/Amber-MD/pytraj
cd pytraj
git checkout 96083c77ee6f355a6cbffd66518401e832a8f8c2
# conda-forge's python bakes its own build-time flags (LTO-related, e.g.
# -ffat-lto-objects) into sysconfig; distutils reuses them by default, but
# they assume conda-forge's own gcc_linux-64/gxx_linux-64 activation scripts
# and error out otherwise (e.g. "unrecognized command-line option
# '-partition=none'"). Override with a clean, minimal set instead.
export CFLAGS="-O2 -fPIC"
export CXXFLAGS="-O2 -fPIC"
python setup.py install
cd "$START_DIR"
rm -rf "$BUILD"

# setuptools can exit 0 even when one extension failed to compile, silently
# leaving an incomplete install -- verify explicitly instead of trusting that.
python -c "import pytraj; print('pytraj installed:', pytraj.__version__)"
