#!/bin/bash -e
# Installs pytraj (github.com/Amber-MD/pytraj) from source into the active
# Python environment. PyPI's pytraj==2.0.5 (2020) doesn't build against
# modern Python/numpy/cython; pytraj's git master tracks cpptraj's current
# API instead (the two commits below are a pair tested together).
#
# This script is only kept before the official release of the pytraj 3 series.
#
# Conda envs get their build dependencies installed here; venvs need the
# C/Fortran toolchain and netcdf/blas/zlib/bzip2 from the system.
#
# Usage:
#   micromamba activate pytraj_env  &&  bash instant_pytraj.sh
#   source .venv/bin/activate       &&  bash instant_pytraj.sh

PY=$(command -v python || command -v python3) || {
  echo "No python interpreter on PATH" >&2
  exit 1
}

# Use sys.prefix to identify the environment prefix
PREFIX=$("$PY" -c 'import sys; print(sys.prefix)')

if [ "$PREFIX" = "${VIRTUAL_ENV:-}" ]; then
  ENV_KIND=venv
elif [ "$PREFIX" = "${CONDA_PREFIX:-}" ]; then
  ENV_KIND=conda
else
  echo "No environment is active: activate a conda/micromamba env or a venv" >&2
  echo "first, so pytraj is not installed into the system Python." >&2
  exit 1
fi
echo "Installing pytraj into the $ENV_KIND environment at $PREFIX"

if [ "$ENV_KIND" = conda ]; then
  # Assumes netcdf4, openblas, zlib, bzip2, make and gcc/g++/gfortran in $PREFIX.
  for candidate in micromamba mamba conda; do
    MGR=$(command -v "$candidate") && break
  done
  [ -n "${MGR:-}" ] || { echo "No micromamba, mamba or conda on PATH" >&2; exit 1; }
  "$MGR" install -y -p "$PREFIX" -c conda-forge cython numpy setuptools

  # configure hardcodes -L$PREFIX/lib64 whichever one conda used (usually lib).
  mkdir -p "$PREFIX/lib64"
  for f in "$PREFIX"/lib/*; do
    [ -e "$f" ] && ln -sf "$f" "$PREFIX/lib64/$(basename "$f")"
  done

  # OpenBLAS ships no libblas.so/liblapack.so; alias them at the same soname.
  for lib in "$PREFIX"/lib/libopenblas.so*; do
    [ -e "$lib" ] || continue
    suffix=${lib##*/libopenblas.so}
    ln -sf "$(basename "$lib")" "$PREFIX/lib/libblas.so$suffix"
    ln -sf "$(basename "$lib")" "$PREFIX/lib/liblapack.so$suffix"
  done

  export LD_LIBRARY_PATH="$PREFIX/lib64:$PREFIX/lib:${LD_LIBRARY_PATH:-}"

  CONFIGURE_ARGS=(--with-netcdf="$PREFIX" --with-blas="$PREFIX"
                  --with-zlib="$PREFIX" --with-bzlib="$PREFIX")
else
  "$PY" -m pip install cython numpy setuptools

  # System libraries: configure's default search paths, unless SYS_PREFIX says
  # they live somewhere else.
  CONFIGURE_ARGS=()
  if [ -n "${SYS_PREFIX:-}" ]; then
    CONFIGURE_ARGS=(--with-netcdf="$SYS_PREFIX" --with-blas="$SYS_PREFIX"
                    --with-zlib="$SYS_PREFIX" --with-bzlib="$SYS_PREFIX")
  fi

  # Finding no HDF5, configure offers to build a bundled one and the piped "n"
  # makes that a hard error. AMBER .nc is NetCDF-3, so drop it; WITH_HDF5=1
  # keeps it where the system has one configure can find.
  [ "${WITH_HDF5:-0}" = 1 ] || CONFIGURE_ARGS+=(-nohdf5)
fi

# Built under $PREFIX rather than a temp dir
rm -rf "$PREFIX/opt/cpptraj"
git clone --filter=tree:0 https://github.com/Amber-MD/cpptraj "$PREFIX/opt/cpptraj"
cd "$PREFIX/opt/cpptraj"
git checkout 19a0bb7fd63396bcf274df34bf51f55e9f5db671
# Auto-answer the interactive FFTW prompt (unused here; no --with-fftw3).
yes n | bash configure -shared -openmp -noarpack "${CONFIGURE_ARGS[@]}" gnu
make libcpptraj -j"$(nproc)"
export CPPTRAJHOME="$PREFIX/opt/cpptraj"

START_DIR="$PWD"
BUILD=$(mktemp -d)
cd "$BUILD"
git clone --filter=tree:0 https://github.com/Amber-MD/pytraj
cd pytraj
git checkout 96083c77ee6f355a6cbffd66518401e832a8f8c2
export CFLAGS="-O2 -fPIC"
export CXXFLAGS="-O2 -fPIC"
# --no-build-isolation keeps CPPTRAJHOME and the flags above visible.
"$PY" -m pip install --no-build-isolation .
cd "$START_DIR"
rm -rf "$BUILD"

# Check the installation
"$PY" -c "import pytraj; print('pytraj installed:', pytraj.__version__)"
